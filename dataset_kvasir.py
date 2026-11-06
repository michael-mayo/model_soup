"""Download and prepare Kvasir-SEG for Ultralytics object detection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import yaml

import log


CONFIG_PATH = Path(__file__).with_name("config.yaml")
DEFAULT_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def load_settings(config_path: Path = CONFIG_PATH) -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return config, config.get("dataset", {}).get("kvasir", {})


def _ca_bundle() -> Path | None:
    candidates = list((Path(sys.prefix) / "lib").glob("python*/site-packages/pip/_vendor/certifi/cacert.pem"))
    candidates.append(Path(sys.prefix) / "ssl" / "cacert.pem")
    return next((path for path in candidates if path.is_file() and path.stat().st_size > 1000), None)


def _download(url: str, destination: Path, *, verify_ssl: bool = True) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    log.message(__file__, f"downloading dataset to {destination}")
    partial = destination.with_suffix(destination.suffix + ".part")
    destination.unlink(missing_ok=True)
    try:
        ca_bundle = _ca_bundle()
        context = (
            ssl.create_default_context(cafile=str(ca_bundle))
            if verify_ssl and ca_bundle
            else ssl.create_default_context()
            if verify_ssl
            else ssl._create_unverified_context()
        )
        if not verify_ssl:
            log.message(__file__, "WARNING: SSL certificate verification disabled by --insecure")
        request = urllib.request.Request(url, headers={"User-Agent": "soup-dataset-downloader/1.0"})
        with urllib.request.urlopen(request, context=context) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        partial.replace(destination)
        if not zipfile.is_zipfile(destination):
            destination.unlink(missing_ok=True)
            raise ValueError("downloaded file is not a valid ZIP archive")
    except Exception:
        partial.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"unsafe archive path: {member.filename}")
        zipped.extractall(destination)


def _find_directory(root: Path, name: str) -> Path:
    matches = [path for path in root.rglob("*") if path.is_dir() and path.name.lower() == name.lower()]
    if not matches:
        raise FileNotFoundError(f"could not find {name}/ in extracted Kvasir-SEG")
    return min(matches, key=lambda path: len(path.parts))


def _find_bbox_json(root: Path) -> Path:
    candidates = [
        path for path in root.rglob("*.json")
        if "bbox" in path.name.lower() or "box" in path.name.lower()
    ]
    if not candidates:
        raise FileNotFoundError("could not find Kvasir-SEG bounding-box JSON")
    return min(candidates, key=lambda path: len(path.parts))


def _image_size(path: Path) -> tuple[int, int]:
    """Read image dimensions for JPEG/PNG without adding an imaging dependency."""

    with path.open("rb") as stream:
        signature = stream.read(24)
        if signature.startswith(b"\x89PNG") and len(signature) >= 24:
            return int.from_bytes(signature[16:20], "big"), int.from_bytes(signature[20:24], "big")
        if signature[:2] != b"\xff\xd8":
            raise ValueError(f"unsupported image format for {path.name}")
        stream.seek(2)
        while True:
            marker_start = stream.read(1)
            if not marker_start:
                break
            if marker_start != b"\xff":
                continue
            marker = stream.read(1)
            while marker == b"\xff":
                marker = stream.read(1)
            if marker in {b"\xd8", b"\xd9"}:
                continue
            length_bytes = stream.read(2)
            if len(length_bytes) != 2:
                break
            length = int.from_bytes(length_bytes, "big")
            if marker[0] in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                stream.read(1)
                height = int.from_bytes(stream.read(2), "big")
                width = int.from_bytes(stream.read(2), "big")
                return width, height
            stream.seek(length - 2, 1)
    raise ValueError(f"could not read image dimensions for {path.name}")


def _bbox_values(annotation: dict[str, Any]) -> tuple[float, float, float, float]:
    if all(key in annotation for key in ("xmin", "ymin", "xmax", "ymax")):
        return tuple(float(annotation[key]) for key in ("xmin", "ymin", "xmax", "ymax"))  # type: ignore[return-value]
    if "bbox" in annotation:
        values = annotation["bbox"]
        if isinstance(values, list) and len(values) == 4:
            x, y, third, fourth = map(float, values)
            if annotation.get("format") in {"xywh", "coco"}:
                return x, y, x + third, y + fourth
            return x, y, third, fourth
    raise ValueError(f"unrecognized Kvasir bbox: {annotation}")


def _annotations(json_path: Path) -> dict[str, list[dict[str, Any]]]:
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    result: dict[str, list[dict[str, Any]]] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        boxes = value.get("bbox", [])
        if isinstance(boxes, dict):
            boxes = [boxes]
        if isinstance(boxes, list):
            result[Path(str(key)).stem.lower()] = [box for box in boxes if isinstance(box, dict)]
    return result


def _split(stem: str) -> str:
    bucket = int(hashlib.sha256(stem.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


def _prepare(raw_root: Path, output_root: Path) -> dict[str, int]:
    images_root = _find_directory(raw_root, "images")
    bbox_json = _find_bbox_json(raw_root)
    annotations = _annotations(bbox_json)
    image_files = sorted(path for path in images_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not image_files:
        raise FileNotFoundError("no images found in extracted Kvasir-SEG")

    counts = {split: 0 for split in SPLITS}
    for image in image_files:
        width, height = _image_size(image)
        split = _split(image.stem.lower())
        image_out = output_root / split / "images" / image.name
        label_out = output_root / split / "labels" / f"{image.stem}.txt"
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, image_out)

        labels: list[str] = []
        for annotation in annotations.get(image.stem.lower(), []):
            x_min, y_min, x_max, y_max = _bbox_values(annotation)
            x_min, x_max = max(0.0, min(x_min, width)), max(0.0, min(x_max, width))
            y_min, y_max = max(0.0, min(y_min, height)), max(0.0, min(y_max, height))
            if x_max <= x_min or y_max <= y_min:
                continue
            x_center = ((x_min + x_max) / 2) / width
            y_center = ((y_min + y_max) / 2) / height
            box_width = (x_max - x_min) / width
            box_height = (y_max - y_min) / height
            labels.append(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
        label_out.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        counts[split] += 1
    return counts


def prepare(config_path: Path = CONFIG_PATH, *, force: bool = False, verify_ssl: bool | None = None) -> Path:
    config, settings = load_settings(config_path)
    project_root = config_path.parent
    paths = config.get("paths", {})
    downloads_root = project_root / paths.get("downloads", "scratch/downloads")
    datasets_root = project_root / paths.get("datasets", "scratch/datasets")
    url = settings.get("url", DEFAULT_URL)
    if verify_ssl is None:
        verify_ssl = bool(settings.get("verify_ssl", True))
    dataset_root = datasets_root / "kvasir"
    raw_root = dataset_root / "raw"
    output_root = dataset_root / "yolo"
    archive = downloads_root / "kvasir-seg.zip"
    dataset_yaml = output_root / "data.yaml"

    if dataset_yaml.exists() and not force:
        log.message(__file__, f"dataset already prepared at {output_root}")
        return dataset_yaml
    if force and output_root.exists():
        shutil.rmtree(output_root)
    if force or not zipfile.is_zipfile(archive):
        _download(url, archive, verify_ssl=verify_ssl)
    extracted_images = any(path.is_dir() and path.name.lower() == "images" for path in raw_root.rglob("*")) if raw_root.exists() else False
    if force or not extracted_images:
        _safe_extract(archive, raw_root)

    counts = _prepare(raw_root, output_root)
    dataset_yaml.write_text(yaml.safe_dump({
        "path": str(output_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {0: "polyp"},
    }, sort_keys=False), encoding="utf-8")
    log.message(__file__, f"prepared {sum(counts.values())} images: {counts}")
    log.message(__file__, f"wrote {dataset_yaml}")
    return dataset_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--insecure", action="store_true", help="disable SSL certificate verification for the download")
    args = parser.parse_args()
    try:
        prepare(args.config, force=args.force, verify_ssl=False if args.insecure else None)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        log.error(__file__, str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

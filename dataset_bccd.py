"""Download and prepare BCCD for Ultralytics object detection."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import ssl
import sys
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml

import log


CONFIG_PATH = Path(__file__).with_name("config.yaml")
DEFAULT_URL = "https://github.com/Shenggan/BCCD_Dataset/archive/refs/heads/master.tar.gz"
SPLITS = ("train", "val", "test")
CLASS_NAMES = ("RBC", "WBC", "Platelets")
CLASS_IDS = {name.lower(): index for index, name in enumerate(CLASS_NAMES)}


def load_settings(config_path: Path = CONFIG_PATH) -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return config, config.get("datasets", {}).get("bccd", {})


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
        if not tarfile.is_tarfile(destination):
            destination.unlink(missing_ok=True)
            raise ValueError("downloaded file is not a valid tar archive")
    except Exception:
        partial.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"unsafe archive path: {member.name}")
        tar.extractall(destination, filter="data")


def _find_directory(root: Path, name: str) -> Path:
    matches = [path for path in root.rglob("*") if path.is_dir() and path.name.lower() == name.lower()]
    if not matches:
        raise FileNotFoundError(f"could not find {name}/ in extracted BCCD")
    return min(matches, key=lambda path: len(path.parts))


def _split_names(split_file: Path) -> set[str]:
    return {line.strip().lower() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()}


def _fallback_split(stem: str) -> str:
    bucket = int(hashlib.sha256(stem.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "val" if bucket < 85 else "test"


def _prepare(raw_root: Path, output_root: Path) -> dict[str, int]:
    annotations_root = _find_directory(raw_root, "Annotations")
    images_root = _find_directory(raw_root, "JPEGImages")
    split_root = _find_directory(raw_root, "Main")
    split_names = {
        split: _split_names(split_root / f"{split}.txt")
        for split in SPLITS
        if (split_root / f"{split}.txt").exists()
    }
    image_by_stem = {path.stem.lower(): path for path in images_root.glob("*") if path.is_file()}
    xml_files = sorted(annotations_root.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError("no Pascal VOC XML annotations found in BCCD")

    counts = {split: 0 for split in SPLITS}
    for xml_path in xml_files:
        root = ET.parse(xml_path).getroot()
        stem = xml_path.stem.lower()
        image = image_by_stem.get(stem)
        if image is None:
            continue
        split = next((candidate for candidate in SPLITS if stem in split_names.get(candidate, set())), None)
        split = split or _fallback_split(stem)
        size = root.find("size")
        width = int(size.findtext("width", "0")) if size is not None else 0
        height = int(size.findtext("height", "0")) if size is not None else 0
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid image dimensions in {xml_path}")

        labels: list[str] = []
        for object_node in root.findall("object"):
            name = (object_node.findtext("name") or "").strip().lower()
            if name not in CLASS_IDS:
                raise ValueError(f"unknown BCCD class {name!r} in {xml_path.name}")
            box = object_node.find("bndbox")
            if box is None:
                continue
            x_min = max(0, min(float(box.findtext("xmin", "0")), width))
            y_min = max(0, min(float(box.findtext("ymin", "0")), height))
            x_max = max(0, min(float(box.findtext("xmax", "0")), width))
            y_max = max(0, min(float(box.findtext("ymax", "0")), height))
            if x_max <= x_min or y_max <= y_min:
                continue
            labels.append(
                f"{CLASS_IDS[name]} {((x_min + x_max) / 2 / width):.6f} "
                f"{((y_min + y_max) / 2 / height):.6f} "
                f"{((x_max - x_min) / width):.6f} {((y_max - y_min) / height):.6f}"
            )

        image_out = output_root / split / "images" / image.name
        label_out = output_root / split / "labels" / f"{image.stem}.txt"
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, image_out)
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
    dataset_root = datasets_root / "bccd"
    raw_root = dataset_root / "raw"
    output_root = dataset_root / "yolo"
    archive = downloads_root / "bccd.tar.gz"
    dataset_yaml = output_root / "data.yaml"

    if dataset_yaml.exists() and not force:
        log.message(__file__, f"dataset already prepared at {output_root}")
        return dataset_yaml
    if force and output_root.exists():
        shutil.rmtree(output_root)
    if force or not archive.exists() or not tarfile.is_tarfile(archive):
        _download(url, archive, verify_ssl=verify_ssl)
    extracted_annotations = any(path.is_dir() and path.name.lower() == "annotations" for path in raw_root.rglob("*")) if raw_root.exists() else False
    if force or not extracted_annotations:
        _safe_extract(archive, raw_root)

    counts = _prepare(raw_root, output_root)
    dataset_yaml.write_text(yaml.safe_dump({
        "path": str(output_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
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
    except (OSError, ValueError, tarfile.TarError) as exc:
        log.error(__file__, str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

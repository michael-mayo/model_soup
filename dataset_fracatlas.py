"""Download and prepare FracAtlas for Ultralytics object detection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

import yaml

import log


CONFIG_PATH = Path(__file__).with_name("config.yaml")
DEFAULT_URL = "https://ndownloader.figshare.com/files/43283628"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def _ca_bundle() -> Path | None:
    candidates = list((Path(sys.prefix) / "lib").glob("python*/site-packages/pip/_vendor/certifi/cacert.pem"))
    candidates.append(Path(sys.prefix) / "ssl" / "cacert.pem")
    return next((path for path in candidates if path.is_file() and path.stat().st_size > 1000), None)


def load_settings(config_path: Path = CONFIG_PATH) -> dict:
    """Load the FracAtlas settings from the central project configuration."""

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return config.get("dataset", {}).get("fracatlas", {})


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
        raise FileNotFoundError(f"could not find {name}/ in extracted FracAtlas")
    return min(matches, key=lambda path: len(path.parts))


def _find_split_file(root: Path, split: str) -> Path | None:
    names = {f"{split}.csv", "valid.csv" if split == "val" else f"{split}.csv"}
    matches = [path for path in root.rglob("*") if path.is_file() and path.name.lower() in names]
    return min(matches, key=lambda path: len(path.parts)) if matches else None


def _split_names(split_file: Path) -> set[str]:
    names: set[str] = set()
    with split_file.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.reader(stream):
            for value in row:
                value = value.strip()
                if value and Path(value).suffix.lower() in IMAGE_SUFFIXES:
                    names.add(Path(value).name.lower())
                    break
    return names


def _fallback_split(image_name: str) -> str:
    """Provide a stable 80/10/10 split if the archive split files are absent."""

    bucket = int(hashlib.sha256(image_name.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


def _copy_dataset(raw_root: Path, output_root: Path) -> dict[str, int]:
    images_root = _find_directory(raw_root, "images")
    annotations_root = _find_directory(raw_root, "Annotations")
    yolo_root = _find_directory(annotations_root, "YOLO")

    image_files = sorted(
        path for path in images_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_files:
        raise FileNotFoundError("no image files found in FracAtlas images/")

    split_names: dict[str, set[str]] = {}
    for split in SPLITS:
        split_file = _find_split_file(raw_root, split)
        if split_file:
            split_names[split] = _split_names(split_file)

    counts = {split: 0 for split in SPLITS}
    for image in image_files:
        image_name = image.name.lower()
        split = next((candidate for candidate in SPLITS if image_name in split_names.get(candidate, set())), None)
        split = split or _fallback_split(image_name)

        destination_image = output_root / split / "images" / image.name
        destination_label = output_root / split / "labels" / f"{image.stem}.txt"
        destination_image.parent.mkdir(parents=True, exist_ok=True)
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, destination_image)

        source_label = next(yolo_root.rglob(f"{image.stem}.txt"), None)
        if source_label:
            shutil.copy2(source_label, destination_label)
        else:
            destination_label.write_text("", encoding="utf-8")
        counts[split] += 1

    return counts


def prepare(config_path: Path = CONFIG_PATH, *, force: bool = False, verify_ssl: bool | None = None) -> Path:
    """Download and prepare FracAtlas, returning its Ultralytics YAML path."""

    settings = load_settings(config_path)
    full_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    url = settings.get("url", DEFAULT_URL)
    if verify_ssl is None:
        verify_ssl = bool(settings.get("verify_ssl", True))
    project_root = config_path.parent
    paths = full_config.get("paths", {})
    downloads_root = project_root / paths.get("downloads", "scratch/downloads")
    datasets_root = project_root / paths.get("datasets", "scratch/datasets")
    dataset_root = datasets_root / "fracatlas"
    raw_root = dataset_root / "raw"
    output_root = dataset_root / "yolo"
    archive = downloads_root / "fracatlas.zip"
    dataset_yaml = output_root / "data.yaml"

    if dataset_yaml.exists() and not force:
        log.message(__file__, f"dataset already prepared at {output_root}")
        return dataset_yaml

    if force and output_root.exists():
        shutil.rmtree(output_root)
    if force or not zipfile.is_zipfile(archive):
        _download(url, archive, verify_ssl=verify_ssl)
    extracted_images = any(
        path.is_dir() and path.name.lower() == "images" for path in raw_root.rglob("*")
    ) if raw_root.exists() else False
    if force or not extracted_images:
        _safe_extract(archive, raw_root)

    counts = _copy_dataset(raw_root, output_root)
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(output_root.resolve()),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "names": {0: "fracture"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    log.message(__file__, f"prepared {sum(counts.values())} images: {counts}")
    log.message(__file__, f"wrote {dataset_yaml}")
    return dataset_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--force", action="store_true", help="redownload and rebuild the prepared dataset")
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

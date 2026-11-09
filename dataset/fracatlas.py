"""FracAtlas bone-fracture X-ray dataset (YOLO annotations)."""

from __future__ import annotations

import csv
from pathlib import Path

from dataset.base import (
    IMAGE_SUFFIXES, SPLITS, Dataset, YoloWriter, find_directory, index_by_stem, list_images, unique,
)


# FracAtlas names its split files train/valid/test.csv.
SPLIT_FILES = {"train": "train.csv", "val": "valid.csv", "test": "test.csv"}


def _split_names(split_file: Path) -> set[str]:
    """Return the lower-cased image names listed in a FracAtlas split CSV."""

    names: set[str] = set()
    with split_file.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.reader(stream):
            value = next((cell.strip() for cell in row if Path(cell.strip()).suffix.lower() in IMAGE_SUFFIXES), None)
            if value:
                names.add(Path(value).stem.lower())
    return names


class FracAtlas(Dataset):
    """Convert FracAtlas.

    The official split CSVs only cover fractured images, so images without
    fractures are assigned by the hash-based fallback split.
    """

    key = "fracatlas"
    archive_name = "fracatlas.zip"

    def convert(self, raw: Path, writer: YoloWriter) -> dict[int, str]:
        """Copy every image with its YOLO label file (empty if none exists)."""

        images = list_images(find_directory(raw, "images"))
        if not images:
            raise FileNotFoundError("no image files found in FracAtlas images/")
        labels = index_by_stem(find_directory(find_directory(raw, "Annotations"), "YOLO").rglob("*.txt"))
        official: dict[str, set[str]] = {}
        for split in SPLITS:
            candidates = list(raw.rglob(SPLIT_FILES[split]))
            if candidates:
                official[split] = _split_names(min(candidates, key=lambda path: len(path.parts)))

        for image in images:
            label = unique(labels.get(image.stem.lower(), []), f"label for {image.name}")
            lines = label.read_text(encoding="utf-8").splitlines() if label else []
            writer.add(self.split_for(image.stem, official), image, lines)
        return {0: "fracture"}

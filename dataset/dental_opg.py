"""Dental OPG X-ray dataset (COCO JSON or YOLO annotations)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dataset.base import Dataset, YoloWriter, index_by_stem, list_images, unique, yolo_box


# Class names used when the archive ships YOLO labels without names.
CLASS_NAMES = ("Healthy Teeth", "Caries", "Impacted Teeth", "Infection", "Fractured Teeth", "BDC/BDR")

# Directory names that identify an official split.
SPLIT_ALIASES = {
    "train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val",
    "test": "test", "testing": "test",
}


def _directory_split(path: Path, raw: Path) -> str | None:
    """Return the split named by one of `path`'s parent directories, if any."""

    parts = (part.lower() for part in path.relative_to(raw).parts[:-1])
    return next((SPLIT_ALIASES[part] for part in parts if part in SPLIT_ALIASES), None)


def _find_coco(raw: Path) -> dict[str, Any] | None:
    """Return the COCO annotation file with the most annotations, if any."""

    best: dict[str, Any] | None = None
    for path in raw.rglob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and {"images", "annotations", "categories"} <= value.keys():
            if best is None or len(value["annotations"]) > len(best["annotations"]):
                best = value
    return best


class DentalOpg(Dataset):
    """Convert Dental OPG, using split directories when present."""

    key = "dental_opg"
    archive_name = "dental_opg_v4.zip"

    def _images(self, raw: Path) -> list[Path]:
        """Return the images of the object-detection part of the archive."""

        roots = [path for path in raw.rglob("*") if path.is_dir() and "object detection" in path.name.lower()]
        images = list_images(min(roots, key=lambda path: len(path.parts)) if roots else raw)
        if not images:
            raise FileNotFoundError("no Dental OPG images found in extracted archive")
        return images

    def _split(self, image: Path, raw: Path) -> str:
        """Return the image's directory split, falling back to the hash split."""

        return _directory_split(image, raw) or self.split_for(image.name)

    def convert(self, raw: Path, writer: YoloWriter) -> dict[int, str]:
        """Convert from COCO JSON if the archive has one, otherwise from YOLO labels."""

        coco = _find_coco(raw)
        return self._convert_yolo(raw, writer) if coco is None else self._convert_coco(raw, writer, coco)

    def _convert_yolo(self, raw: Path, writer: YoloWriter) -> dict[int, str]:
        """Copy images with their YOLO label files, matched by stem and split."""

        labels = index_by_stem(raw.rglob("*.txt"))
        for image in self._images(raw):
            split = self._split(image, raw)
            candidates = labels.get(image.stem.lower(), [])
            same_split = [path for path in candidates if _directory_split(path, raw) == split]
            label = unique(same_split or candidates, f"label for {image.name}")
            writer.add(split, image, label.read_text(encoding="utf-8").splitlines() if label else [])
        return dict(enumerate(CLASS_NAMES))

    def _convert_coco(self, raw: Path, writer: YoloWriter, coco: dict[str, Any]) -> dict[int, str]:
        """Convert COCO boxes, resolving image file names by relative path then by name."""

        images = self._images(raw)
        by_relative = {path.relative_to(raw).as_posix().lower(): path for path in images}
        by_name = index_by_stem(images)
        categories = sorted(coco["categories"], key=lambda category: int(category["id"]))
        class_ids = {int(category["id"]): index for index, category in enumerate(categories)}
        boxes: dict[int, list[dict[str, Any]]] = {}
        for annotation in coco["annotations"]:
            boxes.setdefault(int(annotation["image_id"]), []).append(annotation)

        for record in coco["images"]:
            file_name = str(record["file_name"]).replace("\\", "/").lstrip("./").lower()
            image = by_relative.get(file_name) or unique(
                [path for path in by_name.get(Path(file_name).stem, []) if path.name.lower() == Path(file_name).name],
                f"image {record['file_name']}",
            )
            if image is None:
                raise FileNotFoundError(f"image referenced by annotations was not found: {record['file_name']}")
            labels: list[str] = []
            for annotation in boxes.get(int(record["id"]), []):
                bbox = annotation.get("bbox")
                category = int(annotation["category_id"])
                if category not in class_ids or not isinstance(bbox, list) or len(bbox) < 4:
                    continue
                x, y, box_width, box_height = (float(value) for value in bbox[:4])
                line = yolo_box(
                    class_ids[category], x, y, x + box_width, y + box_height,
                    float(record["width"]), float(record["height"]),
                )
                if line:
                    labels.append(line)
            writer.add(self._split(image, raw), image, labels)
        return {index: str(category["name"]) for index, category in enumerate(categories)}

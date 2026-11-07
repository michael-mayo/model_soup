"""Prepare the Dental OPG X-ray dataset for Ultralytics object detection."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from dataset import IMAGE_SUFFIXES, Dataset, SPLITS


DEFAULT_URL = "https://data.mendeley.com/public-api/zip/c4hhrkxytw/download/4"
CLASS_NAMES = ("Healthy Teeth", "Caries", "Impacted Teeth", "Infection", "Fractured Teeth", "BDC/BDR")
SPLIT_ALIASES = {
    "train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val",
    "test": "test", "testing": "test",
}


class DatasetDentalOpg(Dataset):
    dataset_key = "dental_opg"
    default_url = DEFAULT_URL
    archive_name = "dental_opg_v4.zip"
    module_name = __file__

    @staticmethod
    def _split_for_image(image: Path, raw_root: Path) -> str | None:
        parts = {part.lower() for part in image.relative_to(raw_root).parts[:-1]}
        return next((SPLIT_ALIASES[part] for part in parts if part in SPLIT_ALIASES), None)

    @staticmethod
    def _find_coco_json(raw_root: Path) -> tuple[Path, dict[str, Any]] | None:
        candidates: list[tuple[Path, dict[str, Any]]] = []
        for path in raw_root.rglob("*.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(value, dict) and all(key in value for key in ("images", "annotations", "categories")):
                candidates.append((path, value))
        return max(candidates, key=lambda item: len(item[1].get("annotations", []))) if candidates else None

    @staticmethod
    def _image_index(raw_root: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
        roots = [path for path in raw_root.rglob("*") if path.is_dir() and "object detection" in path.name.lower()]
        image_root = min(roots, key=lambda path: len(path.parts)) if roots else raw_root
        images = sorted(path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise FileNotFoundError("no Dental OPG images found in extracted archive")
        by_relative = {path.relative_to(raw_root).as_posix().lower(): path for path in images}
        by_name: dict[str, list[Path]] = defaultdict(list)
        for path in images:
            by_name[path.name.lower()].append(path)
        return by_relative, by_name

    @staticmethod
    def _resolve_image(file_name: str, by_relative: dict[str, Path], by_name: dict[str, list[Path]]) -> Path:
        normalized = file_name.replace("\\", "/").lstrip("./").lower()
        if normalized in by_relative:
            return by_relative[normalized]
        matches = by_name.get(Path(normalized).name, [])
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(f"image referenced by annotations was not found: {file_name}")
        raise ValueError(f"image filename is ambiguous in annotations: {file_name}")

    def _find_yolo_label(self, image: Path, raw_root: Path, split: str) -> Path | None:
        candidates = sorted(raw_root.rglob(f"{image.stem}.txt"))
        if not candidates:
            return None
        same_split = [candidate for candidate in candidates if self._split_for_image(candidate, raw_root) == split]
        if len(same_split) == 1:
            return same_split[0]
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError(f"label filename is ambiguous for image: {image.name}")

    def _prepare_yolo(self, raw_root: Path, output_root: Path) -> tuple[dict[str, int], dict[int, str]]:
        _, by_name = self._image_index(raw_root)
        counts = {split: 0 for split in SPLITS}
        for candidates in by_name.values():
            for source in candidates:
                split = self._split_for_image(source, raw_root) or self._fallback_split(source.name.lower())
                image_out = output_root / split / "images" / source.name
                label_out = output_root / split / "labels" / f"{source.stem}.txt"
                image_out.parent.mkdir(parents=True, exist_ok=True)
                label_out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, image_out)
                source_label = self._find_yolo_label(source, raw_root, split)
                label_text = source_label.read_text(encoding="utf-8") if source_label else ""
                for line in label_text.splitlines():
                    fields = line.split()
                    if len(fields) != 5 or not fields[0].isdigit() or int(fields[0]) >= len(CLASS_NAMES):
                        raise ValueError(f"invalid YOLO label in {source_label}: {line!r}")
                label_out.write_text(label_text, encoding="utf-8")
                counts[split] += 1
        return counts, {index: name for index, name in enumerate(CLASS_NAMES)}

    def _prepare_coco(self, raw_root: Path, output_root: Path, annotation_path: Path, coco: dict[str, Any]) -> tuple[dict[str, int], dict[int, str]]:
        by_relative, by_name = self._image_index(raw_root)
        categories = sorted(coco["categories"], key=lambda category: int(category["id"]))
        names = {index: str(category["name"]) for index, category in enumerate(categories)}
        category_ids = {int(category["id"]): index for index, category in enumerate(categories)}
        images_by_id = {int(image["id"]): image for image in coco["images"]}
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in coco["annotations"]:
            annotations_by_image[int(annotation["image_id"])].append(annotation)

        counts = {split: 0 for split in SPLITS}
        for image_record in images_by_id.values():
            source = self._resolve_image(str(image_record["file_name"]), by_relative, by_name)
            split = self._split_for_image(source, raw_root) or self._fallback_split(source.name.lower())
            image_out = output_root / split / "images" / source.name
            label_out = output_root / split / "labels" / f"{source.stem}.txt"
            image_out.parent.mkdir(parents=True, exist_ok=True)
            label_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, image_out)
            width, height = float(image_record["width"]), float(image_record["height"])
            labels: list[str] = []
            for annotation in annotations_by_image.get(int(image_record["id"]), []):
                category_id = int(annotation["category_id"])
                bbox = annotation.get("bbox")
                if category_id not in category_ids or not isinstance(bbox, list) or len(bbox) < 4:
                    continue
                x, y, box_width, box_height = (float(value) for value in bbox[:4])
                x_min, y_min = max(0.0, x), max(0.0, y)
                x_max, y_max = min(width, x + box_width), min(height, y + box_height)
                if x_max <= x_min or y_max <= y_min:
                    continue
                labels.append(
                    f"{category_ids[category_id]} {((x_min + x_max) / 2 / width):.6f} {((y_min + y_max) / 2 / height):.6f} "
                    f"{((x_max - x_min) / width):.6f} {((y_max - y_min) / height):.6f}"
                )
            label_out.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
            counts[split] += 1
        self._log(f"using annotations {annotation_path}")
        return counts, names

    def _prepare_dataset(self, raw_root: Path, output_root: Path) -> tuple[dict[str, int], dict[int, str]]:
        coco_result = self._find_coco_json(raw_root)
        return self._prepare_yolo(raw_root, output_root) if coco_result is None else self._prepare_coco(raw_root, output_root, *coco_result)


if __name__ == "__main__":
    raise SystemExit(DatasetDentalOpg().cli())

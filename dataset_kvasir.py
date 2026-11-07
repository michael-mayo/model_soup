"""Prepare Kvasir-SEG for Ultralytics object detection."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from dataset import CONFIG_PATH, IMAGE_SUFFIXES, Dataset


DEFAULT_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"


class DatasetKvasir(Dataset):
    dataset_key = "kvasir"
    default_url = DEFAULT_URL
    archive_name = "kvasir-seg.zip"
    module_name = __file__

    @staticmethod
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

    @staticmethod
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

    def _prepare_dataset(self, raw_root: Path, output_root: Path) -> tuple[dict[str, int], dict[int, str]]:
        images_root = self._find_directory(raw_root, "images", "Kvasir-SEG")
        candidates = [path for path in raw_root.rglob("*.json") if "bbox" in path.name.lower() or "box" in path.name.lower()]
        if not candidates:
            raise FileNotFoundError("could not find Kvasir-SEG bounding-box JSON")
        annotations = self._annotations(min(candidates, key=lambda path: len(path.parts)))
        image_files = sorted(path for path in images_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if not image_files:
            raise FileNotFoundError("no images found in extracted Kvasir-SEG")

        counts = {split: 0 for split in ("train", "val", "test")}
        for image in image_files:
            width, height = self._image_size(image)
            split = self._fallback_split(image.stem.lower())
            image_out = output_root / split / "images" / image.name
            label_out = output_root / split / "labels" / f"{image.stem}.txt"
            image_out.parent.mkdir(parents=True, exist_ok=True)
            label_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, image_out)

            labels: list[str] = []
            for annotation in annotations.get(image.stem.lower(), []):
                x_min, y_min, x_max, y_max = self._bbox_values(annotation)
                x_min, x_max = max(0.0, min(x_min, width)), max(0.0, min(x_max, width))
                y_min, y_max = max(0.0, min(y_min, height)), max(0.0, min(y_max, height))
                if x_max <= x_min or y_max <= y_min:
                    continue
                labels.append(
                    f"0 {((x_min + x_max) / 2 / width):.6f} {((y_min + y_max) / 2 / height):.6f} "
                    f"{((x_max - x_min) / width):.6f} {((y_max - y_min) / height):.6f}"
                )
            label_out.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
            counts[split] += 1
        return counts, {0: "polyp"}


def load_settings(config_path: Path = CONFIG_PATH) -> tuple[dict, dict]:
    dataset = DatasetKvasir(config_path)
    return dataset.config, dataset.settings


def prepare(config_path: Path = CONFIG_PATH, *, force: bool = False, verify_ssl: bool | None = None) -> Path:
    return DatasetKvasir(config_path).prepare(force=force, verify_ssl=verify_ssl)


def main() -> int:
    return DatasetKvasir().cli()


if __name__ == "__main__":
    raise SystemExit(main())

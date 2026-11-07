"""Prepare FracAtlas for Ultralytics object detection."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

from dataset import IMAGE_SUFFIXES, SPLITS, Dataset


DEFAULT_URL = "https://ndownloader.figshare.com/files/43283628"


class DatasetFracAtlas(Dataset):
    dataset_key = "fracatlas"
    default_url = DEFAULT_URL
    archive_name = "fracatlas.zip"
    module_name = __file__

    @staticmethod
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

    def _prepare_dataset(self, raw_root: Path, output_root: Path) -> tuple[dict[str, int], dict[int, str]]:
        images_root = self._find_directory(raw_root, "images", "FracAtlas")
        annotations_root = self._find_directory(raw_root, "Annotations", "FracAtlas")
        yolo_root = self._find_directory(annotations_root, "YOLO", "FracAtlas")
        image_files = sorted(path for path in images_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if not image_files:
            raise FileNotFoundError("no image files found in FracAtlas images/")

        split_names: dict[str, set[str]] = {}
        for split in SPLITS:
            candidates = list(raw_root.rglob("valid.csv" if split == "val" else f"{split}.csv"))
            if candidates:
                split_names[split] = self._split_names(min(candidates, key=lambda path: len(path.parts)))

        counts = {split: 0 for split in SPLITS}
        for image in image_files:
            split = next((candidate for candidate in SPLITS if image.name.lower() in split_names.get(candidate, set())), None)
            split = split or self._fallback_split(image.name.lower())
            image_out = output_root / split / "images" / image.name
            label_out = output_root / split / "labels" / f"{image.stem}.txt"
            image_out.parent.mkdir(parents=True, exist_ok=True)
            label_out.parent.mkdir(parents=True, exist_ok=True)
            source_label = next(yolo_root.rglob(f"{image.stem}.txt"), None)
            shutil.copy2(image, image_out)
            if source_label:
                shutil.copy2(source_label, label_out)
            else:
                label_out.write_text("", encoding="utf-8")
            counts[split] += 1
        return counts, {0: "fracture"}


if __name__ == "__main__":
    raise SystemExit(DatasetFracAtlas().cli())

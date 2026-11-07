"""Prepare BCCD for Ultralytics object detection."""

from __future__ import annotations

import hashlib
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from dataset import CONFIG_PATH, Dataset, SPLITS


DEFAULT_URL = "https://github.com/Shenggan/BCCD_Dataset/archive/refs/heads/master.tar.gz"
CLASS_NAMES = ("RBC", "WBC", "Platelets")
CLASS_IDS = {name.lower(): index for index, name in enumerate(CLASS_NAMES)}


class DatasetBccd(Dataset):
    dataset_key = "bccd"
    default_url = DEFAULT_URL
    archive_name = "bccd.tar.gz"
    archive_type = "tar"
    module_name = __file__

    @staticmethod
    def _fallback_split(image_name: str) -> str:
        bucket = int(hashlib.sha256(image_name.encode("utf-8")).hexdigest()[:8], 16) % 100
        return "train" if bucket < 70 else "val" if bucket < 85 else "test"

    @staticmethod
    def _split_names(split_file: Path) -> set[str]:
        return {line.strip().lower() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()}

    def _prepare_dataset(self, raw_root: Path, output_root: Path) -> tuple[dict[str, int], dict[int, str]]:
        annotations_root = self._find_directory(raw_root, "Annotations", "BCCD")
        images_root = self._find_directory(raw_root, "JPEGImages", "BCCD")
        split_root = self._find_directory(raw_root, "Main", "BCCD")
        split_names = {
            split: self._split_names(split_root / f"{split}.txt")
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
            image = image_by_stem.get(xml_path.stem.lower())
            if image is None:
                continue
            split = next((candidate for candidate in SPLITS if xml_path.stem.lower() in split_names.get(candidate, set())), None)
            split = split or self._fallback_split(xml_path.stem.lower())
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
                x_min = max(0.0, min(float(box.findtext("xmin", "0")), width))
                y_min = max(0.0, min(float(box.findtext("ymin", "0")), height))
                x_max = max(0.0, min(float(box.findtext("xmax", "0")), width))
                y_max = max(0.0, min(float(box.findtext("ymax", "0")), height))
                if x_max <= x_min or y_max <= y_min:
                    continue
                labels.append(
                    f"{CLASS_IDS[name]} {((x_min + x_max) / 2 / width):.6f} {((y_min + y_max) / 2 / height):.6f} "
                    f"{((x_max - x_min) / width):.6f} {((y_max - y_min) / height):.6f}"
                )

            image_out = output_root / split / "images" / image.name
            label_out = output_root / split / "labels" / f"{image.stem}.txt"
            image_out.parent.mkdir(parents=True, exist_ok=True)
            label_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, image_out)
            label_out.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
            counts[split] += 1
        return counts, {index: name for index, name in enumerate(CLASS_NAMES)}


def load_settings(config_path: Path = CONFIG_PATH) -> tuple[dict, dict]:
    dataset = DatasetBccd(config_path)
    return dataset.config, dataset.settings


def prepare(config_path: Path = CONFIG_PATH, *, force: bool = False, verify_ssl: bool | None = None) -> Path:
    return DatasetBccd(config_path).prepare(force=force, verify_ssl=verify_ssl)


def main() -> int:
    return DatasetBccd().cli()


if __name__ == "__main__":
    raise SystemExit(main())

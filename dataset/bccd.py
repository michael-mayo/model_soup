"""BCCD blood-cell dataset (Pascal VOC XML annotations)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from dataset.base import SPLITS, Dataset, YoloWriter, find_directory, index_by_stem, unique, yolo_box


# Class names in YOLO class-id order.
CLASS_NAMES = ("RBC", "WBC", "Platelets")


class Bccd(Dataset):
    """Convert BCCD, using its ImageSets split lists where an image is listed."""

    key = "bccd"
    archive_name = "bccd.tar.gz"
    archive_type = "tar"
    split_percentages = (70, 15)

    def convert(self, raw: Path, writer: YoloWriter) -> dict[int, str]:
        """Convert every annotated image; images without a JPEG are skipped."""

        class_ids = {name.lower(): index for index, name in enumerate(CLASS_NAMES)}
        split_root = find_directory(raw, "Main")
        official = {
            split: {line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
            for split in SPLITS
            if (path := split_root / f"{split}.txt").is_file()
        }
        images = index_by_stem(path for path in find_directory(raw, "JPEGImages").iterdir() if path.is_file())
        xml_files = sorted(find_directory(raw, "Annotations").glob("*.xml"))
        if not xml_files:
            raise FileNotFoundError("no Pascal VOC XML annotations found in BCCD")

        for xml_path in xml_files:
            image = unique(images.get(xml_path.stem.lower(), []), f"image for {xml_path.name}")
            if image is None:
                continue
            root = ET.parse(xml_path).getroot()
            width = float(root.findtext("size/width", "0"))
            height = float(root.findtext("size/height", "0"))
            labels: list[str] = []
            for node in root.iter("object"):
                name = (node.findtext("name") or "").strip().lower()
                if name not in class_ids:
                    raise ValueError(f"unknown BCCD class {name!r} in {xml_path.name}")
                box = node.find("bndbox")
                if box is None:
                    continue
                line = yolo_box(
                    class_ids[name],
                    *(float(box.findtext(tag, "0")) for tag in ("xmin", "ymin", "xmax", "ymax")),
                    width, height,
                )
                if line:
                    labels.append(line)
            writer.add(self.split_for(xml_path.stem, official), image, labels)
        return dict(enumerate(CLASS_NAMES))

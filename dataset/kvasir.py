"""Kvasir-SEG polyp dataset (bounding boxes from its JSON sidecar)."""

from __future__ import annotations

import json
from pathlib import Path

from dataset.base import Dataset, YoloWriter, find_directory, list_images, yolo_box


class Kvasir(Dataset):
    """Convert Kvasir-SEG's bounding boxes; splits are hash-based (no official split)."""

    key = "kvasir"
    archive_name = "kvasir-seg.zip"

    def convert(self, raw: Path, writer: YoloWriter) -> dict[int, str]:
        """Convert every image listed in the bounding-box JSON.

        The JSON maps image id to `{"height", "width", "bbox": [{"xmin", ...}]}`;
        images without an entry are written with no labels.
        """

        candidates = [path for path in raw.rglob("*.json") if "bbox" in path.name.lower()]
        if not candidates:
            raise FileNotFoundError("could not find the Kvasir-SEG bounding-box JSON")
        records = json.loads(min(candidates, key=lambda path: len(path.parts)).read_text(encoding="utf-8"))
        records = {Path(str(key)).stem.lower(): value for key, value in records.items()}
        images = list_images(find_directory(raw, "images"))
        if not images:
            raise FileNotFoundError("no images found in extracted Kvasir-SEG")

        for image in images:
            record = records.get(image.stem.lower(), {})
            labels: list[str] = []
            for box in record.get("bbox", []):
                line = yolo_box(
                    0, float(box["xmin"]), float(box["ymin"]), float(box["xmax"]), float(box["ymax"]),
                    float(record["width"]), float(record["height"]),
                )
                if line:
                    labels.append(line)
            writer.add(self.split_for(image.stem), image, labels)
        return {0: "polyp"}

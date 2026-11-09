"""Shared download, extraction and Ultralytics conversion for example datasets."""

from __future__ import annotations

import hashlib
import shutil
import ssl
import tarfile
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar, Literal

import yaml

from app.config import DatasetLayout, DatasetSource
from app.log import Log


# File suffixes treated as images when scanning extracted archives.
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})

# Ultralytics split names, in the order they are reported.
SPLITS = ("train", "val", "test")

# Marker written once an archive has been fully extracted into raw/.
EXTRACTED_MARKER = ".extracted"

# User-Agent sent with download requests.
USER_AGENT = "soup-dataset-downloader/1.0"


def hash_split(name: str, percentages: tuple[int, int]) -> str:
    """Deterministically assign `name` to a split.

    `percentages` gives the train and val shares; the remainder goes to test.
    Hashing the name keeps the assignment stable across rebuilds.
    """

    train, val = percentages
    bucket = int(hashlib.sha256(name.lower().encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if bucket < train else "val" if bucket < train + val else "test"


def yolo_box(
    class_id: int, x_min: float, y_min: float, x_max: float, y_max: float, width: float, height: float,
) -> str | None:
    """Convert a pixel-space corner box to a normalised YOLO label line.

    The box is clipped to the image; `None` is returned if nothing remains.
    """

    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size {width}x{height}")
    x_min, x_max = max(0.0, min(x_min, width)), max(0.0, min(x_max, width))
    y_min, y_max = max(0.0, min(y_min, height)), max(0.0, min(y_max, height))
    if x_max <= x_min or y_max <= y_min:
        return None
    return (
        f"{class_id} {(x_min + x_max) / 2 / width:.6f} {(y_min + y_max) / 2 / height:.6f} "
        f"{(x_max - x_min) / width:.6f} {(y_max - y_min) / height:.6f}"
    )


def find_directory(root: Path, name: str) -> Path:
    """Return the shallowest directory under `root` called `name` (case-insensitive)."""

    matches = [path for path in root.rglob("*") if path.is_dir() and path.name.lower() == name.lower()]
    if not matches:
        raise FileNotFoundError(f"could not find {name}/ under {root}")
    return min(matches, key=lambda path: len(path.parts))


def list_images(root: Path) -> list[Path]:
    """Return every image file under `root`, sorted."""

    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def index_by_stem(paths: Iterable[Path]) -> dict[str, list[Path]]:
    """Group files by lower-cased stem, for one-pass lookups instead of repeated globbing."""

    index: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        index[path.stem.lower()].append(path)
    return index


def unique(matches: list[Path], description: str) -> Path | None:
    """Return the only element of `matches`, `None` if empty, or raise if ambiguous."""

    if len(matches) > 1:
        raise ValueError(f"ambiguous {description}: {', '.join(str(path) for path in matches)}")
    return matches[0] if matches else None


class YoloWriter:
    """Write images and label files into an Ultralytics dataset directory.

    Tracks per-split counts and the highest class id seen, and refuses to
    overwrite an image already written to the same split.
    """

    def __init__(self, root: Path) -> None:
        """Start a new dataset in `root`, which must not already contain one."""

        self.root = root
        self.counts = {split: 0 for split in SPLITS}
        self.max_class_id = -1

    def add(self, split: str, image: Path, labels: Iterable[str]) -> None:
        """Copy `image` into `split` and write its validated label lines."""

        if split not in self.counts:
            raise ValueError(f"unknown split {split!r}")
        lines = [line.strip() for line in labels if line.strip()]
        for line in lines:
            self._check_label(line, image)

        image_out = self.root / split / "images" / image.name
        label_out = self.root / split / "labels" / f"{image.stem}.txt"
        if image_out.exists() or label_out.exists():
            raise ValueError(f"duplicate image name in {split} split: {image.name}")
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, image_out)
        label_out.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
        self.counts[split] += 1

    def _check_label(self, line: str, image: Path) -> None:
        """Validate one YOLO label line: integer class and four normalised floats."""

        fields = line.split()
        try:
            class_id = int(fields[0])
            coords = [float(value) for value in fields[1:]]
        except (IndexError, ValueError):
            raise ValueError(f"invalid YOLO label for {image.name}: {line!r}") from None
        if class_id < 0 or len(coords) != 4 or not all(0.0 <= value <= 1.0 for value in coords):
            raise ValueError(f"invalid YOLO label for {image.name}: {line!r}")
        self.max_class_id = max(self.max_class_id, class_id)

    def write_yaml(self, names: dict[int, str]) -> Path:
        """Write `data.yaml` and return its path.

        No `path` key is written: Ultralytics then resolves the split
        directories relative to the YAML file, so the dataset can be moved.
        Empty splits are omitted.
        """

        if not self.counts["train"] or not self.counts["val"]:
            raise ValueError(f"train and val splits must both contain images: {self.counts}")
        if self.max_class_id >= len(names):
            raise ValueError(f"label class id {self.max_class_id} exceeds the {len(names)} class names")
        content: dict[str, object] = {
            split: f"{split}/images" for split in SPLITS if self.counts[split]
        }
        content["names"] = names
        data_yaml = self.root / "data.yaml"
        data_yaml.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
        return data_yaml


class Dataset(ABC):
    """Download one object-detection dataset and convert it to Ultralytics format.

    Subclasses set the class variables and implement `convert`; this class
    handles the download, safe extraction, and output directory lifecycle.
    """

    # Config key and directory name of the dataset.
    key: ClassVar[str]
    # File name of the downloaded archive.
    archive_name: ClassVar[str]
    archive_type: ClassVar[Literal["zip", "tar"]] = "zip"
    # Train and val percentages for hash-based splits; the remainder is test.
    split_percentages: ClassVar[tuple[int, int]] = (80, 10)

    def __init__(self, source: DatasetSource, layout: DatasetLayout, downloads: Path) -> None:
        """Bind the dataset to its download source and on-disk layout."""

        self.source = source
        self.layout = layout
        self.archive = downloads / self.archive_name
        self.log = Log(self.key)

    def split_for(self, name: str, official: dict[str, set[str]] | None = None) -> str:
        """Return the split for `name`, preferring an official split list over hashing."""

        lowered = name.lower()
        for split, names in (official or {}).items():
            if lowered in names:
                return split
        return hash_split(lowered, self.split_percentages)

    @abstractmethod
    def convert(self, raw: Path, writer: YoloWriter) -> dict[int, str]:
        """Write every sample from the extracted `raw` directory and return the class names."""

    def _archive_is_valid(self) -> bool:
        """Whether the downloaded archive exists and has the expected format."""

        if not self.archive.is_file():
            return False
        if self.archive_type == "zip":
            return zipfile.is_zipfile(self.archive)
        return tarfile.is_tarfile(self.archive)

    def _download(self, verify_ssl: bool) -> None:
        """Download the archive via a temporary file so partial downloads never look complete."""

        self.archive.parent.mkdir(parents=True, exist_ok=True)
        partial = self.archive.with_name(self.archive.name + ".part")
        self.archive.unlink(missing_ok=True)
        if verify_ssl:
            try:
                import certifi
                context = ssl.create_default_context(cafile=certifi.where())
            except ImportError:
                context = ssl.create_default_context()
        else:
            self.log.message("WARNING: SSL certificate verification disabled")
            context = ssl._create_unverified_context()

        self.log.message(f"downloading {self.source.url} to {self.archive}")
        request = urllib.request.Request(self.source.url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, context=context) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output)
            partial.replace(self.archive)
            if not self._archive_is_valid():
                raise ValueError(f"downloaded file is not a valid {self.archive_type} archive")
        except BaseException:
            partial.unlink(missing_ok=True)
            self.archive.unlink(missing_ok=True)
            raise

    def _extract(self) -> None:
        """Extract the archive into raw/, rejecting members that escape it.

        A marker file is written last so an interrupted extraction is detected
        and redone on the next run.
        """

        raw = self.layout.raw
        shutil.rmtree(raw, ignore_errors=True)
        raw.mkdir(parents=True)
        self.log.message(f"extracting {self.archive.name} to {raw}")
        if self.archive_type == "zip":
            root = raw.resolve()
            with zipfile.ZipFile(self.archive) as source:
                for member in source.infolist():
                    target = (raw / member.filename).resolve()
                    if target != root and root not in target.parents:
                        raise ValueError(f"unsafe archive path: {member.filename}")
                source.extractall(raw)
        else:
            with tarfile.open(self.archive) as source:
                # The "data" filter rejects absolute paths, traversal and special files.
                source.extractall(raw, filter="data")
        (raw / EXTRACTED_MARKER).touch()

    def prepare(self, *, force: bool = False, verify_ssl: bool | None = None) -> Path:
        """Download, extract and convert the dataset; return its data.yaml.

        An existing prepared dataset is reused unless `force` is set, in which
        case the archive is downloaded again and everything is rebuilt.
        `verify_ssl` overrides the configured setting when given.
        """

        if self.layout.data_yaml.is_file() and not force:
            self.log.message(f"already prepared at {self.layout.yolo}")
            return self.layout.data_yaml
        if force or not self._archive_is_valid():
            self._download(self.source.verify_ssl if verify_ssl is None else verify_ssl)
        if force or not (self.layout.raw / EXTRACTED_MARKER).is_file():
            self._extract()

        # Always convert into an empty directory so stale files never survive.
        shutil.rmtree(self.layout.yolo, ignore_errors=True)
        writer = YoloWriter(self.layout.yolo)
        try:
            names = self.convert(self.layout.raw, writer)
            data_yaml = writer.write_yaml(names)
        except BaseException:
            shutil.rmtree(self.layout.yolo, ignore_errors=True)
            raise
        self.log.message(f"prepared {sum(writer.counts.values())} images: {writer.counts}")
        self.log.message(f"wrote {data_yaml}")
        return data_yaml

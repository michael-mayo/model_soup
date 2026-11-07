"""Shared dataset preparation functionality."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import ssl
import sys
import tarfile
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import yaml

import log


CONFIG_PATH = Path(__file__).with_name("config.yaml")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


class Dataset(ABC):
    """Base class for downloading and converting one object-detection dataset."""

    dataset_key: ClassVar[str]
    default_url: ClassVar[str]
    archive_name: ClassVar[str]
    archive_type: ClassVar[str] = "zip"
    module_name: ClassVar[str] = __file__

    def __init__(self, config_path: Path = CONFIG_PATH) -> None:
        self.config_path = config_path
        self.project_root = config_path.parent
        self.config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        self.settings: dict[str, Any] = self.config.get("dataset", {}).get(self.dataset_key, {})

    @property
    def url(self) -> str:
        return str(self.settings.get("url", self.default_url))

    @property
    def dataset_root(self) -> Path:
        paths = self.config.get("paths", {})
        datasets_root = self._path(paths.get("datasets", "scratch/datasets"))
        return datasets_root / self.dataset_key

    def _path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.project_root / path

    def _log(self, message: str) -> None:
        log.message(self.module_name, message)

    def _ca_bundle(self) -> Path | None:
        candidates = list((Path(sys.prefix) / "lib").glob("python*/site-packages/pip/_vendor/certifi/cacert.pem"))
        candidates.append(Path(sys.prefix) / "ssl" / "cacert.pem")
        return next((path for path in candidates if path.is_file() and path.stat().st_size > 1000), None)

    def _archive_is_valid(self, archive: Path) -> bool:
        if self.archive_type == "zip":
            return zipfile.is_zipfile(archive)
        if self.archive_type == "tar":
            return tarfile.is_tarfile(archive)
        raise ValueError(f"unsupported archive type: {self.archive_type}")

    def _download(self, destination: Path, *, verify_ssl: bool) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._log(f"downloading dataset to {destination}")
        partial = destination.with_suffix(destination.suffix + ".part")
        destination.unlink(missing_ok=True)
        try:
            ca_bundle = self._ca_bundle()
            context = (
                ssl.create_default_context(cafile=str(ca_bundle))
                if verify_ssl and ca_bundle
                else ssl.create_default_context()
                if verify_ssl
                else ssl._create_unverified_context()
            )
            if not verify_ssl:
                self._log("WARNING: SSL certificate verification disabled by --insecure")
            request = urllib.request.Request(self.url, headers={"User-Agent": "soup-dataset-downloader/1.0"})
            with urllib.request.urlopen(request, context=context) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output)
            partial.replace(destination)
            if not self._archive_is_valid(destination):
                raise ValueError("downloaded file is not a valid dataset archive")
        except Exception:
            partial.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
            raise

    def _safe_extract(self, archive: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        root = destination.resolve()
        if self.archive_type == "zip":
            with zipfile.ZipFile(archive) as source:
                members = source.infolist()
                for member in members:
                    target = (destination / member.filename).resolve()
                    if target != root and root not in target.parents:
                        raise ValueError(f"unsafe archive path: {member.filename}")
                source.extractall(destination)
            return
        if self.archive_type == "tar":
            with tarfile.open(archive) as source:
                for member in source.getmembers():
                    target = (destination / member.name).resolve()
                    if target != root and root not in target.parents:
                        raise ValueError(f"unsafe archive path: {member.name}")
                source.extractall(destination, filter="data")
            return
        raise ValueError(f"unsupported archive type: {self.archive_type}")

    def _raw_ready(self, raw_root: Path) -> bool:
        return raw_root.exists() and any(raw_root.rglob("*"))

    def _write_yaml(self, output_root: Path, names: dict[int, str]) -> Path:
        dataset_yaml = output_root / "data.yaml"
        dataset_yaml.write_text(yaml.safe_dump({
            "path": str(output_root.resolve()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "names": names,
        }, sort_keys=False), encoding="utf-8")
        return dataset_yaml

    @staticmethod
    def _find_directory(root: Path, name: str, description: str) -> Path:
        matches = [path for path in root.rglob("*") if path.is_dir() and path.name.lower() == name.lower()]
        if not matches:
            raise FileNotFoundError(f"could not find {name}/ in extracted {description}")
        return min(matches, key=lambda path: len(path.parts))

    @staticmethod
    def _fallback_split(image_name: str) -> str:
        bucket = int(hashlib.sha256(image_name.encode("utf-8")).hexdigest()[:8], 16) % 100
        return "train" if bucket < 80 else "val" if bucket < 90 else "test"

    @staticmethod
    def _image_size(path: Path) -> tuple[int, int]:
        """Read JPEG/PNG dimensions without adding an imaging dependency."""

        with path.open("rb") as stream:
            signature = stream.read(24)
            if signature.startswith(b"\x89PNG") and len(signature) >= 24:
                return int.from_bytes(signature[16:20], "big"), int.from_bytes(signature[20:24], "big")
            if signature[:2] != b"\xff\xd8":
                raise ValueError(f"unsupported image format for {path.name}")
            stream.seek(2)
            while True:
                marker_start = stream.read(1)
                if not marker_start:
                    break
                if marker_start != b"\xff":
                    continue
                marker = stream.read(1)
                while marker == b"\xff":
                    marker = stream.read(1)
                if marker in {b"\xd8", b"\xd9"}:
                    continue
                length_bytes = stream.read(2)
                if len(length_bytes) != 2:
                    break
                length = int.from_bytes(length_bytes, "big")
                if marker[0] in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                    stream.read(1)
                    height = int.from_bytes(stream.read(2), "big")
                    width = int.from_bytes(stream.read(2), "big")
                    return width, height
                stream.seek(length - 2, 1)
        raise ValueError(f"could not read image dimensions for {path.name}")

    @abstractmethod
    def _prepare_dataset(self, raw_root: Path, output_root: Path) -> tuple[dict[str, int], dict[int, str]]:
        """Convert the raw dataset and return split counts and class names."""

    def prepare(self, *, force: bool = False, verify_ssl: bool | None = None) -> Path:
        """Download, extract, convert, and return the generated data YAML."""

        if verify_ssl is None:
            verify_ssl = bool(self.settings.get("verify_ssl", True))
        paths = self.config.get("paths", {})
        downloads_root = self._path(paths.get("downloads", "scratch/downloads"))
        raw_root = self.dataset_root / "raw"
        output_root = self.dataset_root / "yolo"
        archive = downloads_root / self.archive_name
        dataset_yaml = output_root / "data.yaml"

        if dataset_yaml.exists() and not force:
            self._log(f"dataset already prepared at {output_root}")
            return dataset_yaml
        if force and output_root.exists():
            shutil.rmtree(output_root)
        if force or not self._archive_is_valid(archive):
            self._download(archive, verify_ssl=verify_ssl)
        if force or not self._raw_ready(raw_root):
            self._safe_extract(archive, raw_root)

        counts, names = self._prepare_dataset(raw_root, output_root)
        dataset_yaml = self._write_yaml(output_root, names)
        self._log(f"prepared {sum(counts.values())} images: {counts}")
        self._log(f"wrote {dataset_yaml}")
        return dataset_yaml

    def cli(self) -> int:
        parser = argparse.ArgumentParser(description=type(self).__doc__ or self.__doc__)
        parser.add_argument("--config", type=Path, default=self.config_path)
        parser.add_argument("--force", action="store_true", help="redownload and rebuild the prepared dataset")
        parser.add_argument("--insecure", action="store_true", help="disable SSL certificate verification for the download")
        args = parser.parse_args()
        if args.config != self.config_path:
            self.__init__(args.config)
        try:
            self.prepare(force=args.force, verify_ssl=False if args.insecure else None)
        except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile, KeyError, TypeError) as exc:
            log.error(self.module_name, str(exc))
            return 1
        return 0

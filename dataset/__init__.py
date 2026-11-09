"""Example object-detection datasets and their Ultralytics converters."""

from __future__ import annotations

from app.config import Config
from dataset.base import Dataset
from dataset.bccd import Bccd
from dataset.dental_opg import DentalOpg
from dataset.fracatlas import FracAtlas
from dataset.kvasir import Kvasir


# Converter class for each dataset key used in config.yaml.
CONVERTERS: dict[str, type[Dataset]] = {cls.key: cls for cls in (FracAtlas, Kvasir, Bccd, DentalOpg)}


def create(name: str, config: Config) -> Dataset:
    """Create the converter for dataset `name`, bound to its configured source and layout."""

    if name not in CONVERTERS:
        raise ValueError(f"no converter for dataset {name!r}; known: {', '.join(CONVERTERS)}")
    if name not in config.datasets:
        raise ValueError(f"dataset {name!r} has no entry under `datasets` in config.yaml")
    return CONVERTERS[name](config.datasets[name], config.layout(name), config.paths.downloads)

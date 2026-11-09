"""Typed loading of `config.yaml` and the standard on-disk dataset layout."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


# Default configuration file, next to the project's entry point.
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass(frozen=True)
class DatasetLayout:
    """Standard directory layout of one dataset under the datasets root.

    Every component (preparation, training, soup) derives its paths from this
    layout, so per-dataset paths never need to be repeated in the config.
    """

    root: Path

    @property
    def raw(self) -> Path:
        """Directory holding the extracted original archive."""

        return self.root / "raw"

    @property
    def yolo(self) -> Path:
        """Directory holding the converted Ultralytics dataset."""

        return self.root / "yolo"

    @property
    def data_yaml(self) -> Path:
        """Ultralytics dataset description file."""

        return self.yolo / "data.yaml"

    @property
    def runs(self) -> Path:
        """Directory holding Ultralytics training runs."""

        return self.root / "runs"

    def model_path(self, name: str) -> Path:
        """Path of the copied checkpoint for the run called `name`."""

        return self.root / "models" / f"{name}.pt"


@dataclass(frozen=True)
class PathsConfig:
    """Resolved top-level storage directories."""

    downloads: Path
    datasets: Path
    weights: Path


@dataclass(frozen=True)
class DatasetSource:
    """Where to download one dataset from."""

    url: str
    verify_ssl: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    """Baseline training hyperparameters and the datasets to train on."""

    model: str = "yolo26n.pt"
    optimizer: str = "MuSGD"
    lr0: float = 0.01
    epochs: int = 300
    patience: int = 15
    batch: int | float = 64
    imgsz: int = 640
    device: int | str = 0
    workers: int = 0
    amp: bool = True
    seed: int = 0
    plots: bool = True
    verbose: bool = True
    run_name: str = "baseline"
    datasets: tuple[str, ...] = ()

    def eval_kwargs(self) -> dict[str, Any]:
        """Ultralytics `val()` arguments shared by every evaluation."""

        return {"imgsz": self.imgsz, "batch": self.batch, "device": self.device, "workers": self.workers}


@dataclass(frozen=True)
class SoupConfig:
    """Which trained baseline the soup starts from."""

    dataset: str
    baseline: str


@dataclass(frozen=True)
class Config:
    """The whole application configuration."""

    path: Path
    paths: PathsConfig
    datasets: dict[str, DatasetSource] = field(default_factory=dict)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    soup: SoupConfig | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def layout(self, dataset: str) -> DatasetLayout:
        """Return the directory layout for the named dataset."""

        return DatasetLayout(self.paths.datasets / dataset)


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a top-level mapping from the YAML, or an empty one if absent."""

    value = raw.get(name) or {}
    if not isinstance(value, dict):
        raise ValueError(f"config section {name!r} must be a mapping")
    return value


def _build(cls: type, values: dict[str, Any], section: str) -> Any:
    """Construct a config dataclass, reporting unknown keys by section name."""

    unknown = set(values) - {item.name for item in fields(cls)}
    if unknown:
        raise ValueError(f"unknown key(s) in config section {section!r}: {', '.join(sorted(unknown))}")
    try:
        return cls(**values)
    except TypeError as exc:
        raise ValueError(f"invalid config section {section!r}: {exc}") from exc


def load_config(path: Path = CONFIG_PATH) -> Config:
    """Load and validate the YAML configuration.

    Relative paths are resolved against the config file's directory so that
    commands behave the same regardless of the current working directory.
    """

    path = path.resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name} must contain a top-level mapping")

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else path.parent / candidate

    path_values = {"downloads": "scratch/downloads", "datasets": "scratch/datasets", "weights": "scratch/weights"}
    path_values.update(_section(raw, "paths"))
    paths = _build(PathsConfig, {key: resolve(str(value)) for key, value in path_values.items()}, "paths")

    datasets = {
        name: _build(DatasetSource, values or {}, f"datasets.{name}")
        for name, values in _section(raw, "datasets").items()
    }

    training_values = dict(_section(raw, "training"))
    training_values["datasets"] = tuple(training_values.get("datasets") or ())
    training = _build(TrainingConfig, training_values, "training")

    soup_values = _section(raw, "soup")
    soup = _build(SoupConfig, soup_values, "soup") if soup_values else None

    return Config(path=path, paths=paths, datasets=datasets, training=training, soup=soup, raw=raw)

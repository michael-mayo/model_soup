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


@dataclass(frozen=True, kw_only=True)
class RunSettings:
    """Ultralytics settings shared by training, fine-tuning and validation."""

    optimizer: str = "MuSGD"
    batch: int | float = 64
    imgsz: int = 640
    device: int | str = 0
    workers: int = 0
    amp: bool = True

    def eval_kwargs(self) -> dict[str, Any]:
        """Ultralytics `val()` arguments shared by every evaluation."""

        return {"imgsz": self.imgsz, "batch": self.batch, "device": self.device, "workers": self.workers}


@dataclass(frozen=True, kw_only=True)
class TrainingConfig(RunSettings):
    """Baseline training hyperparameters and the sample datasets to train on."""

    model: str = "yolo26n.pt"
    lr0: float = 0.01
    epochs: int = 300
    patience: int = 15
    seed: int = 0
    plots: bool = True
    verbose: bool = True
    run_name: str = "baseline"
    datasets: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class SoupConfig(RunSettings):
    """The soup's baseline, dataset, and the settings shared by every member fine-tune.

    `baseline_model` is a .pt file and `dataset` an Ultralytics data.yaml (a
    directory means its data.yaml); neither needs to belong to the sample
    datasets. The inherited run settings apply to fine-tuning and validation.

    Each of `bagging_iterations` bags (a random `bagging_fraction` of the
    training images) is fine-tuned once per augmentation config file, giving
    `bagging_iterations * len(configs)` members.
    """

    baseline_model: Path
    dataset: Path
    output_dir: Path = Path("scratch/current_soup")
    lr0: float = 0.002
    bagging_fraction: float = 0.8
    bagging_iterations: int = 2
    seed: int = 42
    epochs: int = 10
    close_mosaic: int = 2
    configs: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        """Reject settings that cannot produce a soup."""

        if not 0.0 < self.bagging_fraction <= 1.0:
            raise ValueError("soup.bagging_fraction must be in (0, 1]")
        if not 0 <= self.close_mosaic <= self.epochs:
            raise ValueError("soup.close_mosaic must be between 0 and soup.epochs")
        if self.bagging_iterations < 1 or self.epochs < 1 or self.lr0 <= 0:
            raise ValueError("soup.bagging_iterations, soup.epochs and soup.lr0 must be positive")
        if not self.configs:
            raise ValueError("soup.configs must list at least one augmentation config file")


@dataclass(frozen=True)
class SampleDatasetsConfig:
    """Storage, download sources and baseline training for the example datasets."""

    paths: PathsConfig
    sources: dict[str, DatasetSource] = field(default_factory=dict)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def layout(self, dataset: str) -> DatasetLayout:
        """Return the directory layout for the named sample dataset."""

        return DatasetLayout(self.paths.datasets / dataset)


@dataclass(frozen=True)
class Config:
    """The whole application configuration."""

    path: Path
    samples: SampleDatasetsConfig
    soup: SoupConfig | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a mapping from the YAML, or an empty one if absent."""

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

    The file has two top-level sections: `soup` and `sample_datasets` (with
    `paths`, `sources` and `training`). Relative paths are resolved against
    the config file's directory so that commands behave the same regardless
    of the current working directory.
    """

    path = path.resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name} must contain a top-level mapping")
    unknown = set(raw) - {"soup", "sample_datasets"}
    if unknown:
        raise ValueError(f"unknown top-level section(s): {', '.join(sorted(unknown))}")

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else path.parent / candidate

    samples_raw = _section(raw, "sample_datasets")
    unknown = set(samples_raw) - {"paths", "sources", "training"}
    if unknown:
        raise ValueError(f"unknown key(s) in config section 'sample_datasets': {', '.join(sorted(unknown))}")

    path_values = {"downloads": "scratch/downloads", "datasets": "scratch/datasets", "weights": "scratch/weights"}
    path_values.update(_section(samples_raw, "paths"))
    paths = _build(
        PathsConfig, {key: resolve(str(value)) for key, value in path_values.items()}, "sample_datasets.paths",
    )
    sources = {
        name: _build(DatasetSource, values or {}, f"sample_datasets.sources.{name}")
        for name, values in _section(samples_raw, "sources").items()
    }
    training_values = dict(_section(samples_raw, "training"))
    training_values["datasets"] = tuple(training_values.get("datasets") or ())
    training = _build(TrainingConfig, training_values, "sample_datasets.training")
    samples = SampleDatasetsConfig(paths=paths, sources=sources, training=training)

    soup = None
    if soup_values := dict(_section(raw, "soup")):
        for key in ("baseline_model", "dataset"):
            if key in soup_values:
                soup_values[key] = resolve(str(soup_values[key]))
        if "dataset" in soup_values and soup_values["dataset"].is_dir():
            soup_values["dataset"] = soup_values["dataset"] / "data.yaml"
        soup_values["output_dir"] = resolve(str(soup_values.get("output_dir", SoupConfig.output_dir)))
        soup_values["configs"] = tuple(resolve(str(value)) for value in soup_values.get("configs") or ())
        soup = _build(SoupConfig, soup_values, "soup")

    return Config(path=path, samples=samples, soup=soup, raw=raw)

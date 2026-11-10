"""Train baseline Ultralytics detectors for the configured datasets."""

from __future__ import annotations

import csv
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from app import yolo
from app.config import SampleDatasetsConfig
from app.log import Log


LOG = Log(__name__)

# Ultralytics' results.csv column holding validation mAP50-95.
MAP_COLUMN = "metrics/mAP50-95(B)"

# Suffix of the run directory used while training is in progress.
STAGING_SUFFIX = "-incomplete"


@dataclass(frozen=True)
class TrainingResult:
    """Outcome of training one dataset's baseline.

    All mAP values are mAP50-95. `train_map` and `val_map` are both measured on
    the final `best.pt`, so they describe the same checkpoint.
    """

    dataset: str
    model_path: Path | None = None
    run_dir: Path | None = None
    epoch1_val_map: float | None = None
    train_map: float | None = None
    val_map: float | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Whether training completed successfully."""

        return self.error is None


def _epoch1_val_map(results_csv: Path) -> float | None:
    """Read the epoch-1 validation mAP50-95 from Ultralytics' results.csv."""

    if not results_csv.is_file():
        return None
    with results_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            # Older Ultralytics versions pad column names with spaces.
            row = {key.strip(): value.strip() for key, value in row.items() if key}
            if int(float(row["epoch"])) == 1:
                return float(row[MAP_COLUMN]) if row.get(MAP_COLUMN) else None
    return None


def _format_map(value: float | None) -> str:
    """Format an optional mAP value for the summary."""

    return "n/a" if value is None else f"{value:.4f}"


class BaselineTrainer:
    """Train baseline models and publish their best checkpoints."""

    def __init__(self, config: SampleDatasetsConfig) -> None:
        """Bind the trainer to the sample-datasets configuration."""

        self.config = config
        self.settings = config.training

    def _pretrained_model(self) -> Path | str:
        """Return the pretrained model spec, placing bare `.pt` names in the weights directory."""

        spec = Path(self.settings.model)
        if spec.suffix == ".pt" and not spec.is_absolute() and spec.parent == Path("."):
            self.config.paths.weights.mkdir(parents=True, exist_ok=True)
            return self.config.paths.weights / spec.name
        return self.settings.model

    def train(self, dataset: str) -> TrainingResult:
        """Train one dataset and copy its best checkpoint into the models directory.

        Training writes to a staging run directory. The previous run directory
        and published model are only replaced after training succeeds, so a
        crashed or interrupted run never destroys an earlier baseline.
        """

        settings = self.settings
        layout = self.config.layout(dataset)
        if not layout.data_yaml.is_file():
            raise FileNotFoundError(f"dataset YAML not found (run `main.py prepare` first): {layout.data_yaml}")

        run_dir = layout.runs / settings.run_name
        staging_dir = layout.runs / f"{settings.run_name}{STAGING_SUFFIX}"
        if run_dir.parent != layout.runs:
            raise ValueError(f"run_name must be a plain directory name: {settings.run_name!r}")
        # Ultralytics appends to an existing results.csv, so always start clean.
        shutil.rmtree(staging_dir, ignore_errors=True)
        layout.runs.mkdir(parents=True, exist_ok=True)

        model = yolo.load_model(self._pretrained_model())
        model.train(
            data=str(layout.data_yaml), project=str(layout.runs), name=staging_dir.name, exist_ok=True,
            epochs=settings.epochs, patience=settings.patience, batch=settings.batch, imgsz=settings.imgsz,
            optimizer=settings.optimizer, lr0=settings.lr0, device=settings.device, workers=settings.workers,
            amp=settings.amp, seed=settings.seed, plots=settings.plots, verbose=settings.verbose,
        )
        if not (staging_dir / "weights" / "best.pt").is_file():
            raise FileNotFoundError(f"training completed without best.pt in {staging_dir}")

        shutil.rmtree(run_dir, ignore_errors=True)
        staging_dir.rename(run_dir)
        best_weights = run_dir / "weights" / "best.pt"

        best_model = yolo.load_model(best_weights)
        metrics = {
            split: float(best_model.val(
                data=str(layout.data_yaml), split=split, plots=False, verbose=False, **settings.eval_kwargs(),
            ).box.map)
            for split in ("train", "val")
        }

        model_path = layout.model_path(settings.run_name)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        partial = model_path.with_suffix(".pt.part")
        shutil.copy2(best_weights, partial)
        partial.replace(model_path)

        return TrainingResult(
            dataset=dataset, model_path=model_path, run_dir=run_dir,
            epoch1_val_map=_epoch1_val_map(run_dir / "results.csv"),
            train_map=metrics["train"], val_map=metrics["val"],
        )

    def train_all(self, datasets: Sequence[str] | None = None) -> list[TrainingResult]:
        """Train each selected dataset, continuing past failures.

        `datasets` overrides `sample_datasets.training.datasets`; if both are
        empty, every configured dataset is trained.
        """

        selected = list(datasets or self.settings.datasets or self.config.sources)
        missing = [name for name in selected if name not in self.config.sources]
        if missing:
            raise ValueError(f"dataset(s) not configured: {', '.join(missing)}")

        results: list[TrainingResult] = []
        for name in selected:
            LOG.message(f"training {name}")
            try:
                results.append(self.train(name))
            # One failed dataset should not stop the remaining (long) runs.
            except Exception as exc:  # noqa: BLE001
                LOG.error(f"{name}: {exc}")
                results.append(TrainingResult(dataset=name, error=str(exc)))
        return results


def log_summary(results: Sequence[TrainingResult]) -> None:
    """Log a per-dataset summary of training results."""

    LOG.message("baseline training summary")
    for result in results:
        LOG.message(f"[{result.dataset}] {'complete' if result.ok else 'failed'}")
        if not result.ok:
            LOG.message(f"  error: {result.error}")
            continue
        LOG.message(f"  model: {result.model_path}")
        LOG.message(f"  run:   {result.run_dir}")
        LOG.message(f"  epoch 1 val mAP50-95:  {_format_map(result.epoch1_val_map)}")
        LOG.message(f"  best.pt train mAP50-95: {_format_map(result.train_map)}")
        LOG.message(f"  best.pt val mAP50-95:   {_format_map(result.val_map)}")

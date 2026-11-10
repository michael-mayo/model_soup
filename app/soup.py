"""Model soup: bagged, augmentation-diverse fine-tunes of a baseline model.

Each soup member is the baseline fine-tuned on one bag (a random subset of the
training images) with one augmentation config. Members are evaluated the
same way as the baseline. Weight averaging is not implemented yet.
"""

from __future__ import annotations

import csv
import random
import shutil
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import yaml

from app import yolo
from app.config import Config
from app.log import Log


LOG = Log(__name__)

# Fine-tune settings applied to every member unless an augmentation config
# overrides them: warmup would spike the learning rate of an already-trained
# model. (close_mosaic comes from soup.close_mosaic.)
FINE_TUNE_DEFAULTS: dict[str, Any] = {"warmup_epochs": 0}

# Train arguments owned by the soup/training config, not augmentation configs.
RESERVED_KEYS = frozenset({
    "model", "data", "epochs", "lr0", "seed", "optimizer", "batch", "imgsz", "device", "workers", "amp",
    "project", "name", "exist_ok", "save", "val", "plots", "verbose", "resume", "fraction", "augmentations",
})


@dataclass(frozen=True)
class AugmentationConfig:
    """One soup augmentation config file.

    `train_args` are Ultralytics train() hyperparameters; `transforms` are
    Albumentations specs (`{"transform": ClassName, **arguments}`).
    """

    name: str
    train_args: dict[str, Any] = field(default_factory=dict)
    transforms: tuple[dict[str, Any], ...] = ()

    def build_transforms(self) -> list[Any]:
        """Instantiate the Albumentations transforms (lists become tuples)."""

        if not self.transforms:
            return []
        try:
            import albumentations as A
        except ImportError as exc:
            raise RuntimeError(
                f"{self.name}: Albumentations is required for `albumentations` entries; "
                "run `conda run -n soup python -m pip install -r requirements.txt`."
            ) from exc
        built = []
        for spec in self.transforms:
            arguments = {key: tuple(value) if isinstance(value, list) else value
                         for key, value in spec.items() if key != "transform"}
            cls = getattr(A, str(spec.get("transform")), None)
            if cls is None:
                raise ValueError(f"{self.name}: unknown Albumentations transform {spec.get('transform')!r}")
            try:
                built.append(cls(**arguments))
            except Exception as exc:  # Albumentations raises various validation errors.
                raise ValueError(f"{self.name}: invalid {spec['transform']} arguments: {exc}") from exc
        return built


def load_augmentation(path: Path) -> AugmentationConfig:
    """Load and validate an augmentation config file, named after its file stem."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or set(raw) - {"ultralytics", "albumentations"}:
        raise ValueError(f"{path}: expected only `ultralytics` and `albumentations` sections")
    train_args = raw.get("ultralytics") or {}
    transforms = raw.get("albumentations") or []
    if not isinstance(train_args, dict) or not isinstance(transforms, list):
        raise ValueError(f"{path}: `ultralytics` must be a mapping and `albumentations` a list")
    reserved = RESERVED_KEYS & set(train_args)
    if reserved:
        raise ValueError(f"{path}: these keys are set by config.yaml, not here: {', '.join(sorted(reserved))}")
    if not all(isinstance(spec, dict) and "transform" in spec for spec in transforms):
        raise ValueError(f"{path}: each albumentations entry needs a `transform` key")
    config = AugmentationConfig(name=path.stem, train_args=dict(train_args), transforms=tuple(transforms))
    config.build_transforms()  # fail before any training starts
    return config


class EpochProgress:
    """Report fine-tuning progress through Ultralytics trainer callbacks.

    Keeps one terminal line per epoch: it is rewritten as batches complete
    and ended when the epoch finishes. Writes to the stream captured at
    construction, so it stays visible while Ultralytics' output is hidden.
    """

    # Minimum seconds between in-place updates, to avoid flooding the terminal.
    UPDATE_INTERVAL = 0.2

    def __init__(self, label: str, stream: TextIO) -> None:
        """Prefix every line with `label` and write to `stream`."""

        self.label = label
        self.stream = stream
        self.batch = 0
        self.epoch_started = 0.0
        self.last_update = 0.0

    def attach(self, model: Any) -> None:
        """Register the callbacks on an Ultralytics model before training."""

        model.add_callback("on_train_epoch_start", self.on_epoch_start)
        model.add_callback("on_train_batch_end", self.on_batch_end)
        model.add_callback("on_train_epoch_end", self.on_epoch_end)

    def _text(self, trainer: Any, detail: str) -> str:
        """Format the progress line for the current epoch."""

        losses = " ".join(
            f"{str(name).removesuffix('_loss')} {float(value):.3f}" for name, value in (trainer.tloss or {}).items()
        )
        return f"{self.label} epoch {trainer.epoch + 1}/{trainer.epochs}: {detail}  {losses}".rstrip()

    def on_epoch_start(self, trainer: Any) -> None:
        """Reset the batch counter and timer."""

        self.batch = 0
        self.epoch_started = time.monotonic()

    def on_batch_end(self, trainer: Any) -> None:
        """Rewrite the progress line, at most every UPDATE_INTERVAL seconds."""

        self.batch += 1
        now = time.monotonic()
        if now - self.last_update >= self.UPDATE_INTERVAL:
            self.last_update = now
            LOG.progress(self._text(trainer, f"batch {self.batch}/{len(trainer.train_loader)}"), stream=self.stream)

    def on_epoch_end(self, trainer: Any) -> None:
        """End the epoch's line with its duration and mean losses."""

        elapsed = _duration(time.monotonic() - self.epoch_started)
        LOG.progress(self._text(trainer, f"{self.batch} batches in {elapsed}"), stream=self.stream, final=True)


@dataclass(frozen=True)
class ClassMetrics:
    """Val-split detection metrics for one class (or their macro average)."""

    precision: float
    recall: float
    map50: float
    map50_95: float


@dataclass(frozen=True)
class EvaluationResult:
    """Val-split metrics for one model.

    `per_class` maps every class name to its metrics, or to `None` when the
    val split has no instances of that class.
    """

    name: str
    model_path: Path
    per_class: dict[str, ClassMetrics | None]

    @property
    def macro(self) -> ClassMetrics:
        """Unweighted mean over the classes present in the val split."""

        present = [metrics for metrics in self.per_class.values() if metrics is not None]
        if not present:
            return ClassMetrics(0.0, 0.0, 0.0, 0.0)
        return ClassMetrics(*(
            sum(getattr(metrics, key) for metrics in present) / len(present)
            for key in ("precision", "recall", "map50", "map50_95")
        ))


def make_bag(data_yaml: Path, fraction: float, seed: int, bag_dir: Path) -> tuple[Path, int]:
    """Create a bagged copy of the dataset and return its data.yaml and image count.

    A `fraction` of the training images is sampled without replacement. The
    bag holds symlinks to the original images and labels (Ultralytics finds
    labels by swapping `images` for `labels` in the path), and validates on
    the original val split. `train` may be an image directory or an image-list
    file, resolved the way Ultralytics resolves it.
    """

    from ultralytics.data.utils import IMG_FORMATS, img2label_paths

    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    # Ultralytics resolves split paths against `path`, which defaults to the YAML's directory.
    root = Path(data.get("path") or data_yaml.parent)
    root = root if root.is_absolute() else (data_yaml.parent / root)

    def resolve(value: str) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else root / path).resolve()

    if not isinstance(data.get("train"), str) or "val" not in data:
        raise ValueError(f"{data_yaml}: bagging needs a single `train` path and a `val` entry")
    train = resolve(data["train"])
    if train.is_dir():
        candidates = list(train.rglob("*"))
    else:  # an Ultralytics image-list file
        candidates = [resolve(line.strip()) for line in train.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_images = sorted(path for path in candidates if path.suffix[1:].lower() in IMG_FORMATS and path.is_file())
    if not train_images:
        raise FileNotFoundError(f"no training images found for {data_yaml}")
    sample = random.Random(seed).sample(train_images, max(1, round(len(train_images) * fraction)))

    images_dir, labels_dir = bag_dir / "train" / "images", bag_dir / "train" / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    for index, (image, label) in enumerate(zip(sample, img2label_paths([str(path) for path in sample]))):
        # Prefix with the sample index so equal names from different folders cannot collide.
        name = f"{index:06d}_{image.stem}"
        (images_dir / f"{name}{image.suffix}").symlink_to(image.resolve())
        if Path(label).is_file():
            (labels_dir / f"{name}.txt").symlink_to(Path(label).resolve())

    bag_yaml = bag_dir / "data.yaml"
    bag_yaml.write_text(yaml.safe_dump({
        "train": str(images_dir.resolve()),
        "val": [str(resolve(value)) for value in val] if isinstance(val := data["val"], list) else str(resolve(val)),
        "names": data["names"],
    }, sort_keys=False), encoding="utf-8")
    return bag_yaml, len(sample)


class Soup:
    """Fine-tune and evaluate the soup members configured in config.yaml."""

    def __init__(self, config: Config) -> None:
        """Bind to the configuration and load every augmentation config up front."""

        if config.soup is None:
            raise ValueError("config.yaml must define a soup section")
        self.config = config
        self.settings = config.soup
        self.data_yaml = self.settings.dataset
        self.baseline_path = self.settings.baseline_model
        self.augmentations = [load_augmentation(path) for path in self.settings.configs]
        names = [augmentation.name for augmentation in self.augmentations]
        if len(set(names)) != len(names):
            raise ValueError("soup.configs file names must be unique")

    def evaluate(self, name: str, model_path: Path) -> EvaluationResult:
        """Validate `model_path` on the dataset's val split.

        Uses the soup's imgsz/batch/device/workers so that every model is
        measured under the same conditions. Ultralytics output is suppressed.
        """

        if not model_path.is_file():
            raise FileNotFoundError(f"model not found: {model_path}")
        if not self.data_yaml.is_file():
            raise FileNotFoundError(f"dataset YAML not found: {self.data_yaml}")
        with yolo.quiet():
            model = yolo.load_model(model_path)
            results = model.val(
                data=str(self.data_yaml), split="val", plots=False, verbose=False,
                **self.settings.eval_kwargs(),
            )
        names = yolo.class_names(model)
        per_class: dict[str, ClassMetrics | None] = dict.fromkeys(names.values())
        # Only classes with val instances have results; box.maps would fill
        # the others with the overall mAP.
        for position, class_index in enumerate(results.ap_class_index):
            per_class[names[int(class_index)]] = ClassMetrics(*(float(value) for value in results.box.class_result(position)))
        return EvaluationResult(name=name, model_path=model_path, per_class=per_class)

    def _fine_tune(
        self, augmentation: AugmentationConfig, data_yaml: Path, work_dir: Path, name: str, label: str,
    ) -> Path:
        """Fine-tune the baseline on one bag and return the final weights.

        Per-epoch validation is skipped and the last epoch's weights are kept,
        so the val split is never used to pick a checkpoint. Ultralytics'
        console output is suppressed; per-epoch progress is shown instead,
        prefixed with `label`.
        """

        settings = self.settings
        arguments: dict[str, Any] = {
            **FINE_TUNE_DEFAULTS, "close_mosaic": settings.close_mosaic, **augmentation.train_args,
        }
        transforms = augmentation.build_transforms()
        if transforms:
            arguments["augmentations"] = transforms
        terminal = sys.stdout  # captured before quiet() redirects it
        with yolo.quiet():
            model = yolo.load_model(self.baseline_path)
            EpochProgress(label, terminal).attach(model)
            model.train(
                data=str(data_yaml), project=str(work_dir), name=name, exist_ok=True,
                epochs=settings.epochs, lr0=settings.lr0, seed=settings.seed,
                optimizer=settings.optimizer, batch=settings.batch, imgsz=settings.imgsz, device=settings.device,
                workers=settings.workers, amp=settings.amp, verbose=False,
                val=False, plots=False, **arguments,
            )
        weights = work_dir / name / "weights" / "last.pt"
        if not weights.is_file():
            raise FileNotFoundError(f"fine-tune completed without last.pt: {weights}")
        return weights

    def train_members(self) -> list[Path]:
        """Fine-tune any missing members and return the paths of all members.

        A member whose `.pt` file already exists in the output directory is
        reused, not retrained; delete the file to retrain it. Each new member
        is written there as soon as it finishes, so an interrupted run resumes
        where it stopped. Ultralytics run outputs live in a temporary directory
        that is deleted afterwards.
        """

        settings = self.settings
        if not self.baseline_path.is_file():
            raise FileNotFoundError(f"baseline model not found: {self.baseline_path}")
        if not self.data_yaml.is_file():
            raise FileNotFoundError(f"dataset YAML not found: {self.data_yaml}")
        output_dir = settings.output_dir
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        total = settings.bagging_iterations * len(self.augmentations)
        LOG.message(
            f"soup of {total} members ({settings.bagging_iterations} bags x {len(self.augmentations)} configs, "
            f"{settings.epochs} epochs each) from {self.baseline_path.name}"
        )

        output_dir.mkdir(exist_ok=True)
        members: list[Path] = []
        trained = 0
        with tempfile.TemporaryDirectory(prefix=".soup-work-", dir=output_dir.parent) as work:
            work_dir = Path(work)
            for iteration in range(1, settings.bagging_iterations + 1):
                bag: tuple[Path, int] | None = None  # built only if a member of this bag needs training
                for augmentation in self.augmentations:
                    name = f"bag{iteration}_{augmentation.name}"
                    target = output_dir / f"{name}.pt"
                    label = f"member {len(members) + 1}/{total} {name}"
                    if target.is_file():
                        LOG.message(f"{label}: already exists, not retraining")
                        LOG.message(f"  using {target}")
                        members.append(target)
                        continue
                    if bag is None:
                        # Seeded by bag number, so the bag is identical to the one existing members used.
                        bag = make_bag(
                            self.data_yaml, settings.bagging_fraction, settings.seed + iteration,
                            work_dir / f"bag{iteration}",
                        )
                    bag_yaml, size = bag
                    started = time.monotonic()
                    weights = self._fine_tune(augmentation, bag_yaml, work_dir / "runs", name, label)
                    losses = _final_losses(weights.parent.parent / "results.csv")
                    # Publish via a temporary name so an interrupted copy never looks like a finished member.
                    partial = target.with_name(target.name + ".part")
                    shutil.move(weights, partial)
                    members.append(partial.replace(target))
                    trained += 1
                    LOG.message(
                        f"{label}: {size} images, {settings.epochs} epochs in "
                        f"{_duration(time.monotonic() - started)}, final train loss {losses}"
                    )
                    LOG.message(f"  saved {target}")
        LOG.message(f"{trained} members trained, {len(members) - trained} reused, in {output_dir}")
        return members

    def run(self) -> list[EvaluationResult]:
        """Train all members, then evaluate the baseline and every member and log the tables."""

        members = self.train_members()
        LOG.message("validating baseline and soup members")
        results = [self.evaluate("baseline", self.baseline_path)]
        results += [self.evaluate(path.stem, path) for path in members]
        log_tables(results)
        return results


def _final_losses(results_csv: Path) -> str:
    """Summarise the last epoch's training losses from Ultralytics' results.csv."""

    if not results_csv.is_file():
        return "n/a"
    with results_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return "n/a"
    # Column names differ between model families (e.g. dfl_loss vs l1_loss).
    losses = {
        key.strip().removeprefix("train/").removesuffix("_loss"): float(value)
        for key, value in rows[-1].items() if key and key.strip().startswith("train/") and value.strip()
    }
    return " ".join(f"{name} {value:.3f}" for name, value in losses.items()) or "n/a"


def _duration(seconds: float) -> str:
    """Format a duration as e.g. `4m07s`."""

    minutes, seconds = divmod(round(seconds), 60)
    return f"{minutes}m{seconds:02d}s"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]], *, text_columns: int = 1) -> list[str]:
    """Format rows with the first `text_columns` left-aligned and the rest right-aligned."""

    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]
    lines = []
    for row in (headers, *rows):
        cells = [
            str(cell).ljust(width) if index < text_columns else str(cell).rjust(width)
            for index, (cell, width) in enumerate(zip(row, widths))
        ]
        lines.append("  ".join(cells))
    lines.insert(1, "  ".join("-" * width for width in widths))
    return lines


# Column headers shared by both tables.
METRIC_HEADERS = ("P", "R", "mAP50", "mAP50-95")


def _metric_cells(metrics: ClassMetrics | None) -> tuple[str, ...]:
    """Format one row's metric cells, or dashes for a class absent from val."""

    if metrics is None:
        return ("-",) * len(METRIC_HEADERS)
    return tuple(f"{value:.4f}" for value in (metrics.precision, metrics.recall, metrics.map50, metrics.map50_95))


def log_tables(results: Sequence[EvaluationResult]) -> None:
    """Log the per-model per-class table, then the macro-averaged summary last.

    All scores are on the val split. The summary averages each metric over
    the classes present in val, without weighting by instance count.
    """

    LOG.message("per-class val metrics")
    rows = []
    for result in results:
        for index, (class_name, metrics) in enumerate(result.per_class.items()):
            # Show the model name only on its first row so groups stand out.
            rows.append((result.name if index == 0 else "", class_name, *_metric_cells(metrics)))
    for line in _table(("model", "class", *METRIC_HEADERS), rows, text_columns=2):
        LOG.message(f"  {line}")

    LOG.message("summary: macro-averaged val metrics")
    for line in _table(("model", *METRIC_HEADERS), [(r.name, *_metric_cells(r.macro)) for r in results]):
        LOG.message(f"  {line}")

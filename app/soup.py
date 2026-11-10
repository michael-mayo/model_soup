"""Model soup: bagged, augmentation-diverse fine-tunes of a baseline, and ways to combine them.

Each soup member is the baseline fine-tuned on one bag (a random subset of the
training images) with one augmentation config. The members are then combined
by output ensembling (NMS, WBF) and by weight averaging (uniform and greedy
soups). Everything is scored the same way on the val and test splits; only
val scores are used to make choices.
"""

from __future__ import annotations

import csv
import random
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import yaml

from app import ensemble, merging, yolo
from app.config import Config
from app.evaluation import Evaluator, Predictions, ReportRow, log_summary, split_images
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


def make_bag(data_yaml: Path, fraction: float, seed: int, bag_dir: Path) -> tuple[Path, int]:
    """Create a bagged copy of the dataset and return its data.yaml and image count.

    A `fraction` of the training images is sampled without replacement. The
    bag holds symlinks to the original images and labels (Ultralytics finds
    labels by swapping `images` for `labels` in the path), and validates on
    the original val images, listed in a file.
    """

    from ultralytics.data.utils import img2label_paths

    train_images = split_images(data_yaml, "train")
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

    val_list = bag_dir / "val.txt"
    val_list.write_text("".join(f"{path}\n" for path in split_images(data_yaml, "val")), encoding="utf-8")
    names = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"]
    bag_yaml = bag_dir / "data.yaml"
    bag_yaml.write_text(yaml.safe_dump(
        {"train": str(images_dir.resolve()), "val": str(val_list.resolve()), "names": names}, sort_keys=False,
    ), encoding="utf-8")
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

    def _fine_tune(
        self, augmentation: AugmentationConfig, data_yaml: Path, work_dir: Path, name: str, label: str,
    ) -> dict[int, Path]:
        """Fine-tune the baseline on one bag and return its checkpoints by epoch.

        After each epoch in `checkpoint_epochs()` the (EMA) weights are copied
        out; the final epoch's weights are Ultralytics' last.pt. Checkpoints
        are fixed epochs, never chosen by val score: per-epoch validation is
        off. Ultralytics' console output is suppressed; per-epoch progress is
        shown instead, prefixed with `label`.
        """

        settings = self.settings
        arguments: dict[str, Any] = {
            **FINE_TUNE_DEFAULTS, "close_mosaic": settings.close_mosaic, **augmentation.train_args,
        }
        transforms = augmentation.build_transforms()
        if transforms:
            arguments["augmentations"] = transforms
        intermediate = set(settings.checkpoint_epochs()[:-1])
        checkpoints: dict[int, Path] = {}

        def save_checkpoint(trainer: Any) -> None:
            # Called after Ultralytics writes last.pt for the epoch just finished.
            epoch = trainer.epoch + 1
            if epoch in intermediate:
                checkpoints[epoch] = Path(shutil.copy2(trainer.last, work_dir / f"{name}_e{epoch}.pt"))

        terminal = sys.stdout  # captured before quiet() redirects it
        with yolo.quiet():
            model = yolo.load_model(self.baseline_path)
            EpochProgress(label, terminal).attach(model)
            model.add_callback("on_model_save", save_checkpoint)
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
        missing = intermediate - set(checkpoints)
        if missing:
            raise RuntimeError(f"fine-tune did not save checkpoints for epochs {sorted(missing)}")
        from ultralytics.utils.torch_utils import strip_optimizer

        with yolo.quiet():
            for path in checkpoints.values():
                strip_optimizer(path)  # keep the fp16 EMA weights only, like Ultralytics' final last.pt
        checkpoints[settings.epochs] = weights
        return dict(sorted(checkpoints.items()))

    def train_members(self) -> list[Path]:
        """Run any missing fine-tunes and return the paths of all member checkpoints.

        Each fine-tune (one bag x one augmentation config) yields one member
        per checkpoint epoch, named `bag<i>_<config>_e<epoch>.pt`. A fine-tune
        whose checkpoints all exist in the output directory is reused, not
        retrained; deleting any of them retrains it and rewrites them all.
        Checkpoints are written there as soon as their fine-tune finishes, so
        an interrupted run resumes where it stopped. Ultralytics run outputs
        live in a temporary directory that is deleted afterwards.
        """

        settings = self.settings
        if not self.baseline_path.is_file():
            raise FileNotFoundError(f"baseline model not found: {self.baseline_path}")
        if not self.data_yaml.is_file():
            raise FileNotFoundError(f"dataset YAML not found: {self.data_yaml}")
        output_dir = settings.output_dir
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        total = settings.bagging_iterations * len(self.augmentations)
        epochs = settings.checkpoint_epochs()
        LOG.message(
            f"soup of {total * len(epochs)} members: {total} fine-tunes ({settings.bagging_iterations} bags x "
            f"{len(self.augmentations)} configs) x {len(epochs)} checkpoints (epochs {', '.join(map(str, epochs))}) "
            f"from {self.baseline_path.name}"
        )

        output_dir.mkdir(exist_ok=True)
        members: list[Path] = []
        trained = runs = 0
        with tempfile.TemporaryDirectory(prefix=".soup-work-", dir=output_dir.parent) as work:
            work_dir = Path(work)
            for iteration in range(1, settings.bagging_iterations + 1):
                bag: tuple[Path, int] | None = None  # built only if a member of this bag needs training
                for augmentation in self.augmentations:
                    name = f"bag{iteration}_{augmentation.name}"
                    targets = {epoch: output_dir / f"{name}_e{epoch}.pt" for epoch in epochs}
                    runs += 1
                    label = f"fine-tune {runs}/{total} {name}"
                    saved = f"{output_dir}/{name}_e{{{','.join(map(str, epochs))}}}.pt"
                    if all(path.is_file() for path in targets.values()):
                        LOG.message(f"{label}: all {len(epochs)} checkpoints exist, not retraining")
                        LOG.message(f"  using {saved}")
                        members += targets.values()
                        continue
                    if bag is None:
                        # Seeded by bag number, so the bag is identical to the one existing members used.
                        bag = make_bag(
                            self.data_yaml, settings.bagging_fraction, settings.seed + iteration,
                            work_dir / f"bag{iteration}",
                        )
                    bag_yaml, size = bag
                    started = time.monotonic()
                    checkpoints = self._fine_tune(augmentation, bag_yaml, work_dir / "runs", name, label)
                    losses = _final_losses(work_dir / "runs" / name / "results.csv")
                    for epoch, path in checkpoints.items():
                        # Publish via a temporary name so an interrupted copy never looks like a finished member.
                        partial = targets[epoch].with_name(targets[epoch].name + ".part")
                        shutil.move(path, partial)
                        members.append(partial.replace(targets[epoch]))
                    trained += len(checkpoints)
                    LOG.message(
                        f"{label}: {size} images, {settings.epochs} epochs in "
                        f"{_duration(time.monotonic() - started)}, final train loss {losses}"
                    )
                    LOG.message(f"  saved {saved}")
        LOG.message(f"{trained} members trained, {len(members) - trained} reused, in {output_dir}")
        return members

    def _evaluators(self) -> tuple[Evaluator, Evaluator | None]:
        """Return the val evaluator and, if the dataset has a test split, the test evaluator."""

        val = Evaluator(self.data_yaml, self.settings, "val")
        try:
            test: Evaluator | None = Evaluator(self.data_yaml, self.settings, "test")
        except (ValueError, FileNotFoundError):
            test = None
        return val, test

    def _row(
        self, name: str, path: Path, val: Evaluator, test: Evaluator | None,
    ) -> tuple[ReportRow, dict[str, Predictions]]:
        """Evaluate one model on val (and test) and log a one-line result."""

        val_result, val_predictions = val.evaluate(name, path)
        predictions = {"val": val_predictions}
        test_result = None
        if test is not None:
            test_result, predictions["test"] = test.evaluate(name, path)
        row = ReportRow(name, val_result, test_result)
        LOG.message(f"  {_row_text(row)}")
        return row, predictions

    def _build_soup(self, members: list[Path], target: Path, calibration: list[Path]) -> Path:
        """Average `members` into `target`, recalibrating BatchNorm on `calibration` images."""

        settings = self.settings
        with yolo.quiet():
            return merging.build_soup(
                members, target, calibration_images=calibration, imgsz=settings.imgsz,
                batch=min(int(settings.batch), 32), device=settings.device,
            )

    def combine(self, members: list[Path]) -> tuple[list[ReportRow], list[str]]:
        """Evaluate the baseline, members, ensembles and soups on val and test.

        Only val scores are used to make choices (the greedy soup); test
        scores are reported alongside so that choices made on val can be
        checked on unseen data. Returns the rows in table order and notes for
        the summary. Soup checkpoints are saved next to the members as
        soup_uniform.pt and soup_greedy.pt.
        """

        settings = self.settings
        val, test = self._evaluators()
        LOG.message(
            f"evaluating on {len(val.images)} val images"
            + (f" and {len(test.images)} test images" if test else " (the dataset has no test split)")
        )
        rows = [self._row("baseline", self.baseline_path, val, test)[0]]
        member_rows: dict[Path, ReportRow] = {}
        member_predictions: list[dict[str, Predictions]] = []
        for path in members:
            row, predictions = self._row(path.stem, path, val, test)
            member_rows[path] = row
            member_predictions.append(predictions)
        rows += member_rows.values()

        for name, fuse, iou in (("ensemble/NMS", ensemble.ensemble_nms, settings.nms_iou),
                                ("ensemble/WBF", ensemble.ensemble_wbf, settings.wbf_iou)):
            row = ReportRow(
                name,
                val.score(name, fuse([predictions["val"] for predictions in member_predictions], iou)),
                test.score(name, fuse([predictions["test"] for predictions in member_predictions], iou))
                if test else None,
            )
            LOG.message(f"  {_row_text(row)}")
            rows.append(row)

        calibration = merging.sample_images(
            split_images(self.data_yaml, "train"), settings.bn_recalibration_images, settings.seed,
        )
        uniform = self._build_soup(members, settings.output_dir / "soup_uniform.pt", calibration)
        rows.append(self._row("soup/uniform", uniform, val, test)[0])

        # Greedy soup: try members best-first by val score, keeping each only if the soup improves on val.
        ranked = sorted(members, key=lambda path: member_rows[path].val.macro.map50_95, reverse=True)
        LOG.message(f"greedy soup (selecting on val): starting from {ranked[0].stem}")
        with tempfile.TemporaryDirectory(prefix=".soup-greedy-", dir=settings.output_dir.parent) as work:
            def score_subset(subset: list[Path]) -> float:
                candidate = self._build_soup(subset, Path(work) / "candidate.pt", calibration)
                return val.evaluate("soup/greedy", candidate)[0].macro.map50_95

            def report(path: Path, score: float, kept: bool) -> None:
                LOG.message(f"  + {path.stem}: val mAP50-95 {score:.4f}, {'kept' if kept else 'rejected'}")

            selected = merging.greedy_selection(
                ranked, member_rows[ranked[0]].val.macro.map50_95, score_subset, report,
            )
        greedy_path = settings.output_dir / "soup_greedy.pt"
        if len(selected) == 1:
            shutil.copy2(selected[0], greedy_path)
        else:
            self._build_soup(selected, greedy_path, calibration)
        LOG.message(f"greedy soup: {len(selected)} of {len(members)} members ({', '.join(p.stem for p in selected)})")
        rows.append(self._row("soup/greedy", greedy_path, val, test)[0])

        notes = [
            "ensemble val time includes every member's predictions plus fusion",
            "soup/greedy chose its members on val, so its val scores are optimistic"
            + ("; compare on test" if test else ""),
        ]
        return rows, notes

    def run(self) -> list[ReportRow]:
        """Train any missing members, then evaluate everything and log the summary."""

        members = self.train_members()
        rows, notes = self.combine(members)
        log_summary(rows, notes)
        return rows


def _row_text(row: ReportRow) -> str:
    """One-line progress text for an evaluated row."""

    test = f", test {row.test.macro.map50_95:.4f}" if row.test else ""
    return f"{row.name}: mAP50-95 val {row.val.macro.map50_95:.4f}{test}, val time {row.val.seconds:.3f}s"


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

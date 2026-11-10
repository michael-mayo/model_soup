"""Val- and test-split evaluation shared by single models, ensembles and soups.

Every model or method is scored the same way. Each model's per-image val
predictions are captured from Ultralytics' own validator and scored with the
same matching and average-precision code, so a single model reproduces
`model.val()` exactly, and ensembles (which have no single Ultralytics model
to validate) fuse exactly those predictions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from app import yolo
from app.config import RunSettings
from app.log import Log


LOG = Log("soup")

# Ultralytics' validation defaults: keep almost every detection for the PR curve.
PREDICT_CONF = 0.001
MAX_DETECTIONS = 300

# IoU thresholds 0.50:0.05:0.95 used for mAP50-95, as in Ultralytics.
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


@dataclass(frozen=True)
class Detections:
    """One image's detections in original-image pixels, sorted by descending confidence."""

    boxes: np.ndarray    # (N, 4) xyxy
    scores: np.ndarray   # (N,)
    classes: np.ndarray  # (N,) int

    @classmethod
    def empty(cls) -> Detections:
        """Return an empty detection set."""

        return cls(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64))

    def top(self, count: int) -> Detections:
        """Return the `count` most confident detections, sorted."""

        order = np.argsort(-self.scores, kind="stable")[:count]
        return Detections(self.boxes[order], self.scores[order], self.classes[order])


@dataclass(frozen=True)
class Predictions:
    """A model's (or method's) detections for every val image, and the time taken."""

    detections: list[Detections]
    seconds: float


@dataclass(frozen=True)
class ClassMetrics:
    """Detection metrics on one split for one class (or their macro average)."""

    precision: float
    recall: float
    map50: float
    map50_95: float


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics on one split for one model or method.

    `per_class` maps every class name to its metrics, or to `None` when the
    split has no instances of that class. `seconds` is the time to
    produce all val predictions (for an ensemble, every member plus fusion).
    """

    name: str
    per_class: dict[str, ClassMetrics | None]
    seconds: float

    @property
    def macro(self) -> ClassMetrics:
        """Unweighted mean over the classes present in the split."""

        present = [metrics for metrics in self.per_class.values() if metrics is not None]
        if not present:
            return ClassMetrics(0.0, 0.0, 0.0, 0.0)
        return ClassMetrics(*(
            sum(getattr(metrics, key) for metrics in present) / len(present)
            for key in ("precision", "recall", "map50", "map50_95")
        ))


def split_images(data_yaml: Path, split: str) -> list[Path]:
    """Return the sorted image paths of one split of an Ultralytics data.yaml.

    The split may be an image directory, an image-list file, or a list of
    either; relative entries are resolved against `path` (default: the YAML's
    directory), as Ultralytics does.
    """

    from ultralytics.data.utils import IMG_FORMATS

    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data.get("path") or data_yaml.parent)
    root = root if root.is_absolute() else data_yaml.parent / root

    def resolve(value: str) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else root / path).resolve()

    entries = data.get(split)
    if not entries:
        raise ValueError(f"{data_yaml} has no `{split}` split")
    images: list[Path] = []
    for entry in entries if isinstance(entries, list) else [entries]:
        location = resolve(str(entry))
        if location.is_dir():
            images += location.rglob("*")
        else:  # an Ultralytics image-list file
            lines = location.read_text(encoding="utf-8").splitlines()
            images += [resolve(line.strip()) for line in lines if line.strip()]
    return sorted(path for path in images if path.suffix[1:].lower() in IMG_FORMATS and path.is_file())


def _match(detections: Detections, true_classes: np.ndarray, true_boxes: np.ndarray) -> np.ndarray:
    """Mark each detection correct or not at every IoU threshold.

    Same greedy matching as Ultralytics' validator: detections in descending
    confidence order each claim the best still-unclaimed target of their class.
    """

    from ultralytics.utils.metrics import box_iou

    correct = np.zeros((len(detections.scores), len(IOU_THRESHOLDS)), dtype=bool)
    if not len(true_classes) or not len(detections.scores):
        return correct
    iou = box_iou(torch.from_numpy(true_boxes).float(), torch.from_numpy(detections.boxes).float()).numpy()
    iou = iou * (true_classes[:, None] == detections.classes[None, :])
    matched = np.zeros((len(true_classes), len(IOU_THRESHOLDS)), dtype=bool)
    for j in np.flatnonzero((iou >= IOU_THRESHOLDS.min()).any(0)):
        available = np.where(matched, 0, iou[:, j, None])
        k = available.argmax(0)
        correct[j] = available[k, range(len(IOU_THRESHOLDS))] >= IOU_THRESHOLDS
        matched[k, range(len(IOU_THRESHOLDS))] |= correct[j]
    return correct


class Evaluator:
    """Predict on and score one split (val or test) of one dataset."""

    def __init__(self, data_yaml: Path, settings: RunSettings, split: str = "val") -> None:
        """Index the split's images and read the class names."""

        self.data_yaml = data_yaml
        self.settings = settings
        self.split = split
        self.images = split_images(data_yaml, split)
        if not self.images:
            raise FileNotFoundError(f"no {split} images found for {data_yaml}")
        names = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"]
        self.names: dict[int, str] = dict(names) if isinstance(names, dict) else dict(enumerate(names))
        self._order: list[str] | None = None  # val image files, fixed by the first prediction pass
        self._targets: list[tuple[np.ndarray, np.ndarray]] | None = None

    def predict(self, model_path: Path) -> Predictions:
        """Run one model over the split with Ultralytics' own validator.

        The validator's per-image predictions and targets are captured (scaled
        back to original-image pixels) instead of being reduced to metrics, so
        a single model scores exactly as `model.val()` does and ensembles can
        fuse the same predictions. The time is the validator's measured
        preprocess + inference + postprocess time over all the split's images, which
        excludes model loading and warm-up. Ultralytics output is suppressed.
        """

        from ultralytics.models.yolo.detect import DetectionValidator
        from ultralytics.utils import ops

        detections: dict[str, Detections] = {}
        targets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        original_update = DetectionValidator.update_metrics

        def capture(validator: DetectionValidator, preds: list[dict[str, torch.Tensor]], batch: dict) -> None:
            for index, pred in enumerate(preds):
                prepared = validator._prepare_batch(index, batch)
                pred = validator._prepare_pred(pred)

                def native(boxes: torch.Tensor) -> np.ndarray:
                    scaled = ops.scale_boxes(prepared["imgsz"], boxes.clone().float(), prepared["ori_shape"],
                                             ratio_pad=prepared["ratio_pad"])
                    return scaled.cpu().numpy().reshape(-1, 4)

                image = str(prepared["im_file"])
                detections[image] = Detections(
                    native(pred["bboxes"]), pred["conf"].float().cpu().numpy(),
                    pred["cls"].cpu().numpy().astype(np.int64),
                ).top(MAX_DETECTIONS)
                targets[image] = (prepared["cls"].cpu().numpy().astype(np.int64), native(prepared["bboxes"]))
            original_update(validator, preds, batch)

        DetectionValidator.update_metrics = capture
        try:
            with yolo.quiet():
                results = yolo.load_model(model_path).val(
                    data=str(self.data_yaml), split=self.split, plots=False, verbose=False,
                    conf=PREDICT_CONF, max_det=MAX_DETECTIONS, **self.settings.eval_kwargs(),
                )
        finally:
            DetectionValidator.update_metrics = original_update

        order = sorted(detections)
        if self._order is None:
            self._order = order
            self._targets = [targets[image] for image in order]
        elif order != self._order:
            raise RuntimeError(f"{model_path.name} was validated on a different set of images")
        per_image_ms = sum(value for key, value in results.speed.items() if key != "loss")
        return Predictions([detections[image] for image in order], per_image_ms * len(order) / 1000)

    def score(self, name: str, predictions: Predictions) -> EvaluationResult:
        """Compute per-class P, R, mAP50 and mAP50-95 for a set of predictions on the split."""

        from ultralytics.utils.metrics import ap_per_class

        if self._targets is None:
            raise RuntimeError("predict() must run before score()")
        correct, scores, classes, true_classes = [], [], [], []
        for detections, (target_classes, target_boxes) in zip(predictions.detections, self._targets):
            correct.append(_match(detections, target_classes, target_boxes))
            scores.append(detections.scores)
            classes.append(detections.classes)
            true_classes.append(target_classes)
        _, _, precision, recall, _, ap, present, *_ = ap_per_class(
            np.concatenate(correct), np.concatenate(scores), np.concatenate(classes), np.concatenate(true_classes),
        )
        per_class: dict[str, ClassMetrics | None] = dict.fromkeys(self.names.values())
        for position, class_index in enumerate(present):
            per_class[self.names[int(class_index)]] = ClassMetrics(
                float(precision[position]), float(recall[position]),
                float(ap[position, 0]), float(ap[position].mean()),
            )
        return EvaluationResult(name=name, per_class=per_class, seconds=predictions.seconds)

    def evaluate(self, name: str, model_path: Path) -> tuple[EvaluationResult, Predictions]:
        """Predict with and score one model; return the result and its predictions."""

        if not model_path.is_file():
            raise FileNotFoundError(f"model not found: {model_path}")
        predictions = self.predict(model_path)
        return self.score(name, predictions), predictions


@dataclass(frozen=True)
class ReportRow:
    """One summary row: a model or method scored on val and, if available, test."""

    name: str
    val: EvaluationResult
    test: EvaluationResult | None = None


# Metric columns repeated for each split in the summary.
METRIC_HEADERS = ("P", "R", "mAP50", "mAP50-95")


def _metric_cells(result: EvaluationResult | None) -> tuple[str, ...]:
    """Format a result's macro-averaged metrics, or dashes when there is no result."""

    if result is None:
        return ("-",) * len(METRIC_HEADERS)
    macro = result.macro
    return tuple(f"{value:.4f}" for value in (macro.precision, macro.recall, macro.map50, macro.map50_95))


def log_summary(rows: Sequence[ReportRow], notes: Sequence[str] = ()) -> None:
    """Log the summary table: macro-averaged val (and test) metrics and val time per model.

    Each metric is an unweighted mean over the classes present in the split.
    The test columns appear only when rows have test results (the dataset has
    a test split). A group header above the column headers labels the split
    of each column group. `notes` are logged under the table.
    """

    splits = ["val", "test"] if any(row.test is not None for row in rows) else ["val"]
    headers = ("model", *METRIC_HEADERS * len(splits), "val time (s)")
    body = [
        (row.name, *(cell for split in splits for cell in _metric_cells(getattr(row, split))),
         f"{row.val.seconds:.3f}")
        for row in rows
    ]
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *body)]
    gap = "  "

    def span(first: int, count: int, label: str) -> str:
        width = sum(widths[first:first + count]) + len(gap) * (count - 1)
        return f" {label} ".center(width, "-")

    metrics = len(METRIC_HEADERS)
    groups = gap.join([" " * widths[0], *(span(1 + index * metrics, metrics, split)
                                          for index, split in enumerate(splits))])
    lines = [groups.rstrip()]
    for row in (headers, *body):
        lines.append(gap.join(
            str(cell).ljust(width) if index == 0 else str(cell).rjust(width)
            for index, (cell, width) in enumerate(zip(row, widths))
        ))
    lines.insert(2, gap.join("-" * width for width in widths))

    LOG.message("summary: macro-averaged metrics")
    for line in lines:
        LOG.message(f"  {line}")
    for note in notes:
        LOG.message(f"  {note}")

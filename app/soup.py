"""Model soup: evaluation of the baseline that weight averaging will start from."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app import yolo
from app.config import Config
from app.log import Log


LOG = Log(__name__)


@dataclass(frozen=True)
class EvaluationResult:
    """Validation mAP50-95 for one model, overall and per class."""

    model_path: Path
    map50_95: float
    per_class: dict[str, float]


class Soup:
    """Build and evaluate a model soup from configured ingredients.

    Weight averaging is not implemented yet; for now the soup only evaluates
    its baseline model.
    """

    def __init__(self, config: Config) -> None:
        """Bind the soup to the application configuration."""

        if config.soup is None:
            raise ValueError("config.yaml must define a soup section")
        self.config = config
        self.settings = config.soup

    def evaluate(self, model_path: Path, data_yaml: Path) -> EvaluationResult:
        """Validate `model_path` on the val split of `data_yaml`.

        Uses the training imgsz/batch/device/workers so that results match the
        conditions the baseline was trained under. Ultralytics output is
        suppressed.
        """

        if not model_path.is_file():
            raise FileNotFoundError(f"model not found: {model_path}")
        if not data_yaml.is_file():
            raise FileNotFoundError(f"dataset YAML not found: {data_yaml}")
        with yolo.quiet():
            model = yolo.load_model(model_path)
            results = model.val(
                data=str(data_yaml), split="val", plots=False, verbose=False,
                **self.config.training.eval_kwargs(),
            )
        names = yolo.class_names(model)
        per_class = {
            names.get(index, str(index)): float(value) for index, value in enumerate(results.box.maps)
        }
        return EvaluationResult(model_path=model_path, map50_95=float(results.box.map), per_class=per_class)

    def evaluate_baseline(self) -> EvaluationResult:
        """Evaluate the configured baseline model and log its metrics."""

        layout = self.config.layout(self.settings.dataset)
        result = self.evaluate(layout.model_path(self.settings.baseline), layout.data_yaml)
        LOG.message(f"baseline model: {result.model_path}")
        LOG.message(f"  overall val mAP50-95: {result.map50_95:.4f}")
        LOG.message("  per-class val mAP50-95:")
        for name, value in result.per_class.items():
            LOG.message(f"    {name:<20} {value:.4f}")
        return result

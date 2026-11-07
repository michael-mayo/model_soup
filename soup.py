"""Validate and display results for the configured baseline model."""

from __future__ import annotations

import contextlib
import io
import logging
from pathlib import Path
from typing import Any

from log import Log


class Soup:
    """Validate and display results for the configured baseline model."""

    @staticmethod
    def _path(project_root: Path, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else project_root / path

    @staticmethod
    def _dataset_yaml(model_path: Path) -> Path:
        """Infer the dataset YAML from the standard trained-model layout."""

        # .../scratch/datasets/<dataset>/models/<model>.pt
        data_yaml = model_path.parent.parent / "yolo" / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(
                f"could not find the dataset YAML for {model_path}: {data_yaml}"
            )
        return data_yaml


    def validate(self, project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
        """Validate the configured model and print its validation metrics."""

        settings = config.get("soup", {})
        if not isinstance(settings, dict) or not settings.get("baseline_model"):
            raise ValueError("config.yaml must define soup.baseline_model")

        model_path = self._path(project_root, str(settings["baseline_model"]))
        if not model_path.exists():
            raise FileNotFoundError(f"baseline model not found: {model_path}")
        data_yaml = self._dataset_yaml(model_path)

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed in the soup environment; "
                "run `conda run -n soup python -m pip install -r requirements.txt`."
            ) from exc

    # Keep Ultralytics' progress bars, logger, and warnings out of the app's
    # console output. The original logging setting is restored afterwards.
        previous_logging_level = logging.root.manager.disable
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            logging.disable(logging.CRITICAL)
            try:
                model = YOLO(str(model_path))
                results = model.val(
                    data=str(data_yaml),
                    plots=False,
                    verbose=False,
                )
            finally:
                logging.disable(previous_logging_level)

        metrics = getattr(results, "results_dict", {}) or {}
        box_metrics = getattr(results, "box", None)
        if box_metrics is None:
            raise RuntimeError("validation did not return detection metrics")

        names = model.names
        if not isinstance(names, dict):
            names = dict(enumerate(names))

        Log.message(__file__, "Baseline model [soup]")
        Log.message(__file__, f"  model: {model_path}")
        Log.message(__file__, f"  overall val mAP50-95: {box_metrics.map:.4f}")
        Log.message(__file__, "  per-class val mAP50-95:")
        for class_index, class_map in enumerate(box_metrics.maps):
            class_name = names.get(class_index, str(class_index))
            Log.message(__file__, f"    {class_name:<20} {float(class_map):.4f}")
        return metrics

"""Train baseline Ultralytics detectors for the prepared datasets."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).with_name("config.yaml")


def _path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _first_epoch_val_map(run_dir: Path) -> float | None:
    """Read epoch-one validation mAP from Ultralytics' persisted results."""

    results_path = run_dir / "results.csv"
    if not results_path.exists():
        return None
    with results_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(float(row["epoch"])) == 1:
                value = row.get("metrics/mAP50-95(B)", "").strip()
                return float(value) if value else None
    return None


def _best_logged_val_map(run_dir: Path) -> float | None:
    """Return the highest validation mAP recorded during training."""

    results_path = run_dir / "results.csv"
    if not results_path.exists():
        return None
    values: list[float] = []
    with results_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            value = row.get("metrics/mAP50-95(B)", "").strip()
            if value:
                values.append(float(value))
    return max(values) if values else None


def _clear_previous_outputs(runs_path: Path, run_name: str, model_path: Path) -> None:
    """Remove only the configured run and checkpoint before a fresh run."""

    run_dir = (runs_path / run_name).resolve()
    runs_root = runs_path.resolve()
    if run_dir.parent != runs_root:
        raise ValueError(f"run name must identify a direct child of {runs_root}: {run_name!r}")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    if model_path.is_symlink() or model_path.is_file():
        model_path.unlink()
    elif model_path.exists():
        raise ValueError(f"configured model path is not a file: {model_path}")


def load_settings(config_path: Path = CONFIG_PATH) -> tuple[Path, dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return config_path.parent, config.get("dataset", {}).get("trainer", {})


def _train_dataset(project_root: Path, dataset_name: str, settings: dict[str, Any]) -> dict[str, str]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed in the soup environment; "
            "run `conda run -n soup python -m pip install -r requirements.txt`."
        ) from exc

    data_yaml = _path(project_root, settings["data_yaml"])
    run_name = str(settings.get("run_name", "baseline"))
    model_path_value = str(settings.get("model_path", f"scratch/datasets/{dataset_name}/models/{{run_name}}.pt"))
    model_path = _path(
        project_root,
        model_path_value.format(dataset=dataset_name, run_name=run_name),
    )
    runs_path = _path(project_root, settings["runs_dir"])
    weights_dir = _path(project_root, settings.get("weights_dir", "scratch/weights"))
    if not data_yaml.exists():
        raise FileNotFoundError(f"dataset YAML not found: {data_yaml}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    runs_path.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    _clear_previous_outputs(runs_path, run_name, model_path)
    model_spec = Path(str(settings["model"]))
    if not model_spec.is_absolute() and model_spec.suffix == ".pt":
        model_spec = weights_dir / model_spec.name
    model = YOLO(str(model_spec))

    epoch1_map5095: float | None = None

    def collect_epoch_metrics(trainer: Any) -> None:
        """Capture the first validation mAP without interrupting Ultralytics output."""

        nonlocal epoch1_map5095

        epoch = int(getattr(trainer, "epoch", -1)) + 1
        if epoch != 1:
            return
        metrics = getattr(trainer, "metrics", {}) or {}
        map5095 = metrics.get("metrics/mAP50-95(B)")
        if map5095 is not None:
            epoch1_map5095 = float(map5095)

    model.add_callback("on_fit_epoch_end", collect_epoch_metrics)
    results = model.train(
        data=str(data_yaml),
        project=str(runs_path),
        name=run_name,
        exist_ok=bool(settings.get("exist_ok", True)),
        epochs=int(settings.get("epochs", 3)),
        patience=int(settings.get("patience", 15)),
        # Use an explicit integer by default. Fractional values enable
        # Ultralytics AutoBatch, which can choose a batch too large for the
        # host even when GPU memory appears to be available.
        batch=settings.get("batch", 8),
        imgsz=int(settings.get("imgsz", 640)),
        optimizer=settings.get("optimizer", "MuSGD"),
        lr0=float(settings.get("lr0", 0.01)),
        device=settings.get("device", 0),
        # A zero-worker loader is slower but robust on memory-constrained
        # hosts and avoids multiprocessing semaphore leaks on interruption.
        workers=int(settings.get("workers", 0)),
        amp=bool(settings.get("amp", True)),
        seed=int(settings.get("seed", 0)),
        plots=bool(settings.get("plots", True)),
        verbose=bool(settings.get("verbose", True)),
    )

    run_dir = Path(results.save_dir)
    first_epoch_map5095 = _first_epoch_val_map(run_dir) or epoch1_map5095
    best_logged_val_map5095 = _best_logged_val_map(run_dir)
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"training completed without best.pt: {best_weights}")

    best_model = YOLO(str(best_weights))
    eval_settings = {
        "data": str(data_yaml),
        "imgsz": int(settings.get("imgsz", 640)),
        "device": settings.get("device", 0),
        "workers": int(settings.get("workers", 0)),
        "plots": False,
        "verbose": False,
    }
    train_metrics = best_model.val(split="train", **eval_settings)
    val_metrics = best_model.val(split="val", **eval_settings)
    train_map5095 = float(train_metrics.box.map)
    val_map5095 = (
        best_logged_val_map5095
        if best_logged_val_map5095 is not None
        else float(val_metrics.box.map)
    )
    shutil.copy2(best_weights, model_path)
    return {
        "dataset": dataset_name,
        "status": "complete",
        "model": str(model_path),
        "run": str(run_dir),
        "epoch1_val_map5095": f"{first_epoch_map5095:.4f}" if first_epoch_map5095 is not None else "n/a",
        "train_map5095": f"{train_map5095:.4f}",
        "val_map5095": f"{val_map5095:.4f}",
    }


def train_all(
    config_path: Path = CONFIG_PATH,
    selected: list[str] | str | None = None,
) -> list[dict[str, str]]:
    project_root, trainer = load_settings(config_path)
    datasets = trainer.get("datasets", {})
    defaults = {key: value for key, value in trainer.items() if key not in {"datasets", "datasets_to_train"}}
    selected_names = selected or trainer.get("datasets_to_train")
    if isinstance(selected_names, str):
        selected_names = [selected_names]
    if selected_names:
        missing = [name for name in selected_names if name not in datasets]
        if missing:
            raise KeyError(f"dataset(s) not configured: {', '.join(missing)}")
        datasets = {name: datasets[name] for name in selected_names}

    summary: list[dict[str, str]] = []
    for dataset_name, settings in datasets.items():
        try:
            summary.append(_train_dataset(project_root, dataset_name, {**defaults, **settings}))
        except Exception as exc:  # Keep the other baseline runs moving.
            summary.append({
                "dataset": dataset_name,
                "status": "failed",
                "model": "",
                "run": "",
                "error": str(exc),
            })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="train only the specified configured dataset; repeat for multiple datasets",
    )
    args = parser.parse_args()

    summary = train_all(args.config, args.datasets)
    print("\nBaseline training summary")
    print("=========================")
    for result in summary:
        print(f"[{result['dataset']}] {result['status']}")
        if result["model"]:
            print(f"  model: {result['model']}")
            print(f"  runs:  {result['run']}")
            print(f"  epoch 1 val mAP50-95: {result['epoch1_val_map5095']}")
            print(f"  best train mAP50-95: {result['train_map5095']}")
            print(f"  best val mAP50-95:   {result['val_map5095']}")
        elif result.get("error"):
            print(f"  error: {result['error']}")
    return 0 if all(result["status"] == "complete" for result in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())

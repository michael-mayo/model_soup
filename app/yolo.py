"""Thin wrappers around Ultralytics shared by training and soup evaluation."""

from __future__ import annotations

import contextlib
import io
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def load_model(path: str | Path) -> Any:
    """Load an Ultralytics YOLO model, with a clear error if it is not installed.

    The import is deferred so that commands which never touch a model (such as
    dataset preparation) do not pay Ultralytics' import cost.
    """

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed in the soup environment; "
            "run `conda run -n soup python -m pip install -r requirements.txt`."
        ) from exc
    return YOLO(str(path))


@contextlib.contextmanager
def quiet() -> Iterator[None]:
    """Suppress Ultralytics' progress bars, logger output, and warnings.

    Python-level stdout/stderr are captured and logging is disabled for the
    duration of the block; the previous logging state is restored afterwards.
    """

    previous_level = logging.root.manager.disable
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        logging.disable(logging.CRITICAL)
        try:
            yield
        finally:
            logging.disable(previous_level)


def class_names(model: Any) -> dict[int, str]:
    """Return a model's class names as an index-to-name mapping."""

    names = model.names
    return dict(names) if isinstance(names, dict) else dict(enumerate(names))

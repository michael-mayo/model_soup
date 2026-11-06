"""Console logging helpers for the soup application."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO


MODULE_WIDTH = 24


def _module_label(module_name: str | Path) -> str:
    """Return a filename-based, padded module label."""

    return f"[{Path(module_name).stem}]".ljust(MODULE_WIDTH)


def message(module_name: str | Path, text: str, *, stream: TextIO | None = None) -> None:
    """Write one consistently formatted application message."""

    destination = stream or sys.stdout
    print(f"{_module_label(module_name)} {text}", file=destination, flush=True)


def error(module_name: str | Path, text: str, *, stream: TextIO | None = None) -> None:
    """Write one consistently formatted error message."""

    destination = stream or sys.stderr
    print(f"{_module_label(module_name)} {text}", file=destination, flush=True)

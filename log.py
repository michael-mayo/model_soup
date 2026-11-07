"""Console logging helpers for the soup application."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO


MODULE_WIDTH = 24


class Log:
    """Console logging helpers for the soup application."""

    @staticmethod
    def _module_label(module_name: str | Path) -> str:
        return f"[{Path(module_name).stem}]".ljust(MODULE_WIDTH)

    @staticmethod
    def message(module_name: str | Path, text: str, *, stream: TextIO | None = None) -> None:
        destination = stream or sys.stdout
        print(f"{Log._module_label(module_name)} {text}", file=destination, flush=True)

    @staticmethod
    def error(module_name: str | Path, text: str, *, stream: TextIO | None = None) -> None:
        destination = stream or sys.stderr
        print(f"{Log._module_label(module_name)} {text}", file=destination, flush=True)

"""Console logging in the project-wide `[module] message` format."""

from __future__ import annotations

import sys
from typing import TextIO


# Width of the `[module]` label column so messages line up.
LABEL_WIDTH = 24


class Log:
    """Write labelled messages to the console on behalf of one module."""

    def __init__(self, module_name: str) -> None:
        """Create a logger labelled with the last component of `module_name`."""

        self.label = f"[{module_name.rsplit('.', 1)[-1]}]".ljust(LABEL_WIDTH)

    def message(self, text: str, *, stream: TextIO | None = None) -> None:
        """Write an informational message to stdout (or `stream`)."""

        print(f"{self.label} {text}", file=stream or sys.stdout, flush=True)

    def error(self, text: str, *, stream: TextIO | None = None) -> None:
        """Write an error message to stderr (or `stream`)."""

        print(f"{self.label} {text}", file=stream or sys.stderr, flush=True)

"""Primary entry point for the soup application."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from log import Log
from soup import Soup
import yaml


CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the application configuration from YAML."""

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("config.yaml must contain a top-level mapping")
    return loaded


def main() -> int:
    # Configuration is deliberately the first application action.
    try:
        config = load_config()
    except (OSError, ValueError) as exc:
        Log.error(__file__, f"could not read config.yaml: {exc}")
        return 1

    Log.message(__file__, "system initialising")
    for line in yaml.safe_dump(config, sort_keys=False).rstrip().splitlines():
        Log.message(__file__, line)

    try:
        Soup().validate(CONFIG_PATH.parent, config)
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        Log.error(__file__, f"soup validation failed: {exc}")
        return 1

    Log.message(__file__, "complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

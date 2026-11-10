"""Command-line entry point: prepare datasets, train baselines, evaluate the soup."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path

import yaml

import dataset
from app.config import CONFIG_PATH, Config, load_config
from app.log import Log
from app.soup import Soup
from app.trainer import BaselineTrainer, log_summary


LOG = Log("main")

# The config.yaml section each command reads (and logs at startup).
COMMAND_SECTIONS = {"prepare": "sample_datasets", "train": "sample_datasets", "soup": "soup"}


def _prepare(config: Config, args: argparse.Namespace) -> int:
    """Prepare the selected datasets (all configured ones by default)."""

    failures = 0
    for name in args.datasets or list(config.samples.sources):
        try:
            dataset.create(name, config.samples).prepare(force=args.force, verify_ssl=False if args.insecure else None)
        except (OSError, ValueError, KeyError, TypeError, tarfile.TarError, zipfile.BadZipFile) as exc:
            LOG.error(f"{name}: {exc}")
            failures += 1
    return 1 if failures else 0


def _train(config: Config, args: argparse.Namespace) -> int:
    """Train baselines for the selected datasets and log a summary."""

    results = BaselineTrainer(config.samples).train_all(args.datasets)
    log_summary(results)
    return 0 if all(result.ok for result in results) else 1


def _soup(config: Config, args: argparse.Namespace) -> int:
    """Fine-tune the soup members, then validate them and the baseline."""

    Soup(config).run()
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser with its prepare/train/soup subcommands."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="configuration file (default: config.yaml)")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="download and convert datasets to Ultralytics format")
    prepare.add_argument("--dataset", action="append", dest="datasets", metavar="NAME",
                         help="dataset to prepare; repeat for several (default: all configured)")
    prepare.add_argument("--force", action="store_true", help="redownload and rebuild")
    prepare.add_argument("--insecure", action="store_true", help="disable SSL certificate verification")
    prepare.set_defaults(handler=_prepare)

    train = commands.add_parser("train", help="train baseline detectors")
    train.add_argument("--dataset", action="append", dest="datasets", metavar="NAME",
                       help="dataset to train; repeat for several (default: sample_datasets.training.datasets)")
    train.set_defaults(handler=_train)

    soup = commands.add_parser("soup", help="fine-tune and validate the soup members")
    soup.set_defaults(handler=_soup)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, load the configuration first, and run the chosen command."""

    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        LOG.error(f"could not read {args.config}: {exc}")
        return 1

    # Show only the config section the command uses.
    section = COMMAND_SECTIONS[args.command]
    LOG.message(f"system initialising: {args.command}")
    for line in yaml.safe_dump({section: config.raw.get(section)}, sort_keys=False).rstrip().splitlines():
        LOG.message(line)

    try:
        return args.handler(config, args)
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        LOG.error(f"{args.command} failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

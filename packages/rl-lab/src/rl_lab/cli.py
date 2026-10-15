"""rl-lab command line: validate a catalog of diagnostic datasets and experiment bundles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rl_lab.bundle.io import DATASET_SPEC, MANIFEST
from rl_lab.bundle.reader import validate_bundle, validate_dataset


def _validate(catalog: Path) -> int:
    datasets, experiments = catalog / "datasets", catalog / "experiments"
    failed = False
    for directory in sorted(datasets.glob(f"*/{DATASET_SPEC}")):
        report = validate_dataset(directory.parent)
        failed |= _print(f"dataset {directory.parent.name}", report)
    for manifest in sorted(experiments.glob(f"*/{MANIFEST}")):
        report = validate_bundle(manifest.parent, datasets)
        failed |= _print(f"experiment {manifest.parent.name}", report)
    if not any(datasets.glob("*")) and not any(experiments.glob("*")):
        print(f"no datasets or experiments under {catalog}")
    return 1 if failed else 0


def _print(label: str, report) -> bool:
    print(f"{'✓' if report.ok else '✗'} {label}")
    for error in report.errors:
        print(f"    error: {error}")
    for warning in report.warnings:
        print(f"    warning: {warning}")
    return not report.ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rl-lab", description="RL Representation Lab.")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate every dataset and experiment bundle in a catalog.")
    validate.add_argument("catalog", type=Path)
    args = parser.parse_args(argv)
    return _validate(args.catalog)


if __name__ == "__main__":
    sys.exit(main())

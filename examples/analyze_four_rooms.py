"""Analyze a saved Four Rooms run using the AERIS analysis package.

Usage: python examples/analyze_four_rooms.py runs/fb_four_rooms_visual
"""
from __future__ import annotations

import argparse
from pathlib import Path

from aeris.analysis.saved_run import analyze_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--no-plots", action="store_true", help="Compute JSON only, no matplotlib needed")
    args = parser.parse_args()
    report = analyze_run(args.run_directory, gamma=args.gamma, make_plots=not args.no_plots)
    for phase, metrics in report["snapshots"].items():
        print(f"{phase}: {metrics}")
    print(f"Saved {args.run_directory / 'analysis' / 'tabular_policy_v1.json'}")


if __name__ == "__main__":
    main()

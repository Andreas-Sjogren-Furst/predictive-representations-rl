"""Re-render a completed AERIS Four Rooms run without reloading Torch weights.

Usage: uv run --extra fb-torch --extra viz python examples/render_four_rooms.py runs/fb_four_rooms
"""
import argparse
from pathlib import Path

from aeris.visualization.four_rooms import render_four_rooms_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    for path in render_four_rooms_report(args.run_directory):
        print(path)

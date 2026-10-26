"""Train discrete FB on Four Rooms from Python, then analyze the saved run.

The same workflows back the CLI (``aeris train four-rooms``, ``aeris analysis ...``).
Run: uv run --extra fb-torch python examples/fb_four_rooms.py
This is an AERIS diagnostic, NOT a reproduction of published maze results.
"""

from __future__ import annotations

from pathlib import Path

from aeris.workflows.four_rooms import analyze_saved_policies, diagnose_fb_checkpoint
from aeris.workflows.four_rooms_training import FourRoomsFBTraining, train_four_rooms_fb


def main() -> None:
    outcome = train_four_rooms_fb(
        FourRoomsFBTraining(output=Path("runs/fb_four_rooms"), updates=250)
    )
    for label, record in (("initial", outcome.initial), ("final", outcome.final)):
        print(label, {m.name: round(m.value, 4) for m in record.metrics})

    policies = analyze_saved_policies(outcome.run)
    print("final snapshot vs optimal:", policies.report.snapshots["final"])
    fb = diagnose_fb_checkpoint(outcome.run)
    print("occupancy correlation:", fb.exact_occupancy_correlation)
    print(f"Saved run and reports in {outcome.run.root}")


if __name__ == "__main__":
    main()

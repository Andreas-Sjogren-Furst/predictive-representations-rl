"""Registered dashboard panels, by name. Add a panel here to make it available in the app."""

from aeris.dashboard.panels.base import Panel
from aeris.dashboard.panels.evaluation import EvaluationResults
from aeris.dashboard.panels.maps import PolicyMap, TrajectoryAnimation, ZeroShotMap
from aeris.dashboard.panels.tables import ReportView, RunComparison, RunOverview
from aeris.dashboard.panels.training import LossCurves

PANELS: dict[str, Panel] = {
    panel.name: panel
    for panel in (
        RunComparison(),
        LossCurves(),
        EvaluationResults(),
        PolicyMap(),
        TrajectoryAnimation(),
        ZeroShotMap(),
        RunOverview(),
        ReportView(),
    )
}

__all__ = ["PANELS", "Panel"]

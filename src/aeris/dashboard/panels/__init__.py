"""Registered dashboard panels, by name. Add a panel here to make it available in the app."""

from aeris.dashboard.panels.base import Panel
from aeris.dashboard.panels.evaluation import EvaluationResults
from aeris.dashboard.panels.maps import PolicyMap, TrajectoryAnimation
from aeris.dashboard.panels.representations import (
    BSpectrum,
    OccupancyMaps,
    RepresentationProjection,
    RepresentationSimilarity,
    ValueMap,
)
from aeris.dashboard.panels.tables import GroupSummaryTable, ReportView, RunComparison, RunOverview
from aeris.dashboard.panels.tasks import (
    GoalScoresMap,
    SuiteDistribution,
    SuiteScores,
    SuiteTasks,
    TaskPolicy,
)
from aeris.dashboard.panels.training import LossCurves

PANELS: dict[str, Panel] = {
    panel.name: panel
    for panel in (
        RunComparison(),
        GroupSummaryTable(),
        LossCurves(),
        EvaluationResults(),
        PolicyMap(),
        TrajectoryAnimation(),
        SuiteScores(),
        SuiteDistribution(),
        TaskPolicy(),
        GoalScoresMap(),
        SuiteTasks(),
        RunOverview(),
        ReportView(),
        RepresentationProjection(),
        RepresentationSimilarity(),
        BSpectrum(),
        OccupancyMaps(),
        ValueMap(),
    )
}

__all__ = ["PANELS", "Panel"]

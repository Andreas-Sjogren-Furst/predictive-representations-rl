"""Registered dashboard panels, by name. Add a panel here to make it available in the app."""

from aeris.dashboard.panels.base import Panel
from aeris.dashboard.panels.behaviour import FateMap, RolloutReplay, TaskPolicy
from aeris.dashboard.panels.compare import SeedDots
from aeris.dashboard.panels.evaluation import EvaluationResults
from aeris.dashboard.panels.maps import PolicyMap, TrajectoryAnimation
from aeris.dashboard.panels.representations import (
    ActionValues,
    BSpectrum,
    OccupancyMaps,
    RepresentationProjection,
    RepresentationSimilarity,
    ValueMap,
)
from aeris.dashboard.panels.summary import RunSummary
from aeris.dashboard.panels.tables import GroupSummaryTable, ReportView, RunComparison, RunOverview
from aeris.dashboard.panels.tasks import (
    GoalScoresMap,
    SuiteDistribution,
    SuiteScores,
    SuiteTasks,
)
from aeris.dashboard.panels.training import LossCurves

PANELS: dict[str, Panel] = {
    panel.name: panel
    for panel in (
        RunSummary(),
        RunComparison(),
        GroupSummaryTable(),
        SeedDots(),
        LossCurves(),
        EvaluationResults(),
        PolicyMap(),
        TrajectoryAnimation(),
        SuiteScores(),
        SuiteDistribution(),
        TaskPolicy(),
        RolloutReplay(),
        FateMap(),
        GoalScoresMap(),
        SuiteTasks(),
        RunOverview(),
        ReportView(),
        RepresentationProjection(),
        RepresentationSimilarity(),
        BSpectrum(),
        OccupancyMaps(),
        ValueMap(),
        ActionValues(),
    )
}

__all__ = ["PANELS", "Panel"]

from predictive_representations_rl.analysis.base import Analyzer
from predictive_representations_rl.analysis.linear_probe import LinearProbeAnalyzer
from predictive_representations_rl.analysis.pca import PCAAnalyzer

ANALYZERS: dict[str, Analyzer] = {analyzer.name: analyzer for analyzer in (PCAAnalyzer(), LinearProbeAnalyzer())}

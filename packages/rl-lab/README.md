# rl-lab

Gather, inspect and compare what reinforcement-learning agents represent. Work in progress; see
`implementation_lab.md` in the repository root for the specification.

Currently provides `rl_lab.analysis`: pure, algorithm-agnostic computations on representation arrays
(PCA and effective dimensionality, linear probes with leakage-safe splits, linear CKA). Nothing here writes
files, plots, or depends on any training framework.

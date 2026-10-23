# AERIS: FB checkpoint diagnostics (v1)

This diagnostic is deliberately **FB-specific**, whereas `analysis/tabular_occupancy.py` is independent of the algorithm. It evaluates a saved local checkpoint and never alters weights or trains the model.

## Scientific questions

1. Does the learned forward-backward product resemble the exact **continuing-state discounted visitation density** for the task-conditioned target policy?
2. Does greedy online FB value ranking agree with the exact **one-step policy improvement** for that target policy on a goal-state visitation objective?
3. Does the resulting greedy policy also succeed at the **different** terminal-on-goal, finite-horizon navigation objective?
4. Does the backward embedding retain substantial variation across maze states?

These questions are different. A good policy score alone does not validate the FB occupancy approximation, and an inaccurate occupancy approximation does not by itself identify whether F, B, data, sampling, or optimization is responsible.

## Methodology

For `z = B(goal)`, use the saved target forward network and its configured training temperature to obtain

```
pi_z(a | s) = softmax_a(F_target(s,a,z)^T z / temperature).
```

Construct the exact tabular model from `FourRooms`. Its state occupancy is

```
G_z(s,a,x) = E[ sum_{t>=0} gamma^t 1[s_t=x] | s0=s, a0=a, later actions~pi_z ].
```

There are **no terminal states** for this diagnostic occupancy: the goal is just another state that can be revisited. Solve the induced linear system `(I - gamma P_pi) V = I` exactly. Compare `F(s,a,z)^T B(x)` with `G_z(s,a,x) / rho(x)` via Pearson correlation, where `rho(x)=1/number_of_states` for an explicitly verified uniform state-action dataset. This statistic tests pattern agreement, not equality of calibrated numeric values, and does not assert that Pearson correlation is sufficient to validate a representation.

For the goal-state visitation task, compare **per-state centered action scores** against the corresponding exact visitation values under the *same* target policy. Centering removes state-specific baselines that cannot affect greedy action choice. Report their correlation and the fraction of non-goal states in which the greedy online-FB action achieves the best exact one-step policy-improvement value (including ties).

Finally, use the existing exact first-hit/terminal-goal analysis on the greedy policy, always labeled as a **different reward/termination objective**. Capture the effective rank of centered B across states, network action margins, and target-policy entropy as separate descriptive measurements.

## Important limits

- The model must be a **trusted local** PyTorch checkpoint. Never load an arbitrary untrusted `.pt` file; `torch.load(..., weights_only=False)` can execute pickle code.
- The occupancy-density comparison is currently restricted to the `oracle` dataset with verified one-sample-per-state-action coverage and a matching fingerprint. This makes `rho` known. Other dataset distributions require explicit distribution-weighted handling.
- A high Boltzmann temperature may produce nearly uniform target action probabilities. This can be expected by design, not automatically a training defect.
- We do not claim to isolate F errors versus B errors, since they are only identified jointly by their product. Diagnosing each independently requires controlled interventions or additional assumptions.
- A 10- or 250-gradient-update smoke run is not a reproduction of the 2021 paper's results.
- Effective rank does not directly measure task-solving quality. It is a descriptive statistic, not a sufficiency test.
- The v1 report is for **one selected goal** and one saved checkpoint; studying a family of tasks/checkpoints is a separate experiment.

## Usage

```bash
uv run --extra fb-torch python examples/diagnose_fb_four_rooms.py runs/fb_four_rooms_visual
uv run --extra fb-torch --extra viz python examples/diagnose_fb_four_rooms.py runs/fb_four_rooms_visual --plots
```

Outputs are new *derived* artifacts (the original run is not overwritten):

- `analysis/fb_checkpoint_v1.json`
- `figures/fb_action_alignment.png` (only with `--plots`)

The figure marks state-action ranking agreement for **exact target-policy improvement**, not globally optimal goal-reaching actions. For the latter use the existing `policy_gap_final.png` plot.

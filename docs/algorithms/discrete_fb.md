# Discrete Forward–Backward in AERIS

## Scope and provenance

Adapted implementation of Ahmed Touati and Yann Ollivier, *Learning One Representation to Optimize All Rewards* (NeurIPS 2021). References:
- https://github.com/ahmed-touati/controllable_agent/blob/main/grid_modules/fb_agent.py
- https://github.com/ahmed-touati/controllable_agent/blob/main/discrete_action_robots_modules/models.py
- https://github.com/ahmed-touati/controllable_agent/blob/main/grid_modules/mdp_utils.py
- https://github.com/ahmed-touati/controllable_agent

Implementation is **not yet a validated reproduction of paper scores**. Upstream's grid-world environment, online sampling schedules and evaluation protocols have **not** been ported. This module implements its discrete FB network architecture, objective and agent-facing operations so we can test the algorithm through AERIS.

No upstream source file is copied verbatim. Verify upstream licensing and attribution before redistribution.

## Mathematical objective

Given an independent comparison batch of states `s_other`, target-network Boltzmann policy

`pi(a|s',z) = softmax_a( F_target(s',a,z) · z / temperature )`,

and target successor matrix

`M_target[i,j] = E_{a~pi}[F_target(s'_i,a,z_i)] · B_target(s_other_j)`,

the implementation minimizes

`L_FB = 0.5 mean_ij ( F(s_i,a_i,z_i) · B(s_other_j) - gamma M_target[i,j] )^2 - mean_i F(s_i,a_i,z_i) · B(s_i)`

plus the exact *stop-gradient placement* in the reference backward-feature regularizer, weighted by `regularization_coefficient`. Targets are updated **after** the optimizer step using `target = polyak * target + (1 - polyak) * source`.

The objective intentionally does **not** use transition rewards or dones, consistent with the inspected reference implementation. That choice must be reconsidered for absorbing terminals in future environments, rather than silently changing the baseline.

## Integration boundaries

`DiscreteFBAgent` implements `Agent`, `TrainableAgent[FBTrainingBatch]`, `TaskConditioner`, and `Checkpointable` through structural typing. It does **not** hold the environment, dataset, replay or experiment runner. `FBTransitionSampler` turns generic `Transition` records into independently sampled batches. It owns its RNG and exposes `state_dict` separately from the agent. The runner must checkpoint both if exact resumption is desired.

- `ActionRequest.condition`: FB latent vector `z`, not a task ID.
- `goal_condition(goal_observation)`: `B(goal)` as in the upstream grid-world task inference path.
- `infer_condition(task, [(state, reward), ...])`: weighted empirical estimate `mean(r_i B(s_i))` from **authorized** reward evidence. No gradient updates. This is a generalized convenience pathway, not equivalent to the upstream goal-conditioning demonstration.
- `forward_features`, `backward_features`, `action_values`: inspectable torch tensors; caller can detach/cpu/convert when persisting an analysis artifact.
- `update`: returns AERIS `MetricValue` objects, not runner-specific dictionary logs.

## Reference differences

| Area | This integration | Original grid implementation |
| --- | --- | --- |
| Environment | External to agent | Owned by agent |
| Replay and paired sampling | External `FBTransitionSampler` | Agent-owned replay buffer |
| Training loop and evaluation | Not yet integrated | Agent owns both |
| Network design | 3×256 ReLU MLPs by default, all-action `F` output | Same |
| FB objective and partial-detach regularizer | Same expressions | Same expressions |
| Target policy and Polyak update | Boltzmann + same retention convention | Same |
| Task sampler | Cauchy/uniform ball or goal-oriented | Same choices, separate RNG |
| RNG | Agent-owned torch generator | Global PyTorch / NumPy / Python RNGs |
| Exploration | `deterministic=False` controls epsilon-greedy behavior | Explicit `act_e_greedy` |
| Evaluation/GPI | To add with evaluator | In the agent |

Do not compare paper scores until dataset generation, action encoding, reward definition, training schedules, and evaluation all match the original benchmark. The published run uses `--embed-dim 100 --temp 200 --gamma 0.99 --lr 0.0005 --polyak 0.95 --w-sampling cauchy_ball --update-eps 1`, alongside rollout and epoch settings.

## Install / run

Add an optional dependency in your existing `pyproject.toml`:

```toml
[project.optional-dependencies]
fb-torch = ["torch>=2.3,<3", "numpy>=1.26,<3"]
```

On CUDA machines, follow the official PyTorch installation selector so torch builds match driver/CUDA availability. Then:

```bash
uv sync --extra fb-torch
uv run --extra fb-torch pytest -q
uv run --extra fb-torch python examples/discrete_fb_smoke.py
```

No JAX or Gymnasium dependency is required for this first module. The example uses a tiny synthetic discrete transition dataset; it does not assert learned goal-reaching success.

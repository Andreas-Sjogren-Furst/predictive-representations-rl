# Results

Findings from runs of the harness, newest first. Numbers come from the standard run directories under `runs/`
(not in git; see each run's `run.json` for the exact config, command and commits). All results so far are
**single-seed** and should be read as preliminary.

---

## 2026-10-13 — point_mass_maze: one-step FB vs DreamerV3

### Setup

**Environment.** `point_mass_maze` (ExORL custom DMC task): a point mass in a four-room maze, observations
`[x, y, vx, vy]`, 2-D continuous force actions in [-1, 1], episodes start in the top-left room. Four goals:
`reach_{top,bottom}_{left,right}` at (±0.15, ±0.15). Both algorithms use **onestep-fb's task definitions**
(sparse reward, ≈1 within ~1.5 cm of the goal); see *Notes on the setup* below.

**Runs** (`runs/point_mass_maze/...`, seed 0 throughout):

| Run | Mode / data | Budget | Duration | Harness commit |
|---|---|---|---|---|
| `onestep_fb/offline/seed_0` (α = 0) | offline, ExORL RND data (5M transitions), reward-free | 1M gradient steps | 1.4 h (A100) | `a272340` |
| `onestep_fb/offline/alpha{0.03,0.1,0.3,1.0}/seed_0` | as above, `agent.alpha` varied | 1M gradient steps | 1.4–3.0 h (L40S/V100) | `c1b8c1e` |
| `dreamerv3/online/reach_top_left/seed_0` | online, own data, trained on `reach_top_left` reward | 500k agent steps (1M env steps) | 4.4 h (L40S) | `344161c` |

FB uses the README's ExORL walker settings (latent 50, discount 0.98, `q_agg=mean`, `orthonorm_coeff=0.1`,
`tanh_squash=False`); there is no published point_mass_maze command. Dreamer uses `dmc_proprio`.

**Probes.** `runs/point_mass_maze/probes/default.npz`: 4096 windows of 16 steps cut from the held-out RND
validation split (100 episodes), labelled with x, y, vx, vy, speed, room (quadrant), distance to each goal and
`rewarded_<task>` (reward > 1e-6). FB representations are evaluated at the final state and action; Dreamer's
RSSM posterior is filtered over the window with the **deterministic posterior** (`extract: {posterior: mode}`).
Every representation is also extracted from the **untrained network** (same architecture, config and seed).

**Analyses.** PCA (components for 90% variance, participation ratio = effective dimensions); linear probes
(ridge R² for continuous factors, class-balanced logistic regression scored by balanced accuracy for room and
rewarded-labels; 5-fold CV grouped by episode); linear CKA between all representations.

### Task performance

| | top_left | top_right | bottom_left | bottom_right |
|---|---|---|---|---|
| Dreamer (trained on top_left), last 50 train episodes | **947** (917–986) | — | — | — |
| FB α = 0, final zero-shot | 0.3 | 2.5 | 0 | 0 |
| FB sweep (α = 0.03–1.0), final zero-shot | 0.5–0.7 | 0.6–4.3 | 0 | 0–2.0 |

Returns are out of ~1000. A random policy scores 0 on all goals; an untrained FB policy scores ~95 on
top_left only because episodes start next to that goal. Dreamer first exceeded 500 at episode 52 and stayed
there. **No FB setting learned a working zero-shot policy**: every run shows the same noisy `top_right` signal
(peaks 6–18.5 during training) and ~0 elsewhere.

### FB alpha sweep (α ∈ {0, 0.03, 0.1, 0.3, 1.0})

With α = 0 the behaviour-cloning loss grew steadily (40 → 363), i.e. the policy drifted away from the data's
actions; the sweep tested whether adding BC fixes zero-shot performance. **It does not, and α has no
meaningful effect on the representations either.**

Linear probes, mean ± std over the five runs (untrained network in brackets):

| | x | y | vx | room | dist TL | dist BR | rewarded TL | rewarded TR |
|---|---|---|---|---|---|---|---|---|
| `B` | 0.80±0.09 (0.96) | 0.89±0.02 (0.98) | 0.54±0.14 (0.93) | 0.98±0.00 (0.97) | 0.53±0.42 (0.86) | 0.89±0.03 (0.90) | 0.94±0.02 (0.90) | 0.91±0.01 (0.89) |
| `F` | 0.93±0.02 (0.97) | 0.88±0.09 (0.97) | 0.65±0.10 (0.91) | 0.98±0.00 (0.97) | 0.73±0.18 (0.87) | 0.88±0.03 (0.91) | 0.92±0.01 (0.89) | 0.91±0.01 (0.88) |
| Q-values | 0.47±0.06 (0.29) | 0.40±0.06 (0.09) | 0.07±0.04 (0.51) | 0.75±0.05 (0.46) | 0.40±0.10 (0.21) | 0.49±0.05 (0.22) | 0.89±0.03 (0.75) | 0.72±0.03 (0.68) |

Linear CKA:

| | trained vs trained (across α) | trained vs own untrained | trained vs raw observation |
|---|---|---|---|
| `B` | 0.87 (min 0.81) | 0.15 | 0.19 |
| `F` | 0.93 (min 0.87) | 0.11 | 0.27 |
| Q-values | 0.84 (min 0.77) | 0.03 | 0.39 |

- Training strongly reshapes the representations (CKA with the untrained network 0.03–0.15) although the
  probes alone barely show it: untrained `B` already decodes position (R² 0.96), as any random network of the
  state would. **Probe scores must be read against the untrained control.**
- Consistent across all five runs: trained `B`/`F` lose velocity (vx 0.54/0.65 vs 0.93/0.91 untrained) and
  keep room and goal information; the Q-values gain position, room and goal distances and drop velocity.
- PCA: trained `B` needs 22 of 50 dimensions for 90% of its variance (effective 12–14) vs 4 (3.3) untrained;
  `F` 19 vs 3. Part of this spread is imposed by the orthonormality term.
- PCA maps: untrained `B` is speckled (dominated by velocity and action); trained `B` forms smooth regions
  following the maze (top-right room, right corridor, bottom passage), as expected for successor features.
- Not reproducible: distance to the top-left goal from `B` (−0.29 to 0.78 across runs). The −0.29 reported
  for the α = 0 run alone is noise.
- Caveat: all five runs use seed 0, so they share their initialisation (untrained `B` CKA 1.00 across runs)
  and likely the data order; part of the cross-α agreement may come from that.

### FB vs Dreamer

**Dimensionality moves in opposite directions.**

| | untrained: 90% comps / effective dims | trained |
|---|---|---|
| FB `B` (50-d) | 4 / 3.3 | 22 / 14.2 — spread out |
| FB `F` (50-d) | 3 / 2.8 | 19 / 11.6 |
| Dreamer `h` (512-d) | 113 / 28.6 | 11 / 5.8 — compressed |
| Dreamer `z` (1024-d) | — | 22 / 8.1 |

**Linear probes** (α = 0 FB run; R², balanced accuracy for room and rewarded-labels):

| | x | y | vx | speed | room | dist TL | dist BR | rew. TL | rew. TR | rew. BR |
|---|---|---|---|---|---|---|---|---|---|---|
| FB `B` | 0.72 | 0.87 | 0.50 | −0.45 | 0.98 | −0.29* | 0.85 | 0.94 | 0.92 | 0.95 |
| FB `F` | 0.93 | 0.85 | 0.72 | −0.80 | 0.98 | 0.76 | 0.92 | 0.91 | 0.91 | 0.97 |
| FB Q-values | 0.48 | 0.31 | 0.01 | 0.03 | 0.69 | 0.39 | 0.54 | 0.87 | 0.72 | 0.75 |
| Dreamer `h` | 0.99 | 0.97 | 0.94 | 0.63 | 0.98 | 0.99 | 0.95 | 0.97 | 0.94 | 0.90 |
| Dreamer `h` untrained | 0.97 | 0.97 | 0.95 | 0.72 | 0.98 | 0.95 | 0.95 | 0.81 | 0.89 | 0.88 |
| Dreamer `z` | 0.91 | 0.86 | 0.45 | 0.21 | 0.97 | 0.90 | 0.79 | 0.88 | 0.85 | 0.94 |
| Dreamer `z` untrained | 0.97 | 0.96 | 0.93 | 0.68 | 0.97 | 0.94 | 0.95 | 0.81 | 0.85 | 0.83 |
| raw observation | 1.00 | 1.00 | 1.00 | 0.02 | 0.99 | 0.76 | 0.89 | 0.80 | 0.83 | 0.98 |

\* not reproducible across FB runs (see sweep).

- Dreamer's `h` keeps everything linearly readable, including speed (0.63), which is not linear in the raw
  observation (0.02). Its clearest gain over the untrained network is on **its own task**: rewarded-at-top-left
  0.81 → 0.97, distance to the top-left goal 0.95 → 0.99.
- Dreamer splits the state: trained `z` keeps position (x 0.91) but drops velocity (0.45 vs 0.93 untrained);
  motion is carried by `h`.
- FB's trained representations discard speed and much of velocity while keeping room and goal information:
  *where*, not *how fast*.

**Linear CKA, FB × Dreamer:**

| | Dreamer `h` | Dreamer `z` | Dreamer `(h,z)` | Dreamer `h` untrained | raw observation |
|---|---|---|---|---|---|
| FB `B` | 0.24 | 0.21 | 0.25 | 0.33 | 0.20 |
| FB `F` | 0.28 | 0.27 | 0.29 | 0.35 | 0.26 |
| FB Q-values | 0.31 | 0.35 | 0.33 | 0.36 | 0.36 |

CKA with the raw observation: Dreamer `h` 0.64, `z` 0.62, `(h,z)` 0.66, untrained `h` 0.54.

- The trained representations are **less** similar to each other than FB's `B` is to an untrained Dreamer
  (0.24 vs 0.33): training moves the two algorithms apart.
- Dreamer stays close to the geometry of the raw state; FB moves far from it.
- Both are place-organised (PCA maps), along different axes: Dreamer's PC1 separates the left corridor and
  bottom passage from the top rooms, PC2 the bottom passage from the top-right room; FB's components pick out
  the top-right room and its corridor.

**Summary.** FB spreads the state over many dimensions and departs far from the raw state; Dreamer compresses it
into a few dimensions that stay close to the raw state and encode its own task. Their trained representations
organise the maze differently (CKA ≈ 0.2–0.3).

### Caveats

- **One seed per configuration.** The sweep shows single probe cells can swing by up to ±0.4 between runs.
- **Data and objectives differ.** Dreamer collected its own data while solving `reach_top_left` and was trained
  with that reward; FB learned reward-free from RND exploration data covering the whole maze. Some differences
  (e.g. Dreamer encoding the top-left goal) follow from this rather than from the methods.
- **FB's zero-shot policy fails**, so its task side (z, Q-values) is a weak point of comparison; the walker
  hyperparameters probably do not transfer to this domain.
- **Dreamer extraction uses the posterior mode.** With sampled z, two extractions of the trained model differed
  by 0.20 (h) / 0.64 (z) of the spread across probes.
- Linear probes measure *linear* accessibility; a drop can mean information was rearranged, not lost.
- Linear CKA is biased upwards for wide representations; read it against the untrained and observation rows.

### Notes on the setup

- The two repos' `custom_dmc_tasks` share the maze but not the reward: Offline_vs_Online_in_MBRL's
  `point_mass_maze` rewards `1 − distance` everywhere (random policy ≈ 840/1000 on top_left). The harness runs
  Dreamer with onestep-fb's task definitions (`overrides/dreamerv3/custom_dmc_tasks`).
- Dreamer's logged step counts environment steps (action repeat 2); the `run.steps` budget counts agent steps.
- The first FB run used an incomplete config (missing `q_agg=mean`, `orthonorm_coeff=0.1`); it is kept as
  `onestep_fb/offline/seed_0.replaced-20261006-212524` and is not used above.

### Next steps

1. 2–3 more seeds for FB and Dreamer, reporting probe and CKA results as mean ± spread.
2. Reward-free Dreamer (`train_which: wm-r`, with Plan2Explore or offline on the RND data) to separate method
   from data and objective.
3. FB hyperparameters for this domain beyond α (reward temperature for z inference, discount).
4. Non-linear (MLP) probes to tell rearranged information from discarded information.

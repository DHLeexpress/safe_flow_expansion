# Safe Flow Expansion for Static Obstacles

This repository is the compact, source-grounded record of the current method:
SafeMPPI demonstrations, a conditional flow model, and verifier-guided flow
expansion in a static 2-D obstacle scene. Historical Phase A/B/C result bundles
have been removed; the only promoted result is `B1_current_best`.

## Current result

The canonical OOD scene starts at `(0.3, 0.3)`, ends at `(4.7, 4.7)`, and
replaces the unseen center geometry by one radius-1.0 obstacle. Evaluation is
raw, untilted, temperature-1 sampling with a disjoint `M=50` seed bank for each
of seven safety levels.

| checkpoint | SR | CR | timeout | V-safe | V-full | successful U/R | min clearance, all |
|---|---:|---:|---:|---:|---:|---:|---:|
| pretrained r0 | 57.43% | 42.57% | 0% | 18.29% | 13.14% | 94/107 | 0.0171 m |
| **B1 r19** | **96.00%** | **4.00%** | 0% | **86.29%** | **49.71%** | **174/162** | **0.0557 m** |

The round-19 checkpoint was selected by the declared M10 screen before the
disjoint M50 holdout was opened. The recovered delivery is scientifically
complete and hash-validated, but remains marked `RECOVERED_NONCANONICAL`
because a foreign GPU process violated the original artifact-assembly
exclusivity gate after training and evaluation had completed.

![B1 current best](assets/results/b1_current_best/b1_current_best_5x3_gallery.png)

## 1. Static scene and SafeMPPI teacher

The in-distribution scene is a `5 m x 5 m` workspace with radius-0.2 static
disks on the `4 x 4` integer grid and static boundary-wall disks. The robot is a
point double integrator with `dt=0.1`, `|u|<=1`, and a planning horizon `H=10`.
The obstacle, planner, and extra barrier margins are all exactly zero.

### Exact teacher settings

| setting | value |
|---|---:|
| MPPI samples / horizon / temperature | `512 / 10 / 0.1` |
| Gaussian proposal sigma | `0.5 sqrt(3) = 0.8660` per axis |
| sensing and barrier range | `2.0 m` |
| `use_polytope_barrier` / `warm_start` | `true / true` |
| `use_goal_nominal` | `false` |
| centroid gain / temporal smoothing / epsilon | `0.2 / 0.25 / 0.15` |
| centroid anisotropy / urgency floor | `2.5 / 0.02` |
| control-sequence smoothness cost | `0.12` |
| safety / extra barrier / planning margin | `0 / 0 / 0` |
| robot radius | `0` |

These values are frozen in
[`configs/safemppi_static_teacher.json`](configs/safemppi_static_teacher.json).

### Important correction: centroid sampling is active

The teacher loads `best_area_mode4.json`, then `mode1_config` changes the
proposal to Gaussian mode, sets the range, action limits, and noise. It does
**not** zero the centroid parameters. With the current robot-centered nominal
polytope, SafeMPPI computes

\[
\rho_t=\max\!\left(0,\frac{R-s_t}{s_t+\epsilon}\right),\qquad
\tilde p_t=\operatorname{clip}(0.2\rho_t,0.02,1),
\]

and, after the first replan,

\[
p_t=(1-0.25)\tilde p_t+0.25p_{t-1}.
\]

Here `s_t` is the current nominal-polytope minimum face margin. Mode B samples
toward the polytope opening/centroid with probability `p_t`. Static obstacles
only imply zero obstacle velocity; the robot moves, so its robot-centered
polytope and `s_t` change. Therefore `centroid_gain`, `centroid_smooth`, and
`centroid_eps` remain active during demonstration gathering. The separate
`smooth_weight=0.12` is an MPPI control-sequence cost and must not be confused
with `centroid_smooth=0.25`.

At every state SafeMPPI exposes its cost-selected weighted-mean plan and
internal-best plan. Both exact `H=10` plans are passed to the full verifier.
Among certified cost-selected plans, the data collector executes the first
action of the plan with the greatest verified progress. No raw debug rollout is
executed or used as a training target.

## 2. Demonstration data and pretraining

The goal is fixed at `(4.7,4.7)`. Starts are a jittered `32 x 32` grid over
`[0.1,4.9]^2`, filtered to free space with zero initial velocity; there is no
diagonal exclusion. The frozen bank has 881 starts. Collection uses
`gamma in {0.1,0.2,0.3,0.4,0.5,0.7,1.0}`, up to three deterministic retries,
and `T<=800`. It produced 878 successful trajectories per gamma and 341,968
verified planned windows in total.

![Teacher demonstrations](assets/data/full_space_all_gamma_trajectory_overlay.png)

The conditional flow predicts an acceleration window `U in R^(10 x 2)`. Its
condition contains

- five raw values: relative goal, velocity, and gamma;
- a two-dimensional closest-obstacle-boundary vector, averaged only on a
  numerical nearest-obstacle tie;
- a 32-dimensional CNN encoding of the nominal-polytope field.

The resulting context has dimension 39. The velocity network has input 91,
hidden widths `160 -> 96 -> 32`, and a 20-dimensional output. Pretraining uses
reflection-paired data, no equivariance penalty, and exact x/y reflection group
averaging at inference. Each trajectory has equal total loss mass, preventing
long low-gamma trajectories from dominating solely through window count.

The 4.38 GB tensor is not committed. Its authenticated path and digest are in
[`DATA_POINTER.json`](DATA_POINTER.json); on Helios run:

```bash
./scripts/link_helios_data.sh
```

## 3. Safe Flow Expansion (`B1_current_best`)

For round `n`, eight replicas per gamma advance synchronously. At each context,
the current flow generates `K=16` plans. An RBF posterior over the current-model
representation `z=phi_s(U,c)`, `s=0.9`, scores

\[
k(z,z')=\exp\!\left(-\frac{\lVert z-z'\rVert^2}{2\ell^2}\right),
\qquad
\sigma_n^2(z)=k(z,z)-k_z^\top(K_n+\lambda I)^{-1}k_z,
\]

with `ell=0.2003239429`, `lambda=0.01`, and at most 512 positive support points
from the current and previous round (`W=2`). Embeddings and the RBF state are
rebuilt at each round and frozen during that round's gathering.

The verifier queries `B=4` candidates sequentially without replacement using

\[
\pi_j\propto\exp(\sigma_n(z_j)/\beta_n),
\]

where one beta per gamma is calibrated to normalized `ESS=0.25`. Every solved
selected-B query enters cumulative `D`; only full-H positives enter cumulative
`D+`. SOCP errors enter neither. Execution first applies the nominal one-step
`H_P` eligibility gate, then selects the eligible full-H-positive plan with the
minimum exact SafeMPPI execution cost and executes only its first action. If no
such plan exists, that replica terminates as NVP; there is no expert fallback.

Training replays full-H positives from `W=2`. Every eligible positive is used
exactly once per round. Loss mass is equalized hierarchically over

\[
\gamma\;\rightarrow\;(\text{round},\text{episode})
\;\rightarrow\;\text{context}\;\rightarrow\;\text{positive query}.
\]

The batch size is 128, learning rate is `1e-5`, the number of Adam steps is
`ceil(|D_W^+|/128)`, and the visual encoder is frozen. At NVP contexts the
selected-B population supplies the signed negative gradient

\[
g=g_+-\rho g_-,\qquad
\rho=0.01\frac{\lVert g_+\rVert_2}{\lVert g_-\rVert_2}.
\]

The complete generated recipe is
[`configs/b1_current_best_recipe.json`](configs/b1_current_best_recipe.json).

## 4. Reproduction entry points

The files below preserve their original safeMPPI-relative paths under
`source_snapshot/`; copy the snapshot into a clean checkout of the parent
safeMPPI project or use the original source commit shown below.

| stage | entry point |
|---|---|
| static scene and teacher configuration | `overnight_run_07_06/grid_scene.py`, `overnight_run_2026-06-28/best_area_mode4.json` |
| SafeMPPI implementation | `cfm_mppi/safegpc_adapter/safemppi.py` |
| endpoints, expert collection, combine, visualization | `overnight_run_07_06/rev_expansion/codex_challenging/afe_restart/stage2_low7_randomized.py` |
| low7 CFM pretraining | `.../afe_restart/stage3_low7_pretrain.py` |
| balanced reflection-paired r0 launcher | `.../afe_restart/run_low7_reflection_pretrain_pair.sh` |
| RBF expansion trainer | `overnight_run_07_06/rev_expansion/codex_overnight/grid_expand_afe_rbf.py` |
| RBF, beta, execution, negative update | `afe_rbf_core.py`, `afe_adaptive.py`, `afe_execution.py`, `afe_signed_update.py` |
| exact B1 sweep/selection launcher | `run_low7_b1_balanced_sweep.sh`, `analysis/low7_b1_balanced_sweep_driver.py` |
| raw temperature-1 M50 evaluation | `paper_results/low7_raw_m50_eval.py` |

Canonical identities:

- algorithm source: `63ebefa7877c0b923c1c7cdea19228302dd6a0ca`;
- dataset SHA-256: `4b8e2d9be794584fad232bcc46cf78c2c4f422efb3e0642f503c8a77fcd2e8ec`;
- balanced pretrained checkpoint: `524c9c0a4fd071221ac509b9d8e6fbbfb85fdf1811aa04160317f2a9e2d3ef90`;
- selected r19 checkpoint: `60c155472f5ed0e4a1d53581857f09aead7924f8ce11e8e3adf890d5a57fc079`;
- scene SHA-256: `356d6d48b3af2b017b529562b530f35285c86f9107da512a73de6ef664b03e72`.

Verify this repository with:

```bash
python scripts/verify_package.py
python -m pytest tests/test_workbook_contract.py -q
```

## Scope and blind spots

- RBF variance is a novelty score in the learned representation, not a safety
  probability or certificate.
- A verified finite-H window does not prove recursive closed-loop viability;
  NVP remains possible.
- The closest-boundary vector compresses multi-obstacle geometry to two
  numbers; the nominal-polytope grid carries the remaining local geometry.
- `minimum_clearance` is the minimum over every state and obstacle/wall, then
  averaged over all episodes, including failures. At r20, gamma 0.1 is 0.0491 m
  over all episodes and 0.0549 m over successes only.
- The current result is established on static circular obstacles and should not
  be presented as a dynamic-obstacle guarantee.

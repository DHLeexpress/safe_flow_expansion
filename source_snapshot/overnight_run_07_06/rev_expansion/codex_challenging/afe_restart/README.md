# Planned-window AFE restart

> The matched two-arm Claude-grid/Codex-radius-1 AFE2 study no longer runs a
> trainer from this directory. Its canonical shared protocol and launcher are
> [`../../codex_overnight/AFE2_FINAL_PROTOCOL.md`](../../codex_overnight/AFE2_FINAL_PROTOCOL.md).
> `AFE2_RADIUS1_HANDOFF.md` is historical.

This directory is the authoritative clean implementation of Safe Flow
Expansion for the challenging stadium. The legacy trainer is archived and is
not imported into this package.

## Method contract

The non-negotiable identity is:

```text
generated U_plan == acquired U_plan == fully verified U_plan == replayed U_plan
executed action == verified U_plan[0]
```

Every query is one H=10, 2-D action window conditioned on the current
`(grid, low5, history, gamma)` context. The complete plan is rolled through the
double-integrator and submitted to the deterministic bounds-plus-SOCP verifier
before its first action can execute. Progress is stored separately from the
safety label.

The fixed pretrained representation copy defines normalized 32-D features and
the cumulative linear uncertainty matrix

```text
A_n = I + lambda^-1 sum_i z_i z_i^T
sigma_n(U,c)^2 = z^T A_n^-1 z.
```

Every actually verified flow or backup query, positive or negative, updates
`A_n` once. Sigma is used once, in finite Gibbs acquisition. The flow update
uniformly replays every positive **flow** query under the explicit proximal
CFM objective. Certified SafeMPPI backup records remain in the verifier ledger
and uncertainty matrix but never become a hidden demo-distillation channel.

There is no `qbuf`, executed-window reconstruction, gamma curriculum,
easy/frontier split, `demo_frac`, LwF, data/model anchoring, negative-sample
alpha objective, uncertainty-weighted replay, or fixed scientific number of
Adam steps. The current model is fully trainable; only the separate feature
copy used by uncertainty is frozen and hash-locked.

Temperature roles are fixed:

- expansion candidate sampling: `1.0`;
- independent round monitoring and sealed final validity: `1.0`;
- low-temperature rollout rendering: `0.5`, visualization only.

[`METHOD.md`](METHOD.md) is the scientific specification.
[`PROGRESS.md`](PROGRESS.md) records every accepted and rejected stage result.

## Runtime setup

Run modules from the parent `codex_challenging` directory. Physical GPU 1 is
exposed as logical `cuda:0`:

```bash
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06/rev_expansion/codex_challenging
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH="/home/dohyun/miniforge3/lib:/usr/local/cuda/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

The implementation intentionally reuses repository-local policy, scene, and
verifier dependencies. [`deps.py`](deps.py) resolves and hashes those files in
each run manifest; [`verifier.py`](verifier.py) also refuses an ambiguous
`verifier_polytope` import.

## Stage ownership

| Stage | Entry point | Core files used | Principal output/gate |
|---|---|---|---|
| 00 — archive | external ZIP operation, documented in `PROGRESS.md` | legacy workspace only | Integrity-tested pre-restart archive; never a training input. |
| 01 — contract | test suite | `config.py`, `schemas.py`, `dynamics.py`, `verifier.py`, `uncertainty.py`, `store.py`, `acquisition.py`, `controller.py`, `fallback.py`, `proximal_update.py`, `audit.py` | Object identity, full verification, cumulative uncertainty, audit isolation, certified fallback, and fail-closed tests pass. |
| 02 — balanced expert plans | `stage2_planned_demos.py` | `scene.py`, `fallback.py`, `verifier.py`, `schemas.py` | Real R-first/U-first SafeMPPI trajectories and exact verified H=10 planned-window targets. |
| 03 — fresh pretraining | `stage3_pretrain.py` | `policy.py`, `stage2_planned_demos.py`, root `grid_hp_expt.py` | Fresh endpoint-free checkpoint, frozen `phi0` copy, trajectory-disjoint validation, all-gamma R/U promotion gate. |
| 04 — OOD baseline/banks | `stage4_baseline.py` | `scene.py`, `audit.py`, `evaluation.py`, `validity.py` | Frozen pretrained baseline, expert ceiling, round-monitoring bank, and untouched sealed-final bank. |
| 04B — Kazuki/Mizuta | `stage4b_mizuta.py` | `reference/kazuki_baseline.py`, `evaluation.py` | Verifier-free CFM-MPPI scientific T=1 metrics and separate T=0.5 gallery. |
| 05 — Full AFE | `stage5_expand.py` | `controller.py`, `acquisition.py`, `store.py`, `uncertainty.py`, `fallback.py`, `proximal_update.py`, `audit.py`, `evaluation.py` | Per-round ledger, `A_n`, solver telemetry, ordinary rollouts, checkpoints, and monitoring audits. |
| 06 — three controls | `stage6_ablations.py` | `ablations.py`, `decision_budget.py`, the Stage-05 core | Matched realized-decision-budget `-AFE`, `-Progress`, and offline-only `-SOCP` runs. There is no `-Curriculum` because Full has no curriculum. |
| 07 — visualization | `stage7_artifacts.py` | `visualize_expansion.py`, `evaluation.py` | T=0.5 rollout galleries and active-expansion MP4s built read-only from saved T=1 query/replay traces. |
| 08 — final validity | `stage8_sealed_validity.py` | `validity.py`, `audit.py`, selected checkpoints | One-shot T=1 sealed-bank validity, progress-validity, coverage, fallback/fail-closed telemetry, and across-model Full aggregation. |
| 09 — paper outputs | `final_reports.py` | Stage-05/06/07/08 artifacts and `viz_style.py` | `rollouts.png`, `internals.png`, `scatter.png`, Markdown/LaTeX table, and final validity report. |
| expert geometry diagnostic | `expert_radius_sweep.py` | `fallback.py`, `verifier.py`, `scene.py` | Cost-selected externally verified expert sweep across giant-obstacle radii. |
| moving polytopes | `../giant_obstacle_ood/stage1c_window_polytope.py` | native SafeMPPI plus nominal/full-verifier polytope code | Slow synchronized gamma GIF with robot-following blue nominal and green verifier polytopes. Diagnostic only. |

### Core file map

- [`schemas.py`](schemas.py): immutable context/query/replay records and exact
  content hashes.
- [`scene.py`](scene.py): ID/OOD stadium geometry, start `(0.5,0.5)`, goal
  `(4.5,4.5)`, policy context construction, and verifier fingerprints.
- [`dynamics.py`](dynamics.py): the exact H=10 double-integrator rollout and
  first-action execution.
- [`verifier.py`](verifier.py): deterministic strict-bounds plus fitted-polytope
  SOCP label; clearance and progress remain diagnostics.
- [`policy.py`](policy.py): checkpoint validation, ordinary flow sampling,
  frozen feature extraction, and CFM loss on ledger records.
- [`acquisition.py`](acquisition.py): sigma-only Gibbs sampling without
  replacement under a real verifier-call budget.
- [`uncertainty.py`](uncertainty.py): cumulative 32x32 linear design matrix;
  no FIFO, eviction, decimation, or raw-feature buffer.
- [`store.py`](store.py): append-only query ledger, isolated audit ledger,
  deterministic duplicate handling, and uniform positive-flow replay view.
- [`controller.py`](controller.py): receding-horizon query/verify/select/backup
  loop and complete visualization traces.
- [`fallback.py`](fallback.py): SafeMPPI proposal source. It does not certify;
  the shared full verifier does. Runtime expansion admits only cost-selected
  mean/best plans, not raw debug rollouts.
- [`proximal_update.py`](proximal_update.py): full-positive-ledger proximal CFM
  numerical solve with tolerance and update-norm stopping telemetry.
- [`audit.py`](audit.py), [`validity.py`](validity.py), and
  [`evaluation.py`](evaluation.py): untilted audit isolation, interval scope,
  mode coverage, and ordinary closed-loop metrics.
- [`visualize_expansion.py`](visualize_expansion.py): scene-level candidate,
  acquisition, verifier outcome, execution/fallback, and exact replay frames.

## Reproduction sequence

Choose a new output root. Do not reuse an old stage directory after changing
geometry, verifier code, SafeMPPI settings, or policy code because their hashes
are part of the artifact contract.

```bash
RUN="$PWD/afe_restart/stage_results/reproduction_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"
```

### 1. Contract gate

```bash
pytest -q afe_restart/tests
```

The current suite covers both unit contracts and cross-stage artifact guards.
Any failure blocks data generation or checkpoint promotion.

### 2. Generate balanced planned-window demonstrations

The following explicitly locks the original/canonical SafeMPPI recipe used in
the latest radius diagnostic. A different expert recipe must be swept and then
recorded explicitly rather than silently changing a default.

```bash
python -m afe_restart.stage2_planned_demos run \
  --device cuda:0 \
  --outdir "$RUN/02_planned_demos" \
  --max-steps 800 --reach 0.15 \
  --smooth-weight 0.12 --noise-var-mult 3 --retreat-weight 0 \
  --max-debug-candidates 0 --quota 12
```

Important outputs are `manifest.json`, `data/planned_id_balanced.pt`, the
candidate census, and stage summary. The manifest gate requires exact R/U
balance per gamma and hash equality among generated, verified, and training
plans. Failed episodes and raw debug rollouts cannot become targets.

For independent gamma jobs, run one `--gammas` shard per process, then invoke
the `combine` command with every `--shard-manifests` path. The combiner derives
and verifies the common quota/configuration from the shard manifests.

### 3. Pretrain from random weights

```bash
python -m afe_restart.stage3_pretrain run \
  --device cuda:0 \
  --manifest "$RUN/02_planned_demos/manifest.json" \
  --outdir "$RUN/03_pretrain_seed20260716" \
  --epochs 500 --eval-rollouts-per-gamma 24 \
  --visualization-rollouts-per-gamma 8 --seed 20260716
```

Only `data/checkpoint_best.pt` from a passed all-gamma R/U promotion gate may
enter Stage 04/05. `data/phi0_frozen.pt` is the hash-locked uncertainty feature
copy; it is not a frozen online policy encoder. Repeat with an independent
pretraining seed for final across-model validity.

### 4. Create OOD banks and baselines

```bash
python -m afe_restart.stage4_baseline \
  --device cuda:0 \
  --checkpoint "$RUN/03_pretrain_seed20260716/data/checkpoint_best.pt" \
  --outdir "$RUN/04_ood_baseline" \
  --expert-smooth-weight 0.12 --expert-noise-var-mult 3 \
  --expert-retreat-weight 0 --expert-eval-rollouts-per-gamma 8

python -m afe_restart.stage4b_mizuta \
  --device cuda:0 \
  --checkpoint "$RUN/03_pretrain_seed20260716/data/checkpoint_best.pt" \
  --outdir "$RUN/04b_mizuta"
```

Stage 04 emits `data/fixed_audit_bank.pt`,
`data/sealed_final_test_bank.pt`, `data/baseline_rollouts.pt`, and
`data/ood_safemppi_expert_rollouts.pt`. Never evaluate the sealed bank during
tuning. A geometry/radius change requires regenerating both banks and all
downstream runs; [`scene.py`](scene.py) owns the geometry, while Stage 05 also
records its explicit `--giant-radius`.

### 5. Run Full AFE

```bash
python -m afe_restart.stage5_expand \
  --device cuda:0 \
  --checkpoint "$RUN/03_pretrain_seed20260716/data/checkpoint_best.pt" \
  --audit-bank "$RUN/04_ood_baseline/data/fixed_audit_bank.pt" \
  --outdir "$RUN/05_full_seed105000" \
  --giant-radius 1.2 --rounds 6 --seed 105000 \
  --candidate-count 64 --verifier-budget 8 --fallback-verifier-budget 8 \
  --backup-smooth-weight 0.12 --backup-noise-var-mult 3 \
  --backup-retreat-weight 0 --beta 0.2 --ridge-lambda 0.01 \
  --continue-after-gate
```

Acquisition and all scientific audits remain at temperature 1.0. The small
changing-seed closed-loop gate is checkpoint-selection telemetry, not final
validity. Resume only with `--resume-checkpoint` and the same protocol; the
loader rejects conflicting settings.

### 6. Run the matched controls

Select the Full checkpoint first. Then run all three controls from the same
fresh pretrained checkpoint, audit bank, and numerical protocol. The realized
decision count in the selected Full run caps every corresponding control cell.

```bash
FULL_CKPT="$RUN/05_full_seed105000/checkpoints/round_006.pt"

python -m afe_restart.stage6_ablations \
  --device cuda:0 \
  --checkpoint "$RUN/03_pretrain_seed20260716/data/checkpoint_best.pt" \
  --full-reference-dir "$RUN/05_full_seed105000" \
  --full-reference-checkpoint "$FULL_CKPT" \
  --audit-bank "$RUN/04_ood_baseline/data/fixed_audit_bank.pt" \
  --outdir "$RUN/06_controls" --arm all \
  --rounds 6 --seed 105000 \
  --candidate-count 64 --verifier-budget 8 --fallback-verifier-budget 8 \
  --backup-smooth-weight 0.12 --backup-noise-var-mult 3 \
  --backup-retreat-weight 0 --beta 0.2 --ridge-lambda 0.01
```

The controls are exactly:

- `-AFE`: uniform candidate querying; real sigma is still logged;
- `-Progress`: first eligible verified plan instead of progress ranking;
- `-SOCP`: offline strict-bounds training eligibility while the true SOCP
  outcome remains logged and used for all reported validity. It carries no
  runtime-safety claim.

### 7. Render rollout galleries and active expansion

```bash
python -m afe_restart.stage7_artifacts \
  --device cuda:0 \
  --full-run "$RUN/05_full_seed105000" \
  --full-checkpoint "$FULL_CKPT" \
  --ablations-root "$RUN/06_controls" \
  --pretrained-checkpoint "$RUN/03_pretrain_seed20260716/data/checkpoint_best.pt" \
  --outdir "$RUN/07_artifacts" \
  --gallery-rollouts-per-gamma 8 --fps 4
```

The former curriculum video is intentionally replaced by
`active_expansion.mp4`: it displays T=1 candidate plans colored by sigma,
queried verifier outcomes, the certified plan/backup actually selected, and
the exact positive flow plans used by the proximal solver. The rollout gallery
is separately sampled at T=0.5 and cannot supply metrics.

### 8. Run sealed validity once

Create a JSON run manifest matching the schema documented at the top of
[`stage8_sealed_validity.py`](stage8_sealed_validity.py). It must contain one
selected Full run, the three selected controls, and at least one additional
independently pretrained and independently expanded Full replica.

```bash
python -m afe_restart.stage8_sealed_validity \
  --device cuda:0 \
  --run-manifest "$RUN/sealed_runs.json" \
  --sealed-bank "$RUN/04_ood_baseline/data/sealed_final_test_bank.pt" \
  --outdir "$RUN/08_sealed_validity" \
  --plans-per-context 32 --audit-seed 108000
```

This stage is evaluation-only, ordinary/untilted, temperature 1.0, and refuses
to overwrite an existing final report. Its output is
`final_validity_report.pt` plus JSON. Audit samples never enter `A_n`, the
query ledger, or replay.

### 9. Build rollout, internal, scatter, table, and validity reports

[`stage7_artifacts.py`](stage7_artifacts.py) writes a manifest containing the
exact gallery paths. Pass those paths, the selected run/checkpoint paths, the
Stage-02 dataset, Stage-04 baselines, and the Stage-08 sealed artifact to:

```bash
python -m afe_restart.final_reports --help
```

The required arguments deliberately force explicit provenance for Full,
`-AFE`, `-Progress`, and `-SOCP`. Optional expert, pretrained, and Mizuta paths
add the original comparison panels. Outputs are:

- `rollouts.png`, including balanced ID demo seeds and all comparison methods;
- `internals.png`, showing acquisition, ledger/A, proximal, fallback, and
  untilted-audit telemetry rather than a curriculum;
- `scatter.png` with the shared gamma colors and method-specific markers;
- `table.md` and `table.tex`;
- `final_validity_report.md`, sourced only from the sealed artifact.

## Verification and development rules

- Run `pytest -q afe_restart/tests` before promoting or publishing a change.
- Never hand-edit an artifact manifest, ledger, query hash, or stored verifier
  label.
- Never reuse an old monitoring/sealed bank after scene or verifier changes.
- Never use T=0.5 gallery rollouts as acquisition, replay, or a metric source.
- Never turn SafeMPPI backup into implicit expert replay.
- Keep generated data/checkpoints/videos under `stage_results/`; they are
  ignored by Git and may be tens of gigabytes.
- Record negative results and rejected gates in [`PROGRESS.md`](PROGRESS.md)
  instead of silently replacing them.

The latest radius-1.0 diagnostic is a negative expansion result despite a
14/14 canonical expert. Its exact findings and next unresolved mechanism are
at the end of [`PROGRESS.md`](PROGRESS.md); it is the correct starting point
for the next code update.

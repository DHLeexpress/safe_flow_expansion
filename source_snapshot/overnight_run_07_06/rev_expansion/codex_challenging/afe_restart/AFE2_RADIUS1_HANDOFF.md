# Codex radius-1 handoff — historical integration contract

> **Historical and superseded.** This file records the first radius-1
> integration contract, including the now-deleted scene-specific launcher and
> discrete beta-candidate rule. Do not execute it. The canonical shared
> dual-scene protocol and current launcher commands are in
> `../../codex_overnight/AFE2_FINAL_PROTOCOL.md`.

## Goal

Run Claude AFE2's acquisition and two unchanged update recipes on the Codex
center-radius-1 OOD scene and its own pretrained checkpoint. There is one
declared structural correction shared by both arms: the original goal disk is
absorbing, so verification does not reject a plan solely because of the
unexecuted suffix after its first goal hit. This is not a hyperparameter search.

There is one shared implementation:

`../codex_overnight/grid_expand_afe2.py`

Do not copy or independently edit that trainer in `afe_restart`. The two task
adapters live in `../codex_overnight/afe2_scene_profiles.py`:

- `claude_grid_v1`: the scene that produced the Claude AFE2 result.
- `codex_radius1_v1`: replace exactly the four disks at `(2,2)`, `(2,3)`,
  `(3,2)`, `(3,3)` by one disk at `(2.5,2.5)`, physical radius `1.0`; retain
  the other interior disks, boundary walls, and eight plugs. Start is
  `(0.5,0.5)` and goal is `(4.5,4.5)`.

The pretrained checkpoint is the only other task-specific input. The launcher
requires its expected file SHA-256; the trainer also runs the existing Codex
fresh-pretrain promotion gate and rejects a non-promoted checkpoint, a model
whose representation is not 32-D, or frozen parameters. Do not infer a
checkpoint from a filename. Obtain its path and file hash from the matching
Stage-3 manifest (the first documented promoted hash is
`bfbb925a8499205a4639b33b8fe819ae4527fa8cafcabcc8722dd9bedea21efb`; use it
only if that is the intended pretrained replica).

## Frozen recipe

Both Codex arms use the Claude AFE2 values below except for the one explicitly
scene-calibrated acquisition temperature:

| mechanism | value |
|---|---:|
| rounds | 10 |
| gammas | 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0 |
| K / B | 64 / 8 |
| beta / lambda | one radius-1 ESS calibration from {0.01,0.02,0.05} / 10 |
| horizon / reach | 300 / 0.15 |
| replay batch | 128 |
| prox | lr 2e-5, eta 0.01, at most 40 steps, fstep 0.03 |
| afe | lr 1e-4, exactly 250 steps, no prox |
| verified evaluation | 8 fixed-index rollouts per gamma (pilot; report labels this power) |

The launcher first performs one beta-neutral round-0 calibration on the radius-1
checkpoint, persists `beta_calibration.json`, and hash-binds its selected beta
to both arms. This avoids assuming that Claude's beta transfers to the new
scene. If no declared candidate falls inside ESS/K `[0.25,0.5]`, calibration
stops instead of choosing an ad-hoc nearest value; that requires an explicit
protocol decision before either arm runs. Its median gives one equal vote to
each visited control-step K-pool across the fixed gamma sweep (it is not
gamma-balanced). `--lock-reference-recipe` fails
closed if any other value changes. Do not
run the middle-update, loose-prox, feature-concatenation, recovery-start,
pi-execution, or longer-horizon proposals from `codex_overnight/AFE2_HANDOFF.md`
in this replication.

The report's Wilson and bootstrap intervals are conditional descriptive
intervals over the fixed audit contexts and fixed episode indices. With one
training seed and M=8 they are not uncertainty over training runs or a safety
guarantee; add independent seeds before making either claim.

## Terminal-set correction

Let `s_H` be the original full ten-step safety certificate, `G` the unchanged
radius-0.15 goal set, and `tau_G` the plan's first hitting time. Execution uses

```text
e = s_H OR (tau_G <= H AND the prefix 1:tau_G is certified).
```

The second clause is an execution-only terminal rescue. The archive retains
the full-window label and `D+` still contains only `s_H=1`; an unsafe or
unverified suffix is never converted into a positive training target. Log
`full+`, `exec+`, `terminal_reverify`, and `selected_terminal_rescue`
separately. The safety claim ends at the first goal hit. Do not enlarge the goal
radius based on the observed 0.8-m NVP cluster: that would change success rather
than correct the stopping-time mismatch. Safety after task termination would
require a separately specified controlled-invariant terminal set and is not
claimed here.

`B=8` is a candidate-query budget, not a fixed solver-call budget under this
correction: a rejected full window that predicts a goal hit may require one
additional prefix check. The trainer logs `n_socp_solve` and verifier wall time
per gamma; do not compare compute using `B` alone.

## Exact algorithm being replicated

The behavioral reference is the code, not an overstatement in the old README.
At each control step it computes one `sigma/pi` vector over K plans, draws B
indices without replacement in one pre-verifier batch, verifies them, and then
adds completed positive and negative rows to A. It does **not** re-score the
remaining candidates after each query. Preserve this pre-batch acquisition
semantics so the scene comparison uses the same algorithm as Claude's result.

Everything else remains the AFE2 paradigm:

- current-model representation and complete archive re-embedding at round
  boundaries;
- deterministic full-window SOCP/task-box training label, with progress used
  only to rank execution-admissible plans; when a plan hits the absorbing goal,
  execution progress is truncated at that same first hitting time while the
  original full-H progress remains separately logged;
- cumulative safety-positive replay;
- no SafeMPPI, expert action, fallback action, curriculum, anchor, or rollback;
- `NO_VERIFIED_POSITIVE` terminates without executing an action;
- untilted audit data never enter the archive or A.

The archive stores the exact float32 H_P channel, history, and action window
used at acquisition. Rebuilding A after a representation update therefore does
not silently re-embed a float16 approximation. Gathering, replay, update,
audit, fixed probes, and controller evaluation use separately named SHA-256
seed streams; diagnostics cannot change the next round's exploration noise,
and both arms receive the same indexed gathering noise.

One inherited verifier detail is deliberately preserved and disclosed: Claude
e97eead's `GM.in_taskspace` accepts the legacy coordinate interval
`[-0.12,5.12]`, not exact `[0,5]`. The controller uses the same tolerance. This
study does not silently substitute Codex restart's exact-box verifier; changing
that contract requires a separate matched arm.

## Run

Locate the intended Codex pretrained checkpoint first. Do not substitute a
Claude checkpoint. Commit every runtime `.py`/`.sh` file and start from a clean
tree; the locked trainer rejects tracked changes or untracked runtime sources.
Verify `command -v ffmpeg` and `command -v ffprobe` before launch (`brew install
ffmpeg` if absent); the launcher also checks for the `libx264` encoder and
intentionally fails before GPU work when encoding/decoding is unavailable. Use
a new or empty output root; stale calibration/report/video files are rejected
rather than overwritten.
Then run both arms sequentially so they do not share GPU or process state:

```bash
cd overnight_run_07_06/rev_expansion/codex_overnight
export CUDA_VISIBLE_DEVICES=<gpu>
./run_afe2_radius1_pair.sh \
  /absolute/path/to/codex_pretrained_32d.pt \
  EXPECTED_CHECKPOINT_FILE_SHA256 \
  /absolute/path/to/output/afe2_radius1
```

The launcher calibrates beta once, reloads the same checkpoint for each arm,
runs prox first and afe second, validates the pair, and then creates one report
and two seven-gamma videos. The
validator writes `afe2_radius1_pair_manifest.json` only after both runs contain
exactly rounds 0--10 and agree on checkpoint, scene, code, and all non-arm
recipe fields. Each saved
round now contains the exact obstacle array and scene fingerprint, so the video
does not reconstruct a plausible but wrong scene.
There is no hidden wall-clock cutoff: a process failure leaves no valid
`COMPLETE.json`, and the launcher cannot promote a partial arm.

## Completion checks

Before interpreting outcomes, require all of the following:

1. Both `recipe.json` files have the same `source_checkpoint_sha256`, the same
   `scene.sha256`, the same embedded/hash-verified `beta_calibration`, and
   `reference_recipe_locked=true`.
2. The scene contains exactly one `(2.5,2.5,1.0)` disk and none of the four
   replaced central disks.
3. Each `probe.jsonl` contains rounds 0 through 10. A wall-clock-truncated run
   is incomplete, not a result.
4. Every episode status is exactly `reached`, `nvp`, `timeout`, `collision`, or
   `oob`; an exception aborts the run and prevents `COMPLETE.json`.
5. Report round 10 as the endpoint. Round 9 may be shown as a trajectory of the
   learning process, but it must not replace round 10 because it looks better.
6. The video renders every K=64 candidate at every executed control step; B is
   labeled as verifier query objects, while actual SOCP solves are counted
   separately.

Require `DELIVERY_COMPLETE.json`, not merely the matched-pair validation
manifest, before treating the report and videos as delivered.

The scientific target is the same mechanism with one explicit stopping-time
correction, not guaranteed success.
Report, per gamma and arm, controller SR/NVP/CR, untilted validity and adverse
validity, representation cosine, ESS/uplift, and round-to-round stability. The
Claude pattern—prox nearly frozen, afe learning with oscillation/validity
erosion—is a hypothesis to test on radius 1, not an assumed conclusion.

# Phase C code index and blind-spot map

This index explains why every source family was copied. `SOURCE_MANIFEST.json` is the byte-level inventory; this file is the semantic inventory. A file being present does not mean that Phase C endorses every historical mode in that file.

## Provenance classes

- **Direct Phase C runtime:** explicitly hashed by the completed V3 driver.
- **Transitive runtime:** imported by a direct file or required by policy/verifier construction.
- **Data/pretraining provenance:** produced the low7 dataset or checkpoint.
- **Historical/reference:** documents an earlier implementation or supplies compatibility helpers; it is not the Phase C algorithm.
- **Test:** verifies one contract and is not runtime logic.

The direct completed run hashed 12 source files. The wider snapshot is deliberately larger because Python imports and `sys.path` mutations otherwise make the “same code” claim incomplete.

## 1. Direct Phase C runtime

All paths below are relative to `source_snapshot/overnight_run_07_06/rev_expansion/codex_overnight/`.

| File | Role | Blind spot / warning |
|---|---|---|
| `run_low7_rbf_v3_support_sweep.sh` | Validates Python, codecs, CPU count, physical GPUs 1/3, UUIDs, and exclusivity; launches the driver. | Helios-specific interpreter and GPU mapping; fails if the host topology differs. |
| `analysis/low7_rbf_v3_support_sweep_driver.py` | Defines six optimizer-step/demo-support arms, dispatches two GPU queues, screens checkpoints, runs disjoint confirmation, and hashes delivery. | Hard-coded Phase B reference and canonical dataset paths; checkpoint selection inherits the declared finite M10 objective. |
| `grid_expand_afe_rbf.py` | Complete Phase C training loop: synchronous replicas, flow proposals, RBF acquisition, verification, execution, replay, CFM update, checkpoints, and probes. | Large multi-profile file; legacy imports make the executable closure broader than the direct hash list. |
| `afe_demo_support.py` | Loads authenticated TRAIN demos, constructs exact x/y-reflection pairs, and combines demo/positive objectives. | Update-time augmentation does not make the network reflection-equivariant; demo data do not enter acquisition. |
| `afe_core.py` | Shared feature extraction, archive, CFM update, and diagnostic helpers. | Carries compatibility behavior used by earlier AFE versions; callers must inspect the active protocol. |
| `afe_rbf_core.py` | RBF kernel, posterior covariance, memory selection, and sequential Schur-complement conditioning. | Kernel distance is only as meaningful as the current penultimate representation. |
| `afe_execution.py` | Fail-closed nominal-Hp first-step gate and progress/margin execution rankings. | One-step nominal margin is local and does not prove recursive feasibility. |
| `afe_context.py` | Builds the low7 condition, including closest obstacle-boundary vector. | A single closest-boundary vector compresses multi-obstacle geometry. |
| `grid_expand_afe2.py` | Shared model loading, checkpoint contract, verifier calls, scenes, and rollout utilities. | Historical compatibility and dynamic module resolution increase coupling. |
| `paper_results/low7_support_sweep_eval.py` | Raw M10 checkpoint screening, route objective, and exact r0 equivalence audit. | Requires Phase B reference cell archives; screen noise remains finite. |
| `analysis/afe_rbf_sweep_diagnostics.py` | Produces sweep/training diagnostics from probes and metrics. | Gathering plots are not raw policy evaluation. |
| `video_afe2.py` | Renders expansion/gathering frames and MP4. | Shows selected/query behavior and must not be presented as untilted policy behavior. |

`paper_results/low7_raw_m50_eval.py` is also included because the direct evaluator calls its shared raw-evaluation functions for the disjoint M50 confirmation.

## 2. Phase C transitive algorithm modules

| File | Role | Blind spot / warning |
|---|---|---|
| `afe2_calibration.py` | Candidate-pool calibration records and numerical validation. | A calibrated number can still encode an uninformative feature geometry. |
| `afe2_scene_profiles.py` | Canonical scene registry and scene hashes. | Adding a scene without a new hash/provenance record invalidates comparisons. |
| `afe_adaptive.py` | Solves beta for a target normalized ESS over ragged score vectors. | ESS regulates concentration only; it does not enforce U/R coverage. |
| `afe_ensemble_core.py` | Compatibility helpers for the earlier ensemble estimator. | Deep ensemble is not the Phase C estimator; importing the file does not make it active. |
| `afe_route_metrics.py` | Computes early cross-track U/R diagnostics. | U/R is local and scene-specific, not a homotopy proof. |
| `afe_signed_update.py` | Implements optional negative-gradient combination. | Phase C sets `negative_alpha=0`; this code is dormant in the selected run. |
| `grid_expand_afe_ensemble.py` | Earlier ensemble expansion implementation and shared helpers. | Not the Phase C arm; retained because shared imports refer to it. |
| `grid_expand_hardtail.py` | Historical hard-tail/terminal handling helpers. | Its existence must not be confused with an active curriculum or recovery rule. |
| `grid_hp_expt.py` | Nominal-Hp grid/policy helpers. | Shares historical experiment assumptions and absolute import roots. |
| `grid_metrics2.py` | Additional rollout and validity metrics. | Metric names need evaluation-mode context. |
| `paper_results/afe_m20_eval.py` | Portable raw/verified endpoint evaluation utilities. | Some “verified” modes use a controller; only raw mode is the Phase C scientific outcome. |
| `analysis/validate_afe2_pair.py`, `run_afe2_pair.sh` | Historical AFE2 artifact-test fixtures retained so the inherited integrity suite is self-contained. | These launch/validate the superseded dual-arm AFE2 protocol and are not Phase C runtime. |

## 3. Shared flow policy and rollout stack

Paths are relative to `source_snapshot/overnight_run_07_06/`.

| File | Role | Blind spot / warning |
|---|---|---|
| `_paths.py` | Resolves historical project roots. | Encodes workstation/Helios layout assumptions. |
| `grid_feats.py` | Grid/low-dimensional feature construction. | Multiple feature schemas coexist; checkpoint schema must be authenticated. |
| `grid_policy.py` | Base conditional flow policy and CFM sampling/training. | Older low5 defaults exist alongside low7 usage. |
| `grid_policy2.py` | Extended policy/checkpoint compatibility. | Architecture inference can hide schema mismatch unless the checkpoint contract is enforced. |
| `grid_rollout.py` | Double-integrator rollout and execution utilities. | Finite numerical rollout is not a formal closed-loop model guarantee. |
| `grid_scene.py` | Base obstacle/wall scene construction. | Scene mutation outside the hashed profile breaks comparison. |
| `grid_metrics.py` | Success, collision, clearance, and validity helpers. | Whole-trajectory and planned-window validity must not be conflated. |
| `grid_expand.py`, `grid_expand2.py` | Historical expansion helpers imported by later code. | Not the Phase C algorithm; contain earlier assumptions. |
| `sr_cr_eval.py` | Shared SR/CR evaluation utility. | Seed-bank and controller mode determine what SR/CR mean. |
| `uncertainty_nn.py` | Earlier neural uncertainty helper. | Not active in the RBF Phase C arm. |
| `wandb_utils.py` | Optional experiment logging support. | External logging is not part of the authenticated result. |

## 4. Low7 data generation and pretraining

Paths below are under `source_snapshot/overnight_run_07_06/rev_expansion/codex_challenging/afe_restart/`.

| File | Role | Blind spot / warning |
|---|---|---|
| `stage2_low7_randomized.py` | Generates full-grid start/goal SafeMPPI demonstrations, verifies H10 targets, and writes manifests/visuals. | CPU/SOCP intensive; output tensor is 4.38 GB and remains external. |
| `stage2_planned_demos.py` | Core planned-window demonstration record and verifier contract. | Planned target validity is finite-horizon, not trajectory viability. |
| `seed_geometry.py` | Deterministic grid/jitter, free-space filtering, and endpoint geometry. | A fixed goal does not span all goal-conditioned symmetries. |
| `stage3_low7_pretrain.py` | Pair-disjoint trajectory-balanced low7 CFM pretraining and promotion manifest. | Broad spatial data do not enforce equal probability on global routes. |
| `stage3_pretrain.py` | Shared dataset/checkpoint/CFM utilities, including seeded CFM loss. | Supports older schemas; caller must pin low7. |
| `evaluate_low7_pretrained.py` | Raw low7 pretrained qualification and fixed-index gallery. | NFE and seed bank must match before comparing percentages. |
| `run_low7_fixed_goal_grid_pretrain.sh` | Frozen low7 dataset/pretraining launcher. | Helios-specific paths and resource assumptions. |
| `run_low7_randomized_pretrain.sh` | Earlier randomized endpoint launcher. | Historical variant, not the promoted fixed-goal dataset contract. |
| `schemas.py` | Typed records for episodes, targets, and manifests. | Schema correctness does not ensure semantic balance. |
| `policy.py` | Restart package policy wrapper. | Must not replace the authenticated `grid_policy` checkpoint loader silently. |
| `verifier.py` | Restart package verifier wrapper. | Depends on the shared fitted-polytope solver. |
| `dynamics.py` | Restart double-integrator dynamics. | Duplicates shared dynamics and can drift if edited separately. |
| `scene.py` | Restart scene description. | Historical defaults differ from the Phase C profile. |
| `config.py` | Restart experiment configuration. | Not the selected Phase C `recipe.json`. |
| `store.py` | Persistent query/result storage. | Historical store semantics differ from Phase C's cumulative archive plus W2 replay. |
| `uncertainty.py` | Restart uncertainty abstraction. | Not the active Phase C RBF implementation. |
| `validity.py` | Validity bookkeeping. | Label definitions differ across historical experiments. |
| `controller.py` | Earlier controller orchestration. | May include behavior not allowed by Phase C. |
| `fallback.py` | Certified fallback implementation for earlier studies. | Phase C explicitly does not call fallback. |
| `proximal_update.py` | Earlier proximal update. | Phase C has no proximal term. |
| `acquisition.py` | Earlier acquisition logic. | Phase C uses `afe_rbf_core.py` sequential acquisition instead. |
| `audit.py` | Earlier held-out audit helpers. | Audit population and current raw evaluation are different contracts. |
| `decision_budget.py` | Query/compute budget records. | Does not itself guarantee unbiased data collection. |
| `evaluation.py` | Earlier restart evaluation. | Not the V3 disjoint raw M50 evaluator. |
| `ablations.py` | Earlier ablation registry. | Historical factors are not current arm factors. |
| `expert_radius_sweep.py` | Expert scene-radius study. | Uses expert behavior and is not expert-free expansion. |
| `final_reports.py`, `stage7_artifacts.py` | Earlier report/artifact generation. | Visual style may be reused, but results are not Phase C evidence. |
| `visualize_expansion.py` | Earlier expansion visualization. | Controller-induced plots are not raw evaluation. |
| `stage4_baseline.py`, `stage4b_mizuta.py` | Pretrained/baseline planners. | Baselines have distinct sampling/controller semantics. |
| `stage5_expand.py`, `stage6_ablations.py`, `stage8_sealed_validity.py` | Earlier staged expansion/ablation/audit pipeline. | Preserved for provenance only; do not mix with Phase C. |
| `deps.py`, `__init__.py` | Package resolution and namespace. | Dynamic path insertion can create module shadowing. |
| `METHOD.md`, `README.md`, `PROGRESS.md`, `AFE2_RADIUS1_HANDOFF.md` | Historical design and handoff notes. | Narrative may describe superseded algorithms. |
| `reference/pretrain_repr.py` | Original representation/pretraining reference. | Older raw-condition and encoder assumptions. |

## 5. SafeMPPI and polytope/verifier stack

| File | Role | Blind spot / warning |
|---|---|---|
| `cfm_mppi/safegpc_adapter/safemppi.py` | MPPI expert, nominal polytope construction, level-set rejection, and diagnostic records. | Includes fallback-to-safest logic inside the general expert; Phase C never invokes the expert. |
| `cfm_mppi/safegpc_adapter/polytope_v2.py` | Constructs the nominal robot-centered polytope from obstacle support directions. | Nominal faces depend on current sensing and are not the fitted certificate. |
| `cfm_mppi/safegpc_adapter/polytope.py` | Polytope margins and smooth barrier utilities. | Smooth soft-min and normalized minimum barrier are distinct quantities. |
| `cfm_mppi/safegpc_adapter/barrier.py` | Clearance and affine/higher-order barrier utilities. | Phase C's selected execution gate uses nominal polytope Hp, not every barrier variant here. |
| `cfm_mppi/safegpc_adapter/gamma_schedule.py` | Gamma schedule helpers. | Current experiments use fixed per-episode gamma, no curriculum. |
| `overnight_run_2026-07-01/verifier_polytope.py` | Fits trajectory-specific separating faces and checks the SOCP certificate. | Finite-H certificate and solver success do not establish recursive feasibility. |
| `overnight_run_2026-07-01/di_grid_viz.py` | Exact polytope/trajectory rendering. | Visualization can use optional bounded-margin variants; manifests identify the exact mode. |
| `overnight_run_2026-07-01/local_frame.py` | Robot-centered coordinate transform. | Frame mismatch can silently corrupt low7 and verifier geometry. |
| `overnight_run_2026-07-01/polar_grid.py` | Polar/grid representation helpers. | Resolution and clipping discard geometry. |
| `overnight_run_2026-07-01/scenes.py` | Shared scene constructors. | Historical scene names are not authenticated Phase C profiles. |
| `overnight_run_2026-07-01/_paths.py` | Historical import roots. | Absolute-layout coupling. |
| `ieee_compact_polytope_verifier_package/src/demo_verifier_polytope.py` | Original fitted-face solver/checker implementation used by the wrapper. | External solver availability and numerical tolerance matter. |
| `overnight_run_2026-06-28/best_area_mode4.json` | Frozen nominal-polytope tuning payload. | Empirical tuning asset, not a theorem. |

## 6. Additional imported policy/dynamics code

| File | Role | Blind spot / warning |
|---|---|---|
| `overnight_run_today/src/flow_policy.py` | Earlier flow-policy implementation used by compatibility imports. | Not the promoted low7 model definition. |
| `overnight_run_today/src/dynamics.py` | Earlier double-integrator implementation. | Duplicate implementation can drift. |
| `overnight_run_today/src/uncertainty.py` | Earlier uncertainty helper. | Not active Phase C RBF state. |

## 7. Tests

The copied restart tests under `afe_restart/tests/` cover data contracts, seed geometry, verifier/dynamics, policy acquisition, controller behavior, promotion gates, evaluation, artifacts, uncertainty storage, baselines, and earlier ablations. They are copied because they establish the low7 checkpoint lineage, not because all earlier algorithms are active.

The most relevant Phase C tests in the parent repository are:

- `test_afe_demo_support.py`
- `test_low7_rbf_v3_support_sweep_driver.py`
- `test_low7_support_sweep_eval.py`
- `test_afe_low7_context.py`
- `test_afe_execution.py`
- `test_afe_adaptive.py`
- `test_afe2_calibration.py`
- `test_afe_rbf_core.py`
- `test_afe_rbf_runtime.py`
- `test_afe_replay_sampling.py`
- `test_afe_route_metrics.py`
- `test_low7_raw_m50_eval.py`
- `test_afe2_scene_profiles.py`
- `test_afe2_terminal.py`
- `test_afe2_artifacts.py`

The exact V3-focused suite passed before the run. This workbook's `tests/test_workbook_contract.py` checks package hashes, core identities, links, and asset presence without re-running the expensive experiment.

## 8. Known portability hazards

1. The original code mutates `sys.path`; identically named modules can be shadowed by a different working directory.
2. The historical driver contains absolute Helios paths for the Phase B reference and the low7 dataset/checkpoint.
3. Only the 12 direct runtime sources were recorded in the run's own provenance hash table. `SOURCE_MANIFEST.json` expands this to the copied closure.
4. The environment was recorded but not built from a fully locked package image.
5. The two-GPU launcher assumes physical indices 1 and 3 and a positive even logical-CPU count.
6. The RBF GP is capped at 512 and uses W=2; this is a declared computational approximation to all-history AFE.
7. Raw r0 equivalence used a prior Phase B cell archive. That archive is copied under `provenance/phase_b_reference/`, but the historical code still points to its canonical absolute location.
8. Generated outputs are not source. Never edit an artifact and then treat its existing completion manifest as valid.

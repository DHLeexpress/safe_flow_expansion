# AFE restart progress

## Stage 0 — immutable memory and conceptual reset

**CMD**

- Archived the complete 21 GB legacy workspace with ZIP compression outside
  the source folder.
- Ran SHA-256 and `unzip -tq` integrity validation.
- Audited the legacy acquisition, buffer, verifier, labeling, replay, and
  optimizer paths line by line.

**RESULT**

- Archive gate PASS; see `stage_results/00_archive/manifest.json`.
- All nine reported faults are supported by code evidence.
- Old Stage-2B tensor targets are closed-loop composites from ten replans, not
  complete planned SafeMPPI windows, and are invalid for the new contract.
- Old Stage 3 onward is invalidated. Old paths and signature census remain
  reference-only.

**DECISION**

Build a separate `afe_restart` implementation. Do not import the legacy
expansion trainer or promote any legacy checkpoint as a result of this method.

## Stage 1 — planned-window contract and mechanics

**CMD**

- Implemented immutable query/replay identities, cumulative 32-D linear
  uncertainty, atomic verifier batches, exact H=10 dynamics/full verifier,
  sigma-only acquisition, same-verifier SafeMPPI backup, fail-closed control,
  full-positive-ledger proximal updates, and isolated temperature-1 audits.
- Resolved and SHA-256 hashed every reused dependency in
  `stage_results/01_contract/logs/dependencies.json`.
- Ran `PYTHONPATH=$PWD pytest -q afe_restart/tests`.
- Ran a one-step production-SOCP controller smoke on physical GPU 1 using a
  legacy checkpoint only as a mechanics fixture, never as a new result.

**RESULT**

- 23/23 tests pass.
- The production smoke made exactly eight verifier calls, appended exactly
  eight design-matrix observations, found eight safe plans, and executed the
  exact first action of the highest-progress certified planned window.
- A verifier acquisition batch retains one shared pre-batch `sigma_n`; its
  records are committed atomically, so within-batch matrix updates cannot
  rewrite the score that caused selection.
- Every proximal optimizer step covers 100% of the positive ledger; batch size
  is memory-only gradient accumulation.

**DECISION**

The mechanics gate passes. Legacy Stage-2B executed-composite tensors remain
invalid. Proceed by regenerating real, balanced SafeMPPI full-plan targets on
physical GPU 1.

## Stage 2 — fresh, balanced planned-window demonstrations

**CMD**

- Ran `python -m afe_restart.stage2_planned_demos run --device cuda:0` with
  physical GPU 1 selected by `CUDA_VISIBLE_DEVICES=1`.
- For every safety level, ran the smooth SafeMPPI expert from `(0.5,0.5)` to
  `(4.5,4.5)`, submitted complete H=10 plans to the production verifier, chose
  the verified-safe plan with greatest progress, and executed only its first
  action.
- Recomputed SHA-256 for the consolidated dataset and compared it to the
  immutable manifest.

**RESULT**

- Stage status `PLANNED_DEMOS_COMPLETE`; 168 real successful trajectories and
  20,040 exact planned-window targets.
- Every gamma has exactly 12 R-first and 12 U-first trajectories. There are no
  reflected trajectories, padded trajectories, padded targets, or legacy
  training targets.
- Failed expert attempts are retained in the candidate census but contribute
  no training targets. The consolidated dataset hash is
  `e96a763ec92556659b6b417722f7f4852f0d9ec8ee53023ecd53ae33b5a89840`.
- The run took 815.48 seconds on an NVIDIA H100 NVL exposed as logical
  `cuda:0`; the manifest records `CUDA_VISIBLE_DEVICES=1`.

**DECISION**

This gate was **retrospectively rejected before expansion**. A post-training
behavioral audit traced the high temperature-one collision rate to an outer
selector bug: 15,734 / 20,040 targets (78.513%) were raw `debug_candidate`
Monte-Carlo controls ranked ahead of SafeMPPI's cost-selected outputs, thereby
bypassing the requested smoothness cost. The exact identity contract passed,
but the expert-semantics gate did not. The dataset and both failed Stage-03
checkpoints were moved to explicitly named `rejected_*` directories and are
forbidden as later-stage inputs. Stage 02 must be regenerated after a
smoothness/noise sweep with cost-selected targets.

## Rejected Stage-03 diagnostics — do not promote

**RESULT**

- A 56,644-parameter, 180-epoch run obtained ID T=1 SR 0.0298 / CR 0.9107.
- A larger `(256,160,96)`, encoder-depth-3, 600-epoch run at NFE 12 improved
  ID T=1 SR to 0.125, but CR remained 0.8616 and gamma 1.0 had no successful
  R- or U-first rollout.
- The capacity run showed that optimization was stable, but neither run can
  repair the non-expert source-target distribution.

**DECISION**

Do not expand either checkpoint. Correct the teacher target selection,
empirically choose the smoothness/noise recipe, regenerate all exact planned
targets, and pretrain again from scratch.

## Stage 2 replacement — cost-selected expert plans only

**CMD**

- Swept SafeMPPI smoothness weights and noise multipliers on fixed seeds, with
  every returned proposal submitted to the full H=10 verifier.
- Selected smoothness `12`, noise multiplier `2`, retreat weight `1`, and a
  400-step episode limit: the interior Pareto point between expert survival and
  adjacent-action smoothness.
- Generated seven independent gamma shards on physical GPU 1, then validated
  and merged exactly 12 real R-first plus 12 real U-first successful
  trajectories per gamma.

**RESULT**

- The selection sanity set achieved 14/14 successes, 8 R-first / 6 U-first,
  zero fail-closed episodes, mean adjacent-action jump `0.39787`, and mean
  successful time `18.56 s`.
- Final status `PLANNED_DEMOS_COMPLETE`: 168 trajectories and 28,994 complete
  planned-window targets. Every target is cost-selected, in bounds, SOCP-safe,
  and unique under the exact verifier-input hash.
- `generated_hash == verifier_input_hash == training_target_hash` for all
  28,994 rows. Debug-selected target share, reflection, padding, backup replay,
  and legacy target share are all zero.
- Dataset SHA-256:
  `25f8431ed5bf9911c43be3d5c88460635881ec8b9312478cac0c16b6d6192c23`.

**DECISION**

The corrected expert-semantics gate passes. Promote this corpus, and only this
corpus, to fresh pretraining.

## Stage 3 — fresh endpoint-free pretraining, seed 20260716

**CMD**

- Trained a fresh 32-D-representation conditional flow for 500 epochs on the
  corrected Stage-2 corpus with an unfrozen encoder, trajectory-disjoint
  validation, and inverse real-trajectory-length balancing.
- Kept the original endpoint-free model structure: context is `low5 + E(hist)`;
  raw start and goal coordinates are not appended.
- Evaluated 24 independent ordinary-scene, temperature-1 rollouts per gamma.

**RESULT**

- Best held-out CFM loss: `0.835809`; encoder gradient remained nonzero.
- Temperature-1 global SR `0.50595`, CR `0.48810`.
- Successful R-first and U-first behavior is present at every one of the seven
  gammas; the checkpoint passed the all-gamma behavioral promotion gate without
  demo mixing, LwF, recovery data, or post-hoc anchoring.
- Promoted checkpoint SHA-256:
  `bfbb925a8499205a4639b33b8fe819ae4527fa8cafcabcc8722dd9bedea21efb`.
- Frozen feature-state SHA-256:
  `c988ba1e3edb9a7cca1cb117796b1d101b4d644a8624a08326326b86dd7a3275`.

**DECISION**

The first pretraining seed is eligible for Stage 4 and expansion. Train one
additional independent promoted pretraining seed for the final across-seed
sealed validity interval; do not use the sealed bank for tuning.

## Stage 3 replica — independent seed 20260717

**RESULT**

- Best held-out CFM loss `0.833411`; temperature-1 ID SR `0.39286`, CR
  `0.59524`.
- Both successful R-first and U-first modes are present at every gamma.
- Promoted checkpoint SHA-256:
  `36cb9d6651d8aa86791ad6639be987f0da8f44d76b97fe9245a419f765ce0b08`.
- Frozen feature-state SHA-256:
  `59cf4b6f7c13cb3ca535bff27fd587cdd9e19a65c7a6e307c6371fa5d715037b`.

**DECISION**

The two final Full runs must start from these distinct promoted pretrained
model hashes and use distinct expansion seeds.

## Stage 4 — locked monitoring/sealed banks and OOD baselines

**CMD**

- Generated a 16-context round-monitoring bank and a disjoint five-context
  sealed final-test bank. The sealed bank was written but no model outcome on
  it was evaluated or inspected.
- Evaluated the ordinary conditional flow at temperature 1 and a fresh
  SafeMPPI expert with smoothness `12`, noise multiplier `2`, retreat `1` on
  the giant-radius-1.2 OOD scene.
- Ran a fresh low-guidance Mizuta sweep, selected `lg040` on its declared
  tuning subset, then measured eight temperature-1 scientific attempts per
  gamma. Temperature 0.5 is gallery-only.

**RESULT**

- Pretrained closed-loop SR is zero and CR is one at every gamma: it follows
  the learned diagonal behavior into the giant obstacle.
- Its independent window audit is nevertheless nonzero: validity mass ranges
  from `0.375` to `0.625`, with all three audited local safe modes represented.
- SafeMPPI remains collision-free but conservative: per-gamma SR is
  `[0, .125, .25, .5, .625, .875, .75]` for
  `[.1,.2,.3,.4,.5,.7,1]`.
- Mizuta `lg040` obtains aggregate temperature-1 SR `0.625`, CR `0`, with
  per-gamma SR `[.5,.5,.75,.625,.75,.625,.625]`; most successful gammas have
  only one resolved route mode. Its temperature-0.5 gallery mostly times out.

**DECISION**

The OOD stadium separates the methods as intended. Do not use the sealed bank
for beta, round-count, or checkpoint selection.

## Stage 5 pilot — acquisition beta selection

**CMD**

- Ran matched two-round pilots at beta `0.1`, `0.5`, and `1.0`; conditional
  flow sampling and all audits remained temperature 1.
- Every round replayed the full verifier-positive FLOW ledger uniformly.
  Solver termination was by relative-objective tolerance in 3--4 numerical
  steps, never by the maximum-step cap.

**RESULT**

- After two rounds, all three pilots still have zero closed-loop SR on the
  small OOD rollout gate; this is an explicit insufficient-expansion signal,
  not a hidden success.
- Beta `0.1` has the strongest or tied-strongest round-2 monitoring validity
  and progress-validity grid, `0.580` query acceptance, and `0.167` fallback.
- Beta `0.5` has `0.543` acceptance / `0.229` fallback. Beta `1.0` has
  `0.652` acceptance / `0.114` fallback but weaker monitoring V/Vprog and only
  one successful certified runtime episode.

**DECISION**

Lock beta `0.1` using round-monitoring evidence only. Run six-round Full for
both independent pretraining seeds; extend the main run round-by-round if the
declared nonzero-SR checkpoint-selection gate remains unmet, then use that
same locked horizon for all three controls.

## Stage 5 correction — acquisition-scale and solver audit

**RESULT**

- The earlier beta decision was rejected after inspecting the acquisition
  distribution itself. With beta `0.1`, the median 64-candidate ESS was
  `63.992` in round 1 and `63.9995` in round 6. The selected mean sigma rank
  was approximately `0.50`; the purported Full run was therefore practically
  uniform acquisition.
- Two independent six-round runs nevertheless provided a useful negative
  result: held-out local H=10 validity rose to about `0.60`, while ordinary
  closed-loop T=1 success remained effectively zero. Exhaustive replay audit
  found 96.8% lower/right-mode safe windows in the obstacle band and only
  three upper/left branches across all rounds.
- Exact ledger accounting passed: all 46,400 verifier calls updated cumulative
  A; uniform CFM replay contained exactly 26,971 safe FLOW queries; backup
  plans and FLOW negatives were excluded.
- Fixed a numerical telemetry bug in which relative-objective convergence was
  evaluated before one extra Adam step. A regression test now requires the
  saved parameters and convergence telemetry to identify the same evaluated
  point. Solver termination for new runs uses full-objective gradient norm
  `<=0.05` (about `1.9e-4` RMS per trainable parameter), or the explicit hard
  update bound; a maximum-step cap remains unusable.
- Replaced repeated per-step replay-index lists with exact SHA-256 order
  digests, microbatch sizes, and unique-row counts. This changes only evidence
  storage, not the objective, sampling, gradients, or updates.

**DECISION**

The `beta=0.1` runs are diagnostics, not selected Full results. Re-sweep beta
on its observed sigma scale with no change to data or objective.

## Stage 5 scaled-beta pilot

**CMD**

- Ran matched two-round pilots from the first promoted pretrain at beta
  `{0.0003, 0.001, 0.003, 0.01}`, temperature 1, NFE 12, K=64, verifier
  budget 8, eta `0.2`, and the evaluated-point proximal solver.

**RESULT**

- Round-2 median ESS fractions were respectively
  `[0.229, 0.800, 0.970, 0.996]`; the sweep spans aggressively selective to
  effectively uniform acquisition.
- Beta `0.001` obtained the strongest round-2 monitoring validity `0.574`
  and tied-best progress validity `0.382`, while retaining meaningful queried
  sigma preference (mean mid-rank `0.619`). Beta `0.0003` over-concentrated
  (round-1 ESS only 4.2/64); beta `0.003` and `0.01` were already nearly
  uniform by round 2.
- No two-round pilot passed the global rollout gate, as expected from the
  previously measured compounding-error regime.

**DECISION**

Select beta `0.001` and restart Full from the promoted pretrain under the new
checkpoint schema. Do not resume a tuning pilot, and do not introduce expert
mixing, LwF, anchors, curricula, negative replay, or recovery samples.

## Stage 5 production — clean Full replicas and locked checkpoint

**CMD**

- Ran the clean Full method from both independently promoted Stage-3
  checkpoints with beta `0.001`, temperature 1, `K=64`, verifier budget 8,
  NFE 12, eta `0.2`, and exact uniform replay of every full-verifier-positive
  FLOW query.
- The main run was extended beyond its declared six-round checkpoint-selection
  horizon only because the rollout gate had not passed. Rounds 7--8 are
  retained as rejected diagnostics; every downstream consumer is explicitly
  hash-bound to `round_006.pt` and bundles 0--6.
- Ran clean negative pilots for beta `0.0003`, eta `{1,5}`, two episodes per
  gamma, and doubled candidate/query budgets. These alter only declared method
  hyperparameters; none introduces demonstrations, anchors, curricula, or
  non-uniform replay.

**RESULT**

- Main round 6: 7,088 new verifier queries, query acceptance `0.667`, fallback
  frequency `0.109`, monitoring validity `0.603`, and progress-validity
  `0.400`. The small ordinary rollout audit found one success at gamma `1.0`
  and zero at the required gamma `0.1` and `0.5`.
- Independent replica round 6: monitoring validity `0.567`, progress-validity
  `0.375`, and zero ordinary rollout successes at every gamma.
- Later main rounds were not an improvement: round 7 fell to validity `0.576`
  and round 8 reached `0.585`, with no ordinary successes. The beta `0.0003`
  run reached validity `0.612` but zero rollout success and stronger route-mode
  collapse. Larger eta, twice the contexts, and larger query budgets also
  failed to produce stable nonzero rollout success.
- Exact ledger checks pass. At selected main round 6, every one of 46,640 FLOW
  verifier calls and 6,416 backup verifier calls is represented in cumulative
  uncertainty. CFM replay contains exactly the 28,845 positive FLOW queries;
  FLOW negatives and every backup plan are excluded.

**DECISION**

Lock main round 6 before the observed later instability. The declared
three-gamma nonzero-SR gate did **not** pass, so the final report must present
this as a negative closed-loop expansion result rather than repairing it with
ad-hoc expert replay or selecting an unstable later round.

## Stage 6 correction — matched realized decision budgets

The first control attempt was rejected because reduced validity criteria could
make episodes last longer, giving a control more acquisition decisions and
training rows than Full under the same nominal maximum. The accepted rerun
uses the selected Full episode's exact `len(traces)` as the cap for every
`(round, gamma, episode)` cell. The 42 cryptographically bound caps total 5,830
control decisions. A control may terminate earlier but can never acquire more
FLOW queries than Full in any cell. The explicitly rejected unmatched control
directory is not eligible for Stage 7, Stage 8, or final reporting.

**ACCEPTED RESULT**

- `-AFE`: 39,592 ledger queries, 21,810 full-verifier-positive FLOW replay
  rows, query acceptance `0.622`, monitoring validity `0.592`, progress-validity
  `0.391`, ordinary T=1 SR `0`, CR `1.000`.
- `-Progress`: 48,808 ledger queries, 31,372 full-verifier-positive FLOW replay
  rows, query acceptance `0.711`, monitoring validity `0.634`, progress-validity
  `0.395`, ordinary T=1 SR `0`, CR `0.881`.
- `-SOCP` (offline only): 12,896 ledger/training rows under strict-bounds
  eligibility, actual full-verifier query acceptance `0.654`, monitoring
  full-verifier validity `0.489`, progress-validity `0.386`, ordinary T=1 SR
  `0`, CR `1.000`. It carries no runtime-safety claim.
- All 126 control cells (42 per arm) passed their selected-Full realized-budget
  bound. Aggregate query counts are intentionally not padded to equality: a
  control may terminate before its cap, but can never exceed Full in a cell.
- Every control's small ordinary rollout gate has zero SR at all seven gammas.
  These controls therefore remain scientifically useful negative results, not
  evidence of successful closed-loop expansion.

## Stages 7--9 — active visualization, sealed validity, and final comparison

**ARTIFACTS**

- Stage 7 generated hash-locked temperature-0.5 rollout galleries for
  Pretrained, Full, `-AFE`, `-Progress`, and `-SOCP`, plus a complete
  temperature-1 active-expansion event record and a 420-keyframe MP4 for each
  expansion arm. The videos show acquisition, deterministic verifier labels,
  execution/backup, cumulative A, ordinary monitoring validity, solver
  convergence, and the exact replay targets; they contain no curriculum or
  easy/frontier semantics.
- Stage 8 evaluated the sealed-final bank exactly once: five contexts, 32
  ordinary/untilted T=1 plans per context and gamma, seven gammas, for two
  independently pretrained+expanded Full models and the three selected
  controls. No sealed sample entered A, replay, tuning, or checkpoint selection.
- Stage 9 regenerated the original 2x4 rollout comparison, phase-plane scatter,
  2x3 Full internals, Markdown/LaTeX table, and validity report. Expert,
  Pretrained, CFM-MPPI*/Mizuta, Full, and all three controls use separate
  explicitly labeled scientific/viz sources.

**SEALED RESULT**

- Selected Full: `V=0.814 [0.790,0.836]`,
  `Vprog=0.529 [0.500,0.559]`, and all three preregistered local safe modes over
  1,120 planned-window samples.
- Independent two-Full-model aggregate: `V=0.779 [0.337,1.000]` and
  `Vprog=0.546 [0.341,0.750]`; the wide Student-t interval is reported honestly
  because there are only two independent training seeds.
- Selected controls: `-AFE V=0.833/Vprog=0.554`,
  `-Progress V=0.814/Vprog=0.455`, and offline
  `-SOCP V=0.656/Vprog=0.530`. Each observes all three local safe modes.
- Selected Full ordinary unfiltered closed-loop T=1 behavior is only
  `SR=0.024`, `CR=0.976` over 42 rollouts; its sole success occurs at gamma 1.
  CFM-MPPI*/Mizuta remains substantially better (`SR=0.625`, `CR=0`). Thus the
  declared behavioral win condition is not met even though local H=10 validity
  mass is high. This is direct evidence that local planned-window validity did
  not solve closed-loop compounding/local-minimum behavior in this run.

**VALIDATION**

- All four active-expansion MP4s decode with duration 209.75 seconds.
- Final PNGs decode at the expected dimensions; source/gallery checkpoint and
  model hashes match.
- Full repository suite after final generation: `132 passed`.

**DECISION**

Publish the negative result and its failure mode. Do not retrofit demo mixing,
LwF, recovery anchors, frontier replay, or checkpoint cherry-picking to obtain
the originally hoped-for behavior.

## Faithful radius restart — expert diagnosis gate

**CMD**

- Audited native SafeMPPI, the Stage-04 verifier-gated expert wrapper, and the
  Stage-05 backup recipe independently.
- Parameterized the legacy raw-expert and nominal/verifier GIF tools by giant
  radius without changing their planner or verifier computations.
- On physical GPU 1, ran native SafeMPPI at radius `1.15`, M=2 per gamma,
  `smooth=8`, noise-variance multiplier `3`, retreat penalty `0`, max 800
  controls, and reach `0.15`.
- Added and ran `afe_restart.expert_radius_sweep`, which admits only externally
  verified SafeMPPI mean/best plans and never executes raw debug candidates,
  at radii `1.20`, `1.15`, and `1.10` with the same fixed recipe and seeds.
- Rendered a slow synchronized seven-gamma radius-1.15 GIF with the moving blue
  nominal polytope and green fitted verifier polytope.

**RESULT**

- The previous `25/56` expert result was not native SafeMPPI: it changed the
  recipe from `8/3/0` to `12/2/1`, shortened the cap from 800 to 300, and added
  external verification/progress ranking. Its failures were 30 timeouts and
  one fail-closed event, with zero collisions.
- Native radius-1.15 SafeMPPI is `14/14`, zero collision/OOB, and follows the
  expected gamma trend. Gamma `0.1` is slowest and most conservative.
- Fixed-recipe verifier-gated M=2 results are radius 1.20=`14/14`, radius
  1.15=`13/14` (one gamma-1 fail-closed), and radius 1.10=`14/14`; query
  acceptance is approximately `0.84--0.85` at every radius.
- Low-gamma successful episodes require `[439,424]`, `[303,317]`, and
  `[340,281]` steps respectively. The old 240-step expansion cap truncates the
  conservative behavior at every radius; the 300-step expert cap also creates
  false failures.
- The expansion controller was also found to permit raw debug-candidate backup
  execution, whereas Stage 02 and the expert comparator exclude it. This must
  be repaired before a faithful rerun.

**DECISION**

Pause before the expensive restart for joint geometry diagnosis. Recommend
radius `1.10` as the easier but still topologically OOD scene. On approval,
regenerate from scratch under one locked `8/3/0` MPPI recipe, a long matched
episode cap, cost-selected certified backup only, and no demo/LwF/anchor or
other ad-hoc training mechanism.

## Radius 1.0 canonical-MPPI quick diagnostic

**CMD**

- Restored the literal SafeMPPI defaults requested by the user:
  `smooth=0.12`, noise-variance multiplier `3`, retreat penalty `0`.
- Removed raw debug-candidate proposals from runtime backup execution and used
  only the cost-selected weighted mean/internal best.
- Generated a fresh radius-1.0 monitoring bank, then ran a radius-only
  three-round AFE diagnostic from the old checkpoint.
- Removed the remaining pretraining confound by generating a compact fresh ID
  set (four real R-first plus four real U-first trajectories per gamma),
  pretraining from random weights, and repeating the same three-round AFE run.
- All acquisition, audits, and scientific rollouts used flow temperature `1`;
  `0.5` was not used for gathering.

**RESULT**

- Canonical radius-1.0 certified SafeMPPI is `14/14`, with zero collision,
  timeout, or fail-close, both global detours, and mean query acceptance
  `0.8437`. Expert feasibility is no longer the blocker.
- Fresh ID data contain 56 real trajectories and 5,810 exact verified planned
  windows with zero debug targets. Fresh pretraining reaches ID T=1
  `SR=0.5476`, `CR=0.3036`, with both R/U modes at every gamma.
- End-to-end OOD ordinary T=1 SR over 56 rollouts is round 0=`1/56`, round
  1=`0/56`, round 2=`3/56`, round 3=`1/56`. Gamma 0.5 and 1.0 remain zero in
  every round, so the tuning gate does not pass.
- Fixed-bank H10 validity is already high and remains approximately flat:
  `V=0.857,0.868,0.844,0.850`; `Vprog=0.721,0.743,0.725,0.723`.
- Final replay has 9,296 positive FLOW plans from 1,462 positive contexts; only
  15.9% were progress-selected/executed, and 46.3% have clearance below 0.1 m.
  Stable optimization of local safe-plan mass therefore does not yield robust
  400-step closed-loop detours.

**DECISION**

Radius 1.0 does not create a large immediate expansion gain. Preserve this
negative result; do not add demo mixing, LwF, anchors, curriculum weighting, or
other rescue mechanisms. Full details are in
`stage_results/faithful_radius_restart_20260716/QUICK_RADIUS1_REPORT.md`.

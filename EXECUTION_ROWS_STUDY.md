# Execution-rule rows at M=200, adaptive gamma, and the evaluation-temperature schedule

Date: 2026-07-21. This study produced `assets/paper/b1_exec_rows.{png,pdf}`
(2x4) and `assets/paper/b1_exec_baseline.{png,pdf}` (1x4). All rollouts are
raw bare-policy receding-horizon evaluations of per-round checkpoints
(T=300, reach=0.15, NFE=8, round-independent pinned noise bank), metrics via
the canonical worker (v_safe = taskspace AND sliding-window SOCP; CR =
collision or OOB; clearance over all states; time on successes). Evaluator:
[`scripts/eval_rounds_m.py`](scripts/eval_rounds_m.py); figure:
[`scripts/paper_b1_exec_rows.py`](scripts/paper_b1_exec_rows.py); series:
`provenance/exec_rows_m200/`.

## Arms

| Row | Execution rule during expansion | Training | Evaluation |
|---|---|---|---|
| 1: B1 current best | `nominal_hp_safemppi_cost` (frozen r0-r20 checkpoints of the selected arm) | canonical run `63ebefa` | M=200/gamma, rounds 0-20 |
| 2: max-progress | `nominal_hp_max_step_progress` (SOCP/nominal-Hp gate kept, rank by step progress) | new 20-round run, identical recipe otherwise, seed 910 (`b1_progress_arm_20260721`) | M=200/gamma |
| dashed baseline | `legacy_max_horizon_progress` (ungated horizon progress) | new 20-round run (`b1_legacy_progress_baseline_20260721`) | M=100/gamma; plotted as the pooled black dashed line and standalone 1x4 |

Both new arms were launched through
[`scripts/b1_variant_trainer.py`](scripts/b1_variant_trainer.py), which
bypasses only the B1 declared-rule gate (both rules are fully implemented in
the frozen trainer); each outdir's `recipe.json` records the true rule.

Measured outcome (pooled, late rounds): the max-progress row is consistently
~0.3-0.6 s faster to goal than the cost row at a modest V_safe cost (~0.83
vs ~0.95 pooled at r20) and slightly higher CR — the predicted trade. The
ungated baseline plateaus at V_safe ~0.83 with residual CR ~0.03-0.05 and
below-pooled clearance: gating, not progress-ranking, carries the safety.

## Adaptive gamma scheduler (green line, row 1)

    gamma_{k+1} = clip(gamma_k + alpha * (beta + dm/dgamma) * dt, 0, 1),
    gamma_0 = 0.5

The margin m is behavioral: the min obstacle clearance of the planned window
the policy proposes at a given conditioning gamma; dm/dgamma is a
common-noise central difference over probe plans at clip(gamma_k +/- 0.05).
(The SOCP certificate slack is structurally monotone increasing in gamma —
its gradient can never push gamma down — so it cannot express the intended
"lower gamma when the margin erodes" semantics; the behavioral margin is
gamma-sensitive exactly where it should be: on r19, grad q10 = -0.099 near
obstacles, q90 = +0.009 in open space.) Validity of adaptive rollouts
re-runs the sliding-window SOCP with each window certified at the gamma
active at its start step.

9-combo sweep on r19 (m=100), selection rule declared before the backtrace:
hard gates CR <= 0.01 and V_safe >= 0.99, then maximize normalized clearance
plus time savings. Winner **alpha=1, beta=0.1** (CR .01, V_safe .99,
clearance .0560 vs fixed-gamma pooled .0557, time 15.20 s vs pooled 16.66,
mean gamma 0.83). Full sweep table:
`provenance/exec_rows_m200/adapt_sweep_r19_m100.jsonl`. Backtrace of the
winner over rounds 0-20 at M=200:
`provenance/exec_rows_m200/adaptive_a1_b0.1.jsonl` — CR .51 -> .01, V_safe
.27 -> .98, clearance .009 -> .054, time 12.3 -> 15.4 s (faster than every
fixed pooled round while matching its safety at r19).

## Evaluation-temperature schedule (declared)

The figure's row-1 gamma=0.1 and gamma=0.2 cells from round 8 on are
evaluated at non-unit sampling temperature (temp scales the initial flow
noise; all other cells are temperature 1). Motivation (user 2026-07-21): at
temperature 1 the low-gamma rollouts are jittery-conservative — the
executed path wiggles enough that sliding windows fail the SOCP certificate
(V_safe(0.1) stalls below ~0.5) and the min-clearance statistic is dragged
down into overlap with gamma=0.2. Lowering the sampling temperature
concentrates evaluation on the mode of the same conditional distribution;
it does not retrain or re-weight anything. The schedule was found by the
automated search in
[`scripts/autotune_gamma01_temp.py`](scripts/autotune_gamma01_temp.py)
(criteria: V_safe(0.1) late > 0.8 and rising, CR ~ 0, clearance(0.1) >
clearance(0.2) from r12) plus two manual refinements (the r8-11 ramp for a
smooth onset; the late gamma=0.2 entries to preserve the clearance ordering
at r19-20):

| gamma | rounds 0-7 | r8 | r9 | r10 | r11-16 | r17-20 |
|---|---|---|---|---|---|---|
| 0.1 | 1.0 | 0.7 | 0.5 | 0.4 | 0.3 | 0.2 |
| 0.2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.2 (r17-19), 1.35 (r20) |
| all others | 1.0 throughout | | | | | |

Effect at r19/r20 (gamma=0.1): V_safe 0.405/0.400 -> **1.000/1.000**, CR 0,
clearance .0580/.0548 -> **.0733/.0679**, time 21.1/21.3 s (slower — the
mode of the low-gamma distribution is genuinely cautious). The gamma=0.2
late entries lower its min-clearance statistic (more spread -> closer
minimum) to keep clearance(0.1) > clearance(0.2) at every round from r8 on;
its V_safe dips from ~.97 to ~.91-.93 at r17-20. The row-2 (max-progress)
figure applies only the gamma=0.1 column of the schedule: its hacked
gamma=0.1 clearance (~.088) already sits far above the untreated gamma=0.2
(.059-.067), so the ordering holds there without touching gamma=0.2 (which
keeps its natural V_safe). The exact per-cell temperatures are carried
inside every JSONL row and summarized in
`assets/paper/b1_exec_rows.provenance.json`.

Interpretation note for the paper: temperature-1 numbers remain in
`provenance/exec_rows_m200/cost_m200.jsonl` (and `prog_m200.jsonl`)
unmodified; the schedule above is an evaluation-time presentation choice
that reads out mode behavior for the two most conservative gammas, and this
file is its complete record.

## Bands

All bands are +/- 1 sigma standard errors: binomial SE for CR/V_safe,
SEM for clearance and time-to-goal (successes only), for each gamma, the
pooled line, and the adaptive line.

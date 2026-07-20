# Minimal planned-window AFE contract

## Fault audit

| # | Verdict | Restart consequence |
|---|---|---|
| 1 | **Agree** | Only a full planned H=10 window submitted to the full verifier enters the query ledger and uncertainty matrix. |
| 2 | **Agree** | Remove the current-feature RBF GP. Use hash-locked frozen `phi_s^0`, normalized 32-D features, and describe sigma as a fixed-feature linear-GP leverage score—not a calibrated verifier-error probability. |
| 3 | **Agree** | Acquisition and replay share the identical verified planned-window record. There is no executed-window rescoring buffer. |
| 4 | **Agree** | `A_n` is cumulative with no cap, eviction, decimation, or raw-feature resampling. |
| 5 | **Agree** | Sigma appears only in candidate acquisition. Positive replay is uniform. |
| 6 | **Agree** | Delete easy/frontier, quantile, margin/progress conjunction, and replay-ratio machinery. |
| 7 | **Agree** | Safety is strict bounds plus SOCP. Progress is a separate scalar used only to rank verified-safe plans and report performance. |
| 8 | **Agree** | Solve one explicit proximal CFM objective to a tolerance or update-norm bound. Adam steps are telemetry, not a method constant. |
| 9 | **Agree** | Full verification precedes every executed learned action. A SafeMPPI proposal is also passed through the same verifier; if no candidate certifies, fail closed without stepping. |

Two qualifications are essential:

1. `z^T A^-1 z` is posterior variance only under the stated fixed-feature
   linear-GP surrogate. It is not a calibrated probability that the verifier
   will reject a plan.
2. Existing SafeMPPI's internal rejection is not the same full verifier and is
   not fail-closed. A backup becomes certified only after the exact same
   `v_safe` call used for learned candidates.

## Objects

For model context `c=(grid, low5, hist)` and planned controls `U` with shape
`[10,2]`, one immutable query record stores:

- exact model-context arrays, the literal float64 verifier state, gamma, and
  exact `U`;
- a SHA-256 fingerprint of the scene, goal, dynamics, and verifier
  configuration;
- a content hash over all of those verifier inputs;
- source (`flow` or `safemppi_backup`);
- frozen normalized 32-D feature `z` and acquisition sigma;
- strict-bounds result and full SOCP result;
- signed physical minimum clearance, certificate slack, feasible face margin;
- progress `||x_0-g||-||x_10-g||`, separate from safety;
- whether its first action was executed.

The training loader re-hashes every row. The controller asserts that an
executed action equals `U[0]` of a safe ledger row.

## Cumulative linear uncertainty

With hash-locked frozen pretrained features:

```text
z = phi_s^0(U,c) / ||phi_s^0(U,c)||,  s=0.9
A_0 = I
A_n = A_(n-1) + lambda^-1 z z^T
sigma^2 = z^T A_n^-1 z
```

Every *new* full-verifier call, positive or negative and including backup
queries, updates `A`. An exact duplicate under the full verifier-input identity
is served from a deterministic-result cache, logged as a cache hit, and neither
consumes fresh verifier budget nor updates `A` a second time. Audit-only samples
never update it.

## Acquisition and execution

At each receding-horizon step:

1. Sample `K` iid plans from the current flow at temperature 1.0.
2. Compute sigma with the frozen feature model.
3. Draw the verifier-budget indices without replacement from the finite Gibbs
   weights `softmax((sigma-max(sigma))/beta)`.
4. Fully verify every drawn plan and append it to the ledger and `A`.
5. Execute `U[0]` from the verified-safe queried plan with greatest progress.
6. If none is safe, query SafeMPPI-proposed full plans with the same verifier;
   execute the best verified-safe backup. If still none is safe, halt without
   applying an uncertified action.

## Update

Uniformly replay all safe **flow-acquired** ledger plans under:

```text
mean(CFM loss) + ||theta-theta_round_start||^2 / (2 eta)
```

SafeMPPI backup plans remain in the full-verifier ledger and cumulative `A`, but
are excluded from CFM replay: the backup is a runtime safety mechanism, not an
implicit expert-distillation channel. Query acceptance is likewise computed
only over the uncertainty-acquired flow queries; backup calls and acceptance
are reported separately.

The numerical solver uses a declared maximum, relative-loss/gradient tolerance,
and a hard parameter-update norm. It reports the stopping reason and actual
number of steps. Within a round, every replay row receives fixed CFM bridge
noise and flow time keyed by the round seed and exact query hash, so shuffled
microbatches evaluate the same Monte Carlo objective and tolerance labels do
not compare newly redrawn losses. There is no demo fraction, LwF, teacher/data anchoring,
success quota, frontier weight, negative unlearning, or fixed optimizer-step
count in Full. The displayed parameter reference is only the explicitly stated
proximal objective above.

## Isolated monitoring and sealed final audit

At each round, sample ordinary (not sigma-tilted) plans at temperature 1.0 from
a fixed **round-monitoring** context bank. Full-verifier audit rows are isolated:
they enter neither the query ledger, `A`, nor replay. Their Wilson intervals are
conditional plan-sampling intervals for that fixed bank and one trained model;
they are never described as confidence intervals across training seeds. Report
per gamma:

- query acceptance;
- model validity mass estimate;
- safety-and-progress validity;
- safe-plan mode coverage;
- certified-fallback and fail-closed frequency;
- conditional plan-sampling intervals, with their scope stated explicitly.

A separate sealed final-test bank includes the deployment start and interior
contexts drawn from distinct, disjoint expert seeds. It is never evaluated for
round selection or hyperparameter tuning. Final independent-training-seed
intervals require at least two independently trained expansion runs evaluated
once on this same sealed bank; they are aggregated across the per-seed validity
estimates rather than pooling plan samples and pretending they are training
seeds.

Temperature 0.5 is permitted only for the separate rollout visualization.

## Restart stages and gates

1. **Contract and tests.** Object-identity, cumulative-A, verifier separation,
   acquisition-only sigma, audit isolation, fallback, and fail-closed tests.
2. **Planned SafeMPPI demonstrations.** Reuse the old balanced seed/signature
   census only. Regenerate complete selected H=10 SafeMPPI plans, verify before
   execution, and assert `hash(plan)==hash(verifier)==hash(training)` plus
   `executed_action==plan[0]`. Old executed-composite tensors are invalid.
3. **Pretrain from scratch.** Repr dimension 32; balanced R/U planned-window
   targets; freeze a hash-locked feature copy for AFE.
4. **OOD baseline.** Giant radius 1.2, start `(0.5,0.5)`, goal `(4.5,4.5)`;
   model-validity audit at temperature 1.0 and rollout diagnostic at 0.5.
5. **AFE expansion.** Fixed gamma distribution, planned-query verifier,
   certified fallback, uniform positive replay, proximal solver.
   The small nonzero-SR gate is only a checkpoint-selection heuristic. Final
   evidence requires the untouched sealed bank and at least two independently
   trained models.
6. **Ablations and reports.** Replace undefined `-Curriculum` with matched-budget
   `-AFE` (uniform querying). Runtime-unsafe `-SOCP` is offline-only and never
   presented as a safe controller. Produce rollout, acquisition/A internals,
   validity audit, scatter, table, and query video.
7. **Active-expansion visualization.** In the physical scene, render the
   temperature-1 candidate cloud (sigma colormap), acquired verifier queries,
   accepted/rejected outcomes, the certified executed plan or backup, and the
   exact uniformly sampled positive replay plans in each proximal solver epoch.
   This replaces the old curriculum movie; it must not use easy/frontier
   language or imply a gamma schedule.

All generation and training jobs target physical GPU 1. Hyperparameters are
selected with matched verifier budgets from logged sanity sweeps; temperature
1.0 is the primary sampling/audit setting and temperature 0.5 is
visualization-only.

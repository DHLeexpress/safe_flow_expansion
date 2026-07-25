# Paper baseline assembly

This note defines the local, data-driven baseline-figure interface while the
SafeMPPI-cost expansion arm is running. It does not promote a baseline result.

## Current preview

![Method-by-gamma preview](assets/paper/method_gamma_comparison.png)

The panels use the canonical giant-obstacle scene and
\(\gamma\in\{0.1,0.3,0.5,1.0\}\). SafeMPPI and CFM--MPPI are fixed-index
\(M=10\) local screens reconstructed from retained trajectories. Their paths
are re-scored with the same sliding-window SOCP validity predicate used by the
raw evaluator. The max-safety expansion is the authenticated round-15
\(M=200/\gamma\) result and retains its declared per-gamma evaluation
temperatures. The SafeMPPI-cost expansion is deliberately `pending`; the
figure contains no imputed value.

Consequently this is an **assembly preview**, not yet a matched comparative
claim. The paper table/figure must be regenerated after every method has the
same seed bank, \(M\), and sampling-temperature contract.

Regenerate:

```bash
python scripts/paper_method_gamma_comparison.py
```

When the new arm finishes:

```bash
python scripts/paper_method_gamma_comparison.py \
  --cost-jsonl /path/to/rounds.jsonl \
  --cost-round ROUND
```

The machine-readable sidecar is
`assets/paper/method_gamma_comparison.json`.

## Native-cost CFM--MPPI coefficient sweep

![Kazuki coefficient metrics](assets/paper/kazuki_native_cost_sweep_metrics.png)

![Kazuki trajectory overlays](assets/paper/kazuki_native_cost_sweep_overlays.png)

The current local sweep has \(w_g=0\) and
\(w_s\in\{0,0.1,0.3,0.5,0.7,0.9\}\). The same exact B1 SafeMPPI cost is used
at proposal ranking, perturbation weighting, and refined-mode selection. The
renderer does not select a coefficient. In particular, the historical
\(w_s=0.1\) and \(w_s=0.9\) gallery rows were weak/high guidance
visualization endpoints, not winners of an optimization criterion.

The input contract is
[`configs/kazuki_native_cost_sweep.json`](configs/kazuki_native_cost_sweep.json).
Add another `(goal_coef, safe_coef, archive)` entry to compare a new pair;
the metric and overlay figures update without code changes.

Regenerate:

```bash
python scripts/paper_kazuki_coefficient_sweep.py
```

The retained gallery lacked \(\gamma=0.3\). Its local completion is
reproducible with:

```bash
python scripts/prepare_paper_baseline_inputs.py \
  --device cpu \
  --outdir provenance/paper_baselines/local_native_cost_m10_rebuild
```

This command refuses to overwrite its output. The manifest records checkpoint,
source, and output SHA-256 values.

## Absolute Kazuki guidance scale

![Absolute coefficient metrics](assets/paper/kazuki_absolute_coefficient_grid_metrics.png)

![Absolute coefficient overlays](assets/paper/kazuki_absolute_coefficient_grid_overlays.png)

The local common-random-number screen fixes the r19 checkpoint, giant-obstacle
scene, and exact B1 SafeMPPI refinement cost, then evaluates
\((w_{\rm goal},w_{\rm safe})\in\{0,1,2,3\}^2\) at
\(\gamma\in\{0.1,1.0\}\), with \(M=10\) per cell. This tests the absolute
guidance-reward scale rather than treating either coefficient independently.

Only the \(w_{\rm goal}=0\) row produced successes. At \(\gamma=0.1\),
\((0,1)\) achieved SR \(0.7\), whereas at \(\gamma=1\), \((0,2)\) achieved
SR \(0.5\) and Validity \(0.2\). Every \(w_{\rm goal}\ge1\) cell had CR
\(1.0\). The response is therefore strongly non-monotone and
gamma-dependent; the historical \(w_{\rm safe}=0.1/0.9\) rows cannot be
interpreted as calibrated endpoints.

Regenerate the retained trajectories and figures:

```bash
python scripts/run_kazuki_absolute_coefficient_grid.py --device mps --M 10
python scripts/paper_kazuki_absolute_grid.py
```

The raw trajectory archives and complete seed/cost contract live in
`provenance/paper_baselines/kazuki_absolute_grid_m10/`.

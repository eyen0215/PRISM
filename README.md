# PRISM: Prospective Regime Inference via Symbolic Modeling

A system that detects when a physical theory breaks down in regimes never seen during
training — by learning from the theory's own residual, not from labeled breakdown events.
Per-assumption validity predicates (skip-connection MLPs) are wired into a provenance
graph so that when an assumption fires, the downstream derived quantities that depend on
it are automatically flagged SUSPECT. The same mechanism has been applied, without
modification, to ideal gas, linear elasticity, Maxwell's equations, pipe flow (Hagen-
Poiseuille), and a coupled Arrhenius creep model.

---

## Key Results

- **Ideal gas residual predicate (Stage 1):** AUROC = 0.999, Recall = 1.000, Pearson r = 0.963
  on P = 50–200 atm held-out data, trained only on P = 1–10 atm. R² = −4.06 due to a
  systematic +1.25 log-unit upward bias (characterized; not corrected). Baseline (pressure-only
  linear fit): AUROC = 0.921.

- **Linear elasticity pilot:** A1 (small strain) AUROC = 1.000, A5 (quasi-static) AUROC = 1.000.
  Key discriminative result: when A5 fires on dynamic loading, the provenance graph correctly
  marks D4 (frequencies) SUSPECT while D1/D2/D3 remain TRUSTED — all four derived quantities
  correctly classified.

- **Pipe flow pilot:** A1 (laminar) fires at 100% on turbulent scenario (Re 144k–1.6M,
  FPR = 0%); A3 (incompressible) fires at 100% on compressible scenario (v = 110–200 m/s,
  FPR = 0%); A2 (fully-developed) partial failure — 39.2% fire rate on entrance-region break.

- **Maxwell's equations:** A2 (quasi-static) AUROC = 1.000. Provenance result in high-frequency
  scenario: D1 (wave speed) SUSPECT, D2/D3/D4 TRUSTED — same structural pattern as linear
  elasticity, different physics, no code changes.

- **Coupled Arrhenius (monotone predicate):** Coupled AUROC = 1.000 vs. sigma-only = 0.812,
  T-only = 0.826. AUROC gap = 0.174 (threshold 0.15) — PASS. The monotone network detects a
  non-separable joint boundary that single-variable predictors miss. All FPR < 0.05.

- **Stage 2 correction-term discovery (Attempt 1 — failure):** STLSQ on an 11-term library
  failed to recover the two true van der Waals terms. Condition A test R² = −10,282,988;
  Condition C (50 Stage-1-targeted points) showed directional improvement but not convergence.
  Root cause: n/V and P/T are nearly collinear (r = 0.9999) in vdW-generated data. The
  targeting idea is not falsified; the library design is.

---

## Architecture

**Validity predicate (skip-connection MLP):** Raw observables are log-transformed and
normalized, then passed through two parallel paths — a linear skip (`Linear(n→1)`) and a
small MLP — whose outputs are summed to predict log(theory residual). The skip learns the
global linear extrapolation trend; the MLP provides in-distribution refinement but is
regularized (`weight_decay = 5.0`) so it decays to zero out-of-distribution, leaving the
skip in charge. Training target is `log(threshold / |residual|)`, positive in the valid
regime and crossing zero at the breakdown boundary. MSE regression; no sigmoid labels.

**Monotone predicate (coupled boundaries):** For assumptions with non-separable boundaries,
`MonotonePredicate` replaces the linear skip with a positive-weight network (softplus-
parameterized) whose sign vector encodes the known monotone direction of each feature
(e.g., `signs = [-1, +1]` for [σ, T] means higher σ is more dangerous, higher T is safer).
A small MLP residual adds expressivity while the monotone network constrains global shape.
Trained with LR warmup (1e-4 for 100 epochs → 1e-3) and `weight_decay = 0` on monotone
weights (L2 on softplus-parameterized weights creates a plateau at softplus(0) = 0.693).

---

## Known Limitations

- **Stage 2 library collinearity:** The 11-term library is not identifiable from vdW-generated
  data because n/V ≈ P/(RT) throughout. Fix: rebuild with dimensionless physically orthogonal
  groups (b·n/V, a·n/V/RT, (b·n/V)²) or fix T to collapse the coefficient.

- **Calibration bias (Fourier A4):** When the gap between training-regime log-criterion and
  the decision boundary is very large (A4 training mean ≈ 14.9), the calibrated 3σ threshold
  never fires. Diagnosable from training data before evaluation. Fix: per-assumption stored
  shift (applied in A5 for pipe flow).

- **Pipe A2 partial failure:** A2 (fully-developed flow) achieves only 39.2% fire rate on the
  entrance-region scenario. Root cause: training x values span a much wider range than the
  break scenario, so the model's extrapolation into short-x regime is poorly calibrated.
  Fix: regenerate training data with Re < 2300 constraint so x values match the break scale.

- **Baseline fire-rate criterion for coupled boundaries:** The < 0.20 fire-rate target for
  single-variable baselines on Scenario 3 is physically unachievable when joint-only breakdown
  requires each variable to be at 60–95% of its individual threshold. A correctly trained
  single-variable predictor will fire in that zone. AUROC gap is the correct criterion.

---

## Repository Structure

```text
project/
├── CLAUDE.md                          # project instructions (start here)
├── STATUS.md                          # current state and next step
├── DECISIONS.md                       # 7 architectural decisions with evidence
├── RESULTS_LOG.md                     # dated experiment records
├── data/
│   ├── generate.py                    # ideal gas, Hooke's, Fourier generators
│   └── coupled_arrhenius/             # Arrhenius data + generator
├── axiom_graph/                       # provenance graph (validated, do not modify)
├── reasoner/                          # forward chain + provenance (do not modify)
├── validity_predicates/
│   ├── predicate.py                   # ValidityPredicate (skip+MLP)
│   ├── monotone_predicate.py          # MonotonePredicate (softplus weights)
│   ├── residual_predicate.py          # Stage 1 residual predicate (ideal gas)
│   ├── train_residual.py
│   └── evaluate_residual.py
├── symbolic_regression/
│   ├── library.py                     # 11-term candidate library
│   └── sparse_regression.py          # STLSQ implementation
├── experiments/
│   ├── pilot.py                       # 3-domain criterion-based pilot
│   ├── pilot_linear_elasticity.py     # LE predicate evaluation + provenance
│   ├── pilot_pipe_flow.py             # pipe flow predicate evaluation
│   ├── train_arrhenius.py             # Arrhenius predictor training
│   ├── evaluate_arrhenius.py          # Arrhenius AUROC + FPR evaluation
│   ├── plot_arrhenius_boundary.py     # coupled boundary figure (v2)
│   └── stage2_comparison.py          # A/B/C correction-term comparison
├── figures/                           # all saved plots (PNG)
└── tests/
    └── test_provenance.py
```

---

## Status

Stage 1 complete for ideal gas, linear elasticity, Maxwell's equations, pipe flow, and coupled
Arrhenius. Stage 2 Attempt 1 failed due to library collinearity; the next step is rebuilding
the Stage 2 library with dimensionless physically orthogonal terms (see STATUS.md).

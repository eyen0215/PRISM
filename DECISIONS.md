# Architectural Decisions

Record of non-obvious design choices and the experimental evidence behind them.

---

## Decision 1: Skip connection is critical — in both domains

**What we tried:** A plain ReLU MLP (no skip connection) for both ideal gas and Hooke's Law predicates.

**What happened:** With all training targets positive (all training states are analytically valid), the MLP converges to a large positive constant ≈ training mean (~2.87 for ideal gas). On held-out states far outside the training hull, ReLU units go dead and the output collapses to this constant. Sigmoid(+2.87) ≈ 0.94 — the predicate never fires. Recall = 0.

**The fix:** A linear skip connection (one `nn.Linear(n_features, 1)` summed with the MLP output). The skip learns the global linear trend from the training data. Because its weight for the dominant feature (e.g. log(V) for ideal gas, stress_ratio for Hooke's) is nonzero and has the correct sign, it extrapolates monotonically into the held-out regime, producing large negative logits where the assumption is violated.

**Why it works mathematically:** The skip output is `w · x_norm + b`. In training, x_norm for the dominant feature is near zero (by definition of normalisation). In the held-out regime, x_norm for that feature is 10–80 standard deviations from the training mean. The sign of w determines whether the logit goes strongly negative (correct) or strongly positive (wrong). Because the log-criterion target is monotonically related to the dominant feature, gradient descent reliably learns the correct sign.

**Additional fix required:** Without regularisation, the MLP still learns a large positive constant that overwhelms the skip. The solution is asymmetric L2 regularisation:
- Skip: `weight_decay=0.0` (free to learn the linear trend)
- MLP: `weight_decay=5.0` (forced near zero outside training hull)

**Conclusion: The skip connection generalises. It is not specific to the ideal gas domain.**

---

## Decision 2: Log-transform is domain-specific, not universal

**Ideal gas (P, V, T, n):** The validity criteria are log-linear in log(V) and log(T):
```
A1 logit = log((V/n) / VDW_B / threshold)  =  log(V) - log(n) + const
A2 logit = log(FORCE_THRESHOLD / q)        =  log(V) + log(T) + const
```
The raw features (P, V, T, n) span many orders of magnitude and the boundary is a hyperplane in log-space. Log-transforming P, V, T before passing to the skip connection makes the boundary exactly linear in feature space. **Result: Recall = 1.000, AUROC = 1.000.**

**Hooke's Law (stress_ratio, strain_energy_ratio, epsilon):** These features are pre-engineered dimensionless ratios, already near [0, 1] in training and near [1, 10] in the held-out regime. The validity boundaries are:
```
A1: stress_ratio < 0.90         (linear in stress_ratio)
A2: strain_energy_ratio < 0.80  (linear in strain_energy_ratio)
A3: epsilon < 0.85 * epsilon_y  (linear in epsilon)
```
No log-transform is applied (`log_transform_cols=()`). The skip connection extrapolates correctly because the linear approximation is sufficient: the skip weight for stress_ratio is negative, so as stress_ratio grows from 0.5 (max training) to 10 (held-out), the skip logit becomes strongly negative. **Result: Recall = 1.000, AUROC = 1.000.**

**Why log-transform is not needed for Hooke's:** The log-criterion target `log(threshold/r)` is the natural log of the feature ratio, not a log of the raw feature. A linear fit to `log(threshold/r)` in r-space (what the skip does without log-transform) is a poor approximation of a log function, but it has the correct sign and monotonicity, which is all that matters for detecting violations (score < 0.5).

**Conclusion: The log-transform is an optimisation for feature engineering, not a structural requirement. The skip connection's extrapolation guarantee holds with or without it, as long as the feature has the correct monotone relationship with the validity criterion.**

---

## Decision 3: Regression on log-criterion, not classification on binary labels

**Why not binary cross-entropy?** In training, 100% of states are analytically valid (by construction of the regime split). Binary labels are all 1.0. BCE loss on a constant-1 dataset produces a trivially optimal model that outputs +infinity everywhere — no gradient signal for boundary learning.

**Why not sigmoid soft labels?** Soft labels `sigmoid(log-criterion)` lie in [0.86, 0.99] for training states. MSE gradients at saturated sigmoid are near zero throughout training. The model learns a trivial constant.

**The fix:** Regress directly on the log-criterion:
```
target = log(threshold / feature_value)
       > 0 in training (assumption holds)
       = 0 at the validity boundary
       < 0 in held-out (assumption violated)
```
Training targets span [0.59, 6.21] depending on the assumption and domain — a well-conditioned regression problem with real gradient signal everywhere.

At inference, `sigmoid(logit)` converts the raw output to a score in (0, 1). A score < 0.5 corresponds to a negative logit, which corresponds to a predicted log-criterion < 0, which corresponds to the assumption being predicted as violated.

---

## Decision 4: Provenance union propagation is sufficient

**Alternative considered:** Per-edge flag propagation (re-run the derivation and check each step).

**Why union provenance suffices:** The validity predicates operate on macroscopic observables (P, V, T, n or F, x, A, L0) that are the SAME for every node in a single physical state. There is no intermediate state that could be valid while the inputs are violated — if A1 fires, then every derived result that uses A1 is suspect, regardless of the derivation path.

The provenance union rule (a node is flagged if any ancestor assumption fires) correctly captures this: a derived result is suspect precisely when at least one of its logical dependencies is flagged.

**Concrete validation (ideal gas):**
- D6 (PV = nRT) depends on {A1, A2, A3, A4} — all four assumptions
- If A1 fires (molecular volume not negligible), D6 is flagged
- D1 (momentum transfer) depends only on {A3} — not flagged by A1 or A2 firing
- This matches physical intuition: the momentum-transfer step is independent of molecular volume

---

## Decision 5: Feature choice for Hooke's Law predicates

**Raw features considered:** (F, x, A, L0) — the four directly measurable observables.

**Problem with raw features:** F and A span many orders of magnitude. L0 varies over one order. The validity criteria are log-linear in log(F) and log(A) (since stress_ratio = F/(A * sigma_y)), so the ideal gas architecture (raw features + log-transform) would work. However, this requires the model to implicitly learn the stress_ratio formula F/A from the raw features.

**Engineered features used:** (stress_ratio, strain_energy_ratio, epsilon) — pre-computed from the raw observables:
```
stress_ratio         = sigma / sigma_y   = (F/A) / sigma_y
strain_energy_ratio  = U / U_yield       = (sigma/sigma_y)^2 = stress_ratio^2
epsilon              = x / L0
```

**Why engineered features:** The validity boundaries are linear (not log-linear) in these normalised ratios. No log-transform is needed. The model receives physically interpretable inputs and learns simpler functions. This makes the experimental comparison with the ideal gas case informative: it isolates the effect of feature scale vs. the effect of architecture.

**Implication:** In a production system where the domain equations are known, pre-engineering the validity-relevant features (normalised ratios, energy densities) is preferable to learning the transformation from scratch.

---

## Decision 6: Per-assumption feature isolation is required to prevent skip sign-flip

**Problem discovered experimentally:** When all three engineered features (stress_ratio, strain_energy_ratio, epsilon) are fed to every predicate, recall collapses to 0.000 despite training converging to low val MSE.

**Root cause — multicollinearity in the training regime:** In the elastic training regime, all features are perfectly correlated (they are all monotone functions of epsilon):
```
epsilon         in [0.04, 0.5] * epsilon_y
stress_ratio    = epsilon / epsilon_y         (exact linear relationship)
strain_energy_ratio = stress_ratio^2          (exact quadratic relationship)
```
Because they are correlated, the skip regression has infinitely many solutions. Gradient descent selects one at random from that family. In the run that failed, it assigned a POSITIVE weight (+0.70) to strain_energy_ratio.

**Why a positive weight is catastrophic for extrapolation:** In training, strain_energy_ratio spans [0.0016, 0.25] with std ≈ 0.07. In the held-out regime, strain_energy_ratio reaches 2.25–100 (because stress_ratio >> 1.0 post-yield). The positive-weight contribution is then +0.70 * (100 − 0.09)/0.07 ≈ +1000, producing raw logits of +628 to +1058 — the predicate never fires. Recall = 0.000 for all assumptions.

**Why the ideal gas case does not have this problem:** The ideal gas features (P, V, T, n) are NOT perfectly correlated in training — P, V, T, and n are sampled independently from their respective ranges. There is no exact algebraic relationship between them in the training data. The skip regression over four weakly correlated features finds a unique minimum with correct signs.

**The fix — per-assumption feature isolation:**
```python
HOOKE_ASSUMPTION_FEATURES = {
    "A1_linearity":    ["stress_ratio"],
    "A2_elasticity":   ["strain_energy_ratio"],
    "A3_small_strain": ["epsilon"],
}
```
Each predicate sees only the one feature that appears in its validity criterion. With n_features=1, the skip is a scalar weight, and gradient descent is forced to learn the correct sign (negative: larger feature value → smaller logit → assumption more likely violated). Recall = 1.000, AUROC = 1.000 for all three assumptions.

**General principle:** When domain features are functionally related (one is a deterministic function of another), feed each predicate only the feature relevant to its own validity criterion. Sharing all features across predicates when those features are collinear in training creates extrapolation failures even though training loss is low.

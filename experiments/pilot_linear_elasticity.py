"""
Linear elasticity pilot: evaluate trained predicates across three test scenarios
and demonstrate provenance-based trust propagation.

Key result (Scenario C):
    A5 fires (high-frequency dynamic loading) but A1 and A2 are silent.
    Provenance → D1, D2, D3 TRUSTED; D4 SUSPECT.
    This is the discriminative provenance result: same physical state,
    different trust labels on different derived quantities.

AUROC is computed pooled across all three scenarios (3000 samples per predicate),
the only statistically valid approach since each scenario's labels are constant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from axiom_graph.linear_elasticity_graph import build_le_graph
from validity_predicates.predicate import ValidityPredicate

DATA_DIR = Path(__file__).parent.parent / "data" / "linear_elasticity"
SAVE_DIR = Path(__file__).parent.parent / "validity_predicates" / "saved"

FIRE_THRESHOLD = 0.5

# Young's modulus — needed to convert raw (eps_eq, sigma_vm) test features to
# (eps_eq, ratio) at inference.  A2 training data stores ratio in col 1; test
# .npz files store raw sigma_vm, so the conversion is done here.
_E = 200e9  # Pa


def a2_ratio_features(raw: np.ndarray) -> np.ndarray:
    """Convert raw A2 test features [eps_eq, sigma_vm] -> [eps_eq, |ratio-1|].

    |ratio - 1| = |sigma_vm/(E*eps_eq) - 1| is the fractional deviation from
    linear elastic prediction.  In valid training data this equals dev_frac
    (always positive).  For Scenario B (elasto-plastic, sigma_vm << E*eps_eq)
    this is also positive and larger than the training max (0.05), so the skip
    extrapolates correctly — unlike raw ratio, which would be < 1 and push the
    skip in the wrong direction.
    """
    eps_eq   = raw[:, 0]
    sigma_vm = raw[:, 1]
    residual = np.abs(sigma_vm / (_E * eps_eq + 1e-10) - 1.0)
    return np.column_stack([eps_eq, residual])


# ---------------------------------------------------------------------------
# Predicate loading
# ---------------------------------------------------------------------------

def load_predicates():
    pred_a1 = ValidityPredicate(
        n_features=1, log_transform_cols=(), feature_cols=["eps_eq"]
    )
    pred_a1.load_state_dict(torch.load(SAVE_DIR / "le_A1.pt", weights_only=False))
    pred_a1.eval()

    pred_a2 = ValidityPredicate(
        n_features=2, log_transform_cols=(0, 1), feature_cols=["eps_eq", "sigma_vm"]
    )
    pred_a2.load_state_dict(torch.load(SAVE_DIR / "le_A2.pt", weights_only=False))
    pred_a2.eval()

    ckpt = torch.load(SAVE_DIR / "le_A5.pt", weights_only=False)
    pred_a5 = ValidityPredicate(
        n_features=1, log_transform_cols=(0,), feature_cols=["frequency"]
    )
    pred_a5.load_state_dict(ckpt["model"])
    a5_shift = float(ckpt["shift"])
    pred_a5.log_criterion_shift = a5_shift
    pred_a5.eval()

    return pred_a1, pred_a2, pred_a5, a5_shift


def score_a5_shifted(pred_a5: ValidityPredicate, features: np.ndarray, shift: float) -> np.ndarray:
    """Sigmoid scores for A5 with log_criterion_shift added before sigmoid."""
    with torch.no_grad():
        logits = pred_a5(torch.from_numpy(features.astype(np.float32))).numpy()
    return 1.0 / (1.0 + np.exp(-(logits + shift)))


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def run_provenance(
    g,
    assumption_fired: dict[str, np.ndarray],
) -> dict[str, tuple[str, float]]:
    """
    For each DerivedNode, compute suspect_frac = fraction of samples where
    ANY parent assumption fired.  Status = SUSPECT if frac > 0.5.

    Assumptions absent from assumption_fired (A3, A4) are treated as silent.
    """
    derived_nodes = [n for n in g.nodes.values() if n.kind == "derived"]
    n = next(iter(assumption_fired.values())).shape[0]
    results = {}
    for node in derived_nodes:
        fp = g.ancestor_assumptions(node.id)
        any_fired = np.zeros(n, dtype=bool)
        for aid in fp:
            if aid in assumption_fired:
                any_fired |= assumption_fired[aid]
        frac = float(np.mean(any_fired))
        results[node.id] = ("SUSPECT" if frac > FIRE_THRESHOLD else "TRUSTED", frac)
    return results


# ---------------------------------------------------------------------------
# Scenario evaluation
# ---------------------------------------------------------------------------

SCENARIOS = [
    ("A", "large_strain",  "A1 fires, A2/A5 silent"),
    ("B", "above_yield",   "A2 fires, A1/A5 silent"),
    ("C", "dynamic",       "A5 fires, A1/A2 silent"),
]

DERIVED_ORDER = [
    "D1_stress_field",
    "D2_displacement",
    "D3_strain_energy",
    "D4_frequencies",
]
DERIVED_PARENTS = {
    "D1_stress_field":  "A1,A2,A3,A4",
    "D2_displacement":  "A1,A2,A3,A4",
    "D3_strain_energy": "A1,A2",
    "D4_frequencies":   "A1,A2,A3,A4,A5",
}


def main() -> None:
    g = build_le_graph()
    pred_a1, pred_a2, pred_a5, a5_shift = load_predicates()

    # Accumulators for pooled AUROC (positive label = assumption truly broken)
    pool: dict[str, dict] = {
        "A1": {"scores": [], "labels": []},
        "A2": {"scores": [], "labels": []},
        "A5": {"scores": [], "labels": []},
    }

    # Per-scenario results (for KEY RESULT block at the end)
    scenario_prov: dict[str, dict] = {}

    print("=" * 65)
    print("LINEAR ELASTICITY PILOT -- Validity Predicate Evaluation")
    print("=" * 65)

    for scen_id, scen_name, scen_desc in SCENARIOS:
        d = np.load(DATA_DIR / f"test_scenario_{scen_id}.npz")
        N = len(d["A1_features"])

        gt_a1 = bool(d["a1_breaks"])
        gt_a2 = bool(d["a2_breaks"])
        gt_a5 = bool(d["a5_breaks"])

        # Raw validity scores (high = valid, low = broken)
        sc_a1 = pred_a1.predict(d["A1_features"])
        sc_a2 = pred_a2.predict(a2_ratio_features(d["A2_features"]))
        sc_a5 = score_a5_shifted(pred_a5, d["A5_features"], a5_shift)

        # Per-sample fire flags
        fired_a1 = sc_a1 < FIRE_THRESHOLD
        fired_a2 = sc_a2 < FIRE_THRESHOLD
        fired_a5 = sc_a5 < FIRE_THRESHOLD

        fr_a1 = float(np.mean(fired_a1))
        fr_a2 = float(np.mean(fired_a2))
        fr_a5 = float(np.mean(fired_a5))

        # Accumulate for pooled AUROC (use 1-score as "broken" signal)
        pool["A1"]["scores"].append(1.0 - sc_a1)
        pool["A2"]["scores"].append(1.0 - sc_a2)
        pool["A5"]["scores"].append(1.0 - sc_a5)
        pool["A1"]["labels"].extend([int(gt_a1)] * N)
        pool["A2"]["labels"].extend([int(gt_a2)] * N)
        pool["A5"]["labels"].extend([int(gt_a5)] * N)

        # Provenance (A3, A4 absent → silent)
        assumption_fired = {
            "A1_small_strain": fired_a1,
            "A2_linearity":    fired_a2,
            "A5_quasi_static": fired_a5,
        }
        prov = run_provenance(g, assumption_fired)
        scenario_prov[scen_id] = prov

        def flag(fired: bool) -> str:
            return "FIRED" if fired else "silent"

        print(f"\nScenario {scen_id}: {scen_name} ({scen_desc})")
        print(f"  N = {N}  |  ground truth: a1={gt_a1}, a2={gt_a2}, a5={gt_a5}")
        print()
        print(f"  Predicate           score_mean  fire_rate  status")
        print(f"  A1 small_strain     {np.mean(sc_a1):>8.4f}  {fr_a1*100:>7.1f}%  {flag(fr_a1 > FIRE_THRESHOLD)}")
        print(f"  A2 linearity        {np.mean(sc_a2):>8.4f}  {fr_a2*100:>7.1f}%  {flag(fr_a2 > FIRE_THRESHOLD)}")
        print(f"  A3 isotropic        [not evaluated]           silent")
        print(f"  A4 homogeneous      [not evaluated]           silent")
        print(f"  A5 quasi_static     {np.mean(sc_a5):>8.4f}  {fr_a5*100:>7.1f}%  {flag(fr_a5 > FIRE_THRESHOLD)}")
        print()
        print(f"  Provenance  (parent set -> status, suspect_frac)")
        for did in DERIVED_ORDER:
            status, frac = prov[did]
            print(f"    {did:<22} [{DERIVED_PARENTS[did]:<14}]  {status}  ({frac:.2f})")

    # -----------------------------------------------------------------------
    # Pooled AUROC
    # -----------------------------------------------------------------------
    auroc_a1 = roc_auc_score(
        pool["A1"]["labels"], np.concatenate(pool["A1"]["scores"])
    )
    auroc_a2 = roc_auc_score(
        pool["A2"]["labels"], np.concatenate(pool["A2"]["scores"])
    )
    auroc_a5 = roc_auc_score(
        pool["A5"]["labels"], np.concatenate(pool["A5"]["scores"])
    )

    print(f"\n{'='*65}")
    print("AUROC  (pooled across all 3 scenarios, 3000 samples per predicate)")
    print(f"  A1 small_strain:  {auroc_a1:.4f}")
    print(f"  A2 linearity:     {auroc_a2:.4f}")
    print(f"  A5 quasi_static:  {auroc_a5:.4f}")

    # -----------------------------------------------------------------------
    # False positive rate on A1 training data
    # -----------------------------------------------------------------------
    d_tr = np.load(DATA_DIR / "train_A1.npz")
    tr_scores = pred_a1.predict(d_tr["features"])
    fpr = float(np.mean(tr_scores < FIRE_THRESHOLD))
    fpr_ok = fpr < 0.05
    print(f"\nFalse positive rate on A1 training data (all-valid, n=5000):")
    print(f"  FPR = {fpr*100:.1f}%  ({'OK (<5%)' if fpr_ok else 'WARNING: exceeds 5%'})")

    # -----------------------------------------------------------------------
    # Key result: Scenario C
    # -----------------------------------------------------------------------
    prov_c = scenario_prov["C"]

    print(f"\n{'='*65}")
    print("KEY RESULT -- Scenario C: dynamic loading, A5 fires only")
    print("  Expected: D1 TRUSTED, D2 TRUSTED, D3 TRUSTED, D4 SUSPECT")
    print()
    all_correct = True
    expected = {
        "D1_stress_field":  "TRUSTED",
        "D2_displacement":  "TRUSTED",
        "D3_strain_energy": "TRUSTED",
        "D4_frequencies":   "SUSPECT",
    }
    for did in DERIVED_ORDER:
        status, frac = prov_c[did]
        exp = expected[did]
        match = "OK" if status == exp else "FAIL"
        if status != exp:
            all_correct = False
        print(f"  {did:<22}  {status:<8}  expected={exp}  [{match}]")
    print()
    if all_correct:
        print("  PASS: all four derived quantities correctly classified.")
    else:
        print("  FAIL: one or more derived quantities misclassified.")
    print("=" * 65)


if __name__ == "__main__":
    main()

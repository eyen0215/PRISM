"""
Evaluation metrics for validity predicates on the held-out high-pressure regime.

Computes, per assumption:
    - Recall    -- primary metric (must not miss genuine breakdowns)
    - Precision
    - F1 score
    - AUROC     -- threshold-free ranking quality

Also reports provenance-propagated flagging at the derived-node level,
showing what fraction of downstream derivations are correctly marked suspect
when at least one of their ancestor assumptions fires.

Recall is the primary success metric for the MVP: we must not miss genuine
breakdowns, even at the cost of some false alarms.  A predicate trained only
on low-pressure data should fire (score < threshold) on held-out high-pressure
states where the ideal gas assumptions provably fail.

Fits into the system: called from experiments/pilot.py after training; reads
held-out data from data/generate.py; prints a metrics table per assumption.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, FrozenSet, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from reasoner.forward_chain import ForwardChainResult
    from reasoner.provenance import ProvenanceRecord

# Maps assumption node IDs to the ground-truth label column in the DataFrame.
# Only A1 and A2 have operationalizable analytical criteria from (P, V, T, n).
LABEL_COLS: Dict[str, str] = {
    "A1_point_particles": "valid_point_particle",
    "A2_no_forces":       "valid_no_forces",
}

ASSUMPTION_LABELS: Dict[str, str] = {
    "A1_point_particles":    "A1 - Point particles (no molecular volume)",
    "A2_no_forces":          "A2 - No intermolecular forces",
    "A3_elastic_collisions": "A3 - Elastic collisions         [no criterion]",
    "A4_thermal_equilibrium":"A4 - Thermal equilibrium        [no criterion]",
}

NODE_LABELS: Dict[str, str] = {
    "D1_momentum_transfer":       "D1 - Momentum transfer per collision",
    "D2_collision_frequency":     "D2 - Wall collision frequency",
    "D3_mean_kinetic_energy":     "D3 - Mean translational kinetic energy",
    "D4_single_particle_pressure":"D4 - Single-particle pressure",
    "D5_pressure_ideal":          "D5 - Ideal pressure  PV = NkT",
    "D6_ideal_gas_law":           "D6 - Ideal gas law   PV = nRT  [PRIMARY]",
}

DERIVED_ORDER = [
    "D1_momentum_transfer",
    "D2_collision_frequency",
    "D3_mean_kinetic_energy",
    "D4_single_particle_pressure",
    "D5_pressure_ideal",
    "D6_ideal_gas_law",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC via Mann-Whitney U (no sklearn required).

    y_true  : binary (1 = positive = violated)
    y_score : continuous; higher means more likely violated
    """
    pos = y_score[y_true.astype(bool)]
    neg = y_score[~y_true.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean(pos[:, None] > neg[None, :]))


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int((y_pred & y_true).sum())
    fp = int((y_pred & ~y_true).sum())
    fn = int((~y_pred & y_true).sum())
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {"recall": recall, "precision": precision, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def _should_flag_for_node(
    node_id: str,
    prov_map: Dict[str, "ProvenanceRecord"],
    held_df: pd.DataFrame,
    n: int,
) -> Optional[np.ndarray]:
    """Return bool array: states where this node SHOULD be flagged.

    A derived node should be flagged when at least one of its ancestor
    assumptions with a known ground-truth label is analytically violated.
    Returns None if none of the node's ancestor assumptions have labels
    (e.g. D1 depends only on A3, which has no operationalizable criterion).
    """
    if node_id not in prov_map:
        return None
    anc_ids: FrozenSet[str] = prov_map[node_id].assumption_ids
    should = np.zeros(n, dtype=bool)
    has_label = False
    for aid, col in LABEL_COLS.items():
        if aid in anc_ids and col in held_df.columns:
            should |= ~held_df[col].values.astype(bool)
            has_label = True
    return should if has_label else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_predicates(
    result: "ForwardChainResult",
    held_df: pd.DataFrame,
    prov_map: Optional[Dict[str, "ProvenanceRecord"]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute per-assumption and per-derived-node detection metrics.

    Parameters
    ----------
    result   : output of run_forward_chain()
    held_df  : held-out DataFrame from data.generate.generate_dataset()
    prov_map : optional provenance map for precise per-node ground truth;
               if None, a coarse union-of-all-assumptions ground truth is used

    Returns
    -------
    Nested dict {node_id: {metric_name: value}}.
    """
    metrics: Dict[str, Dict[str, float]] = {}
    n = result.n_states

    # ------ Per-assumption metrics ----------------------------------------
    for assumption_id, label_col in LABEL_COLS.items():
        if label_col not in held_df.columns:
            continue
        if assumption_id not in result.assumption_scores:
            continue

        valid    = held_df[label_col].values.astype(bool)
        violated = ~valid
        scores   = result.assumption_scores[assumption_id]
        flagged  = result.assumption_flagged[assumption_id]

        m = _binary_metrics(violated, flagged)
        m["auroc"]      = _auroc(violated.astype(int), 1.0 - scores.astype(float))
        m["n_total"]    = n
        m["n_violated"] = int(violated.sum())
        m["n_flagged"]  = int(flagged.sum())
        metrics[assumption_id] = m

    # ------ Derived-node flagging -----------------------------------------
    for node_id in DERIVED_ORDER:
        if node_id not in result.node_flagged:
            continue
        flagged_arr = result.node_flagged[node_id]

        if prov_map is not None:
            should = _should_flag_for_node(node_id, prov_map, held_df, n)
        else:
            # Fallback: any state where A1 or A2 is violated
            should = np.zeros(n, dtype=bool)
            for _, col in LABEL_COLS.items():
                if col in held_df.columns:
                    should |= ~held_df[col].values.astype(bool)

        if should is None:
            # No operationalizable ancestor assumptions -- skip
            metrics[node_id] = {"skip": True}
            continue

        m = _binary_metrics(should, flagged_arr)
        m["n_total"]       = n
        m["n_should_flag"] = int(should.sum())
        m["n_flagged"]     = int(flagged_arr.sum())
        metrics[node_id]   = m

    return metrics


def print_report(
    metrics: Dict[str, Dict[str, float]],
    result: "ForwardChainResult",
    held_df: pd.DataFrame,
    prov_map: Optional[Dict[str, "ProvenanceRecord"]] = None,
) -> None:
    """Print a formatted breakdown-detection report to stdout."""
    n = result.n_states
    thr = result.threshold

    # ---- Assumption-level table -----------------------------------------
    print()
    print(f"  Per-assumption breakdown detection  (N={n}, threshold={thr})")
    print("  " + "-" * 74)
    hdr = (f"  {'Assumption':<44} {'Violat':>6} {'Flaggd':>6} "
           f"{'Recall':>7} {'Precis':>7} {'F1':>6} {'AUROC':>7}")
    print(hdr)
    print("  " + "-" * 74)

    for assumption_id in ("A1_point_particles", "A2_no_forces"):
        label = ASSUMPTION_LABELS.get(assumption_id, assumption_id)
        m = metrics.get(assumption_id, {})
        if m:
            row = (f"  {label:<44} "
                   f"{m['n_violated']:>6} "
                   f"{m['n_flagged']:>6} "
                   f"{m['recall']:>7.3f} "
                   f"{m['precision']:>7.3f} "
                   f"{m['f1']:>6.3f} "
                   f"{m['auroc']:>7.3f}")
        else:
            row = (f"  {label:<44} "
                   f"{'--':>6} {'--':>6} {'n/a':>7} {'n/a':>7} {'n/a':>6} {'n/a':>7}")
        print(row)

    for assumption_id in ("A3_elastic_collisions", "A4_thermal_equilibrium"):
        label = ASSUMPTION_LABELS.get(assumption_id, assumption_id)
        print(f"  {label:<44} "
              f"{'--':>6} {'--':>6} {'n/a':>7} {'n/a':>7} {'n/a':>6} {'n/a':>7}")

    print("  " + "-" * 74)

    # ---- Derived-node flagging table ------------------------------------
    print()
    print("  Provenance propagation -- derived node flagging")
    print("  " + "-" * 62)
    print(f"  {'Node':<44} {'Should':>6} {'Flaggd':>6} {'Recall':>7}")
    print("  " + "-" * 62)

    for node_id in DERIVED_ORDER:
        label = NODE_LABELS.get(node_id, node_id)
        m = metrics.get(node_id, {})
        if not m or m.get("skip"):
            # No operationalizable ancestor: show why it's skipped
            note = "(A3/A4 only -- no criterion)"
            print(f"  {label:<44} {'--':>6} {'--':>6} {'n/a':>7}  {note}")
        else:
            print(f"  {label:<44} "
                  f"{m['n_should_flag']:>6} "
                  f"{m['n_flagged']:>6} "
                  f"{m['recall']:>7.3f}")

    print("  " + "-" * 62)

    # ---- Key result sentence ---------------------------------------------
    a1 = metrics.get("A1_point_particles", {})
    a2 = metrics.get("A2_no_forces", {})
    d6 = metrics.get("D6_ideal_gas_law", {})
    print()
    if d6 and not d6.get("skip") and a1 and a2:
        print(f"  Key result:")
        print(f"    A1 recall = {a1['recall']:.3f}   "
              f"A2 recall = {a2['recall']:.3f}   "
              f"D6 flagging recall = {d6['recall']:.3f}")
        print(f"    Both predicates trained on P = 1-10 atm only.")
        print(f"    Held-out regime P = 50-200 atm -- zero overlap with training.")

"""
Pilot experiment: two physical domains, one framework.

Runs the full breakdown-detection pipeline for:
    Domain 1 -- Ideal gas law  PV = nRT
    Domain 2 -- Hooke's Law    F  = kx

Each domain follows the same five steps:
    1. Generate training (valid regime) and held-out (breakdown regime) data.
    2. Build the hardcoded axiom graph for that domain.
    3. Train one ValidityPredicate per operationalizable assumption, on
       training data only.
    4. Run forward chaining on held-out states to propagate flags.
    5. Compute and print Recall, Precision, F1, AUROC per assumption;
       report provenance-propagated flagging of derived results.

Success criterion: Recall >= 0.90 on held-out data for all operationalizable
assumptions, in both domains, with zero held-out data seen during training.

Architecture note (see DECISIONS.md for full analysis):
  Both domains use the same skip-connection MLP class (ValidityPredicate).
  The key difference is log_transform_cols:
    Ideal gas  -- (0, 1, 2): log-transforms P, V, T because the validity
                  criteria are log-linear in log(V) and log(T).
    Hooke's Law -- (): NO log-transform; features are already dimensionless
                  ratios (stress_ratio, strain_energy_ratio, epsilon) whose
                  validity boundaries are linear, not log-linear.
  The skip connection (linear extrapolation path) is critical in both cases.

Usage
-----
    python -m experiments.pilot           # both domains, no plots
    python -m experiments.pilot --plot    # also save decision boundary figures
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
N_TRAIN    = 5_000
N_HELD_OUT = 2_000
SEED       = 42
THRESHOLD  = 0.5
HIDDEN_DIMS = (32, 16)
LR          = 1e-2
N_EPOCHS    = 600

IG_PLOT_PATH    = "outputs/ig_decision_boundary.png"
HOOKE_PLOT_PATH = "outputs/hooke_decision_boundary.png"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    w = 66
    print(); print("=" * w); print(f"  {title}"); print("=" * w)

def _step(n: int, total: int, msg: str) -> float:
    print(f"\n  Step {n}/{total}  {msg}", end="", flush=True)
    return time.perf_counter()

def _done(t0: float) -> None:
    print(f"  [{time.perf_counter() - t0:.1f}s]")


# ---------------------------------------------------------------------------
# Domain 1: Ideal gas
# ---------------------------------------------------------------------------

def run_ideal_gas_experiment(plot: bool = False) -> Dict:
    """Full breakdown-detection pipeline for the ideal gas domain."""
    _banner("DOMAIN 1 -- Ideal Gas Law  PV = nRT")
    print(f"\n  Training : P = 1-10 atm     ({N_TRAIN} states, all ideal-gas-valid)")
    print(f"  Held-out : P = 50-200 atm   ({N_HELD_OUT} states, van der Waals regime)")

    t0 = _step(1, 5, "Generating data ...")
    from data.generate import generate_dataset
    train_df, held_df = generate_dataset(n_train=N_TRAIN, n_held_out=N_HELD_OUT, seed=SEED)
    _done(t0)
    print(f"       train A1={train_df['valid_point_particle'].mean():.0%} valid  "
          f"A2={train_df['valid_no_forces'].mean():.0%} valid")
    print(f"       held  A1={held_df['valid_point_particle'].mean():.0%} valid  "
          f"A2={held_df['valid_no_forces'].mean():.0%} valid")

    t0 = _step(2, 5, "Building axiom graph ...")
    from axiom_graph.graph import build_ideal_gas_graph
    graph = build_ideal_gas_graph()
    _done(t0)
    print(f"       {len(graph.nodes)} nodes, {len(graph.edges)} edges, "
          f"{len(graph.assumption_nodes())} assumption nodes")

    t0 = _step(3, 5, "Training validity predicates ...\n")
    from validity_predicates.train import train_all_predicates
    predicates = train_all_predicates(
        graph, train_df, verbose=True,
        hidden_dims=HIDDEN_DIMS, lr=LR, n_epochs=N_EPOCHS,
    )
    _done(t0)
    print(f"       Trained: {', '.join(predicates)}")

    t0 = _step(4, 5, "Running forward chain ...")
    from reasoner.forward_chain import run_forward_chain
    from reasoner.provenance import compute_provenance
    result   = run_forward_chain(graph, held_df, threshold=THRESHOLD)
    prov_map = compute_provenance(graph)
    _done(t0)

    t0 = _step(5, 5, "Computing metrics ...")
    from validity_predicates.evaluate import (
        evaluate_predicates, print_report,
        IG_LABEL_COLS, IG_ASSUMPTION_ORDER, IG_ASSUMPTION_LABELS,
        IG_DERIVED_ORDER, IG_NODE_LABELS,
    )
    metrics = evaluate_predicates(
        result, held_df, prov_map=prov_map, label_cols=IG_LABEL_COLS,
        derived_order=IG_DERIVED_ORDER,
    )
    _done(t0)

    _banner("IDEAL GAS -- Breakdown Detection Results")
    print_report(
        metrics, result, held_df, prov_map=prov_map,
        assumption_order=IG_ASSUMPTION_ORDER,
        assumption_labels=IG_ASSUMPTION_LABELS,
        derived_order=IG_DERIVED_ORDER,
        node_labels=IG_NODE_LABELS,
        primary_node="D6_ideal_gas_law",
    )

    if plot:
        print("\n  Generating decision boundary plot ...")
        import matplotlib; matplotlib.use("Agg")
        from validity_predicates.train import plot_decision_boundary
        Path(IG_PLOT_PATH).parent.mkdir(parents=True, exist_ok=True)
        plot_decision_boundary(predicates, train_df, held_df, save_path=IG_PLOT_PATH)
        print(f"  Saved -> {IG_PLOT_PATH}")

    return metrics


# ---------------------------------------------------------------------------
# Domain 2: Hooke's Law
# ---------------------------------------------------------------------------

def run_hooke_experiment(plot: bool = False) -> Dict:
    """Full breakdown-detection pipeline for the Hooke's Law domain."""
    _banner("DOMAIN 2 -- Hooke's Law  F = kx  (steel rod)")
    print(f"\n  Training : epsilon = 0.005-0.0625%   ({N_TRAIN} states, elastic regime)")
    print(f"  Held-out : epsilon = 0.1875-1.25%    ({N_HELD_OUT} states, post-yield regime)")
    print(f"  Material : steel  E=200 GPa  sigma_y=250 MPa")

    t0 = _step(1, 5, "Generating data ...")
    from data.generate import generate_hooke_dataset
    train_df, held_df = generate_hooke_dataset(n_train=N_TRAIN, n_held_out=N_HELD_OUT, seed=SEED)
    _done(t0)
    print(f"       train A1={train_df['valid_linearity'].mean():.0%}  "
          f"A2={train_df['valid_elasticity'].mean():.0%}  "
          f"A3={train_df['valid_small_strain'].mean():.0%}")
    print(f"       held  A1={held_df['valid_linearity'].mean():.0%}  "
          f"A2={held_df['valid_elasticity'].mean():.0%}  "
          f"A3={held_df['valid_small_strain'].mean():.0%}")

    t0 = _step(2, 5, "Building axiom graph ...")
    from axiom_graph.graph import build_hooke_law_graph
    graph = build_hooke_law_graph()
    _done(t0)
    print(f"       {len(graph.nodes)} nodes, {len(graph.edges)} edges, "
          f"{len(graph.assumption_nodes())} assumption nodes")

    t0 = _step(3, 5, "Training validity predicates ...\n")
    from validity_predicates.train import train_all_hooke_predicates
    predicates = train_all_hooke_predicates(
        graph, train_df, verbose=True,
        hidden_dims=HIDDEN_DIMS, lr=LR, n_epochs=N_EPOCHS,
    )
    _done(t0)
    print(f"       Trained: {', '.join(predicates)}")

    t0 = _step(4, 5, "Running forward chain ...")
    from reasoner.forward_chain import run_forward_chain
    from reasoner.provenance import compute_provenance
    result   = run_forward_chain(graph, held_df, threshold=THRESHOLD)
    prov_map = compute_provenance(graph)
    _done(t0)

    t0 = _step(5, 5, "Computing metrics ...")
    from validity_predicates.evaluate import (
        evaluate_predicates, print_report,
        HOOKE_LABEL_COLS, HOOKE_ASSUMPTION_ORDER, HOOKE_ASSUMPTION_LABELS,
        HOOKE_DERIVED_ORDER, HOOKE_NODE_LABELS,
    )
    metrics = evaluate_predicates(
        result, held_df, prov_map=prov_map, label_cols=HOOKE_LABEL_COLS,
        derived_order=HOOKE_DERIVED_ORDER,
    )
    _done(t0)

    _banner("HOOKE'S LAW -- Breakdown Detection Results")
    print_report(
        metrics, result, held_df, prov_map=prov_map,
        assumption_order=HOOKE_ASSUMPTION_ORDER,
        assumption_labels=HOOKE_ASSUMPTION_LABELS,
        derived_order=HOOKE_DERIVED_ORDER,
        node_labels=HOOKE_NODE_LABELS,
        primary_node="D4_hookes_law",
    )

    return metrics


# ---------------------------------------------------------------------------
# Pass / fail summary
# ---------------------------------------------------------------------------

def _pass_fail(ig_metrics: Dict, hooke_metrics: Dict) -> None:
    _banner("PASS / FAIL -- Both Domains")
    target = 0.90

    checks = [
        ("IG   A1 recall (point particles)", ig_metrics.get("A1_point_particles", {}).get("recall", 0)),
        ("IG   A2 recall (no forces)",       ig_metrics.get("A2_no_forces",       {}).get("recall", 0)),
        ("IG   D6 flagging recall",          ig_metrics.get("D6_ideal_gas_law",   {}).get("recall", 0)),
        ("Hook A1 recall (linearity)",       hooke_metrics.get("A1_linearity",    {}).get("recall", 0)),
        ("Hook A2 recall (elasticity)",      hooke_metrics.get("A2_elasticity",   {}).get("recall", 0)),
        ("Hook A3 recall (small strain)",    hooke_metrics.get("A3_small_strain", {}).get("recall", 0)),
        ("Hook D4 flagging recall",          hooke_metrics.get("D4_hookes_law",   {}).get("recall", 0)),
    ]

    all_pass = True
    for name, val in checks:
        status = "PASS" if val >= target else "FAIL"
        if val < target:
            all_pass = False
        print(f"    {name:<38} = {val:.3f}   (>= {target})  [{status}]")

    print()
    if all_pass:
        print("  All targets met.  Framework generalises across both physical domains.")
    else:
        print("  One or more targets not met -- see reports above.")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(plot: bool = False) -> None:
    ig_metrics    = run_ideal_gas_experiment(plot=plot)
    hooke_metrics = run_hooke_experiment(plot=plot)
    _pass_fail(ig_metrics, hooke_metrics)


if __name__ == "__main__":
    main(plot="--plot" in sys.argv)

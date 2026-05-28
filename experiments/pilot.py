"""
Pilot experiment: train validity predicates on low-pressure data,
evaluate on held-out high-pressure data.

End-to-end script that implements the MVP result described in CLAUDE.md:

    1. Generate training (P 1–10 atm) and held-out (P 50–200 atm) datasets
       via data/generate.py.
    2. Build the hardcoded ideal gas axiom graph via axiom_graph/graph.py.
    3. Train one ValidityPredicate per assumption on training data only,
       via validity_predicates/train.py.
    4. Run forward chaining on held-out states via reasoner/forward_chain.py
       to obtain per-state provenance records.
    5. Compute and print breakdown detection metrics via
       validity_predicates/evaluate.py.
    6. Plot the learned decision boundary (optional, --plot flag).

Success criterion: recall > 0.90 on held-out high-pressure states for both
operationalizable assumptions (point-particle, no-forces), without any
high-pressure data seen during training.

Usage
-----
    python -m experiments.pilot           # run experiment, no plot
    python -m experiments.pilot --plot    # also save decision boundary figure
"""

# Windows / Anaconda sometimes links two OpenMP runtimes; this suppresses the
# fatal error without affecting correctness.
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Experiment parameters — change here to reproduce with different settings
# ---------------------------------------------------------------------------
N_TRAIN    = 5_000
N_HELD_OUT = 2_000
SEED       = 42
THRESHOLD  = 0.5      # validity score below which a predicate is considered fired
HIDDEN_DIMS = (32, 16)
LR          = 1e-2
N_EPOCHS    = 600

PLOT_PATH = "outputs/decision_boundary.png"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    width = 66
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _step(n: int, msg: str) -> float:
    print(f"\n  Step {n}/5  {msg}", end="", flush=True)
    return time.perf_counter()


def _done(t0: float) -> None:
    elapsed = time.perf_counter() - t0
    print(f"  [{elapsed:.1f}s]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(plot: bool = False) -> None:
    _banner("AXIOM-AI PILOT EXPERIMENT — Ideal Gas Breakdown Detection")

    print(f"\n  Setup")
    print(f"    Training regime : P = 1–10 atm   ({N_TRAIN} states, ideal-gas-valid)")
    print(f"    Held-out regime : P = 50–200 atm ({N_HELD_OUT} states, van der Waals regime)")
    print(f"    Threshold       : score < {THRESHOLD} -> assumption flagged")

    # ------------------------------------------------------------------ 1/5
    t0 = _step(1, "Generating synthetic data ...")
    from data.generate import generate_dataset
    train_df, held_df = generate_dataset(
        n_train=N_TRAIN, n_held_out=N_HELD_OUT, seed=SEED
    )
    _done(t0)
    print(f"       train : {len(train_df)} states  "
          f"(A1 valid: {train_df['valid_point_particle'].mean():.1%},  "
          f"A2 valid: {train_df['valid_no_forces'].mean():.1%})")
    print(f"       held  : {len(held_df)} states  "
          f"(A1 valid: {held_df['valid_point_particle'].mean():.1%},  "
          f"A2 valid: {held_df['valid_no_forces'].mean():.1%})")

    # ------------------------------------------------------------------ 2/5
    t0 = _step(2, "Building axiom graph ...")
    from axiom_graph.graph import build_ideal_gas_graph
    graph = build_ideal_gas_graph()
    _done(t0)
    n_nodes = len(graph.nodes)
    n_edges = len(graph.edges)
    n_assump = len(graph.assumption_nodes())
    print(f"       {n_nodes} nodes  ({n_assump} assumption nodes)  {n_edges} derivation edges")

    # ------------------------------------------------------------------ 3/5
    t0 = _step(3, "Training validity predicates ...\n")
    from validity_predicates.train import train_all_predicates
    predicates = train_all_predicates(
        graph, train_df,
        verbose=True,
        hidden_dims=HIDDEN_DIMS,
        lr=LR,
        n_epochs=N_EPOCHS,
    )
    _done(t0)
    print(f"       Trained {len(predicates)} predicate(s): "
          + ", ".join(predicates.keys()))

    # ------------------------------------------------------------------ 4/5
    t0 = _step(4, "Running forward chain on held-out data ...")
    from reasoner.forward_chain import run_forward_chain
    from reasoner.provenance import compute_provenance
    result   = run_forward_chain(graph, held_df, threshold=THRESHOLD)
    prov_map = compute_provenance(graph)
    _done(t0)

    # ------------------------------------------------------------------ 5/5
    t0 = _step(5, "Computing metrics ...")
    from validity_predicates.evaluate import evaluate_predicates, print_report
    metrics = evaluate_predicates(result, held_df, prov_map=prov_map)
    _done(t0)

    # ------------------------------------------------------------------ Report
    _banner("BREAKDOWN DETECTION RESULTS")
    print_report(metrics, result, held_df, prov_map=prov_map)

    # ------------------------------------------------------------------ Optional plot
    if plot:
        print()
        print("  Generating decision boundary plot ...")
        import matplotlib
        matplotlib.use("Agg")
        from validity_predicates.train import plot_decision_boundary
        Path(PLOT_PATH).parent.mkdir(parents=True, exist_ok=True)
        plot_decision_boundary(
            predicates, train_df, held_df, save_path=PLOT_PATH
        )
        print(f"  Saved -> {PLOT_PATH}")

    # ------------------------------------------------------------------ Pass/fail
    a1_recall = metrics.get("A1_point_particles", {}).get("recall", 0.0)
    a2_recall = metrics.get("A2_no_forces", {}).get("recall", 0.0)
    d6_recall = metrics.get("D6_ideal_gas_law", {}).get("recall", 0.0)

    _banner("PASS / FAIL")
    target = 0.90
    results_line = [
        ("A1 recall", a1_recall, target),
        ("A2 recall", a2_recall, target),
        ("D6 flagging recall", d6_recall, target),
    ]
    all_pass = True
    for name, val, tgt in results_line:
        status = "PASS" if val >= tgt else "FAIL"
        if val < tgt:
            all_pass = False
        print(f"    {name:<22} = {val:.3f}   (target >= {tgt})  [{status}]")

    print()
    if all_pass:
        print("  All targets met.  Core MVP result achieved.")
    else:
        print("  One or more targets not met.  See report above.")
    print()


if __name__ == "__main__":
    plot_flag = "--plot" in sys.argv
    main(plot=plot_flag)

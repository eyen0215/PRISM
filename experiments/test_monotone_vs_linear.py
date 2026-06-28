"""
Test MonotonePredicate vs ValidityPredicate (linear skip) on pipe flow A2.

A2 = fully_developed assumption.
Training target: log(x / L_entry), all positive (valid regime only).
L_entry = 0.06 * Re * D  (not given to the model — must be discovered).

Success criterion from ARCHITECTURE_V4.md:
  MonotonePredicate AUROC within 0.02 of linear skip AUROC.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from validity_predicates.predicate import ValidityPredicate
from validity_predicates.monotone_predicate import MonotonePredicate
from validity_predicates.train_monotone import train_monotone_predicate

DATA_DIR = Path("data/pipe_flow")
SAVE_DIR  = Path("validity_predicates/saved")

# A2 config
N_FEATURES    = 5
FEATURE_COLS  = ["x", "v", "D", "rho", "mu"]
LOG_COLS      = (0, 1, 2, 3, 4)
SIGNS         = [+1, -1, -1, -1, +1]
# x: more downstream → safer (+1)
# v: higher velocity → higher Re → more dangerous (-1)
# D: higher diameter → higher Re → more dangerous (-1)
# rho: higher density → higher Re → more dangerous (-1)
# mu: higher viscosity → lower Re → safer (+1)

THRESHOLD = 0.5   # predicate fires when score < THRESHOLD


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    d_tr  = np.load(DATA_DIR / "train_A2.npz")
    d_br  = np.load(DATA_DIR / "test_scenario_A2break.npz")
    d_gr  = np.load(DATA_DIR / "grid_A2_boundary.npz")

    feats_train   = d_tr["features"].astype(np.float32)          # (5000, 5)
    log_criterion = d_tr["log_criterion"].astype(np.float32)     # (5000,)
    break_feats   = d_br["A2_features"].astype(np.float32)       # (1000, 5)
    grid_feats    = d_gr["A2_features"].astype(np.float32)       # (2500, 5)
    x_grid        = d_gr["x_grid"]                               # (50,)
    v_grid        = d_gr["v_grid"]                               # (50,)
    true_label    = d_gr["true_label"]                           # (2500,)

    # Same 1000-sample valid holdout used in train_pipe_predicates.py
    valid_hold = feats_train[:1000]

    return feats_train, log_criterion, valid_hold, break_feats, grid_feats, x_grid, v_grid, true_label


# ---------------------------------------------------------------------------
# Load existing linear-skip model
# ---------------------------------------------------------------------------

def load_linear_model() -> ValidityPredicate:
    ck = torch.load(SAVE_DIR / "pipe_A2.pt", map_location="cpu", weights_only=False)
    m  = ValidityPredicate(n_features=N_FEATURES, log_transform_cols=LOG_COLS)
    m.load_state_dict(ck["state_dict"])
    m.feat_mean.copy_(torch.tensor(ck["feat_mean"]))
    m.feat_std.copy_(torch.tensor(ck["feat_std"]))
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_metrics(valid_feats, break_feats, model) -> dict:
    n_v  = len(valid_feats)
    n_b  = len(break_feats)
    y_true    = np.concatenate([np.ones(n_v), np.zeros(n_b)])
    all_feats = np.vstack([valid_feats, break_feats])

    scores = model.predict(all_feats)
    auroc  = roc_auc_score(y_true, scores)

    valid_scores = scores[:n_v]
    break_scores = scores[n_v:]

    fire_rate = float((break_scores < THRESHOLD).mean())   # predicate fires on break
    fpr       = float((valid_scores < THRESHOLD).mean())   # false alarms on valid

    # Trivial baseline: single best column in the correct monotone direction
    # For A2, x (col 0) is the natural candidate — more downstream → safer
    col_vals = all_feats[:, 0]
    raw_auroc = roc_auc_score(y_true, col_vals)
    trivial_auroc = max(raw_auroc, 1.0 - raw_auroc)

    return {
        "auroc":        auroc,
        "trivial_auroc": trivial_auroc,
        "gap":          auroc - trivial_auroc,
        "fire_rate":    fire_rate,
        "fpr":          fpr,
    }


# ---------------------------------------------------------------------------
# Monotonicity check
# ---------------------------------------------------------------------------

def check_monotonicity(model: MonotonePredicate) -> bool:
    """Sweep x and v individually; verify scores move in the expected direction."""
    N = 30

    # Fixed values: v=2.0, D=0.01, rho=1.2, mu=1.81e-5
    base_v, base_D, base_rho, base_mu = 2.0, 0.01, 1.2, 1.81e-5

    # (a) Vary x from 0.1 to 5.0 at fixed v — score should INCREASE (sign[0]=+1)
    x_vals = np.linspace(0.1, 5.0, N).astype(np.float32)
    feats_x = np.column_stack([
        x_vals,
        np.full(N, base_v, np.float32),
        np.full(N, base_D, np.float32),
        np.full(N, base_rho, np.float32),
        np.full(N, base_mu, np.float32),
    ])
    scores_x = model.predict(feats_x)
    mono_x   = all(scores_x[i] <= scores_x[i + 1] for i in range(N - 1))

    # (b) Vary v from 0.1 to 10.0 at fixed x=1.0 — score should DECREASE (sign[1]=-1)
    v_vals = np.linspace(0.1, 10.0, N).astype(np.float32)
    feats_v = np.column_stack([
        np.full(N, 1.0, np.float32),
        v_vals,
        np.full(N, base_D, np.float32),
        np.full(N, base_rho, np.float32),
        np.full(N, base_mu, np.float32),
    ])
    scores_v = model.predict(feats_v)
    mono_v   = all(scores_v[i] >= scores_v[i + 1] for i in range(N - 1))

    passed = mono_x and mono_v

    print(f"\nMonotonicity on grid:")
    print(f"  x sweep  (sign=+1, should increase): "
          f"{'OK' if mono_x else 'FAIL'}  "
          f"{scores_x[0]:.4f} -> {scores_x[-1]:.4f}")
    print(f"  v sweep  (sign=-1, should decrease): "
          f"{'OK' if mono_v else 'FAIL'}  "
          f"{scores_v[0]:.4f} -> {scores_v[-1]:.4f}")
    print(f"Monotonicity on grid: {'PASS' if passed else 'FAIL'}")

    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Pipe flow A2: MonotonePredicate vs LinearSkip (ValidityPredicate)")
    print("=" * 65)

    torch.manual_seed(42)
    np.random.seed(42)

    feats_train, log_criterion, valid_hold, break_feats, \
        grid_feats, x_grid, v_grid, true_label = load_data()

    print(f"\nData: {len(feats_train)} train, {len(valid_hold)} valid holdout, "
          f"{len(break_feats)} break")
    print(f"log_criterion: min={log_criterion.min():.3f}  "
          f"max={log_criterion.max():.3f}  mean={log_criterion.mean():.3f}")

    # -----------------------------------------------------------------------
    # Linear skip — load existing
    # -----------------------------------------------------------------------
    print("\n--- Linear skip (loaded from pipe_A2.pt) ---")
    linear_model = load_linear_model()
    w = linear_model.skip.weight.data.numpy().ravel()
    print(f"  skip weights [x,v,D,rho,mu]: {np.round(w, 3)}")

    # -----------------------------------------------------------------------
    # Monotone predicate — train from scratch
    # -----------------------------------------------------------------------
    print("\n--- MonotonePredicate training (600 epochs) ---")
    mono_model = MonotonePredicate(
        n_features=N_FEATURES,
        signs=SIGNS,
        log_transform_cols=LOG_COLS,
        feature_cols=FEATURE_COLS,
    )
    train_monotone_predicate(
        mono_model,
        feats_train,
        log_criterion,
        lr=1e-3,
        weight_decay_mono=0.1,
        weight_decay_mlp=1.0,
        epochs=600,
        batch_size=256,
        verbose=True,
    )

    # -----------------------------------------------------------------------
    # Evaluate both
    # -----------------------------------------------------------------------
    print("\nEvaluating on valid holdout + break set ...")
    lin_m  = compute_metrics(valid_hold, break_feats, linear_model)
    mono_m = compute_metrics(valid_hold, break_feats, mono_model)

    # -----------------------------------------------------------------------
    # Comparison table
    # -----------------------------------------------------------------------
    w = 14
    print("\n")
    print("=" * 65)
    print(f"{'Metric':<28} {'Linear skip':>{w}} {'Monotone net':>{w}}")
    print("-" * 65)
    rows = [
        ("AUROC",               "auroc"),
        ("Trivial baseline AUROC", "trivial_auroc"),
        ("Gap over trivial",    "gap"),
        ("Fire rate on A2break","fire_rate"),
        ("False positive rate", "fpr"),
    ]
    for label, key in rows:
        lv = lin_m[key]
        mv = mono_m[key]
        print(f"  {label:<26} {lv:>{w}.4f} {mv:>{w}.4f}")
    print("=" * 65)

    # -----------------------------------------------------------------------
    # Success criterion
    # -----------------------------------------------------------------------
    auroc_gap = abs(mono_m["auroc"] - lin_m["auroc"])
    print(f"\nSuccess criterion: |AUROC_mono - AUROC_linear| <= 0.02")
    print(f"  Delta = {auroc_gap:.4f}  →  {'PASS' if auroc_gap <= 0.02 else 'FAIL'}")

    # -----------------------------------------------------------------------
    # Monotonicity check
    # -----------------------------------------------------------------------
    check_monotonicity(mono_model)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SAVE_DIR / "monotone_pipe_A2.pt"
    torch.save({
        "state_dict":        mono_model.state_dict(),
        "n_features":        N_FEATURES,
        "signs":             SIGNS,
        "log_transform_cols": list(LOG_COLS),
        "feature_cols":      FEATURE_COLS,
        "feat_mean":         mono_model.feat_mean.numpy().tolist(),
        "feat_std":          mono_model.feat_std.numpy().tolist(),
    }, save_path)
    print(f"\nSaved MonotonePredicate → {save_path}")


if __name__ == "__main__":
    main()

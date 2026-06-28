"""
Head-to-head: ValidityPredicate (linear skip) vs MonotonePredicate on the
synthetic exponential breakdown boundary.  No log-transform on either feature.

Validity condition:  x * exp(-0.5*v) > 1.0
log_criterion     :  log(x) - 0.5*v

log_transform_cols=() for BOTH models — raw (x, v) features.

Why the linear skip fails:
  It learns a*x_norm + b*v_norm — a straight line in (x, v) space.
  The true boundary is x = exp(0.5*v) — an exponential curve that curves
  away from any straight line as v grows.  In the training range (v ≤ 4)
  a straight line approximates the curve passably; beyond v=4 it diverges.

Why the monotone network succeeds:
  With signs=[+1, -1], it is constrained to be increasing in x and
  decreasing in v — the correct monotonicity directions.  It can learn
  any monotone function, including the exponential shape.

Success criteria (ARCHITECTURE_V4.md):
  MonotonePredicate AUROC > 0.85
  LinearSkip        AUROC < 0.70
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from validity_predicates.predicate import ValidityPredicate
from validity_predicates.monotone_predicate import MonotonePredicate
from validity_predicates.train_monotone import train_monotone_predicate

DATA_DIR = Path("data/synthetic_nonlinear")
SAVE_DIR  = Path("validity_predicates/saved")
K         = 0.5
THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data():
    tr = np.load(DATA_DIR / "train_nonlinear.npz")
    te = np.load(DATA_DIR / "test_nonlinear.npz")
    return (
        tr["features"].astype(np.float32),
        tr["log_criterion"].astype(np.float32),
        te["features"].astype(np.float32),
        te["is_valid"],
    )


# ---------------------------------------------------------------------------
# Linear-skip training
# ---------------------------------------------------------------------------

def _log_transform(features: np.ndarray, log_cols: tuple) -> np.ndarray:
    x = features.copy().astype(np.float32)
    for c in log_cols:
        x[:, c] = np.log(np.clip(x[:, c], 1e-9, None))
    return x


def train_linear(features, targets, log_cols=(), epochs=300, label="linear") -> ValidityPredicate:
    torch.manual_seed(0)
    model = ValidityPredicate(n_features=2, log_transform_cols=log_cols)

    x_log     = _log_transform(features, log_cols)
    feat_mean = x_log.mean(axis=0).astype(np.float32)
    feat_std  = (x_log.std(axis=0) + 1e-8).astype(np.float32)
    model.set_normalization(feat_mean, feat_std)

    opt = torch.optim.Adam([
        {"params": model.skip.parameters(), "weight_decay": 0.0},
        {"params": model.mlp.parameters(),  "weight_decay": 5.0},
    ], lr=1e-3)
    loss_fn = nn.MSELoss()

    X_t = torch.from_numpy(features.astype(np.float32))
    y_t = torch.from_numpy(targets.astype(np.float32))
    n   = len(features)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i : i + 256]
            opt.zero_grad()
            loss = loss_fn(model(X_t[idx]), y_t[idx])
            loss.backward()
            opt.step()
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                tr_loss = loss_fn(model(X_t), y_t).item()
            print(f"  [{label}] epoch {epoch:3d}  train_mse={tr_loss:.5f}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_metrics(test_feats, is_valid, model) -> dict:
    scores        = model.predict(test_feats)
    y_true        = is_valid.astype(int)
    auroc         = roc_auc_score(y_true, scores)

    col_vals      = test_feats[:, 0]   # trivial: x alone
    raw_auroc     = roc_auc_score(y_true, col_vals)
    trivial_auroc = max(raw_auroc, 1.0 - raw_auroc)

    fire_rate = float((scores[~is_valid] < THRESHOLD).mean())
    fpr       = float((scores[ is_valid] < THRESHOLD).mean())

    return {
        "auroc":          auroc,
        "trivial_auroc":  trivial_auroc,
        "gap":            auroc - trivial_auroc,
        "fire_rate":      fire_rate,
        "fpr":            fpr,
    }


def ood_fire_rate(model, feats_ood) -> float:
    scores = model.predict(feats_ood)
    return float((scores < THRESHOLD).mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 68)
    print("Synthetic exponential boundary: x*exp(-0.5*v) > 1")
    print("log_criterion = log(x) - 0.5*v   |   log_transform_cols=()")
    print("Linear skip: straight line in (x,v)   |   Monotone: any curve")
    print("=" * 68)

    torch.manual_seed(42)
    np.random.seed(42)

    feats_train, log_crit_train, feats_test, is_valid_test = load_data()
    n_valid = is_valid_test.sum()
    print(f"\nTrain: {len(feats_train)}  (v in [0.1, 4.0], all valid)")
    print(f"Test:  {len(feats_test)} (valid={n_valid}, break={len(feats_test)-n_valid})")

    # -------------------------------------------------------------------
    # Linear skip — no log-transform; fits a*x_norm + b*v_norm
    # -------------------------------------------------------------------
    print("\n--- Linear skip (log_transform_cols=()) ---")
    lin = train_linear(
        feats_train, log_crit_train,
        log_cols=(), epochs=300, label="linear",
    )
    w_lin = lin.skip.weight.data.numpy().ravel()
    print(f"  skip weights [x, v]: {np.round(w_lin, 4)}")

    # -------------------------------------------------------------------
    # Monotone predicate — no log-transform; signs=[+1,-1]
    # -------------------------------------------------------------------
    print("\n--- MonotonePredicate (log_transform_cols=(), signs=[+1,-1]) ---")
    mono = MonotonePredicate(
        n_features=2,
        signs=[+1, -1],
        log_transform_cols=(),
        feature_cols=["x", "v"],
    )
    train_monotone_predicate(
        mono, feats_train, log_crit_train,
        lr=1e-3, weight_decay_mono=0.1, weight_decay_mlp=1.0,
        epochs=600, batch_size=256, verbose=True,
    )

    # -------------------------------------------------------------------
    # In-distribution evaluation
    # -------------------------------------------------------------------
    print("\nEvaluating on test set (v in [0.0, 5.0], straddles boundary) ...")
    m_lin  = compute_metrics(feats_test, is_valid_test, lin)
    m_mono = compute_metrics(feats_test, is_valid_test, mono)

    # -------------------------------------------------------------------
    # Comparison table
    # -------------------------------------------------------------------
    W = 16
    print("\n")
    print("=" * 68)
    print(f"{'Metric':<30} {'Linear skip':>{W}} {'Monotone net':>{W}}")
    print("-" * 68)
    rows = [
        ("AUROC",                "auroc"),
        ("Trivial (x alone)",    "trivial_auroc"),
        ("Gap over trivial",     "gap"),
        ("Fire rate on invalid", "fire_rate"),
        ("False positive rate",  "fpr"),
    ]
    for label, key in rows:
        lv = m_lin[key]
        mv = m_mono[key]
        print(f"  {label:<28} {lv:>{W}.4f} {mv:>{W}.4f}")
    print("=" * 68)

    # -------------------------------------------------------------------
    # Success criteria
    # -------------------------------------------------------------------
    mono_pass = m_mono["auroc"] > 0.85
    lin_pass  = m_lin["auroc"]  < 0.70
    print("\nSuccess criteria (ARCHITECTURE_V4.md):")
    print(f"  MonotonePredicate AUROC > 0.85: {m_mono['auroc']:.4f}  "
          f"→  {'PASS' if mono_pass else 'FAIL'}")
    print(f"  LinearSkip AUROC < 0.70       : {m_lin['auroc']:.4f}  "
          f"→  {'PASS' if lin_pass else 'FAIL'}")

    # -------------------------------------------------------------------
    # Skip weight inspection
    # -------------------------------------------------------------------
    print("\nLinear skip weight inspection:")
    print(f"  [x, v]  skip weights: {w_lin[0]:+.4f}  {w_lin[1]:+.4f}")
    print(f"  True criterion in raw space: log(x) - 0.5*v — NOT linear in (x, v).")
    print(f"  Skip forced to fit a straight-line approximation to an exponential curve.")

    # -------------------------------------------------------------------
    # OOD extrapolation test
    # v in [7.0, 10.0] — well outside training range of v in [0.1, 4.0]
    # True boundary at v=7: x = exp(3.5) ≈ 33.1
    # True boundary at v=10: x = exp(5.0) ≈ 148.4
    # Sample x in [1, 50] — all below the boundary, so all truly invalid.
    # -------------------------------------------------------------------
    print("\n" + "-" * 68)
    print("OOD EXTRAPOLATION TEST")
    print("  Training range: v in [0.1, 4.0]")
    print("  OOD range:      v in [7.0, 10.0]")
    print("  True boundary at v=7:  x = exp(3.5) ≈ 33.1")
    print("  True boundary at v=10: x = exp(5.0) ≈ 148.4")
    print("  OOD samples: x in [1, 50], v in [7, 10]  — ALL truly invalid")
    print("-" * 68)

    rng = np.random.default_rng(99)
    n_ood = 200
    v_ood = rng.uniform(7.0, 10.0, n_ood).astype(np.float32)
    # Cap x at 30: minimum boundary in this v range is exp(0.5*7)=33.1,
    # so x in [1, 30] guarantees all samples are below the boundary.
    x_ood = rng.uniform(1.0, 30.0, n_ood).astype(np.float32)
    feats_ood = np.column_stack([x_ood, v_ood])

    # Verify all are truly invalid
    lc_ood = np.log(x_ood) - K * v_ood
    n_truly_invalid = (lc_ood < 0).sum()
    print(f"\n  Samples generated: {n_ood}")
    print(f"  Truly invalid (lc < 0): {n_truly_invalid}/{n_ood}")
    assert n_truly_invalid == n_ood, f"OOD generation error: {n_ood - n_truly_invalid} valid samples slipped through"

    fr_lin  = ood_fire_rate(lin,  feats_ood)
    fr_mono = ood_fire_rate(mono, feats_ood)

    print(f"\n  Linear skip fire rate  (OOD): {fr_lin:.4f}")
    print(f"  Monotone network fire rate (OOD): {fr_mono:.4f}")

    advantage = fr_mono - fr_lin
    demonstrated = fr_mono > fr_lin + 0.05   # monotone notably better
    print(f"\n  Monotone advantage (Δfire_rate): {advantage:+.4f}")
    print(f"OOD extrapolation — monotone advantage: "
          f"{'DEMONSTRATED' if demonstrated else 'NOT SHOWN'}")

    # -------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    torch.save({
        "state_dict":  lin.state_dict(),
        "n_features":  2,
        "log_cols":    (),
        "feat_mean":   lin.feat_mean.numpy().tolist(),
        "feat_std":    lin.feat_std.numpy().tolist(),
    }, SAVE_DIR / "linear_nonlinear.pt")

    torch.save({
        "state_dict":         mono.state_dict(),
        "n_features":         2,
        "signs":              [+1, -1],
        "log_transform_cols": (),
        "feature_cols":       ["x", "v"],
        "feat_mean":          mono.feat_mean.numpy().tolist(),
        "feat_std":           mono.feat_std.numpy().tolist(),
    }, SAVE_DIR / "monotone_nonlinear.pt")

    print(f"\nSaved → {SAVE_DIR / 'linear_nonlinear.pt'}")
    print(f"Saved → {SAVE_DIR / 'monotone_nonlinear.pt'}")


if __name__ == "__main__":
    main()

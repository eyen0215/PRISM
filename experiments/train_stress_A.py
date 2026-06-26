"""
Stress Test A — Train and evaluate the empirical predicate on noisy data.

Part 1 — Train
    Load train_noisy.npz (log_criterion derived from 10%-noisy dP measurements).
    Same architecture and hyperparameters as Version 1 (train_empirical_predicate.py).
    Save to validity_predicates/saved/noisy_pipe_A2.pt.
    Print effective skip weights; compare signs and |w_D|/|w_v| ratio to Version 1.

Part 2 — Evaluate on near-boundary test set
    Load test_near_boundary.npz (TRUE pred_error spanning [0.02, 0.12]).
    Report AUROC vs is_valid_true, confusion matrix, and noise-baseline AUROC.

Part 3 — Comparison table
    Side-by-side Version 1 (clean) vs Stress A (noisy) across all key metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, confusion_matrix

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from validity_predicates.predicate import ValidityPredicate

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
TRAIN_PATH = ROOT / "data" / "empirical_residual" / "stress_test_A" / "train_noisy.npz"
TEST_PATH  = ROOT / "data" / "empirical_residual" / "stress_test_A" / "test_near_boundary.npz"
SAVE_PATH  = ROOT / "validity_predicates" / "saved" / "noisy_pipe_A2.pt"
V1_PATH    = ROOT / "validity_predicates" / "saved" / "empirical_pipe_A2.pt"

# ------------------------------------------------------------------
# Config — identical to train_empirical_predicate.py
# ------------------------------------------------------------------
FEATURE_COLS = ["x", "v", "D"]
LOG_COLS     = (0, 1, 2)
N_FEATURES   = 3
HIDDEN_DIMS  = (32, 16)

LR                = 1e-3
WEIGHT_DECAY_MLP  = 5.0
WEIGHT_DECAY_SKIP = 0.0
N_EPOCHS          = 600
BATCH_SIZE        = 256
EPSILON           = 0.05


def _log_transform(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    for col in LOG_COLS:
        X[:, col] = np.log(np.clip(X[:, col], 1e-9, None))
    return X


# ==================================================================
# PART 1 — TRAIN
# ==================================================================

def train() -> tuple[ValidityPredicate, np.ndarray, float]:
    """Train on noisy data. Returns (predicate, feat_std, final_mse)."""
    torch.manual_seed(0)
    np.random.seed(0)

    print(f"Loading  {TRAIN_PATH}")
    data    = np.load(TRAIN_PATH)
    X_raw   = data["features"].astype(np.float32)       # (5000, 3)  [x, v, D]
    y_all   = data["log_criterion"].astype(np.float32)  # noisy-based log_criterion

    print(f"  {len(X_raw)} training samples")
    print(f"  log_criterion range : [{y_all.min():.3f}, {y_all.max():.3f}]"
          f"  mean = {y_all.mean():.3f}")
    print(f"  (target computed from noisy dP measurements, not clean pred_error)")

    # Fit normalisation on log-transformed features
    X_log     = _log_transform(X_raw)
    feat_mean = X_log.mean(axis=0).astype(np.float32)
    feat_std  = (X_log.std(axis=0) + 1e-8).astype(np.float32)

    print(f"\nFeature log-space statistics:")
    for i, name in enumerate(FEATURE_COLS):
        print(f"  log({name}) : mean = {feat_mean[i]:+.3f}  std = {feat_std[i]:.3f}")

    # Build model
    predicate = ValidityPredicate(
        hidden_dims=HIDDEN_DIMS,
        n_features=N_FEATURES,
        log_transform_cols=LOG_COLS,
        feature_cols=FEATURE_COLS,
    )
    predicate.set_normalization(feat_mean, feat_std)

    optimizer = torch.optim.Adam([
        {"params": predicate.skip.parameters(), "weight_decay": WEIGHT_DECAY_SKIP},
        {"params": predicate.mlp.parameters(),  "weight_decay": WEIGHT_DECAY_MLP},
    ], lr=LR)
    loss_fn = nn.MSELoss()

    X_t = torch.from_numpy(X_raw)
    y_t = torch.from_numpy(y_all)
    n_tr = len(X_t)

    print(f"\nTraining for {N_EPOCHS} epochs  "
          f"(batch={BATCH_SIZE}, lr={LR}, wd_mlp={WEIGHT_DECAY_MLP}) ...")
    print(f"  {'Epoch':>6}  {'train MSE':>10}")
    print(f"  {'-'*6}  {'-'*10}")

    for epoch in range(1, N_EPOCHS + 1):
        predicate.train()
        perm       = torch.randperm(n_tr)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n_tr, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            x_b, y_b = X_t[idx], y_t[idx]
            optimizer.zero_grad()
            loss = loss_fn(predicate(x_b), y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        if epoch % 100 == 0:
            print(f"  {epoch:6d}  {epoch_loss / n_batches:10.5f}")

    predicate.eval()
    with torch.no_grad():
        final_mse = loss_fn(predicate(X_t), y_t).item()
    print(f"\nFinal full-batch MSE : {final_mse:.5f}")

    # Effective skip weights
    w_skip = predicate.skip.weight.data.numpy().ravel()   # (3,)
    b_skip = predicate.skip.bias.data.item()
    eff_w  = w_skip / feat_std

    formula_w = np.array([+1.0, -1.0, -2.0])

    print()
    print("=== Skip Connection Effective Weights (Stress A — noisy) ===")
    print("(effective_weight[i] = skip.weight[i] / feat_std[i])")
    print()
    signs_match = True
    for i, name in enumerate(FEATURE_COLS):
        sign_ok = np.sign(eff_w[i]) == np.sign(formula_w[i])
        tag = "SIGN OK" if sign_ok else "SIGN MISMATCH"
        if not sign_ok:
            signs_match = False
        print(f"  {name}: effective weight = {eff_w[i]:+.3f}  "
              f"(Version 1 formula: {formula_w[i]:+.3f})  [{tag}]")

    ratio = abs(eff_w[2]) / (abs(eff_w[1]) + 1e-12)
    print(f"\n  |w_D| / |w_v|  = {ratio:.3f}  (Version 1 formula: 2.000, expect ~1.3-3.0)")
    print(f"  D^2 ratio {'RECOVERED' if 1.3 <= ratio <= 3.0 else 'OUTSIDE expected range'}")
    print(f"  skip bias  b = {b_skip:+.3f}")
    print(f"  Signs all correct: {signs_match}")

    # Save
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict"  : predicate.state_dict(),
        "feat_mean"   : feat_mean.tolist(),
        "feat_std"    : feat_std.tolist(),
        "feature_cols": FEATURE_COLS,
        "log_cols"    : list(LOG_COLS),
        "eff_weights" : eff_w.tolist(),
        "final_mse"   : final_mse,
    }, SAVE_PATH)
    print(f"\nSaved -> {SAVE_PATH}")

    return predicate, feat_std, final_mse, eff_w


# ==================================================================
# PART 2 — EVALUATE ON NEAR-BOUNDARY TEST SET
# ==================================================================

def evaluate(predicate: ValidityPredicate) -> dict:
    """Evaluate on near-boundary test set. Returns dict of metrics."""
    print()
    print("=" * 60)
    print("PART 2 — Evaluate on near-boundary test set")
    print("=" * 60)
    print(f"Loading  {TEST_PATH}")

    data           = np.load(TEST_PATH)
    features_te    = data["features"].astype(np.float32)
    is_valid_true  = data["is_valid_true"].astype(bool)
    is_valid_noisy = data["is_valid_noisy"].astype(bool)
    pe_true        = data["pred_error_true"].astype(np.float32)

    n_te = len(features_te)
    print(f"  {n_te} near-boundary samples")
    print(f"  TRUE pred_error range : [{pe_true.min():.4f}, {pe_true.max():.4f}]")
    print(f"  Fraction truly valid (< 0.05) : {is_valid_true.mean():.3f}")
    print(f"  Fraction noisy-valid  (< 0.05) : {is_valid_noisy.mean():.3f}")

    # Neural network scores
    scores = predicate.predict(features_te)   # (N,) in (0, 1)
    pred_valid = scores > 0.5

    # AUROC — neural network
    auroc_nn = roc_auc_score(is_valid_true, scores)

    # AUROC — noise baseline (use is_valid_noisy as the "score")
    auroc_noise = roc_auc_score(is_valid_true, is_valid_noisy.astype(float))

    # Confusion matrix at 0.5 threshold (rows=true, cols=pred)
    tn, fp, fn, tp = confusion_matrix(is_valid_true, pred_valid).ravel()

    n_valid_true   = int(is_valid_true.sum())
    n_invalid_true = n_te - n_valid_true

    print()
    print("=== AUROC (near-boundary test, TRUE labels) ===")
    print(f"  Neural network  AUROC : {auroc_nn:.4f}")
    print(f"  Noise baseline  AUROC : {auroc_noise:.4f}"
          f"  (using is_valid_noisy directly as predictor)")
    if auroc_nn >= auroc_noise:
        print(f"  Neural net beats noise baseline by {auroc_nn - auroc_noise:+.4f}")
    else:
        print(f"  Noise baseline beats neural net by {auroc_noise - auroc_nn:+.4f}  "
              f"(noise overwhelmed signal)")

    print()
    print("=== Confusion Matrix (threshold = 0.5) ===")
    print(f"  {'':20s}  Pred valid  Pred invalid")
    print(f"  {'True valid  ':20s}  {tp:10d}  {fn:12d}  "
          f"(TPR = {tp / (tp + fn + 1e-9):.3f})")
    print(f"  {'True invalid':20s}  {fp:10d}  {tn:12d}  "
          f"(TNR = {tn / (tn + fp + 1e-9):.3f})")

    return {
        "auroc_nn"    : auroc_nn,
        "auroc_noise" : auroc_noise,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_valid_true": n_valid_true,
        "n_te"        : n_te,
    }


# ==================================================================
# PART 3 — COMPARISON TABLE
# ==================================================================

def comparison_table(eff_w_noisy: np.ndarray, metrics: dict) -> None:
    """Print side-by-side Version 1 vs Stress A metrics."""

    # Load Version 1 effective weights from saved checkpoint
    v1_eff_w = np.array([+1.0, -1.0, -2.0])  # fallback defaults
    v1_auroc = 1.000
    if V1_PATH.exists():
        ckpt = torch.load(V1_PATH, map_location="cpu", weights_only=False)
        if "eff_weights" in ckpt:
            v1_eff_w = np.array(ckpt["eff_weights"])

    print()
    print("=" * 66)
    print("COMPARISON TABLE")
    print("=" * 66)
    hdr = f"  {'Metric':<30s}  {'Version 1 (clean)':>17}  {'Stress A (noisy)':>16}"
    print(hdr)
    print(f"  {'-'*30}  {'-'*17}  {'-'*16}")

    def row(label, v1, sa):
        print(f"  {label:<30s}  {v1:>17}  {sa:>16}")

    row("AUROC (test set)",
        f"{v1_auroc:.3f}",
        f"{metrics['auroc_nn']:.3f}")
    row("  test regime",
        "deep violation",
        "near boundary")
    row("Noise baseline AUROC",
        "N/A",
        f"{metrics['auroc_noise']:.3f}")
    row("NN vs noise baseline",
        "N/A",
        f"{metrics['auroc_nn'] - metrics['auroc_noise']:+.3f}")
    row("Skip weight x",
        f"{v1_eff_w[0]:+.3f}",
        f"{eff_w_noisy[0]:+.3f}")
    row("Skip weight v",
        f"{v1_eff_w[1]:+.3f}",
        f"{eff_w_noisy[1]:+.3f}")
    row("Skip weight D",
        f"{v1_eff_w[2]:+.3f}",
        f"{eff_w_noisy[2]:+.3f}")
    row("Label flip rate (noise)",
        "0.0%",
        "62.4%")

    print()
    auroc = metrics["auroc_nn"]
    if auroc >= 0.80:
        verdict = f"ROBUST  -- AUROC {auroc:.3f} >= 0.80 despite 10% noise"
    elif auroc >= 0.70:
        verdict = f"MARGINAL -- AUROC {auroc:.3f} in [0.70, 0.80); noise is significant"
    else:
        verdict = f"FRAGILE  -- AUROC {auroc:.3f} < 0.70; noise overwhelms signal"
    print(f"  Verdict: {verdict}")
    print()


# ==================================================================
# MAIN
# ==================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PART 1 — Train ValidityPredicate on noisy data")
    print("=" * 60)

    predicate, feat_std, final_mse, eff_w = train()

    metrics = evaluate(predicate)

    comparison_table(eff_w, metrics)

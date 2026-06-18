"""Train linear elasticity validity predicates A1, A2, A5.

A5 re-centering rationale
--------------------------
The A5 training data (f in [0.1, 100] Hz) produces inertia ratios ~1e-12 to
1e-3, so log(0.01 / ratio) ranges from ~2 to ~16 with mean ≈ 14.9.  Training
on these targets forces the model to output ~14.9 in the valid regime, which
means sigmoid(logit) ≈ 1 everywhere in training and the skip connection cannot
learn a useful gradient toward the breakdown boundary.

Fix: subtract mean(log_criterion) from A5 targets before training, centering
them around 0.  At inference, add the shift back before thresholding so the
decision boundary is still at log_criterion = 0.  The stored shift is needed
by downstream evaluation code.

For A1 and A2 the raw means (1.175 and 5.249) are in range; no re-centering.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from validity_predicates.predicate import ValidityPredicate

DATA_DIR     = Path(__file__).parent.parent / "data" / "linear_elasticity"
SAVE_DIR     = Path(__file__).parent.parent / "validity_predicates" / "saved"

LR           = 1e-3
WD_SKIP      = 0.0
WD_MLP       = 5.0
EPOCHS       = 300
BATCH_SIZE   = 256
HOLDOUT_FRAC = 0.20


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def compute_norm_stats(
    X_raw: np.ndarray,
    log_cols: tuple,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean/std on the log-transformed version of X (matching forward())."""
    X = X_raw.copy().astype(np.float64)
    for col in log_cols:
        X[:, col] = np.log(np.clip(X[:, col], 1e-9, None))
    mean = X.mean(axis=0).astype(np.float32)
    std  = (X.std(axis=0) + 1e-8).astype(np.float32)
    return mean, std


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one(
    name: str,
    features_raw: np.ndarray,
    log_criterion_raw: np.ndarray,
    log_cols: tuple,
    feature_col_names: list,
    recenter: bool,
) -> tuple[ValidityPredicate, float]:
    """
    Train one ValidityPredicate.  Returns (predicate, shift) where shift is
    mean(log_criterion) for A5 and 0.0 for A1/A2.
    """
    N        = len(features_raw)
    n_hold   = int(N * HOLDOUT_FRAC)
    n_train  = N - n_hold
    n_feat   = features_raw.shape[1]

    X_train_raw = features_raw[:n_train].astype(np.float32)
    X_hold_raw  = features_raw[n_train:].astype(np.float32)
    lc_train    = log_criterion_raw[:n_train]
    lc_hold     = log_criterion_raw[n_train:]

    # Re-centering shift (computed on full loaded dataset)
    raw_mean = float(np.mean(log_criterion_raw))
    shift    = raw_mean if recenter else 0.0

    y_train = (lc_train - shift).astype(np.float32)

    print(f"\n--- {name} ---")
    print(f"  n_features={n_feat}, log_cols={log_cols}")
    print(f"  mean log_criterion (raw):      {raw_mean:.3f}")
    print(f"  mean log_criterion (adjusted): {float(np.mean(y_train)):.3f}")
    if raw_mean > 10.0 and not recenter:
        print(f"  WARNING: mean(log_criterion) > 10 -- calibration bias likely")

    # Normalization stats on log-transformed training features
    feat_mean, feat_std = compute_norm_stats(X_train_raw, log_cols)

    # Predicate
    predicate = ValidityPredicate(
        n_features=n_feat,
        log_transform_cols=log_cols,
        feature_cols=feature_col_names,
    )
    predicate.set_normalization(feat_mean, feat_std)

    # Optimizer: two param groups with different weight decay
    optimizer = torch.optim.Adam(
        [
            {"params": list(predicate.skip.parameters()), "weight_decay": WD_SKIP},
            {"params": list(predicate.mlp.parameters()),  "weight_decay": WD_MLP},
        ],
        lr=LR,
    )
    loss_fn = nn.MSELoss()

    X_t = torch.from_numpy(X_train_raw)
    y_t = torch.from_numpy(y_train)

    predicate.train()
    for epoch in range(EPOCHS):
        perm       = torch.randperm(n_train)
        total_loss = 0.0
        n_batches  = 0
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            xb, yb = X_t[idx], y_t[idx]
            optimizer.zero_grad()
            # Tensor path in __call__ → super().__call__ → forward() → raw logit
            logits = predicate(xb)
            loss   = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
        if (epoch + 1) % 100 == 0:
            avg = total_loss / n_batches
            print(f"  epoch {epoch+1:3d}/{EPOCHS}  loss={avg:.4f}")

    # Holdout evaluation
    predicate.eval()
    with torch.no_grad():
        X_hold_t       = torch.from_numpy(X_hold_raw)
        logits_hold    = predicate(X_hold_t).numpy()

    # Recover original-scale log_criterion estimate, then sigmoid
    corrected_logits = logits_hold + shift
    scores = 1.0 / (1.0 + np.exp(-corrected_logits))

    print(
        f"  holdout n={len(X_hold_raw)}: "
        f"score mean={np.mean(scores):.4f}  std={np.std(scores):.4f}"
    )

    return predicate, shift


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    summary: list[tuple] = []

    # --- A1: small_strain ---
    d_a1 = np.load(DATA_DIR / "train_A1.npz")
    pred_a1, _ = train_one(
        name="A1 (small_strain)",
        features_raw=d_a1["features"],
        log_criterion_raw=d_a1["log_criterion"],
        log_cols=(),
        feature_col_names=["eps_eq"],
        recenter=False,
    )
    path_a1 = SAVE_DIR / "le_A1.pt"
    torch.save(pred_a1.state_dict(), path_a1)
    print(f"  Saved -> {path_a1}")
    raw_a1 = float(np.mean(d_a1["log_criterion"]))
    summary.append(("A1", raw_a1, raw_a1, "le_A1.pt"))

    # --- A2: linearity ---
    d_a2 = np.load(DATA_DIR / "train_A2.npz")
    pred_a2, _ = train_one(
        name="A2 (linearity)",
        features_raw=d_a2["features"],
        log_criterion_raw=d_a2["log_criterion"],
        log_cols=(0, 1),
        feature_col_names=["eps_eq", "sigma_vm"],
        recenter=False,
    )
    path_a2 = SAVE_DIR / "le_A2.pt"
    torch.save(pred_a2.state_dict(), path_a2)
    print(f"  Saved -> {path_a2}")
    raw_a2 = float(np.mean(d_a2["log_criterion"]))
    summary.append(("A2", raw_a2, raw_a2, "le_A2.pt"))

    # --- A5: quasi_static (with re-centering) ---
    d_a5 = np.load(DATA_DIR / "train_A5.npz")
    pred_a5, shift_a5 = train_one(
        name="A5 (quasi_static)  [re-centered]",
        features_raw=d_a5["features"],
        log_criterion_raw=d_a5["log_criterion"],
        log_cols=(0,),
        feature_col_names=["frequency"],
        recenter=True,
    )
    pred_a5.log_criterion_shift = shift_a5
    path_a5 = SAVE_DIR / "le_A5.pt"
    torch.save({"model": pred_a5.state_dict(), "shift": shift_a5}, path_a5)
    print(f"  Saved -> {path_a5}")
    raw_a5 = float(np.mean(d_a5["log_criterion"]))
    summary.append(("A5", raw_a5, raw_a5 - shift_a5, "le_A5.pt"))

    # Summary table
    print("\n" + "=" * 70)
    print(f"  {'Predicate':<12} {'mean log_crit (raw)':>20} {'mean log_crit (adj)':>20}  Saved")
    print("  " + "-" * 66)
    for pred_name, raw_m, adj_m, fname in summary:
        print(f"  {pred_name:<12} {raw_m:>20.3f} {adj_m:>20.3f}  {fname}")
    print("=" * 70)

    # Verify all three files exist
    for p in [path_a1, path_a2, path_a5]:
        assert p.exists(), f"Expected saved file missing: {p}"
    print("All three predicates saved and verified.")


if __name__ == "__main__":
    main()

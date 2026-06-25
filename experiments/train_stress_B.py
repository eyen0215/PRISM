"""
Stress Test B — Train and evaluate the empirical predicate on the Colebrook-White system.

The Blasius/Colebrook-White prediction error is NOT log-linear in (Re, eps) space.
The validity boundary is a transcendental curve, not a hyperplane.

Part 1 — Train
    Load train_colebrook.npz (features=[Re, eps], log_criterion = log(0.05/pred_error)).
    ValidityPredicate with n_features=2, log_transform_cols=[0, 1].
    Same hyperparameters as Version 1.
    Save to validity_predicates/saved/colebrook_pred.pt.

    Print effective skip weights [Re, eps].
    Print MLP vs skip contribution magnitude on training data.
    The skip will find an approximate linear boundary; the MLP will correct the curve.

Part 2 — Evaluate
    AUROC on combined valid (train) + invalid (test) pool.
    Trivial baseline: threshold on Re alone (ignoring eps).
    Gap = model AUROC - trivial AUROC.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from validity_predicates.predicate import ValidityPredicate

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
TRAIN_PATH = ROOT / "data" / "empirical_residual" / "stress_test_B" / "train_colebrook.npz"
TEST_PATH  = ROOT / "data" / "empirical_residual" / "stress_test_B" / "test_colebrook.npz"
SAVE_PATH  = ROOT / "validity_predicates" / "saved" / "colebrook_pred.pt"

# ------------------------------------------------------------------
# Config — identical to train_empirical_predicate.py
# ------------------------------------------------------------------
FEATURE_COLS = ["Re", "eps"]
LOG_COLS     = (0, 1)
N_FEATURES   = 2
HIDDEN_DIMS  = (32, 16)

LR                = 1e-3
WEIGHT_DECAY_MLP  = 5.0
WEIGHT_DECAY_SKIP = 0.0
N_EPOCHS          = 600
BATCH_SIZE        = 256


def _log_transform(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    for col in LOG_COLS:
        X[:, col] = np.log(np.clip(X[:, col], 1e-9, None))
    return X


def _skip_mlp_contributions(predicate: ValidityPredicate, X_raw: np.ndarray):
    """Return per-sample skip and MLP outputs on the normalised features."""
    predicate.eval()
    with torch.no_grad():
        x = torch.from_numpy(X_raw.astype(np.float32))
        for col in predicate._log_transform_cols:
            x[..., col] = torch.log(x[..., col].clamp(min=1e-9))
        x_norm = (x - predicate.feat_mean) / (predicate.feat_std + 1e-8)
        skip_out = predicate.skip(x_norm).squeeze(-1).numpy()   # (N,)
        mlp_out  = predicate.mlp(x_norm).squeeze(-1).numpy()    # (N,)
    return skip_out, mlp_out


# ==================================================================
# PART 1 — TRAIN
# ==================================================================

def train() -> tuple[ValidityPredicate, np.ndarray, np.ndarray]:
    """Train on Colebrook data. Returns (predicate, feat_std, eff_w)."""
    torch.manual_seed(0)
    np.random.seed(0)

    print(f"Loading  {TRAIN_PATH}")
    data   = np.load(TRAIN_PATH)
    X_raw  = data["features"].astype(np.float32)       # (5000, 2) [Re, eps]
    y_all  = data["log_criterion"].astype(np.float32)  # log(0.05 / pred_error)

    print(f"  {len(X_raw)} training samples  (all valid: pred_error < 0.05)")
    print(f"  Re  range  : [{X_raw[:,0].min():.0f}, {X_raw[:,0].max():.0f}]")
    print(f"  eps range  : [{X_raw[:,1].min():.2e}, {X_raw[:,1].max():.2e}]")
    print(f"  log_criterion range : [{y_all.min():.3f}, {y_all.max():.3f}]"
          f"  mean = {y_all.mean():.3f}")

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

    # ------------------------------------------------------------------
    # Effective skip weights
    # ------------------------------------------------------------------
    w_skip = predicate.skip.weight.data.numpy().ravel()   # (2,)
    b_skip = predicate.skip.bias.data.item()
    eff_w  = w_skip / feat_std

    print()
    print("=== Skip Connection Effective Weights ===")
    print("(effective_weight[i] = skip.weight[i] / feat_std[i])")
    print()
    print("NOTE: Colebrook-White is NOT log-linear — integer weights are")
    print("      NOT expected. The skip finds an approximate linear boundary;")
    print("      the MLP corrects the nonlinear curve.")
    print()
    for i, name in enumerate(FEATURE_COLS):
        direction = "breakdown increases with" if eff_w[i] < 0 else "breakdown decreases with"
        print(f"  {name:5s}: effective weight = {eff_w[i]:+.4f}  "
              f"({'neg: ' + direction + ' ' + name if eff_w[i] < 0 else 'pos: valid increases with ' + name})")
    print(f"  bias: {b_skip:+.4f}")

    # ------------------------------------------------------------------
    # Skip vs MLP contribution on training data
    # ------------------------------------------------------------------
    skip_out, mlp_out = _skip_mlp_contributions(predicate, X_raw)

    rms_skip = float(np.sqrt(np.mean(skip_out ** 2)))
    rms_mlp  = float(np.sqrt(np.mean(mlp_out  ** 2)))
    ratio    = rms_mlp / (rms_skip + 1e-12)

    print()
    print("=== Skip vs MLP Contribution (training data) ===")
    print(f"  RMS skip output : {rms_skip:.4f}")
    print(f"  RMS MLP output  : {rms_mlp:.4f}")
    print(f"  MLP/skip ratio  : {ratio:.4f}")
    if ratio > 1.0:
        print(f"  -> MLP > skip: nonlinearity is significant; "
              f"skip alone is not enough to describe the boundary")
    elif ratio > 0.3:
        print(f"  -> MLP ~= skip: both contribute meaningfully")
    else:
        print(f"  -> MLP << skip: skip dominates; "
              f"boundary is approximately log-linear")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict"  : predicate.state_dict(),
        "feat_mean"   : feat_mean.tolist(),
        "feat_std"    : feat_std.tolist(),
        "feature_cols": FEATURE_COLS,
        "log_cols"    : list(LOG_COLS),
        "eff_weights" : eff_w.tolist(),
        "final_mse"   : final_mse,
        "rms_skip"    : rms_skip,
        "rms_mlp"     : rms_mlp,
    }, SAVE_PATH)
    print(f"\nSaved -> {SAVE_PATH}")

    return predicate, feat_std, eff_w


# ==================================================================
# PART 2 — EVALUATE
# ==================================================================

def evaluate(predicate: ValidityPredicate) -> dict:
    """AUROC on combined valid+invalid pool; gap over Re-only trivial baseline."""
    print()
    print("=" * 60)
    print("PART 2 — Evaluate")
    print("=" * 60)

    # Load both datasets and merge into a single pool for AUROC
    tr_data = np.load(TRAIN_PATH)
    te_data = np.load(TEST_PATH)

    X_tr = tr_data["features"].astype(np.float32)   # (5000, 2) — all valid
    X_te = te_data["features"].astype(np.float32)   # (1000, 2) — all invalid

    labels_tr = np.ones(len(X_tr),  dtype=bool)
    labels_te = np.zeros(len(X_te), dtype=bool)

    X_all    = np.concatenate([X_tr, X_te], axis=0)   # (6000, 2)
    y_all    = np.concatenate([labels_tr, labels_te])  # (6000,) bool

    print(f"  Combined pool : {len(y_all)} samples  "
          f"({y_all.sum()} valid + {(~y_all).sum()} breakdown)")
    print(f"  Re  range : [{X_all[:,0].min():.0f}, {X_all[:,0].max():.0f}]")
    print(f"  eps range : [{X_all[:,1].min():.2e}, {X_all[:,1].max():.2e}]")

    # Model scores
    scores = predicate.predict(X_all)   # sigmoid outputs in (0, 1)
    auroc_model = roc_auc_score(y_all, scores)

    # ------------------------------------------------------------------
    # Trivial baseline: score = -log(Re)
    # Reasoning: higher Re -> more likely breakdown (ignores eps)
    # Higher score -> predicts valid, so negate Re.
    # ------------------------------------------------------------------
    Re_all = X_all[:, 0]
    trivial_scores = -np.log(Re_all)           # valid region has lower Re
    auroc_trivial  = roc_auc_score(y_all, trivial_scores)
    gap = auroc_model - auroc_trivial

    print()
    print("=== AUROC Results ===")
    print(f"  Model AUROC         : {auroc_model:.4f}")
    print(f"  Trivial (Re only)   : {auroc_trivial:.4f}  (score = -log(Re), ignores eps)")
    print(f"  Gap (model - triv.) : {gap:+.4f}")

    # Diagnose polarity: AUROC < 0.5 means scores are systematically INVERTED
    # (invalid samples ranked higher than valid). Report both raw and flipped.
    if auroc_model < 0.5:
        auroc_flipped = 1.0 - auroc_model
        print(f"  POLARITY INVERTED — model assigns higher scores to INVALID samples.")
        print(f"  Flipped AUROC (1 - model)  : {auroc_flipped:.4f}  "
              f"(ranking ability if sign were corrected)")
        print(f"  Cause: skip weights have wrong sign — the model extrapolates in the")
        print(f"    wrong direction outside the training distribution.")
        print(f"    Within training (Re 4k-80k, eps 1e-7-1e-5) the gradient signal")
        print(f"    is very weak; invalid samples lie in a higher-(Re,eps) region that")
        print(f"    the model ranks as 'more valid' due to wrong-sign extrapolation.")
    elif gap > 0.05:
        print(f"  -> Model meaningfully beats trivial baseline "
              f"(eps information is being used)")
    elif gap > 0:
        print(f"  -> Model marginally beats trivial baseline")
    else:
        print(f"  -> Model does not beat trivial baseline")

    # Score distribution by class
    s_valid   = scores[y_all]
    s_invalid = scores[~y_all]
    print()
    print("Score distribution (sigmoid output):")
    print(f"  Valid   samples: mean={s_valid.mean():.4f}  "
          f"std={s_valid.std():.4f}  median={np.median(s_valid):.4f}")
    print(f"  Invalid samples: mean={s_invalid.mean():.4f}  "
          f"std={s_invalid.std():.4f}  median={np.median(s_invalid):.4f}")

    return {
        "auroc_model"  : auroc_model,
        "auroc_trivial": auroc_trivial,
        "gap"          : gap,
    }


# ==================================================================
# SUMMARY TABLE
# ==================================================================

def summary(eff_w: np.ndarray, predicate: ValidityPredicate,
            X_raw_train: np.ndarray, metrics: dict) -> None:

    skip_out, mlp_out = _skip_mlp_contributions(predicate, X_raw_train)
    rms_skip = float(np.sqrt(np.mean(skip_out ** 2)))
    rms_mlp  = float(np.sqrt(np.mean(mlp_out  ** 2)))

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  AUROC                      : {metrics['auroc_model']:.4f}")
    print(f"  Trivial baseline AUROC     : {metrics['auroc_trivial']:.4f}")
    print(f"  Gap                        : {metrics['gap']:+.4f}")
    print(f"  Skip weight Re             : {eff_w[0]:+.4f}")
    print(f"  Skip weight eps            : {eff_w[1]:+.4f}")
    print(f"  MLP/skip contribution ratio: {rms_mlp / (rms_skip + 1e-12):.4f}")
    print()
    print("  Key question — does the model discriminate well despite")
    print("  a non-log-linear boundary?")
    auroc_effective = max(metrics['auroc_model'], 1.0 - metrics['auroc_model'])
    if metrics['auroc_model'] < 0.5:
        print(f"  INVERTED — AUROC {metrics['auroc_model']:.3f}; "
              f"model ranks invalid HIGHER than valid.")
        print(f"  Ranking ability is {auroc_effective:.3f} (flipped), but sign is WRONG.")
        print(f"  Root cause: skip weights have wrong sign (+Re, +eps instead of -Re, -eps).")
        print(f"  The training range (Re 4k-80k, eps 1e-7-1e-5) is too far from the")
        print(f"  breakdown boundary; the gradient is too weak to set the correct sign.")
        print(f"  Fix: extend training range closer to boundary (e.g. Re up to 200k)")
        print(f"  so the model sees declining log_criterion as Re/eps increase.")
    elif auroc_effective >= 0.85:
        print(f"  YES — AUROC {metrics['auroc_model']:.3f} >= 0.85 despite transcendental boundary")
    elif auroc_effective >= 0.70:
        print(f"  PARTIALLY — AUROC {metrics['auroc_model']:.3f} in [0.70, 0.85); "
              f"MLP compensation visible")
    else:
        print(f"  NO — AUROC {metrics['auroc_model']:.3f} < 0.70; "
              f"transcendental boundary is too complex for this architecture")
    print()
    print("  Does MLP compensate for what skip can't capture?")
    ratio = rms_mlp / (rms_skip + 1e-12)
    if ratio > 1.0:
        print(f"  YES — MLP/skip = {ratio:.3f}; MLP is dominant, "
              f"nonlinearity drives the prediction")
    elif ratio > 0.3:
        print(f"  PARTIALLY — MLP/skip = {ratio:.3f}; both contribute; "
              f"skip provides linear direction, MLP refines the curve")
    else:
        print(f"  NO — MLP/skip = {ratio:.3f}; skip dominates; "
              f"log-linear approximation is sufficient in the training hull")


# ==================================================================
# MAIN
# ==================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PART 1 — Train ValidityPredicate on Colebrook-White data")
    print("=" * 60)

    predicate, feat_std, eff_w = train()

    # Reload training features for the summary
    X_raw_train = np.load(TRAIN_PATH)["features"].astype(np.float32)

    metrics = evaluate(predicate)

    summary(eff_w, predicate, X_raw_train, metrics)

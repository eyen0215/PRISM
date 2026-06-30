"""
Train three validity predictors on coupled Arrhenius data.

Predictor 1 -- sigma-only  : ValidityPredicate, features=[sigma]
Predictor 2 -- T-only      : ValidityPredicate, features=[T]
Predictor 3 -- coupled     : MonotonePredicate, features=[sigma, T], signs=[-1, +1]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from validity_predicates.predicate import ValidityPredicate
from validity_predicates.monotone_predicate import MonotonePredicate
from validity_predicates.train_monotone import train_monotone_predicate

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "coupled_arrhenius")
SAVED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "validity_predicates", "saved")
os.makedirs(SAVED_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Load training data
# ---------------------------------------------------------------------------

def load_train():
    d = np.load(os.path.join(DATA_DIR, "train_coupled.npz"))
    features      = d["features"].astype(np.float32)   # (N, 2): [sigma, T]
    log_criterion = d["log_criterion"].astype(np.float32)
    return features, log_criterion


# ---------------------------------------------------------------------------
# Training loop for ValidityPredicate (mini-batch, matches monotone style)
# ---------------------------------------------------------------------------

def train_validity_predicate(
    predicate: ValidityPredicate,
    features: np.ndarray,
    log_criterion: np.ndarray,
    lr: float = 1e-3,
    weight_decay_skip: float = 0.0,
    weight_decay_mlp: float = 5.0,
    epochs: int = 600,
    batch_size: int = 256,
    verbose: bool = True,
) -> ValidityPredicate:
    # Compute normalization (no log-transform for Arrhenius features)
    feat = features.astype(np.float32).copy()
    mean = feat.mean(axis=0)
    std  = feat.std(axis=0)
    predicate.set_normalization(mean, std)

    predicate.train()

    optimizer = torch.optim.Adam(
        [
            {"params": predicate.skip.parameters(), "weight_decay": weight_decay_skip},
            {"params": predicate.mlp.parameters(),  "weight_decay": weight_decay_mlp},
        ],
        lr=lr,
    )

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(log_criterion, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            # ValidityPredicate.__call__ with tensors routes to nn.Module.__call__ -> forward()
            preds = predicate.forward(xb)
            loss  = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(features)

        if verbose and epoch % 100 == 0:
            print("  epoch %4d/%d  loss=%.6f" % (epoch, epochs, epoch_loss))

    predicate.eval()
    return predicate


# ---------------------------------------------------------------------------
# Compute final MSE on the full training set
# ---------------------------------------------------------------------------

def final_mse(model, features: np.ndarray, log_criterion: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        X = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(log_criterion, dtype=torch.float32)
        preds = model.forward(X)
        return nn.functional.mse_loss(preds, y).item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    features, lc = load_train()
    sigma_col = features[:, 0:1]   # shape (N, 1)
    T_col     = features[:, 1:2]   # shape (N, 1)

    # Pearson correlations between individual features and log_criterion
    r_sigma = float(np.corrcoef(sigma_col[:, 0], lc)[0, 1])
    r_T     = float(np.corrcoef(T_col[:, 0],     lc)[0, 1])
    print("=== Training-data correlations with log_criterion ===")
    print("  Pearson r(sigma, log_crit) = %.4f" % r_sigma)
    print("  Pearson r(T,     log_crit) = %.4f" % r_T)
    if abs(r_sigma) < 0.3 and abs(r_T) < 0.3:
        print("  Both |r| < 0.3: single-variable predictors have weak signal.")
        print("  This is EXPECTED -- breakdown is jointly driven by sigma and T.")
    print()

    # -----------------------------------------------------------------------
    # Predictor 1: sigma-only
    # -----------------------------------------------------------------------
    print("--- Predictor 1: sigma-only ---")
    pred_sigma = ValidityPredicate(
        hidden_dims=(32, 16),
        n_features=1,
        log_transform_cols=[],
        feature_cols=["sigma"],
    )
    train_validity_predicate(
        pred_sigma, sigma_col, lc,
        lr=1e-3, weight_decay_skip=0.0, weight_decay_mlp=5.0,
        epochs=600, batch_size=256, verbose=True,
    )
    mse_sigma = final_mse(pred_sigma, sigma_col, lc)
    save_path_sigma = os.path.join(SAVED_DIR, "arrhenius_sigma_only.pt")
    torch.save(pred_sigma.state_dict(), save_path_sigma)
    print("  Final MSE = %.4f  -> saved\n" % mse_sigma)

    # -----------------------------------------------------------------------
    # Predictor 2: T-only
    # -----------------------------------------------------------------------
    print("--- Predictor 2: T-only ---")
    pred_T = ValidityPredicate(
        hidden_dims=(32, 16),
        n_features=1,
        log_transform_cols=[],
        feature_cols=["T"],
    )
    train_validity_predicate(
        pred_T, T_col, lc,
        lr=1e-3, weight_decay_skip=0.0, weight_decay_mlp=5.0,
        epochs=600, batch_size=256, verbose=True,
    )
    mse_T = final_mse(pred_T, T_col, lc)
    save_path_T = os.path.join(SAVED_DIR, "arrhenius_T_only.pt")
    torch.save(pred_T.state_dict(), save_path_T)
    print("  Final MSE = %.4f  -> saved\n" % mse_T)

    # -----------------------------------------------------------------------
    # Predictor 3: Coupled monotone [sigma, T]  (fixed training)
    #
    # Changes from first attempt:
    #   - weight_decay_mono=0.0  (monotonicity IS the structural regularizer;
    #     L2 on softplus weights caused the plateau/instability)
    #   - LR warmup: 1e-4 for first 100 epochs, then 1e-3 for epochs 101-600
    #   - hidden_dims_mono=(32, 16)  (more capacity for the 2D coupled boundary)
    # -----------------------------------------------------------------------
    print("--- Predictor 3: Coupled MonotonePredicate [sigma, T] (fixed) ---")
    pred_coupled = MonotonePredicate(
        n_features=2,
        signs=[-1, +1],
        log_transform_cols=(),
        feature_cols=["sigma", "T"],
        hidden_dims_mono=(32, 16),
        hidden_dims_mlp=(16, 8),
    )

    # Normalization (no log-transform)
    feat_np = features.astype(np.float32)
    pred_coupled.set_normalization(feat_np.mean(axis=0), feat_np.std(axis=0))
    pred_coupled.train()

    mono_params = list(pred_coupled.monotone.parameters())
    mlp_params  = list(pred_coupled.mlp.parameters())

    # Phase 1: warmup with lr=1e-4, no weight decay on mono
    opt_warm = torch.optim.Adam(
        [
            {"params": mono_params, "weight_decay": 0.0},
            {"params": mlp_params,  "weight_decay": 1.0},
        ],
        lr=1e-4,
    )
    # Phase 2: main training with lr=1e-3
    opt_main = torch.optim.Adam(
        [
            {"params": mono_params, "weight_decay": 0.0},
            {"params": mlp_params,  "weight_decay": 1.0},
        ],
        lr=1e-3,
    )

    X_c = torch.tensor(features, dtype=torch.float32)
    y_c = torch.tensor(lc,       dtype=torch.float32)
    loader_c = DataLoader(TensorDataset(X_c, y_c), batch_size=256, shuffle=True)
    loss_fn_c = nn.MSELoss()

    WARMUP_EPOCHS = 100
    TOTAL_EPOCHS  = 600
    prev_loss = None

    for epoch in range(1, TOTAL_EPOCHS + 1):
        opt = opt_warm if epoch <= WARMUP_EPOCHS else opt_main
        epoch_loss = 0.0
        for xb, yb in loader_c:
            opt.zero_grad()
            loss = loss_fn_c(pred_coupled.forward(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(features)

        if epoch % 100 == 0:
            phase = "warmup" if epoch <= WARMUP_EPOCHS else "main  "
            print("  epoch %4d/%d  [%s]  loss=%.6f" % (epoch, TOTAL_EPOCHS, phase, epoch_loss))
            if prev_loss is not None and epoch > 200 and epoch_loss > prev_loss + 1e-4:
                print("  WARNING: loss increased from %.6f to %.6f at epoch %d"
                      % (prev_loss, epoch_loss, epoch))
            prev_loss = epoch_loss

    pred_coupled.eval()
    mse_coupled = final_mse(pred_coupled, features, lc)
    save_path_coupled = os.path.join(SAVED_DIR, "arrhenius_coupled.pt")
    torch.save(pred_coupled.state_dict(), save_path_coupled)
    print("  Final MSE = %.4f  -> saved to %s" % (mse_coupled, save_path_coupled))
    if mse_coupled < 0.8:
        print("  PASS (MSE %.4f < 0.80 threshold)" % mse_coupled)
    else:
        print("  FAIL (MSE %.4f >= 0.80 threshold)" % mse_coupled)
    print()

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("=== Summary ===")
    print("%-18s | %-10s | %-14s | %s" % ("Predictor", "Features", "Final MSE loss", "Notes"))
    print("-" * 65)
    print("%-18s | %-10s | %-14.4f | %s" % ("sigma-only",    "[sigma]",    mse_sigma,  "baseline"))
    print("%-18s | %-10s | %-14.4f | %s" % ("T-only",        "[T]",        mse_T,      "baseline"))
    print("%-18s | %-10s | %-14.4f | %s" % ("Coupled (mono)", "[sigma, T]", mse_coupled,"key predictor"))
    print()
    print("Correlations:")
    print("  r(sigma, log_crit) = %.4f" % r_sigma)
    print("  r(T,     log_crit) = %.4f" % r_T)

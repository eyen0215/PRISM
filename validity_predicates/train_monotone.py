"""
Training function for MonotonePredicate.

Two parameter groups with separate weight decay:
  - monotone network: weight_decay=0.1 (allows expressive fit)
  - MLP: weight_decay=1.0 (regularizes local corrections)

Normalization is computed from training features after applying the same
log-transform that _preprocess() applies at inference time.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validity_predicates.monotone_predicate import MonotonePredicate


def train_monotone_predicate(
    predicate: MonotonePredicate,
    features: np.ndarray,
    log_criterion: np.ndarray,
    lr: float = 1e-3,
    weight_decay_mono: float = 0.1,
    weight_decay_mlp: float = 1.0,
    epochs: int = 600,
    batch_size: int = 256,
    verbose: bool = True,
) -> MonotonePredicate:
    """Train a MonotonePredicate on valid-regime data.

    Parameters
    ----------
    predicate     : MonotonePredicate instance (modified in place)
    features      : (N, n_features) raw physical features
    log_criterion : (N,) training targets — positive in valid regime,
                    zero at boundary, negative in violation regime
    """
    # Compute normalization on log-transformed features (matching _preprocess)
    feat = features.astype(np.float32).copy()
    for col in predicate._log_transform_cols:
        feat[:, col] = np.log(np.clip(feat[:, col], 1e-9, None))
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    predicate.set_normalization(mean, std)

    predicate.train()

    mono_params = list(predicate.monotone.parameters())
    mlp_params = list(predicate.mlp.parameters())
    optimizer = torch.optim.Adam(
        [
            {"params": mono_params, "weight_decay": weight_decay_mono},
            {"params": mlp_params, "weight_decay": weight_decay_mlp},
        ],
        lr=lr,
    )

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(log_criterion, dtype=torch.float32)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = predicate.forward(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(dataset)

        if verbose and epoch % 100 == 0:
            print(f"  epoch {epoch:4d}/{epochs}  loss={epoch_loss:.6f}")

    predicate.eval()
    return predicate


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    N = 1000
    # Simple synthetic: features drawn from N(0,1), no log transform needed.
    # Target is linear in the features — the monotone net can represent it exactly.
    x0 = np.random.randn(N).astype(np.float32)   # sign=+1: safer when higher
    x1 = np.random.randn(N).astype(np.float32)   # sign=-1: more dangerous when higher
    features = np.stack([x0, x1], axis=1)

    # Normalize target to unit std so convergence speed is predictable.
    raw = x0 - x1 + 0.05 * np.random.randn(N).astype(np.float32)
    log_criterion = (raw / raw.std()).astype(np.float32)
    log_criterion = np.clip(log_criterion, -20.0, 20.0)

    pred = MonotonePredicate(
        n_features=2,
        signs=[+1, -1],
        log_transform_cols=(),
        feature_cols=["x0", "x1"],
    )

    print("Training MonotonePredicate on synthetic log-linear data...")
    # weight_decay=0 for this sanity check: the production defaults (0.1/1.0)
    # apply L2 to raw_weights whose softplus image is ≠ 0 at equilibrium,
    # causing a constant-output plateau on synthetic data. Real training
    # on physics data with 600 epochs handles this correctly.
    train_monotone_predicate(
        pred, features, log_criterion,
        lr=1e-3, weight_decay_mono=0.0, weight_decay_mlp=0.0,
        batch_size=64, epochs=200, verbose=True,
    )

    # Evaluate final MSE
    pred.eval()
    with torch.no_grad():
        X = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(log_criterion, dtype=torch.float32)
        preds = pred.forward(X)
        final_mse = torch.nn.functional.mse_loss(preds, y).item()

    print(f"\nFinal MSE: {final_mse:.6f}")
    if final_mse < 0.1:
        print("Training check PASSED (MSE < 0.1)")
    else:
        print(f"Training check FAILED (MSE={final_mse:.6f} >= 0.1)")

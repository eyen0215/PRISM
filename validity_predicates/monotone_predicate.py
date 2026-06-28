"""
MonotonePredicate: skip-connection predicate where the skip is a monotone network.

Architecture
------------
Two paths summed to produce the output logit:

  Monotone path -- MonotoneNet with all-positive weights (enforced via softplus).
                   Input is sign-flipped so the network is always increasing in
                   each signed input. Correct extrapolation by construction:
                   the monotone constraint replaces the linearity assumption.

  MLP path      -- small ReLU network, same role as before (local nonlinearity
                   near training data). L2-regularised (weight_decay=1.0) so it
                   contributes less than MonotoneNet in extrapolation.

Sign vector convention
----------------------
  sign[i] = +1  if increasing feature i -> higher validity (safer)
  sign[i] = -1  if increasing feature i -> lower validity (more dangerous)

The network sees x_signed[i] = sign[i] * x_norm[i], so the monotone constraint
("output increases with every input") translates to the correct physics.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Tuple


class MonotoneNet(nn.Module):
    """Small network with positive weights -- monotone increasing in all inputs.

    Positivity is enforced by applying softplus to raw weight parameters in the
    forward pass; bias terms are unconstrained.
    """

    def __init__(self, n_features: int, hidden_dims: Tuple[int, ...] = (16, 8)) -> None:
        super().__init__()
        dims = [n_features, *list(hidden_dims), 1]
        self.raw_weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for i in range(len(dims) - 1):
            # Initialize negative so softplus(raw) ≈ exp(-3) ≈ 0.05 per weight.
            # Prevents layer-wise signal amplification at init (positive-weight networks
            # have no sign cancellation, so large initial weights → huge outputs).
            self.raw_weights.append(
                nn.Parameter(torch.randn(dims[i + 1], dims[i]) * 0.1 - 3.0)
            )
            self.biases.append(nn.Parameter(torch.zeros(dims[i + 1])))
        self.activation = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, (w_raw, b) in enumerate(zip(self.raw_weights, self.biases)):
            w_pos = F.softplus(w_raw)
            x = F.linear(x, w_pos, b)
            if i < len(self.raw_weights) - 1:
                x = self.activation(x)
        return x


class MonotonePredicate(nn.Module):
    """Skip-connection style predicate where the skip is a MonotoneNet.

    Parameters
    ----------
    n_features         : number of input features
    signs              : list of +1 or -1, one per feature
                         +1 = increasing feature means safer (higher validity)
                         -1 = increasing feature means more dangerous
    log_transform_cols : feature indices to log-transform before processing
    feature_cols       : feature names (informational)
    hidden_dims_mono   : hidden dims for monotone network (default: (16, 8))
    hidden_dims_mlp    : hidden dims for residual MLP (default: (16, 8))
    """

    def __init__(
        self,
        n_features: int,
        signs: Sequence[int],
        log_transform_cols: Sequence[int] = (),
        feature_cols: Sequence[str] | None = None,
        hidden_dims_mono: Tuple[int, ...] = (16, 8),
        hidden_dims_mlp: Tuple[int, ...] = (16, 8),
    ) -> None:
        super().__init__()
        assert len(signs) == n_features, "signs must have one entry per feature"
        self.n_features = n_features
        self.register_buffer("signs", torch.tensor(signs, dtype=torch.float32))
        self._log_transform_cols = tuple(log_transform_cols)
        self.feature_cols = list(feature_cols) if feature_cols is not None else [f"x{i}" for i in range(n_features)]

        self.monotone = MonotoneNet(n_features, hidden_dims_mono)

        mlp_dims = [n_features, *list(hidden_dims_mlp), 1]
        mlp_layers: list[nn.Module] = []
        for i in range(len(mlp_dims) - 1):
            mlp_layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i < len(mlp_dims) - 2:
                mlp_layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp_layers)

        self.register_buffer("feat_mean", torch.zeros(n_features))
        self.register_buffer("feat_std", torch.ones(n_features))

    def set_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Store feature normalisation statistics from the training split."""
        self.feat_mean.copy_(torch.tensor(mean, dtype=torch.float32))
        self.feat_std.copy_(torch.tensor(std, dtype=torch.float32))

    def _preprocess(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.clone().float()
        for col in self._log_transform_cols:
            x[..., col] = torch.log(x[..., col].clamp(min=1e-9))
        x_norm = (x - self.feat_mean) / (self.feat_std + 1e-8)
        x_signed = x_norm * self.signs
        return x_norm, x_signed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logit (unbounded). Positive -> valid, negative -> violated."""
        x_norm, x_signed = self._preprocess(x)
        mono_out = self.monotone(x_signed)
        mlp_out = self.mlp(x_norm)
        return (mono_out + mlp_out).squeeze(-1)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return validity scores in (0, 1) for a batch of feature vectors."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(features.astype(np.float32))
            logits = self.forward(x).numpy()
            return torch.sigmoid(torch.tensor(logits)).numpy()

    def __call__(self, features):
        if isinstance(features, np.ndarray):
            return self.predict(features)
        return super().__call__(features)


if __name__ == "__main__":
    torch.manual_seed(42)

    # signs=[+1, -1]: x0 safer when higher, x1 more dangerous when higher
    pred = MonotonePredicate(n_features=2, signs=[+1, -1])
    pred.eval()

    N = 20
    sweep = torch.linspace(-2.0, 2.0, N)

    # Fix x0=0, vary x1: scores should decrease (sign[1]=-1)
    x1_sweep = torch.zeros(N, 2)
    x1_sweep[:, 1] = sweep
    with torch.no_grad():
        scores_x1 = torch.sigmoid(pred.forward(x1_sweep)).numpy()

    monotone_x1 = all(scores_x1[i] >= scores_x1[i + 1] for i in range(N - 1))

    # Fix x1=0, vary x0: scores should increase (sign[0]=+1)
    x0_sweep = torch.zeros(N, 2)
    x0_sweep[:, 0] = sweep
    with torch.no_grad():
        scores_x0 = torch.sigmoid(pred.forward(x0_sweep)).numpy()

    monotone_x0 = all(scores_x0[i] <= scores_x0[i + 1] for i in range(N - 1))

    if monotone_x0 and monotone_x1:
        print("Monotonicity check PASSED")
    else:
        print("Monotonicity check FAILED")
        if not monotone_x0:
            diffs = [scores_x0[i + 1] - scores_x0[i] for i in range(N - 1)]
            violations = [(i, d) for i, d in enumerate(diffs) if d < 0]
            print(f"  x0 (sign=+1) violations at indices: {violations[:5]}")
        if not monotone_x1:
            diffs = [scores_x1[i + 1] - scores_x1[i] for i in range(N - 1)]
            violations = [(i, d) for i, d in enumerate(diffs) if d > 0]
            print(f"  x1 (sign=-1) violations at indices: {violations[:5]}")

    print(f"  x0 sweep scores (min={scores_x0.min():.4f}, max={scores_x0.max():.4f}): "
          f"{scores_x0[0]:.4f} -> {scores_x0[-1]:.4f}")
    print(f"  x1 sweep scores (min={scores_x1.min():.4f}, max={scores_x1.max():.4f}): "
          f"{scores_x1[0]:.4f} -> {scores_x1[-1]:.4f}")

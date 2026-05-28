"""
ValidityPredicate: a skip-connection MLP mapping physical features -> validity score in [0, 1].

Architecture
------------
Input features : domain-specific physical or engineered observables.
                 Optionally log-transformed inside forward() for features that
                 vary over orders of magnitude with log-linear validity criteria
                 (e.g. ideal gas: log P, log V, log T).
                 For domains with linearly-bounded criteria on engineered features
                 (e.g. Hooke's Law with stress_ratio, strain_energy_ratio, epsilon),
                 log_transform_cols=() skips the transform entirely.

Two computation paths are summed to produce the output logit:

  Skip path  -- one linear layer (n_features -> 1) with no hidden units.
                Guarantees that the dominant behaviour is LINEAR in the
                (optionally log-transformed) normalised features, ensuring
                correct extrapolation when test conditions fall far outside
                the training distribution.

  MLP path   -- two hidden ReLU layers, also mapping n_features -> 1.
                Learns nonlinear residual corrections within the training hull.
                Heavily L2-regularised during training (weight_decay=5.0) so it
                stays near zero on out-of-distribution inputs and cannot override
                the skip connection's extrapolation signal.

The output is an unbounded logit.  Sigmoid is applied in predict() to convert
to a score in (0, 1):
    score > 0.5  ->  assumption probably satisfied
    score < 0.5  ->  assumption probably violated  (predicate fires)
    score = 0.5  ->  exactly at the analytical validity threshold

Why the skip connection is critical (both domains)
---------------------------------------------------
All training data is analytically valid, so all regression targets are positive.
Without regularisation the MLP collapses to a large positive constant (~training
mean), overwhelming the skip even on strongly out-of-distribution inputs.
With weight_decay=5.0 on the MLP, the skip dominates outside the training hull
and extrapolates the learned linear trend into the held-out regime.

Why log_transform_cols differs by domain
-----------------------------------------
Ideal gas (P, V, T, n):  criteria are log-linear in log(V) and log(T), so
  log-transforming those columns makes the boundary a hyperplane in feature
  space that the skip can capture exactly.

Hooke's Law (stress_ratio, strain_energy_ratio, epsilon):  features are already
  normalised dimensionless ratios; the boundary is linear (not log-linear) in
  these features.  No log-transform is applied; the skip still extrapolates
  correctly because the linear approximation gives the right sign for
  out-of-distribution inputs.

Fits into the system: validity_predicates/train.py fits this model;
axiom_graph/nodes.py Node.attach_predicate() wires it to the graph;
reasoner/provenance.py propagates flags downstream.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Sequence, Tuple

# ---------------------------------------------------------------------------
# Ideal-gas domain defaults (kept for backward compatibility)
# ---------------------------------------------------------------------------

FEATURE_COLS = ["P", "V", "T", "n"]
N_FEATURES   = len(FEATURE_COLS)

# Columns to log-transform for ideal gas (P, V, T vary over orders of magnitude)
_LOG_COLS = (0, 1, 2)

# ---------------------------------------------------------------------------
# Hooke's Law domain constants
# ---------------------------------------------------------------------------

HOOKE_FEATURE_COLS = ["stress_ratio", "strain_energy_ratio", "epsilon"]
HOOKE_N_FEATURES   = len(HOOKE_FEATURE_COLS)
HOOKE_LOG_COLS     = ()   # no log-transform: features are already dimensionless ratios


class ValidityPredicate(nn.Module):
    """Skip-connection MLP validity predicate for one physical assumption.

    Parameters
    ----------
    hidden_dims        : sizes of MLP hidden layers (default: (32, 16))
    n_features         : number of input features (default: 4 for ideal gas)
    log_transform_cols : indices of columns to log-transform in forward()
                         (default: (0, 1, 2) for ideal gas P, V, T;
                          pass () to disable all log-transforms)
    """

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (32, 16),
        n_features: int = N_FEATURES,
        log_transform_cols: Sequence[int] = _LOG_COLS,
        feature_cols: Sequence[str] = FEATURE_COLS,
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self._log_transform_cols = tuple(log_transform_cols)
        self.feature_cols = list(feature_cols)

        # Residual linear skip -- guarantees linear extrapolation
        self.skip = nn.Linear(n_features, 1)

        # Nonlinear MLP path -- residual corrections within training hull
        mlp_dims = [n_features, *hidden_dims, 1]
        mlp_layers: list[nn.Module] = []
        for i in range(len(mlp_dims) - 1):
            mlp_layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i < len(mlp_dims) - 2:
                mlp_layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp_layers)

        # Normalisation buffers -- fitted on (optionally log-transformed) training features
        self.register_buffer("feat_mean", torch.zeros(n_features))
        self.register_buffer("feat_std",  torch.ones(n_features))

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Store feature normalisation statistics from the training split."""
        self.feat_mean.copy_(torch.tensor(mean, dtype=torch.float32))
        self.feat_std.copy_(torch.tensor(std,  dtype=torch.float32))

    # ------------------------------------------------------------------
    # Forward pass -- returns raw logit (unbounded)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw log-criterion logit for a batch of feature vectors.

        Positive logit -> score > 0.5 (assumption probably valid).
        Negative logit -> score < 0.5 (assumption probably violated).

        Parameters
        ----------
        x : shape (..., n_features), raw physical/engineered features

        Returns
        -------
        Tensor of shape (...,), unbounded real values.
        """
        x = x.clone().float()
        for col in self._log_transform_cols:
            x[..., col] = torch.log(x[..., col].clamp(min=1e-9))
        x_norm = (x - self.feat_mean) / (self.feat_std + 1e-8)
        return (self.skip(x_norm) + self.mlp(x_norm)).squeeze(-1)

    # ------------------------------------------------------------------
    # Numpy-facing inference helpers -- return scores in (0, 1)
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return validity scores in (0, 1) for a batch of feature vectors.

        Parameters
        ----------
        features : shape (N, n_features), dtype float32 or float64

        Returns
        -------
        np.ndarray of shape (N,), values in (0, 1).
        """
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(features.astype(np.float32))
            return torch.sigmoid(self.forward(x)).numpy()

    def predict_single(self, *feature_values: float) -> float:
        """Score a single state given as positional floats -- returns a float in (0, 1)."""
        arr = np.array([list(feature_values)], dtype=np.float32)
        return float(self.predict(arr)[0])

    def __call__(self, features):
        """Allow the predicate to be used directly as a Node.validity_predicate.

        Accepts a numpy array and returns sigmoid scores (not raw logits).
        1-D input of length n_features -> scalar float;
        2-D (N, n_features) -> ndarray of N floats.
        """
        if isinstance(features, np.ndarray):
            if features.ndim == 1 and features.shape[0] == self.n_features:
                return self.predict_single(*features.tolist())
            return self.predict(features)
        # Tensor input: let nn.Module.__call__ route to forward() (raw logit)
        return super().__call__(features)

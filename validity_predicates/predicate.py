"""
ValidityPredicate: a small MLP mapping (P, V, T, n) → validity score in [0, 1].

Architecture
------------
Input features : raw (P, V, T, n).  Inside forward() the model log-transforms
                 P, V, and T (all vary over orders of magnitude or have
                 log-linear relationships with the validity criteria).

Two computation paths are summed to produce the output logit:

  Skip path  — one linear layer (N_FEATURES → 1) with no hidden units.
               Guarantees that the dominant behaviour is LINEAR in the
               normalised log-features, ensuring correct extrapolation when
               test conditions fall far outside the training distribution.

  MLP path   — two hidden ReLU layers, also mapping N_FEATURES → 1.
               Learns nonlinear residual corrections within the training hull.

The output is an unbounded logit.  Sigmoid is applied in predict() to convert
to a score in (0, 1):
    score > 0.5  →  assumption probably satisfied
    score < 0.5  →  assumption probably violated  (predicate fires)
    score = 0.5  →  exactly at the analytical validity threshold

Why the skip connection is critical
------------------------------------
Both validity criteria are log-linear in log(V) (and log(T) for A2):

    A1 logit = log(V/n / VDW_B / threshold)  ≈  log(V) + const
    A2 logit = log(threshold / q)             ≈  log(V) + log(T) + const

A ReLU MLP without a skip connection can go dead outside the training hull,
producing a constant output regardless of input.  The skip connection captures
the linear trend and extrapolates it correctly: very small V (high-pressure
held-out states) → large negative logit → score << 0.5 → assumption flagged.

Fits into the system: validity_predicates/train.py fits this model;
axiom_graph/nodes.py Node.attach_predicate() wires it to the graph;
reasoner/provenance.py propagates flags downstream.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

# Column order expected by train.py make_features()
FEATURE_COLS = ["P", "V", "T", "n"]
N_FEATURES = len(FEATURE_COLS)

# Apply log transform to P, V, T before the linear layers.
# All three vary non-linearly relative to the validity criteria.
_LOG_COLS = (0, 1, 2)   # indices of P, V, T in FEATURE_COLS


class ValidityPredicate(nn.Module):
    """Skip-connection MLP validity predicate for one ideal-gas assumption.

    Parameters
    ----------
    hidden_dims : sizes of MLP hidden layers (default: 32 and 16 units)
    """

    def __init__(self, hidden_dims: Tuple[int, ...] = (32, 16)) -> None:
        super().__init__()

        # Residual linear skip — guarantees linear extrapolation
        self.skip = nn.Linear(N_FEATURES, 1)

        # Nonlinear MLP path — residual corrections within training hull
        mlp_dims = [N_FEATURES, *hidden_dims, 1]
        mlp_layers: list[nn.Module] = []
        for i in range(len(mlp_dims) - 1):
            mlp_layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i < len(mlp_dims) - 2:
                mlp_layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp_layers)

        # Normalisation buffers — fitted on log-transformed training features
        self.register_buffer("feat_mean", torch.zeros(N_FEATURES))
        self.register_buffer("feat_std",  torch.ones(N_FEATURES))

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Store feature normalisation statistics from the training split.

        `mean` and `std` must be computed on the LOG-TRANSFORMED feature
        matrix (log applied to P, V, T columns) to match what forward() does.
        """
        self.feat_mean.copy_(torch.tensor(mean, dtype=torch.float32))
        self.feat_std.copy_(torch.tensor(std,  dtype=torch.float32))

    # ------------------------------------------------------------------
    # Forward pass — returns raw logit (unbounded)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw log-criterion logit for a batch of (P, V, T, n) states.

        Positive logit → score > 0.5 (assumption probably valid).
        Negative logit → score < 0.5 (assumption probably violated).

        Parameters
        ----------
        x : shape (..., 4), raw physical state features
            (P in atm, V in litres, T in Kelvin, n in moles)

        Returns
        -------
        Tensor of shape (...,), unbounded real values.
        """
        x = x.clone().float()
        for col in _LOG_COLS:
            x[..., col] = torch.log(x[..., col].clamp(min=1e-9))
        x_norm = (x - self.feat_mean) / (self.feat_std + 1e-8)
        return (self.skip(x_norm) + self.mlp(x_norm)).squeeze(-1)

    # ------------------------------------------------------------------
    # Numpy-facing inference helpers — return scores in (0, 1)
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return validity scores in (0, 1) for a batch of raw (P, V, T, n) states.

        Parameters
        ----------
        features : shape (N, 4), dtype float32 or float64

        Returns
        -------
        np.ndarray of shape (N,), values in (0, 1).
        """
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(features.astype(np.float32))
            return torch.sigmoid(self.forward(x)).numpy()

    def predict_single(self, P: float, V: float, T: float, n: float) -> float:
        """Score a single (P, V, T, n) state — returns a float in (0, 1)."""
        return float(self.predict(np.array([[P, V, T, n]], dtype=np.float32))[0])

    def __call__(self, features) -> float | np.ndarray:
        """Allow the predicate to be used directly as a Node.validity_predicate.

        Accepts a numpy array and returns sigmoid scores (not raw logits).
        1-D input of length 4 → scalar float; 2-D (N, 4) → ndarray of N floats.
        """
        if isinstance(features, np.ndarray):
            if features.ndim == 1 and features.shape[0] == N_FEATURES:
                return self.predict_single(*features.tolist())
            return self.predict(features)
        # Tensor input: let nn.Module.__call__ route to forward() (raw logit)
        return super().__call__(features)

"""
Candidate feature library for Stage 2 symbolic regression (ideal gas, Pilot 1).

build_library(P, V, T, n) computes all candidate terms from the raw observables
and returns a dict mapping term name -> numpy array of computed values.

Terms (9 total):
  n/V, (n/V)^2, (n/V)^3      -- vdW-like candidates (expected to appear)
  n*T/V, T/V, 1/T, n/(V*T)   -- plausible distractors; n/(V*T) absorbs T-dependence of n/V coef
  n/T, V/n, P*n               -- more distractors

Removed from original 11-term library: P/T (r=0.9999 with n/V in training data),
P*V (near-constant ~nRT in training).

The library is intentionally NOT hand-picked to contain only the answer:
sparsity has to select the right subset from among the distractors.
"""

from __future__ import annotations

import numpy as np


def build_library(
    P: np.ndarray,
    V: np.ndarray,
    T: np.ndarray,
    n: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return a dict mapping term name to 1-D numpy array of computed values."""
    rho = n / V  # molar density -- used repeatedly
    return {
        "n/V":       rho,
        "(n/V)^2":   rho ** 2,
        "(n/V)^3":   rho ** 3,
        "n*T/V":     n * T / V,
        "T/V":       T / V,
        "1/T":       1.0 / T,
        "n/(V*T)":   rho / T,
        "n/T":       n / T,
        "V/n":       V / n,
        "P*n":       P * n,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    data = dict(np.load("data/ideal_gas_residual_train.npz"))
    lib = build_library(data["P"], data["V"], data["T"], data["n"])

    print(f"{'Term':<12}  shape")
    print("-" * 30)
    for name, arr in lib.items():
        print(f"{name:<12}  {arr.shape}")

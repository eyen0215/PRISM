"""
Sequential Thresholded Least Squares (STLSQ) for sparse symbolic regression.

Algorithm (SINDy-style):
  1. Fit y ≈ Θ @ c via ordinary least squares.
  2. Zero out any coefficient with |c_i| < threshold.
  3. Refit using only the surviving columns.
  4. Repeat until the active set stabilizes.

Usage:
  result = stlsq(Theta, y, term_names, threshold=0.01, max_iter=20)
  # result is a dict: {"term": coefficient}, containing only non-zero terms.
"""

from __future__ import annotations

import numpy as np


def stlsq(
    Theta: np.ndarray,
    y: np.ndarray,
    term_names: list[str],
    threshold: float = 0.01,
    max_iter: int = 20,
) -> dict[str, float]:
    """
    Fit y ≈ Theta @ c sparsely via sequential thresholded least squares.

    Parameters
    ----------
    Theta      : (N, M) feature matrix, one column per library term
    y          : (N,) target values
    term_names : length-M list of term name strings
    threshold  : coefficients with |c| < threshold are zeroed and dropped
    max_iter   : safety cap on iterations

    Returns
    -------
    dict mapping term_name -> coefficient for every non-zero term
    """
    Theta = np.asarray(Theta, dtype=float)
    y = np.asarray(y, dtype=float)
    n_terms = Theta.shape[1]
    active = np.ones(n_terms, dtype=bool)

    for _ in range(max_iter):
        c_full = np.zeros(n_terms)
        if active.sum() == 0:
            break
        c_full[active], _, _, _ = np.linalg.lstsq(Theta[:, active], y, rcond=None)

        new_active = active & (np.abs(c_full) >= threshold)
        if np.array_equal(new_active, active):
            break
        active = new_active

    return {term_names[i]: float(c_full[i]) for i in range(n_terms) if active[i]}


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N = 500
    x1 = rng.uniform(0.5, 3.0, N)
    x2 = rng.uniform(0.5, 3.0, N)

    y = 2.0 * x1 + 0.5 * x2**2 + rng.normal(0, 0.05, N)

    # Candidate library — includes the true terms plus distractors
    Theta = np.column_stack([x1, x2, x1**2, x2**2, x1 * x2, np.ones(N)])
    names = ["x1", "x2", "x1^2", "x2^2", "x1*x2", "const"]

    result = stlsq(Theta, y, names, threshold=0.1)

    print("True model:  y = 2.0*x1 + 0.5*x2^2  (+ noise sigma=0.05)")
    print(f"\nSTLSQ recovered ({len(result)} term(s)):")
    for name, coef in result.items():
        print(f"  {name:<8}  coef = {coef:.4f}")
    dropped = [n for n in names if n not in result]
    print(f"\nDropped (below threshold): {dropped}")

"""
Generate synthetic dataset with an EXPONENTIAL breakdown boundary.

Validity condition:  x * exp(-k * v) > threshold
  k = 0.5, threshold = 1.0

log_criterion = log(x) - k*v - log(threshold)
              = log(x) - 0.5*v

Features: [x, v]
  log_transform_cols = ()    <- NO log-transform on either feature
  signs = [+1, -1]           <- larger x safer, larger v more dangerous

Why the linear skip fails with log_transform_cols=():
  The linear skip learns a*x_norm + b*v_norm — a straight line in (x, v) space.
  The true boundary is x = exp(0.5*v) — an exponential curve.
  A straight line cannot match an exponential curve across the full feature range.

Why the monotone network succeeds:
  With signs=[+1, -1] it learns any monotone function of (x, -v),
  which includes the exponential curve x = exp(0.5*v).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

K = 0.5
THRESHOLD = 1.0
SEED = 42
OUT_DIR = Path(__file__).parent


def log_criterion(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.log(x) - K * v - np.log(THRESHOLD)


def generate_train(n_target: int = 5000, rng: np.random.Generator = None) -> dict:
    """Valid-regime samples: log_criterion > 0 everywhere."""
    if rng is None:
        rng = np.random.default_rng(SEED)

    xs, vs, lcs = [], [], []
    while len(xs) < n_target:
        # Oversample to account for rejection
        batch = max(n_target * 3, 10_000)
        log_x = rng.uniform(0.5, 3.0, batch)   # x in [exp(0.5), exp(3)] ≈ [1.6, 20]
        v     = rng.uniform(0.1, 4.0, batch)
        x     = np.exp(log_x)
        lc    = log_criterion(x, v)
        valid = lc > 0
        xs.append(x[valid])
        vs.append(v[valid])
        lcs.append(lc[valid])

    x_all  = np.concatenate(xs)[:n_target].astype(np.float32)
    v_all  = np.concatenate(vs)[:n_target].astype(np.float32)
    lc_all = np.clip(np.concatenate(lcs)[:n_target], -20.0, 20.0).astype(np.float32)

    return {
        "features":      np.column_stack([x_all, v_all]),
        "log_criterion": lc_all,
        "is_valid":      np.ones(n_target, dtype=bool),
    }


def generate_test(n: int = 1000, rng: np.random.Generator = None) -> dict:
    """Samples straddling the boundary: log_criterion spans approximately [-2, 2]."""
    if rng is None:
        rng = np.random.default_rng(SEED + 1)

    # Sample v freely, then pick x near the boundary x_boundary = exp(k*v)
    # with log-scale perturbation so log_criterion is roughly uniform in [-2, 2]
    v    = rng.uniform(0.0, 5.0, n).astype(np.float32)
    # x_boundary = exp(k*v); perturb log(x) by offset in [-2, 2]
    offset = rng.uniform(-2.0, 2.0, n).astype(np.float32)
    log_x  = K * v + offset          # log(x) = k*v + offset → lc = offset
    x      = np.exp(log_x).astype(np.float32)
    lc     = np.clip(log_criterion(x, v), -20.0, 20.0).astype(np.float32)

    return {
        "features":      np.column_stack([x, v]),
        "log_criterion": lc,
        "is_valid":      (lc > 0),
    }


def generate_grid(n_side: int = 50) -> dict:
    """Regular grid over (x, v) space for boundary visualisation.

    Linear spacing for x because we are no longer working in log-x space.
    The exponential boundary will curve visibly against the straight-line
    decision boundary learned by the linear skip.
    """
    x_vals = np.linspace(0.5, 25.0, n_side).astype(np.float32)
    v_vals = np.linspace(0.0, 6.0, n_side).astype(np.float32)

    xx, vv = np.meshgrid(x_vals, v_vals)   # (n_side, n_side)
    x_flat = xx.ravel().astype(np.float32)
    v_flat = vv.ravel().astype(np.float32)

    lc_flat    = log_criterion(x_flat, v_flat).astype(np.float32)
    true_label = (lc_flat > 0).astype(np.int32)

    # True boundary: x_boundary(v) = exp(k*v)
    true_boundary_x = np.exp(K * v_vals).astype(np.float32)

    return {
        "features":        np.column_stack([x_flat, v_flat]),
        "log_criterion":   np.clip(lc_flat, -20.0, 20.0),
        "true_label":      true_label,
        "x_grid":          x_vals,
        "v_grid":          v_vals,
        "true_boundary_x": true_boundary_x,   # x_boundary for each v in v_grid
    }


def main() -> None:
    rng = np.random.default_rng(SEED)

    print("Exponential breakdown boundary: x * exp(-0.5*v) > 1.0")
    print("  log_criterion = log(x) - 0.5*v")
    print("  log_transform_cols=() — no transform on either feature")
    print("  Linear skip fits a*x + b*v (straight line); boundary is x=exp(0.5*v) (curve).\n")

    print("True boundary x = exp(0.5*v):")
    for v_val in [0, 1, 2, 3, 4, 5, 6]:
        x_bnd = np.exp(K * v_val)
        print(f"  v={v_val}  x_boundary={x_bnd:.3f}")

    print()

    # Train
    train = generate_train(5000, rng)
    path_tr = OUT_DIR / "train_nonlinear.npz"
    np.savez(path_tr, **train)
    lc = train["log_criterion"]
    print(f"train_nonlinear.npz: {len(lc)} samples  "
          f"log_criterion min={lc.min():.4f}  max={lc.max():.4f}  mean={lc.mean():.4f}")
    print(f"  all valid: {train['is_valid'].all()}")

    # Test
    test = generate_test(1000, rng)
    path_te = OUT_DIR / "test_nonlinear.npz"
    np.savez(path_te, **test)
    lc = test["log_criterion"]
    n_valid = test["is_valid"].sum()
    print(f"\ntest_nonlinear.npz: {len(lc)} samples  "
          f"log_criterion min={lc.min():.4f}  max={lc.max():.4f}  mean={lc.mean():.4f}")
    print(f"  valid={n_valid}  break={len(lc)-n_valid}")

    # Grid
    grid = generate_grid(50)
    path_gr = OUT_DIR / "grid_nonlinear.npz"
    np.savez(path_gr, **grid)
    lc = grid["log_criterion"]
    print(f"\ngrid_nonlinear.npz: {len(lc)} points  "
          f"x in [{grid['x_grid'].min():.2f}, {grid['x_grid'].max():.2f}]  "
          f"v in [{grid['v_grid'].min():.2f}, {grid['v_grid'].max():.2f}]")
    print(f"  valid fraction: {grid['true_label'].mean():.3f}")

    print(f"\nSaved:\n  {path_tr}\n  {path_te}\n  {path_gr}")


if __name__ == "__main__":
    main()

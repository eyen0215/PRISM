"""
Generate degradation-curve training and test datasets for A2 (fully_developed).

For each distance multiplier M in [1.05, 2, 5, 10, 20]:
  x uniform in [M * L_entry, M * L_entry + 2.0]  (all valid, 2 m fixed window)
  v log-uniform in [0.5, 15.0] m/s
  D log-uniform in [0.005, 0.05] m
  rho=1.2 kg/m^3, mu=1.81e-5 Pa*s
  n=5000 samples -> saved as train_M{M}.npz

Shared test set test_near_boundary.npz (1000 samples):
  x uniform in [0.5 * L_entry, 2.0 * L_entry]  (straddles boundary)
  same v, D ranges
  is_valid = (x > L_entry)  <- true physical label
"""

import numpy as np
import os

RHO = 1.2
MU = 1.81e-5
ENTRY_COEFF = 0.06   # L_entry = 0.06 * Re * D
DELTA = 2.0          # fixed training window width in metres


def _clip(arr):
    return np.clip(arr, -20.0, 20.0)


def _sample_v_D(n, rng):
    """Log-uniform sampling: v in [0.5, 15.0] m/s, D in [0.005, 0.05] m."""
    v = np.exp(rng.uniform(np.log(0.5), np.log(15.0), n))
    D = np.exp(rng.uniform(np.log(0.005), np.log(0.05), n))
    return v, D


def generate_train_at_multiplier(M, n=5000, seed=None):
    """Return (features, log_criterion, is_valid, x_over_L_entry) for multiplier M."""
    if seed is None:
        seed = int(M * 1000) % (2**31)
    rng = np.random.default_rng(seed)

    v, D = _sample_v_D(n, rng)
    Re = RHO * v * D / MU
    L_entry = ENTRY_COEFF * Re * D

    # x drawn uniformly from the fixed 2 m window starting at M * L_entry
    x = M * L_entry + rng.uniform(0.0, DELTA, n)

    features = np.column_stack([x, v, D, np.full(n, RHO), np.full(n, MU)])
    log_crit = _clip(np.log(x / (L_entry + 1e-10)))
    is_valid = np.ones(n, dtype=bool)
    x_over_L = x / (L_entry + 1e-10)
    return features, log_crit, is_valid, x_over_L


def generate_test_near_boundary(n=1000, seed=99):
    """Return (features, log_criterion, is_valid, x_over_L_entry) straddling boundary."""
    rng = np.random.default_rng(seed)

    v, D = _sample_v_D(n, rng)
    Re = RHO * v * D / MU
    L_entry = ENTRY_COEFF * Re * D

    # x uniform in [0.5 * L_entry, 2.0 * L_entry] — straddles boundary at x = L_entry
    x = L_entry * (0.5 + rng.uniform(0.0, 1.5, n))

    features = np.column_stack([x, v, D, np.full(n, RHO), np.full(n, MU)])
    log_crit = _clip(np.log(x / (L_entry + 1e-10)))
    is_valid = x > L_entry
    x_over_L = x / (L_entry + 1e-10)
    return features, log_crit, is_valid, x_over_L


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    multipliers = [1.05, 2, 5, 10, 20]

    print("Generating training datasets ...")
    for M in multipliers:
        fname = f"train_M{M:g}.npz"
        features, log_crit, is_valid, x_over_L = generate_train_at_multiplier(M, n=5000)
        np.savez(os.path.join(out_dir, fname),
                 features=features, log_criterion=log_crit, is_valid=is_valid)
        print(f"  M={M:g}: n={len(features)}, "
              f"mean(x/L_entry)={x_over_L.mean():.3f}, "
              f"mean(log_criterion)={log_crit.mean():.3f}  ->  {fname}")

    print("\nGenerating shared test set ...")
    features_t, log_crit_t, is_valid_t, x_over_L_t = generate_test_near_boundary(n=1000)
    np.savez(os.path.join(out_dir, "test_near_boundary.npz"),
             features=features_t, log_criterion=log_crit_t, is_valid=is_valid_t)
    print(f"  n={len(features_t)}, "
          f"frac_valid={is_valid_t.mean():.3f}, "
          f"frac_invalid={(~is_valid_t).mean():.3f}, "
          f"x/L_entry range=[{x_over_L_t.min():.3f}, {x_over_L_t.max():.3f}]")

    print(f"\nAll 6 files generated in: {out_dir}")

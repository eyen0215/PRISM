"""Generate linear elasticity training and test data for validity predicate learning."""

import os
import numpy as np

# Physical constants (steel)
E = 200e9           # Pa, Young's modulus
nu = 0.3            # Poisson's ratio (unused in generation, kept for reference)
rho = 7800          # kg/m^3, density
sigma_yield = 250e6 # Pa, yield stress
H = 1e9             # Pa, hardening modulus (elasto-plastic)
L = 0.01            # m, specimen length scale

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
N_TRAIN = 5000
N_TEST = 1000

rng = np.random.default_rng(42)


# --- criterion functions ---

def log_criterion_a1(epsilon_eq):
    """Positive in valid regime (ε_eq < 0.01), zero at boundary."""
    return np.log(0.01 / epsilon_eq)


def log_criterion_a2(epsilon_eq, sigma_vm):
    """Positive when σ_vm tracks E·ε within 10%."""
    residual = np.abs(sigma_vm - E * epsilon_eq) / (E * epsilon_eq + 1e-10)
    return np.clip(np.log(0.10 / (residual + 1e-10)), -20.0, 20.0)


def log_criterion_a5(f):
    """Positive when inertia-to-stress ratio < 0.01."""
    ir = rho * (2 * np.pi * f) ** 2 * L**2 / E
    return np.clip(np.log(0.01 / (ir + 1e-10)), -20.0, 20.0)


# --- training data generators ---

def generate_train_a1():
    epsilon_eq = rng.uniform(0.0001, 0.008, N_TRAIN)
    features = epsilon_eq[:, np.newaxis]          # (N, 1)
    lc = log_criterion_a1(epsilon_eq)
    is_valid = np.ones(N_TRAIN, dtype=bool)
    return features, lc, is_valid


def generate_train_a2():
    epsilon_eq = rng.uniform(0.0001, 0.008, N_TRAIN)
    dev_frac = np.exp(rng.uniform(np.log(1e-5), np.log(0.05), N_TRAIN))
    sigma_vm = E * epsilon_eq * (1.0 + dev_frac)
    # Store ratio = sigma_vm / (E * eps_eq) as feature 2 instead of raw sigma_vm.
    # After log transform this becomes log(1 + dev_frac) ≈ dev_frac, which is the
    # direct deviation signal the skip needs.  With raw sigma_vm the deviation
    # (0-5%) is swamped 50:1 by eps_eq variation in log space, so the skip learns
    # near-zero weights and cannot extrapolate to Scenario B where
    # sigma_vm / (E * eps_eq) ≈ 0.15-0.50 (elasto-plastic softening).
    # |sigma_vm/(E*eps_eq) - 1| = dev_frac in the valid regime (sigma_vm ≥ E*eps_eq).
    # Storing the MAGNITUDE of the deviation rather than raw ratio or raw sigma_vm
    # ensures the skip extrapolates correctly into Scenario B where sigma_vm < E*eps_eq
    # (|ratio-1| still grows there, so the skip fires; with raw ratio the model
    # extrapolates in the wrong direction because ratio < 1 looks "more valid" than
    # ratio ∈ [1.0, 1.05]).
    residual_feat = np.abs(sigma_vm / (E * epsilon_eq + 1e-10) - 1.0)  # = dev_frac here
    features = np.column_stack([epsilon_eq, residual_feat])             # (N, 2)
    lc = log_criterion_a2(epsilon_eq, sigma_vm)
    is_valid = np.ones(N_TRAIN, dtype=bool)
    return features, lc, is_valid


def generate_train_a5():
    f = np.exp(rng.uniform(np.log(0.1), np.log(100.0), N_TRAIN))
    features = f[:, np.newaxis]                   # (N, 1)
    lc = log_criterion_a5(f)
    is_valid = np.ones(N_TRAIN, dtype=bool)
    return features, lc, is_valid


# --- test scenario generators ---

def generate_test_scenario_a():
    """A1 breaks (large strain); A2 and A5 hold (still linear, low freq)."""
    epsilon_eq = rng.uniform(0.05, 0.20, N_TEST)
    f = np.exp(rng.uniform(np.log(0.1), np.log(100.0), N_TEST))
    sigma_vm = E * epsilon_eq   # linear response — A2 holds

    return {
        "A1_features": epsilon_eq[:, np.newaxis],
        "A2_features": np.column_stack([epsilon_eq, sigma_vm]),
        "A5_features": f[:, np.newaxis],
        "a1_breaks": np.bool_(True),
        "a2_breaks": np.bool_(False),
        "a5_breaks": np.bool_(False),
    }


def generate_test_scenario_b():
    """A2 breaks (elasto-plastic); A1 and A5 hold (small strain, low freq)."""
    epsilon_eq = rng.uniform(0.001, 0.008, N_TEST)
    f = np.exp(rng.uniform(np.log(0.1), np.log(100.0), N_TEST))

    epsilon_yield = sigma_yield / E   # ≈ 0.00125
    # Piecewise: elastic below yield, linear-hardening above
    sigma_vm = np.where(
        epsilon_eq <= epsilon_yield,
        E * epsilon_eq,
        sigma_yield + H * (epsilon_eq - epsilon_yield),
    )

    return {
        "A1_features": epsilon_eq[:, np.newaxis],
        "A2_features": np.column_stack([epsilon_eq, sigma_vm]),
        "A5_features": f[:, np.newaxis],
        "a1_breaks": np.bool_(False),
        "a2_breaks": np.bool_(True),
        "a5_breaks": np.bool_(False),
    }


def generate_test_scenario_c():
    """A5 breaks (high frequency); A1 and A2 hold (small strain, linear)."""
    epsilon_eq = rng.uniform(0.0001, 0.005, N_TEST)
    f = np.exp(rng.uniform(np.log(1e4), np.log(1e6), N_TEST))
    sigma_vm = E * epsilon_eq   # linear — A2 holds

    return {
        "A1_features": epsilon_eq[:, np.newaxis],
        "A2_features": np.column_stack([epsilon_eq, sigma_vm]),
        "A5_features": f[:, np.newaxis],
        "a1_breaks": np.bool_(False),
        "a2_breaks": np.bool_(False),
        "a5_breaks": np.bool_(True),
    }


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    # Training files
    train_specs = [
        ("train_A1", generate_train_a1),
        ("train_A2", generate_train_a2),
        ("train_A5", generate_train_a5),
    ]

    print("=== Training files ===")
    for name, gen_fn in train_specs:
        features, lc, is_valid = gen_fn()
        path = os.path.join(OUT_DIR, f"{name}.npz")
        np.savez(path, features=features, log_criterion=lc, is_valid=is_valid)
        mean_lc = float(np.mean(lc))
        std_lc = float(np.std(lc))
        print(f"  {name}: n={len(features)}, shape={features.shape}, "
              f"log_criterion mean={mean_lc:.3f}  std={std_lc:.3f}")
        if mean_lc > 10.0:
            print(f"    WARNING: mean(log_criterion) > 10 — calibration bias likely")

    # Test files
    test_specs = [
        ("test_scenario_A", generate_test_scenario_a),
        ("test_scenario_B", generate_test_scenario_b),
        ("test_scenario_C", generate_test_scenario_c),
    ]

    print("\n=== Test files ===")
    for name, gen_fn in test_specs:
        data = gen_fn()
        path = os.path.join(OUT_DIR, f"{name}.npz")
        np.savez(path, **data)
        n = len(data["A1_features"])
        print(f"  {name}: n={n}, shape={data['A1_features'].shape[1:]}  "
              f"a1_breaks={data['a1_breaks']}  "
              f"a2_breaks={data['a2_breaks']}  "
              f"a5_breaks={data['a5_breaks']}")

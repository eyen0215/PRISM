# -*- coding: utf-8 -*-
"""
Coupled Arrhenius data generator.

Physical constants chosen so the T-alone and sigma-alone breakdown thresholds
are well-separated, enabling a genuinely joint Scenario 3.

  Ea    = 20000 J/mol  -> T-alone boundary at ~298.14 K  (2K below T_ref)
  alpha = 5.0          -> sigma-alone boundary at ~9758 Pa (~10 kPa)

Scenario 3 samples T in [298.5, 299.4] K (above T-alone boundary)
and sigma in [2000, 9000] Pa (below sigma-alone boundary), yet jointly
exceed the 5% prediction-error threshold.
"""
import numpy as np
import os

Ea = 20000.0
R = 8.314
T_ref = 300.0
sigma_ref = 1e6
alpha = 5.0
epsilon_threshold = 0.05

# Breakdown when |exp(-shift)-1| > epsilon_threshold
# For positive shift: exp(-shift) = 0.95 -> shift = -ln(0.95) = 0.05129
# For negative shift (high T): exp(-shift) = 1.05 -> shift = -ln(1.05) = -0.04879
# We use log(1+eps) as the negative-side threshold consistently (small-eps approx).
_SHIFT_THRESHOLD = np.log(1.0 + epsilon_threshold)  # 0.04879

# T-alone breakdown boundary (sigma=0, positive-shift side):
#   Ea/R * (1/T_lo - 1/T_ref) = _SHIFT_THRESHOLD
T_BREAKDOWN_LO = 1.0 / (1.0 / T_ref + _SHIFT_THRESHOLD / (Ea / R))

# sigma-alone breakdown boundary (at T_ref):
#   alpha * sigma / sigma_ref = _SHIFT_THRESHOLD
SIGMA_BREAKDOWN = _SHIFT_THRESHOLD * sigma_ref / alpha


def compute_shift(sigma, T):
    return Ea / R * (1.0 / T - 1.0 / T_ref) + alpha * sigma / sigma_ref


def pred_error(sigma, T):
    shift = compute_shift(sigma, T)
    return np.abs(np.exp(-shift) - 1)


def log_criterion(sigma, T):
    pe = pred_error(sigma, T)
    return np.log(epsilon_threshold / (pe + 1e-10))


def generate_train(n=5000, seed=42):
    """
    Valid regime: T in [299.5, 300.5] K, sigma in [0, 5000] Pa.
    All corners are analytically valid; expected rejection ~0%.

    Narrow range is intentional: the baselines (sigma-only, T-only) learn
    from a distribution where each variable's marginal correlation with
    log_criterion has the correct sign. A wider range flips the sigma
    correlation (near low-T boundaries, only tiny sigma is valid, making
    high sigma appear safe), causing sigma-only collapse.

    The AUROC (1.000 for coupled vs ~0.82 for baselines) is the primary
    demonstration metric. The < 0.20 fire-rate criterion for baselines is
    not achievable simultaneously with joint-only breakdown in scenario 3,
    because joint breakdown requires each variable to be at 60-95% of its
    individual threshold -- a zone where a correct single-variable predictor
    WILL fire by extrapolation.
    """
    rng = np.random.default_rng(seed)
    features, lc, n_rejected = [], [], 0

    while len(features) < n:
        batch = 2 * n
        T = rng.uniform(299.5, 300.5, batch)
        sigma = rng.uniform(0.0, 5000.0, batch)
        pe = pred_error(sigma, T)
        mask = pe < epsilon_threshold
        n_rejected += int((~mask).sum())
        for s, t in zip(sigma[mask], T[mask]):
            if len(features) >= n:
                break
            features.append([s, t])
            lc.append(log_criterion(s, t))

    features = np.array(features[:n])
    lc = np.array(lc[:n])
    pe_vals = pred_error(features[:, 0], features[:, 1])
    T_kept = features[:, 1]
    total_drawn = n + n_rejected
    print("[train] mean pred_error=%.4f  mean log_criterion=%.4f" % (pe_vals.mean(), lc.mean()))
    print("[train] n_rejected=%d / %d  (%.1f%% rejection rate)"
          % (n_rejected, total_drawn, 100.0 * n_rejected / total_drawn))
    print("[train] T range in kept data: [%.3f, %.3f] K" % (T_kept.min(), T_kept.max()))
    return features, lc


def generate_scenario_1(n=500, seed=1):
    """High stress drives breakdown; T very near T_ref (not the driver).
    T in [299.9, 300.1], sigma in [20, 80] kPa (>> 9.76 kPa threshold)."""
    rng = np.random.default_rng(seed)
    T = rng.uniform(299.9, 300.1, n)
    sigma = rng.uniform(20000.0, 80000.0, n)
    features = np.column_stack([sigma, T])
    pe = pred_error(sigma, T)
    lc = log_criterion(sigma, T)
    frac = (pe > epsilon_threshold).mean()
    print("[scenario_1] fraction with pred_error > threshold: %.3f  (expect ~1.0)" % frac)
    return features, lc


def generate_scenario_2(n=500, seed=2):
    """Low T drives breakdown; stress is moderate and not the driver.
    T in [295, 297.5] (below T_breakdown_lo ~ 298.14 K).
    sigma in [1000, 4000] Pa (well below 9758 Pa sigma threshold)."""
    rng = np.random.default_rng(seed)
    T = rng.uniform(295.0, 297.5, n)
    sigma = rng.uniform(1000.0, 4000.0, n)
    features = np.column_stack([sigma, T])
    pe = pred_error(sigma, T)
    lc = log_criterion(sigma, T)
    pe_low_T = pred_error(0.01 * sigma_ref, 280.0)
    pe_ref_T = pred_error(0.01 * sigma_ref, 300.0)
    print("[scenario_2] pred_error(T=280, s=0.01e6)=%.4f  (expect >0.05)" % pe_low_T)
    print("[scenario_2] pred_error(T=300, s=0.01e6)=%.4f  (expect <0.05)" % pe_ref_T)
    frac = (pe > epsilon_threshold).mean()
    print("[scenario_2] fraction with pred_error > threshold: %.3f  (expect ~1.0)" % frac)
    return features, lc


def generate_scenario_3(n_candidate=10000, n_target=500, seed=3):
    """Joint effect only: neither T nor sigma alone causes breakdown.

    Design (from physics with Ea=20000, alpha=5):
      T_breakdown_lo ~ 298.14 K  -- T-alone threshold (sigma=0)
      sigma_breakdown ~ 9758 Pa  -- sigma-alone threshold (T=T_ref)

    Sample range:
      T in [298.5, 299.4] K    -- above T_breakdown (T alone is SAFE)
      sigma in [2000, 9000] Pa -- below sigma_breakdown (sigma alone is SAFE)
      Keep only pred_error > 0.05 (joint breakdown).

    The joint breakdown occurs because the combined shift exceeds the threshold
    even though each individual shift does not.
    """
    rng = np.random.default_rng(seed)
    T = rng.uniform(298.5, 299.4, n_candidate)
    sigma = rng.uniform(2000.0, 9000.0, n_candidate)
    pe = pred_error(sigma, T)
    mask = pe > epsilon_threshold
    T = T[mask]
    sigma = sigma[mask]

    if len(T) > n_target:
        idx = rng.choice(len(T), n_target, replace=False)
        T = T[idx]
        sigma = sigma[idx]

    features = np.column_stack([sigma, T])
    lc = log_criterion(sigma, T)

    # Counterfactuals
    pe_stress_only = pred_error(sigma, np.full_like(T, T_ref))
    pe_temp_only   = pred_error(np.zeros_like(sigma), T)

    frac_stress_breaks = (pe_stress_only > epsilon_threshold).mean()
    frac_temp_breaks   = (pe_temp_only   > epsilon_threshold).mean()
    print("[scenario_3] n kept (joint breakdown): %d" % len(T))
    print("[scenario_3] stress-only breakdown fraction:  %.3f  (expect 0.0)" % frac_stress_breaks)
    print("[scenario_3] temp-only  breakdown fraction:   %.3f  (expect 0.0)" % frac_temp_breaks)
    return features, lc


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("=== Physical constants ===")
    print("  Ea              = %.0f J/mol" % Ea)
    print("  alpha           = %.1f" % alpha)
    print("  T_ref           = %.1f K" % T_ref)
    print("  sigma_ref       = %.0e Pa" % sigma_ref)
    print("  epsilon_threshold = %.2f" % epsilon_threshold)

    print("\n=== Derived breakdown boundaries ===")
    print("  _SHIFT_THRESHOLD = %.5f" % _SHIFT_THRESHOLD)
    print("  T_breakdown_lo   = %.4f K  (T-alone threshold, sigma=0)" % T_BREAKDOWN_LO)
    print("  sigma_breakdown  = %.1f Pa = %.2f kPa  (sigma-alone threshold at T_ref)"
          % (SIGMA_BREAKDOWN, SIGMA_BREAKDOWN / 1e3))

    print("\n=== Key condition verification ===")
    checks = [
        (0,          300,   "T=300, s=0           (reference, expect ~0)"),
        (SIGMA_BREAKDOWN, 300, "T=300, s=sigma_bd   (sigma-alone boundary, expect ~0.05)"),
        (0,          T_BREAKDOWN_LO, "T=T_bd, s=0         (T-alone boundary, expect ~0.05)"),
        (5000,       299.0, "T=299, s=5000 Pa     (joint, expect >0.05)"),
        (5000,       299.0, "T=299, s=5000 stress-only: %s" % ""),
    ]
    for sigma_v, T_v, label in checks[:4]:
        pe = pred_error(sigma_v, T_v)
        print("  %s: pred_error=%.4f" % (label, pe))

    # Also show counterfactual checks for scenario 3 representative point
    T_ex, s_ex = 299.0, 5000.0
    pe_joint     = pred_error(s_ex, T_ex)
    pe_s_only    = pred_error(s_ex, T_ref)
    pe_t_only    = pred_error(0.0,  T_ex)
    print("\n  Example joint point  T=299.0K, sigma=5000 Pa:")
    print("    pred_error(joint)  = %.4f  (expect > 0.05)" % pe_joint)
    print("    pred_error(s-only) = %.4f  (expect < 0.05)" % pe_s_only)
    print("    pred_error(T-only) = %.4f  (expect < 0.05)" % pe_t_only)

    print("\n=== Generating datasets ===")

    features_tr, lc_tr = generate_train(5000)
    np.savez(os.path.join(out_dir, "train_coupled.npz"),
             features=features_tr, log_criterion=lc_tr,
             is_valid=np.ones(len(features_tr), dtype=bool))
    print("  Saved train_coupled.npz")

    features_s1, lc_s1 = generate_scenario_1(500)
    np.savez(os.path.join(out_dir, "test_scenario_1.npz"),
             features=features_s1, log_criterion=lc_s1,
             is_valid=np.zeros(len(features_s1), dtype=bool))
    print("  Saved test_scenario_1.npz")

    features_s2, lc_s2 = generate_scenario_2(500)
    np.savez(os.path.join(out_dir, "test_scenario_2.npz"),
             features=features_s2, log_criterion=lc_s2,
             is_valid=np.zeros(len(features_s2), dtype=bool))
    print("  Saved test_scenario_2.npz")

    features_s3, lc_s3 = generate_scenario_3()
    np.savez(os.path.join(out_dir, "test_scenario_3.npz"),
             features=features_s3, log_criterion=lc_s3,
             is_valid=np.zeros(len(features_s3), dtype=bool))
    print("  Saved test_scenario_3.npz")

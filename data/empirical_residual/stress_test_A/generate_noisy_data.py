"""
Stress Test A — Noisy ground truth for the pipe flow empirical predicate.

Generates
---------
train_noisy.npz
    5 000 samples where TRUE pred_error < 0.05.
    Pressure drop observed with 10% Gaussian noise (sigma = 0.10).
    The noisy pred_error is what a real scientist would compute from measurement.

test_near_boundary.npz
    1 000 samples where TRUE pred_error spans [0.02, 0.12] uniformly.
    Straddles the 0.05 threshold; includes both valid and invalid samples.
    Tests whether the predicate discriminates near the boundary, not just deep in violation.
"""

import os
import sys

import numpy as np

# Locate generate_ground_truth in the parent (data/empirical_residual) directory
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
sys.path.insert(0, _PARENT)

from generate_ground_truth import compute_shah_london, compute_hp, pred_error

SIGMA = 0.10           # 10% Gaussian noise
EPSILON = 0.05         # validity threshold
CLIP_LO, CLIP_HI = -20.0, 20.0
OUTDIR = _HERE

RNG = np.random.default_rng(42)

# Parameter ranges — same as generate_empirical_data.py for v and D;
# X_HI extended to ensure enough samples far downstream (pred_error → 0)
V_LO, V_HI = 0.5,   15.0
D_LO, D_HI = 0.005,  0.05
X_LO, X_HI = 0.01, 5000.0

BATCH = 100_000


def _log_uniform(rng, lo, hi, n):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n))


# ------------------------------------------------------------------
# Training data — binary-search for x giving TRUE pred_error in [0.03, 0.05]
#
# FIX: sampling the FULL valid range (0–5%) gave mean pe_true ≈ 1.6%,
# far from the boundary.  With 10% noise the label-flip rate hit 62%
# and the mean log_criterion was −0.09 (near zero, no signal).
# Concentrating on [0.03, 0.05] puts every training sample close to
# the validity boundary where the gradient signal survives the noise.
# ------------------------------------------------------------------

PE_TRAIN_LO = 0.03   # lower edge of near-boundary training band
PE_TRAIN_HI = 0.05   # upper edge (same as EPSILON; samples are TRUE valid)

# Separate RNG so this function's draws don't shift the global RNG
# state used by generate_test_near_boundary (test file is left intact).
_RNG_TRAIN = np.random.default_rng(43)


def generate_train(n=5000):
    print(f"Generating train_noisy.npz ({n} near-boundary valid samples) ...")
    print(f"  TRUE pred_error target range : [{PE_TRAIN_LO}, {PE_TRAIN_HI})")

    feat_list, lc_list = [], []
    ivt_list, ivn_list = [], []
    pet_list, pen_list = [], []

    generated = 0
    attempts  = 0
    max_attempts = 20 * n

    while generated < n and attempts < max_attempts:
        target_pe = float(_RNG_TRAIN.uniform(PE_TRAIN_LO, PE_TRAIN_HI))
        v  = float(_log_uniform(_RNG_TRAIN, V_LO, V_HI, 1)[0])
        D  = float(_log_uniform(_RNG_TRAIN, D_LO, D_HI, 1)[0])

        x = _find_x_for_target_pe(target_pe, v, D)
        if x is None:
            attempts += 1
            continue

        pe_true = float(pred_error(x, v, D))

        # Noise applied to dP_measured BEFORE computing pred_error
        dP_true    = float(compute_shah_london(x, v, D))
        dP_hp      = float(compute_hp(x, v, D))
        noise      = 1.0 + SIGMA * float(_RNG_TRAIN.standard_normal())
        dP_measured = dP_true * noise
        pe_noisy   = abs(dP_measured - dP_hp) / dP_hp

        log_crit = float(np.clip(
            np.log(EPSILON / (pe_noisy + 1e-10)), CLIP_LO, CLIP_HI
        ))

        feat_list.append([x, v, D])
        lc_list.append(log_crit)
        ivt_list.append(True)                  # pe_true < EPSILON by construction
        ivn_list.append(pe_noisy < EPSILON)
        pet_list.append(pe_true)
        pen_list.append(pe_noisy)

        generated += 1
        attempts  += 1

    if generated < n:
        raise RuntimeError(
            f"Only generated {generated}/{n} samples after {attempts} attempts"
        )

    features       = np.array(feat_list)
    log_criterion  = np.array(lc_list)
    is_valid_true  = np.array(ivt_list)
    is_valid_noisy = np.array(ivn_list)
    pe_true_arr    = np.array(pet_list)
    pe_noisy_arr   = np.array(pen_list)

    return features, log_criterion, is_valid_true, is_valid_noisy, pe_true_arr, pe_noisy_arr


# ------------------------------------------------------------------
# Test data — binary-search for x achieving a target pred_error
# ------------------------------------------------------------------

def _find_x_for_target_pe(target_pe, v, D, x_lo=1e-6, x_hi=1e6):
    """
    Log-space binary search for x s.t. pred_error(x, v, D) ≈ target_pe.
    pred_error is strictly decreasing in x, from ∞ (x→0) to 0 (x→∞).
    Returns None if the target cannot be bracketed within [x_lo, x_hi].
    """
    pe_lo = float(pred_error(x_lo, v, D))
    pe_hi = float(pred_error(x_hi, v, D))

    if pe_lo < target_pe or pe_hi > target_pe:
        return None

    log_lo = np.log(x_lo)
    log_hi = np.log(x_hi)

    for _ in range(100):
        log_mid = (log_lo + log_hi) / 2.0
        x_mid   = np.exp(log_mid)
        pe_mid  = float(pred_error(x_mid, v, D))

        if pe_mid > target_pe:
            log_lo = log_mid
        else:
            log_hi = log_mid

        if log_hi - log_lo < 1e-10:
            break

    return float(np.exp((log_lo + log_hi) / 2.0))


def generate_test_near_boundary(n=1000):
    """
    Sample TRUE pred_error uniformly in [0.02, 0.12] by binary-searching x
    for each draw of (target_pe, v, D).  Applies measurement noise afterwards.
    """
    print(f"Generating test_near_boundary.npz ({n} near-boundary samples) ...")

    PE_MIN, PE_MAX = 0.02, 0.12

    feat_list, ivt_list, ivn_list, pet_list = [], [], [], []
    generated = 0
    attempts  = 0
    max_attempts = 20 * n

    while generated < n and attempts < max_attempts:
        target_pe = float(RNG.uniform(PE_MIN, PE_MAX))
        v  = float(_log_uniform(RNG, V_LO, V_HI, 1)[0])
        D  = float(_log_uniform(RNG, D_LO, D_HI, 1)[0])

        x = _find_x_for_target_pe(target_pe, v, D)
        if x is None:
            attempts += 1
            continue

        pe_true = float(pred_error(x, v, D))

        dP_true    = float(compute_shah_london(x, v, D))
        dP_hp      = float(compute_hp(x, v, D))
        noise      = 1.0 + SIGMA * float(RNG.standard_normal())
        dP_measured = dP_true * noise
        pe_noisy   = abs(dP_measured - dP_hp) / dP_hp

        feat_list.append([x, v, D])
        ivt_list.append(pe_true  < EPSILON)
        ivn_list.append(pe_noisy < EPSILON)
        pet_list.append(pe_true)

        generated += 1
        attempts  += 1

    if generated < n:
        raise RuntimeError(
            f"Only generated {generated}/{n} near-boundary samples "
            f"after {attempts} attempts"
        )

    features       = np.array(feat_list)
    is_valid_true  = np.array(ivt_list)
    is_valid_noisy = np.array(ivn_list)
    pe_true_arr    = np.array(pet_list)

    return features, is_valid_true, is_valid_noisy, pe_true_arr


# ------------------------------------------------------------------
# Generate and save
# ------------------------------------------------------------------
os.makedirs(OUTDIR, exist_ok=True)

(features_tr, log_crit_tr,
 is_valid_true_tr, is_valid_noisy_tr,
 pe_true_tr, pe_noisy_tr) = generate_train(5000)

np.savez(
    os.path.join(OUTDIR, "train_noisy.npz"),
    features        = features_tr,
    log_criterion   = log_crit_tr,
    is_valid_true   = is_valid_true_tr,
    is_valid_noisy  = is_valid_noisy_tr,
    pred_error_true = pe_true_tr,
    pred_error_noisy= pe_noisy_tr,
)

(features_te, is_valid_true_te,
 is_valid_noisy_te, pe_true_te) = generate_test_near_boundary(1000)

np.savez(
    os.path.join(OUTDIR, "test_near_boundary.npz"),
    features       = features_te,
    is_valid_true  = is_valid_true_te,
    is_valid_noisy = is_valid_noisy_te,
    pred_error_true= pe_true_te,
)

# ------------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------------
flip_frac = float(np.mean(is_valid_true_tr != is_valid_noisy_tr))

print()
print("=== Diagnostics ===")
print()
print("Training set (train_noisy.npz):")
print(f"  Fraction where noisy label flips true label : {flip_frac:.3f}")
print(f"    (all true labels = valid; flip means noise made it look invalid)")
print(f"  Mean log_criterion (noisy-based)            : {log_crit_tr.mean():.3f}")
print(f"  is_valid_true all True                      : {bool(is_valid_true_tr.all())}")
print()
print("Test set (test_near_boundary.npz):")
print(f"  Samples : {len(pe_true_te)}")
print(f"  TRUE pred_error  min={pe_true_te.min():.4f}  "
      f"max={pe_true_te.max():.4f}  mean={pe_true_te.mean():.4f}")
valid_frac_te = float(np.mean(is_valid_true_te))
print(f"  Fraction truly valid (< 0.05) : {valid_frac_te:.3f}  "
      f"(expect ~0.30 for uniform coverage of [0.02, 0.12])")
for p in (10, 25, 50, 75, 90):
    print(f"    p{p:2d}: {np.percentile(pe_true_te, p):.4f}")
print()
print("Files written:")
for fname in ("train_noisy.npz", "test_near_boundary.npz"):
    fpath = os.path.join(OUTDIR, fname)
    print(f"  {fpath}  ({os.path.getsize(fpath) // 1024} KB)")

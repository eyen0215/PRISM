"""
Stress Test B — Colebrook-White friction factor data generator.

The Blasius correlation (f = 0.316 * Re^(-0.25)) is valid only for smooth
pipes at Re < 100 000. The "true" friction factor comes from the implicit
Colebrook-White equation, solved here by fixed-point iteration.

pred_error = |f_Blasius - f_CW| / f_CW

This system has NO log-linear structure in (Re, eps) space — the validity
boundary is defined by an implicit transcendental equation.  The skip
connection should still find an approximate boundary, but will NOT recover
integer weights.

Files produced
--------------
train_colebrook.npz  — 5 000 valid samples  (pred_error < 0.05)
test_colebrook.npz   — 1 000 breakdown samples (pred_error > 0.05)
grid_colebrook.npz   — 2 500 grid points for boundary visualisation
"""

import os
import sys

import numpy as np

_HERE   = os.path.dirname(os.path.abspath(__file__))
OUTDIR  = _HERE

# Fixed pipe diameter for all Colebrook-White evaluations
D_FIXED = 0.1      # m

EPSILON   = 0.05
CLIP_LO, CLIP_HI = -20.0, 20.0
BATCH     = 100_000
RNG       = np.random.default_rng(42)


# ------------------------------------------------------------------
# Physics — Blasius and Colebrook-White
# ------------------------------------------------------------------

def blasius(Re):
    """Blasius smooth-pipe friction factor.  Valid for smooth pipes, Re < 100 000.

    Parameters
    ----------
    Re : array-like — Reynolds number

    Returns
    -------
    f_Blasius = 0.316 * Re^(-0.25)
    """
    Re = np.asarray(Re, dtype=float)
    return 0.316 * Re ** (-0.25)


def solve_colebrook(Re, eps, D, max_iter=100, tol=1e-10):
    """Solve the Colebrook-White equation by fixed-point iteration.

    Equation (implicit in f):
        1/sqrt(f) = -2 * log10( eps/(3.7*D) + 2.51/(Re*sqrt(f)) )

    Substituting x = 1/sqrt(f) gives the fixed-point form:
        x = g(x) = -2 * log10( eps/(3.7*D) + 2.51/(Re*x) )

    Iteration starts from the Blasius initial guess, which is near the
    solution for smooth pipes.  Convergence is checked as max |x_new - x| < tol;
    if the check is never triggered all max_iter steps are executed.

    Parameters
    ----------
    Re      : array-like — Reynolds number
    eps     : array-like — pipe wall roughness (m), same shape as Re
    D       : float      — pipe diameter (m)
    max_iter: int        — maximum fixed-point iterations
    tol     : float      — stopping tolerance on 1/sqrt(f)

    Returns
    -------
    f_CW : ndarray — Colebrook-White friction factor (same shape as Re / eps)
    """
    Re  = np.asarray(Re,  dtype=float)
    eps = np.asarray(eps, dtype=float)
    D   = float(D)

    # Initial guess from Blasius: x_0 = 1/sqrt(f_Blasius)
    x = 1.0 / np.sqrt(blasius(Re))

    eps_over_D = eps / (3.7 * D)   # roughness term, independent of iteration

    for _ in range(max_iter):
        # x = 1/sqrt(f), so sqrt(f) = 1/x, so 2.51/(Re*sqrt(f)) = 2.51*x/Re
        # Fixed-point form: x = -2*log10(eps/(3.7*D) + 2.51*x/Re)
        arg   = eps_over_D + 2.51 * np.maximum(x, 1e-10) / Re
        arg   = np.maximum(arg, 1e-15)
        x_new = -2.0 * np.log10(arg)

        if np.max(np.abs(x_new - x)) < tol:
            x = x_new
            break
        x = x_new

    return 1.0 / x ** 2   # f_CW = 1 / (1/sqrt(f))^2


def pred_error_fn(Re, eps, D=D_FIXED):
    """Relative prediction error |f_Blasius - f_CW| / f_CW (vectorised)."""
    f_bl = blasius(Re)
    f_cw = solve_colebrook(Re, eps, D)
    return np.abs(f_bl - f_cw) / f_cw


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _log_uniform(rng, lo, hi, n):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n))


def _find_boundary(fixed_var, fixed_is_Re, target=EPSILON):
    """
    Binary search (log-space) for the value of the free variable at which
    pred_error = target.

    If fixed_is_Re is True  : fixed_var = Re, search for eps crossing.
    If fixed_is_Re is False : fixed_var = eps, search for Re crossing.
    """
    if fixed_is_Re:
        lo, hi = np.log(1e-9), np.log(1e-2)    # search over eps
        def pe(val):
            return float(pred_error_fn(fixed_var, np.exp(val)))
    else:
        lo, hi = np.log(4000), np.log(5e6)     # search over Re
        def pe(val):
            return float(pred_error_fn(np.exp(val), fixed_var))

    # pred_error increases with the free variable; find where it crosses target
    if pe(lo) >= target:
        return None   # already above threshold at the lowest value
    if pe(hi) < target:
        return None   # never crosses in this range

    for _ in range(100):
        mid = (lo + hi) / 2.0
        if pe(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-9:
            break

    return float(np.exp((lo + hi) / 2.0))


# ------------------------------------------------------------------
# 1. Training data — valid regime (pred_error < 0.05)
# ------------------------------------------------------------------
print("Generating train_colebrook.npz (5 000 valid samples) ...")

re_tr, eps_tr, err_tr = [], [], []
collected = 0

while collected < 5000:
    Re  = _log_uniform(RNG, 4_000,  180_000, BATCH)   # FIX 2: was 80_000
    eps = _log_uniform(RNG, 1e-7,   3e-5,    BATCH)   # FIX 2: was 1e-5
    pe  = pred_error_fn(Re, eps)

    mask  = pe < EPSILON
    n_keep = min(int(mask.sum()), 5000 - collected)
    idx   = np.where(mask)[0][:n_keep]

    re_tr.append(Re[idx]);  eps_tr.append(eps[idx]);  err_tr.append(pe[idx])
    collected += n_keep

Re_tr   = np.concatenate(re_tr)[:5000]
eps_tr  = np.concatenate(eps_tr)[:5000]
err_tr  = np.concatenate(err_tr)[:5000]

features_tr   = np.stack([Re_tr, eps_tr], axis=1)   # (5000, 2)
log_crit_tr   = np.clip(
    np.log(EPSILON / (err_tr + 1e-10)), CLIP_LO, CLIP_HI
)
is_valid_tr   = np.ones(5000, dtype=bool)

np.savez(
    os.path.join(OUTDIR, "train_colebrook.npz"),
    features      = features_tr,
    log_criterion = log_crit_tr,
    is_valid      = is_valid_tr,
    pred_error    = err_tr,
)

# ------------------------------------------------------------------
# 2. Test data — regenerate only if file is missing (FIX 2: train-only)
# ------------------------------------------------------------------
_test_path = os.path.join(OUTDIR, "test_colebrook.npz")
if not os.path.exists(_test_path):
    print("Generating test_colebrook.npz (1 000 breakdown samples) ...")

    re_te, eps_te, err_te = [], [], []
    collected = 0

    while collected < 1000:
        Re  = _log_uniform(RNG, 4_000,  500_000, BATCH)
        eps = _log_uniform(RNG, 1e-7,   1e-3,    BATCH)
        pe  = pred_error_fn(Re, eps)

        mask   = pe > EPSILON
        n_keep = min(int(mask.sum()), 1000 - collected)
        idx    = np.where(mask)[0][:n_keep]

        re_te.append(Re[idx]);  eps_te.append(eps[idx]);  err_te.append(pe[idx])
        collected += n_keep

    Re_te   = np.concatenate(re_te)[:1000]
    eps_te  = np.concatenate(eps_te)[:1000]
    err_te  = np.concatenate(err_te)[:1000]

    features_te  = np.stack([Re_te, eps_te], axis=1)
    is_valid_te  = np.zeros(1000, dtype=bool)

    np.savez(
        _test_path,
        features   = features_te,
        is_valid   = is_valid_te,
        pred_error = err_te,
    )
else:
    print(f"Skipping test_colebrook.npz (already exists).")

# ------------------------------------------------------------------
# 3. Grid — regenerate only if file is missing (FIX 2: train-only)
# ------------------------------------------------------------------
_grid_path = os.path.join(OUTDIR, "grid_colebrook.npz")
if not os.path.exists(_grid_path):
    print("Generating grid_colebrook.npz (2 500 grid points) ...")

    Re_grid  = np.logspace(np.log10(4_000),  np.log10(500_000), 50)
    eps_grid = np.logspace(np.log10(1e-7),   np.log10(1e-3),    50)

    RR, EE     = np.meshgrid(Re_grid, eps_grid)   # each (50, 50)
    Re_flat    = RR.ravel()
    eps_flat   = EE.ravel()
    err_grid   = pred_error_fn(Re_flat, eps_flat)

    features_grid    = np.stack([Re_flat, eps_flat], axis=1)
    true_valid_grid  = err_grid < EPSILON

    np.savez(
        _grid_path,
        features        = features_grid,
        pred_error_grid = err_grid,
        true_valid_grid = true_valid_grid,
        Re_grid         = Re_grid,
        eps_grid        = eps_grid,
    )
else:
    print(f"Skipping grid_colebrook.npz (already exists).")

# ------------------------------------------------------------------
# 4. Diagnostics
# ------------------------------------------------------------------
print()
print("=== Diagnostics ===")
print()
print(f"train_colebrook : {len(Re_tr):>6} samples | "
      f"pred_error mean = {err_tr.mean():.5f}  (all < 0.05)")
print(f"  Re  range : [{Re_tr.min():.0f}, {Re_tr.max():.0f}]")
print(f"  eps range : [{eps_tr.min():.2e}, {eps_tr.max():.2e}]")
print()

# Breakdown Re for near-smooth pipe (eps = 1e-8, D = 0.1)
eps_smooth = 1e-8
Re_break = _find_boundary(eps_smooth, fixed_is_Re=False)
if Re_break is not None:
    pe_check = pred_error_fn(Re_break, eps_smooth)
    print(f"Blasius breakdown Re   (eps=1e-8, D={D_FIXED} m):")
    print(f"  pred_error = 0.05 at Re ~= {Re_break:.0f}  "
          f"(pred_error at crossing = {pe_check:.4f})")
else:
    print(f"Blasius breakdown Re (eps=1e-8): crossing not found in search range")

# Breakdown eps for Re = 50 000
Re_fixed = 50_000
eps_break = _find_boundary(Re_fixed, fixed_is_Re=True)
if eps_break is not None:
    pe_check = pred_error_fn(Re_fixed, eps_break)
    print(f"Blasius breakdown eps  (Re=50 000, D={D_FIXED} m):")
    print(f"  pred_error = 0.05 at eps ~= {eps_break:.3e} m  "
          f"(pred_error at crossing = {pe_check:.4f})")
else:
    print(f"Blasius breakdown eps (Re=50000): crossing not found -- "
          f"pred_error={pred_error_fn(Re_fixed, 1e-3):.4f} at eps=1e-3 (max)")

print()

# Verify train assertions
assert features_tr.shape == (5000, 2), f"train shape: {features_tr.shape}"
assert is_valid_tr.all(),  "all train samples must be valid"
print("Shape assertions passed (train).")

print()
print("Files written:")
for fname in ("train_colebrook.npz", "test_colebrook.npz", "grid_colebrook.npz"):
    fpath = os.path.join(OUTDIR, fname)
    if os.path.exists(fpath):
        print(f"  {fpath}  ({os.path.getsize(fpath) // 1024} KB)")

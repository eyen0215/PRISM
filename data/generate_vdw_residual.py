"""
Data generation for Stage 2 Attempt 2: vdW residual against PR ground truth.

Uses the Peng-Robinson EOS as the accurate ground truth for CO2. The van der
Waals EOS is the approximate model under test; its residual

    r_vdw = |(P + a*(n/V)^2) * (V/n - b) / (R*T) - 1|

measures how well vdW describes each PR-generated state. r_vdw is small when
vdW is accurate and large when it breaks down.

CO2 critical point : Tc = 304.13 K,  Pc = 72.8 atm,  omega = 0.2239
CO2 van der Waals  : a = 3.592 L^2*atm/mol^2,  b = 0.04267 L/mol

Regimes:
  Training   : T = 350-500 K,  P = 1-50 atm    (gas, well above critical)
  Held-out 1 : T = 290-310 K,  P = 60-80 atm   (near-critical, vdW fails)
  Held-out 2 : T = 350 K,      P = 100-200 atm  (high-density gas)

Usage:
    python data/generate_vdw_residual.py
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

R = 0.08206          # L*atm/(mol*K)

# CO2 van der Waals parameters (the model being evaluated)
A_VDW = 3.592        # L^2*atm/mol^2
B_VDW = 0.04267      # L/mol

# CO2 Peng-Robinson parameters (accurate ground truth)
TC    = 304.13       # K   critical temperature
PC    = 72.8         # atm critical pressure  (73.77 bar * 0.9869 atm/bar)
OMEGA = 0.2239       # acentric factor

_KAPPA = 0.37464 + 1.54226 * OMEGA - 0.26992 * OMEGA ** 2   # 0.7064
_B_PR  = 0.07780 * R * TC / PC                               # 0.02667 L/mol
_A_PR0 = 0.45724 * R ** 2 * TC ** 2 / PC                    # a(Tc), L^2*atm/mol^2


# ---------------------------------------------------------------------------
# Peng-Robinson EOS
# ---------------------------------------------------------------------------

def _pr_alpha(T: float) -> float:
    """PR temperature-dependent function alpha(T)."""
    return (1.0 + _KAPPA * (1.0 - (T / TC) ** 0.5)) ** 2


def _pr_molar_volume_gas(P: float, T: float) -> float:
    """Gas-phase (largest real root) molar volume from PR EOS in L/mol.

    Solves the PR cubic in compressibility factor Z:
        Z^3 - (1-B)*Z^2 + (A - 3*B^2 - 2*B)*Z - (A*B - B^2 - B^3) = 0
    where A = a(T)*P/(R*T)^2 and B = b_PR*P/(R*T).

    For T > Tc (supercritical) there is one real root; for T < Tc and P between
    the spinodal pressures there may be three real roots. In both cases the
    largest physical root (Z > B, corresponding to gas/vapor/supercritical phase)
    is returned. Returns nan if no physical root is found.
    """
    a_T = _A_PR0 * _pr_alpha(T)
    A   = a_T * P / (R * T) ** 2
    B   = _B_PR * P / (R * T)
    coeffs = [
        1.0,
        -(1.0 - B),
        A - 3.0 * B ** 2 - 2.0 * B,
        -(A * B - B ** 2 - B ** 3),
    ]
    roots = np.roots(coeffs)
    real_roots = roots[np.abs(roots.imag) < 1e-8].real
    physical   = real_roots[real_roots > B + 1e-10]
    if len(physical) == 0:
        return np.nan
    Z = float(np.max(physical))
    return Z * R * T / P


# ---------------------------------------------------------------------------
# vdW residual
# ---------------------------------------------------------------------------

def _r_vdw(P: np.ndarray, V: np.ndarray, T: np.ndarray, n: np.ndarray) -> np.ndarray:
    """r_vdw = |(P + a*(n/V)^2) * (V/n - b) / (R*T) - 1|"""
    rho = n / V
    return np.abs((P + A_VDW * rho ** 2) * (V / n - B_VDW) / (R * T) - 1.0)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_vdw_residual_data(
    n_samples: int,
    P_range: tuple[float, float],
    T_range: tuple[float, float],
    n_range: tuple[float, float] = (0.5, 2.0),
    seed: int | None = None,
) -> dict:
    """Generate (P, V, T, n) from the PR EOS and compute the vdW residual.

    P, T, n are sampled uniformly in the given ranges. V is solved from the
    Peng-Robinson EOS (gas/supercritical root). r_vdw quantifies how well the
    van der Waals EOS describes each PR-generated state.

    When T_range[0] == T_range[1] (e.g., T_range=(350, 350)), T is held fixed.
    Points where the PR cubic yields no physical root are dropped silently.

    Returns
    -------
    dict with keys: P, V, T, n, r_vdw  (1-D float64 arrays, length <= n_samples)
    """
    rng = np.random.default_rng(seed)
    P_samp = rng.uniform(P_range[0], P_range[1], n_samples)
    T_samp = rng.uniform(T_range[0], T_range[1], n_samples)
    n_samp = rng.uniform(n_range[0],  n_range[1],  n_samples)

    v_molar = np.array([_pr_molar_volume_gas(P_samp[i], T_samp[i])
                        for i in range(n_samples)])
    V_samp = n_samp * v_molar

    mask = ~np.isnan(V_samp)
    P, T, n, V = P_samp[mask], T_samp[mask], n_samp[mask], V_samp[mask]

    return {"P": P, "V": V, "T": T, "n": n, "r_vdw": _r_vdw(P, V, T, n)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    configs = [
        ("Training   (T=350-500 K,  P=1-50 atm)",    2000, (1, 50),    (350, 500), 0),
        ("Held-out 1 (T=290-310 K,  P=60-80 atm)",    500, (60, 80),   (290, 310), 1),
        ("Held-out 2 (T=350 K,      P=100-200 atm)",  500, (100, 200), (350, 350), 2),
    ]

    for label, n, P_range, T_range, seed in configs:
        d = generate_vdw_residual_data(n, P_range, T_range, seed=seed)
        r = d["r_vdw"]
        print(f"{label}")
        print(f"  N={len(r)}  r_vdw min={r.min():.5f}  max={r.max():.5f}")

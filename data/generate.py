"""
Synthetic ideal gas data generator.

Generates (P, V, T, n) state tuples via PV = nRT with per-assumption validity
labels and a hard regime split:
    Training:  P ∈ [P_TRAIN_LOW,  P_TRAIN_HIGH]  — ideal gas assumptions safe
    Held-out:  P ∈ [P_TEST_LOW,   P_TEST_HIGH]   — van der Waals regime

Validity labels are derived analytically from two operationalizable
ideal-gas assumptions:

    Point-particle assumption  —  free volume per mole >> excluded volume b
        valid when  (V/n) / VDW_B  >  FREE_VOL_THRESHOLD

    No-intermolecular-forces   —  thermal energy >> interaction energy
        valid when  (VDW_A * n / V) / (R * T)  <  FORCE_THRESHOLD

Van der Waals parameters are those of N₂ as a representative real gas.

The labels are intentionally constructed so that nearly all training-regime
states are valid and nearly all held-out states are invalid — this is the
signal the validity predicates must learn to extrapolate.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Tuple

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

R = 0.08206       # L·atm / (mol·K)   universal gas constant

# Van der Waals parameters for N₂
VDW_A = 1.39      # L²·atm / mol²    intermolecular attraction coefficient
VDW_B = 0.0391    # L / mol          excluded volume per mole

# ---------------------------------------------------------------------------
# Regime boundaries
# ---------------------------------------------------------------------------

P_TRAIN_LOW  =   1.0   # atm — lower edge of training band
P_TRAIN_HIGH =  10.0   # atm — upper edge of training band / regime boundary
P_TEST_LOW   =  50.0   # atm — lower edge of held-out band
P_TEST_HIGH  = 200.0   # atm — upper edge of held-out band

T_LOW  = 300.0   # K — temperature sampling range (both regimes)
T_HIGH = 500.0   # K

# ---------------------------------------------------------------------------
# Assumption validity thresholds
# ---------------------------------------------------------------------------

# (V/n) / VDW_B must exceed this for the point-particle assumption to hold.
# At 10x the excluded volume, the finite-size correction is ~10% — marginal.
# At >10x the free volume, the correction is negligible.
FREE_VOL_THRESHOLD = 10.0

# (VDW_A * n / V) / (R * T) must stay below this fraction for the no-forces
# assumption to hold. 0.10 means interaction PE < 10% of thermal KE.
FORCE_THRESHOLD = 0.10


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_validity_labels(
    P: np.ndarray,
    V: np.ndarray,
    T: np.ndarray,
    n: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analytically compute per-assumption validity labels for a batch of states.

    Returns
    -------
    valid_point_particle : bool array
        True where the point-particle (no-volume) assumption holds.
    valid_no_forces : bool array
        True where the no-intermolecular-forces assumption holds.
    """
    free_vol_ratio = (V / n) / VDW_B              # dimensionless
    force_ratio    = (VDW_A * n / V) / (R * T)    # dimensionless

    valid_point_particle = free_vol_ratio > FREE_VOL_THRESHOLD
    valid_no_forces      = force_ratio    < FORCE_THRESHOLD

    return valid_point_particle, valid_no_forces


def generate_regime(
    n_samples: int,
    P_low: float,
    P_high: float,
    T_low: float = T_LOW,
    T_high: float = T_HIGH,
    n_moles: float = 1.0,
    regime_label: str = "train",
    rng: np.random.Generator = None,
    noise_frac: float = 0.005,
) -> pd.DataFrame:
    """
    Sample `n_samples` physical states uniformly in [P_low, P_high] × [T_low, T_high].

    Volume is derived from the ideal gas law (V = nRT/P), making PV=nRT the
    ground truth for the training regime. The same formula is applied in the
    held-out regime — the discrepancy between this V and the true van der Waals
    volume is the signal the predicates must detect, not something pre-encoded
    in the data.

    A small Gaussian noise (noise_frac ≈ 0.5%) is added to every quantity to
    simulate realistic measurement uncertainty.

    Parameters
    ----------
    n_samples    : number of states to generate
    P_low, P_high: pressure range in atm
    T_low, T_high: temperature range in K
    n_moles      : moles of gas (fixed per sample)
    regime_label : 'train' or 'held_out'
    rng          : seeded Generator for reproducibility
    noise_frac   : fractional std of Gaussian noise on each variable

    Returns
    -------
    DataFrame with columns: P, V, T, n, regime, valid_point_particle, valid_no_forces
    """
    if rng is None:
        rng = np.random.default_rng(42)

    P = rng.uniform(P_low, P_high, n_samples)
    T = rng.uniform(T_low, T_high, n_samples)
    n = np.full(n_samples, n_moles, dtype=float)

    V_ideal = (n * R * T) / P  # ideal gas law

    # Add small independent measurement noise to each observable
    P = P * (1.0 + rng.normal(0.0, noise_frac, n_samples))
    V = V_ideal * (1.0 + rng.normal(0.0, noise_frac, n_samples))
    T = T * (1.0 + rng.normal(0.0, noise_frac, n_samples))

    valid_pp, valid_nf = compute_validity_labels(P, V, T, n)

    return pd.DataFrame(
        {
            "P": P,
            "V": V,
            "T": T,
            "n": n,
            "regime": regime_label,
            "valid_point_particle": valid_pp,
            "valid_no_forces": valid_nf,
        }
    )


def generate_dataset(
    n_train: int = 5000,
    n_held_out: int = 2000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate the full training / held-out dataset pair.

    The two RNG states are drawn from a single seeded Generator so that the
    split is fully reproducible without correlating the two regimes.

    Returns
    -------
    train_df    : low-pressure states (P_TRAIN_LOW – P_TRAIN_HIGH atm)
    held_out_df : high-pressure states (P_TEST_LOW – P_TEST_HIGH atm)
    """
    rng = np.random.default_rng(seed)
    train_df = generate_regime(
        n_train, P_TRAIN_LOW, P_TRAIN_HIGH, regime_label="train", rng=rng
    )
    held_out_df = generate_regime(
        n_held_out, P_TEST_LOW, P_TEST_HIGH, regime_label="held_out", rng=rng
    )
    return train_df, held_out_df


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_regime_boundary(
    train_df: pd.DataFrame,
    held_out_df: pd.DataFrame,
    save_path: str = None,
) -> None:
    """
    Produce a two-panel figure showing the P–V regime split and assumption validity.

    Left panel  — P–V scatter, training vs held-out, with the regime boundary
                  drawn as a dashed horizontal line at P = P_TRAIN_HIGH.
    Right panel — same scatter colored by whether ALL ideal-gas assumptions are
                  valid (blue) or at least one is violated (red), plus the regime
                  boundary for reference.

    Both axes use log–log scale to show the full dynamic range of the dataset.

    Parameters
    ----------
    train_df    : DataFrame returned by generate_regime(..., regime_label='train')
    held_out_df : DataFrame returned by generate_regime(..., regime_label='held_out')
    save_path   : if given, save the figure to this path (PNG/PDF/SVG)
    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Ideal Gas Regime Split and Assumption Validity  |  PV = nRT synthetic data",
        fontsize=13,
    )

    # ---- left panel: regime split ----------------------------------------
    ax_left.scatter(
        train_df["V"], train_df["P"],
        s=5, alpha=0.35, color="steelblue", label=f"Training  P = {P_TRAIN_LOW}–{P_TRAIN_HIGH} atm",
    )
    ax_left.scatter(
        held_out_df["V"], held_out_df["P"],
        s=5, alpha=0.35, color="tomato", label=f"Held-out  P = {P_TEST_LOW}–{P_TEST_HIGH} atm",
    )
    ax_left.axhline(
        P_TRAIN_HIGH, color="black", linewidth=1.8, linestyle="--",
        label=f"Regime boundary  P = {P_TRAIN_HIGH} atm",
    )
    ax_left.set_xlabel("Volume  V  (L)")
    ax_left.set_ylabel("Pressure  P  (atm)")
    ax_left.set_title("Training vs Held-out Regime")
    ax_left.set_xscale("log")
    ax_left.set_yscale("log")
    ax_left.legend(fontsize=8, markerscale=3)

    # Annotate regime labels directly on the plot
    ax_left.text(
        0.97, 0.20, "Training\n(ideal gas valid)",
        transform=ax_left.transAxes, ha="right", va="bottom",
        fontsize=8, color="steelblue",
    )
    ax_left.text(
        0.97, 0.80, "Held-out\n(van der Waals regime)",
        transform=ax_left.transAxes, ha="right", va="top",
        fontsize=8, color="tomato",
    )

    # ---- right panel: assumption validity --------------------------------
    all_df = pd.concat([train_df, held_out_df], ignore_index=True)
    both_valid = all_df["valid_point_particle"] & all_df["valid_no_forces"]

    ax_right.scatter(
        all_df.loc[both_valid, "V"],
        all_df.loc[both_valid, "P"],
        s=5, alpha=0.35, color="steelblue",
    )
    ax_right.scatter(
        all_df.loc[~both_valid, "V"],
        all_df.loc[~both_valid, "P"],
        s=5, alpha=0.35, color="tomato",
    )
    ax_right.axhline(
        P_TRAIN_HIGH, color="black", linewidth=1.8, linestyle="--",
    )

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
               markersize=7, label="All assumptions valid"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tomato",
               markersize=7, label="≥1 assumption violated"),
        Line2D([0], [0], color="black", linewidth=1.8, linestyle="--",
               label=f"Regime boundary  P = {P_TRAIN_HIGH} atm"),
    ]
    ax_right.legend(handles=legend_handles, fontsize=8)
    ax_right.set_xlabel("Volume  V  (L)")
    ax_right.set_ylabel("Pressure  P  (atm)")
    ax_right.set_title("Assumption Validity in P–V Space")
    ax_right.set_xscale("log")
    ax_right.set_yscale("log")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_df, held_out_df = generate_dataset(n_train=5000, n_held_out=2000)

    for label, df in [("Training", train_df), ("Held-out", held_out_df)]:
        n = len(df)
        pp = df["valid_point_particle"].mean()
        nf = df["valid_no_forces"].mean()
        p_range = f"{df['P'].min():.1f} – {df['P'].max():.1f}"
        print(f"{label} ({n} samples, P = {p_range} atm)")
        print(f"  valid_point_particle : {pp:.1%}")
        print(f"  valid_no_forces      : {nf:.1%}")
        print()

    visualize_regime_boundary(train_df, held_out_df)

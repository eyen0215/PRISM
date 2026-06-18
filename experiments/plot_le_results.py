"""
Generate two diagnostic figures for the linear elasticity pilot.

  figures/le_provenance_matrix.png  — 4x3 TRUSTED/SUSPECT grid (hardcoded)
  figures/le_decision_boundaries.png — predicate scores vs primary feature
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from validity_predicates.predicate import ValidityPredicate

DATA_DIR = Path(__file__).parent.parent / "data" / "linear_elasticity"
SAVE_DIR = Path(__file__).parent.parent / "validity_predicates" / "saved"
FIG_DIR  = Path(__file__).parent.parent / "figures"

E           = 200e9
sigma_yield = 250e6
rho         = 7800.0
L           = 0.01

# Physical decision boundaries
A1_BOUNDARY = 0.01
A2_BOUNDARY = sigma_yield / E                                    # ~0.00125
A5_BOUNDARY = np.sqrt(0.01 * E / (rho * L**2)) / (2 * np.pi)  # ~8061 Hz


# ---------------------------------------------------------------------------
# Predicate loading
# ---------------------------------------------------------------------------

def load_predicates():
    pred_a1 = ValidityPredicate(
        n_features=1, log_transform_cols=(), feature_cols=["eps_eq"]
    )
    pred_a1.load_state_dict(torch.load(SAVE_DIR / "le_A1.pt", weights_only=False))
    pred_a1.eval()

    pred_a2 = ValidityPredicate(
        n_features=2, log_transform_cols=(0, 1), feature_cols=["eps_eq", "sigma_vm"]
    )
    pred_a2.load_state_dict(torch.load(SAVE_DIR / "le_A2.pt", weights_only=False))
    pred_a2.eval()

    ckpt = torch.load(SAVE_DIR / "le_A5.pt", weights_only=False)
    pred_a5 = ValidityPredicate(
        n_features=1, log_transform_cols=(0,), feature_cols=["frequency"]
    )
    pred_a5.load_state_dict(ckpt["model"])
    a5_shift = float(ckpt["shift"])
    pred_a5.eval()

    return pred_a1, pred_a2, pred_a5, a5_shift


def score_a5(pred, features: np.ndarray, shift: float) -> np.ndarray:
    """Sigmoid with log_criterion_shift applied before thresholding."""
    with torch.no_grad():
        logits = pred(torch.from_numpy(features.astype(np.float32))).numpy()
    return 1.0 / (1.0 + np.exp(-(logits + shift)))


# ---------------------------------------------------------------------------
# Figure 1 — provenance matrix (hardcoded from spec)
# ---------------------------------------------------------------------------

# Row order: D1, D2, D3, D4
# Column order: Scenario A (A1 fires), Scenario B (A2 fires), Scenario C (A5 fires)
# A1 is in {D1,D2,D3,D4} footprints  → all SUSPECT in Scenario A
# A2 is in {D1,D2,D3,D4} footprints  → all SUSPECT in Scenario B
# A5 is only in D4 footprint          → D1/D2/D3 TRUSTED, D4 SUSPECT in Scenario C
PROV_LABELS = [
    "D1  stress field",
    "D2  displacement",
    "D3  strain energy",
    "D4  frequencies",
]
PROV_MATRIX = [
    #  Scen A      Scen B      Scen C
    ["SUSPECT",  "SUSPECT",  "TRUSTED"],   # D1
    ["SUSPECT",  "SUSPECT",  "TRUSTED"],   # D2
    ["SUSPECT",  "SUSPECT",  "TRUSTED"],   # D3
    ["SUSPECT",  "SUSPECT",  "SUSPECT"],   # D4
]
COL_LABELS = [
    "Scenario A\n(large strain, A1 fires)",
    "Scenario B\n(above yield, A2 fires)",
    "Scenario C\n(dynamic, A5 fires)",
]
CELL_COLOR  = {"TRUSTED": "#4caf50", "SUSPECT": "#e53935"}
CELL_TCOLOR = {"TRUSTED": "white",   "SUSPECT": "white"}


def plot_provenance_matrix() -> None:
    n_rows, n_cols = 4, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 5.5))
    fig.suptitle(
        "Provenance Propagation -- Linear Elasticity",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            status = PROV_MATRIX[r][c]
            ax.set_facecolor(CELL_COLOR[status])
            ax.text(
                0.5, 0.5, status,
                ha="center", va="center",
                fontsize=12, fontweight="bold",
                color=CELL_TCOLOR[status],
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(1.8)
                spine.set_color("white")

        # Row label on the left column
        axes[r, 0].set_ylabel(
            PROV_LABELS[r],
            fontsize=9,
            rotation=0,
            ha="right",
            va="center",
            labelpad=8,
        )

    # Column labels on top row
    for c, label in enumerate(COL_LABELS):
        axes[0, c].set_title(label, fontsize=9, pad=5)

    plt.tight_layout(rect=[0.10, 0, 1, 1])  # leave space for row labels

    out = FIG_DIR / "le_provenance_matrix.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2 — decision boundaries
# ---------------------------------------------------------------------------

def _subsample(X: np.ndarray, n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    return idx


def plot_decision_boundaries(pred_a1, pred_a2, pred_a5, a5_shift) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        "Validity Predicate Decision Boundaries -- Linear Elasticity",
        fontsize=12, fontweight="bold",
    )

    # ---- A1 ----------------------------------------------------------------
    ax = axes[0]

    d_tr = np.load(DATA_DIR / "train_A1.npz")
    X_tr = d_tr["features"]
    idx  = _subsample(X_tr)
    sc_tr = pred_a1.predict(X_tr[idx])
    ax.scatter(X_tr[idx, 0], sc_tr, s=10, alpha=0.6, color="steelblue",
               label="Training (valid)", zorder=2)

    d_te = np.load(DATA_DIR / "test_scenario_A.npz")
    X_te = d_te["A1_features"]
    sc_te = pred_a1.predict(X_te)
    ax.scatter(X_te[:, 0], sc_te, s=10, alpha=0.5, color="crimson",
               label="Scenario A (A1 fires)", zorder=3)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2,
               label="Threshold 0.5")
    ax.axvline(A1_BOUNDARY, color="dimgray", linestyle=":", linewidth=1.5,
               label=f"eps_eq = {A1_BOUNDARY}")
    ax.set_xlabel("Equivalent strain (eps_eq)", fontsize=9)
    ax.set_ylabel("Predicate score", fontsize=9)
    ax.set_ylim(-0.05, 1.10)
    ax.set_title("A1 -- small strain", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5, loc="center right")

    # ---- A2 ----------------------------------------------------------------
    ax = axes[1]

    d_tr = np.load(DATA_DIR / "train_A2.npz")
    X_tr = d_tr["features"]
    idx  = _subsample(X_tr)
    sc_tr = pred_a2.predict(X_tr[idx])
    ax.scatter(X_tr[idx, 0], sc_tr, s=10, alpha=0.6, color="steelblue",
               label="Training (valid)", zorder=2)

    d_te = np.load(DATA_DIR / "test_scenario_B.npz")
    X_te = d_te["A2_features"]
    sc_te = pred_a2.predict(X_te)
    ax.scatter(X_te[:, 0], sc_te, s=10, alpha=0.5, color="crimson",
               label="Scenario B (A2 fires)", zorder=3)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2,
               label="Threshold 0.5")
    ax.axvline(A2_BOUNDARY, color="dimgray", linestyle=":", linewidth=1.5,
               label=f"eps_yield = {A2_BOUNDARY:.5f}")
    ax.set_xlabel("Equivalent strain (eps_eq)", fontsize=9)
    ax.set_ylabel("Predicate score", fontsize=9)
    ax.set_ylim(-0.05, 1.10)
    ax.set_title("A2 -- linearity", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5, loc="center right")

    # ---- A5 ----------------------------------------------------------------
    ax = axes[2]

    d_tr = np.load(DATA_DIR / "train_A5.npz")
    X_tr = d_tr["features"]
    idx  = _subsample(X_tr)
    sc_tr = score_a5(pred_a5, X_tr[idx], a5_shift)
    ax.scatter(X_tr[idx, 0], sc_tr, s=10, alpha=0.6, color="steelblue",
               label="Training (valid)", zorder=2)

    d_te = np.load(DATA_DIR / "test_scenario_C.npz")
    X_te = d_te["A5_features"]
    sc_te = score_a5(pred_a5, X_te, a5_shift)
    ax.scatter(X_te[:, 0], sc_te, s=10, alpha=0.5, color="crimson",
               label="Scenario C (A5 fires)", zorder=3)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2,
               label="Threshold 0.5")
    ax.axvline(A5_BOUNDARY, color="dimgray", linestyle=":", linewidth=1.5,
               label=f"f_boundary = {A5_BOUNDARY:.0f} Hz")
    ax.set_xlabel("Frequency (Hz)", fontsize=9)
    ax.set_ylabel("Predicate score", fontsize=9)
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.10)
    ax.set_title("A5 -- quasi-static", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper right")

    plt.tight_layout()

    out = FIG_DIR / "le_decision_boundaries.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading predicates...")
    pred_a1, pred_a2, pred_a5, a5_shift = load_predicates()
    print(f"  A5 shift = {a5_shift:.3f}")

    print("\nFigure 1: provenance matrix...")
    plot_provenance_matrix()

    print("Figure 2: decision boundaries...")
    plot_decision_boundaries(pred_a1, pred_a2, pred_a5, a5_shift)

    print("\nVerifying output files:")
    for name in ["le_provenance_matrix.png", "le_decision_boundaries.png"]:
        p = FIG_DIR / name
        assert p.exists(), f"Missing: {p}"
        size_kb = p.stat().st_size / 1024
        print(f"  {name}  ({size_kb:.0f} KB)")

    print("\nDone.")


if __name__ == "__main__":
    main()

"""
Plot learned decision boundaries for linear skip vs monotone network
on the synthetic exponential breakdown boundary.

Saves: figures/nonlinear_boundary_comparison.png
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
from validity_predicates.monotone_predicate import MonotonePredicate

SAVE_DIR = Path("validity_predicates/saved")
FIG_PATH = Path("figures/nonlinear_boundary_comparison.png")
K = 0.5

# Grid resolution for evaluation (finer than the saved npz)
N_V, N_X = 200, 200
V_MIN, V_MAX = 0.0, 6.0
X_MIN, X_MAX = 0.5, 25.0


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

def load_linear() -> ValidityPredicate:
    ck = torch.load(SAVE_DIR / "linear_nonlinear.pt", map_location="cpu", weights_only=False)
    m  = ValidityPredicate(n_features=2, log_transform_cols=ck.get("log_cols", ()))
    m.load_state_dict(ck["state_dict"])
    m.feat_mean.copy_(torch.tensor(ck["feat_mean"]))
    m.feat_std.copy_(torch.tensor(ck["feat_std"]))
    m.eval()
    return m


def load_mono() -> MonotonePredicate:
    ck = torch.load(SAVE_DIR / "monotone_nonlinear.pt", map_location="cpu", weights_only=False)
    m  = MonotonePredicate(
        n_features=2,
        signs=ck["signs"],
        log_transform_cols=tuple(ck.get("log_transform_cols", ())),
        feature_cols=ck.get("feature_cols", ["x", "v"]),
    )
    m.load_state_dict(ck["state_dict"])
    m.feat_mean.copy_(torch.tensor(ck["feat_mean"]))
    m.feat_std.copy_(torch.tensor(ck["feat_std"]))
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Score grid: Z[i,j] = score at (x_vals[i], v_vals[j])
# pcolormesh(v_vals, x_vals, Z) places Z[i,j] at (v_vals[j], x_vals[i]) ✓
# ---------------------------------------------------------------------------

def score_grid(model, x_vals, v_vals):
    V, X = np.meshgrid(v_vals, x_vals)          # X[i,j]=x_vals[i], V[i,j]=v_vals[j]
    feats = np.column_stack([X.ravel(), V.ravel()]).astype(np.float32)
    scores = model.predict(feats)
    return scores.reshape(len(x_vals), len(v_vals))  # Z[i,j] at (x_vals[i], v_vals[j])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    lin  = load_linear()
    mono = load_mono()

    v_vals = np.linspace(V_MIN, V_MAX, N_V)
    x_vals = np.linspace(X_MIN, X_MAX, N_X)

    print("Scoring linear skip on grid ...")
    Z_lin  = score_grid(lin,  x_vals, v_vals)
    print("Scoring monotone network on grid ...")
    Z_mono = score_grid(mono, x_vals, v_vals)

    # True boundary curve
    v_curve = np.linspace(V_MIN, V_MAX, 500)
    x_true  = np.exp(K * v_curve)          # x = exp(0.5*v)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.88, wspace=0.08)

    cmap   = "RdYlGn"
    vmin, vmax = 0.0, 1.0

    panels = [
        (axes[0], Z_lin,  "Linear skip — fails on exponential boundary"),
        (axes[1], Z_mono, "Monotone network — handles exponential boundary"),
    ]

    for ax, Z, title in panels:
        im = ax.pcolormesh(
            v_vals, x_vals, Z,
            cmap=cmap, vmin=vmin, vmax=vmax,
            shading="auto", rasterized=True,
        )

        # True boundary — black dashed
        mask = (x_true >= X_MIN) & (x_true <= X_MAX)
        ax.plot(
            v_curve[mask], x_true[mask],
            color="black", lw=2.2, ls="--", zorder=3,
            label=r"True boundary  $x = e^{0.5v}$",
        )

        # Learned 0.5-score contour — blue solid
        cs = ax.contour(
            v_vals, x_vals, Z,
            levels=[0.5], colors=["steelblue"], linewidths=[2.0],
            linestyles=["-"], zorder=4,
        )
        # Proxy for legend
        ax.plot([], [], color="steelblue", lw=2.0, ls="-", label="Learned 0.5-score contour")

        ax.set_yscale("log")
        ax.set_xlim(V_MIN, V_MAX)
        ax.set_ylim(X_MIN, X_MAX)
        ax.set_xlabel("v", fontsize=12)
        ax.set_title(title, fontsize=11, pad=8)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
        ax.grid(True, which="major", alpha=0.20, color="white")

    axes[0].set_ylabel("x  (log scale)", fontsize=12)

    # Single colorbar on the right
    cbar_ax = fig.add_axes([0.90, 0.12, 0.018, 0.74])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("Predicate score  (1 = valid, 0 = broken)", fontsize=10)

    caption = (
        "True breakdown boundary is exponential: $x = e^{0.5v}$.\n"
        "A linear layer in raw feature space cannot represent this curve.\n"
        "The monotone network approximates the curve."
    )
    fig.text(0.50, 0.01, caption, ha="center", va="bottom", fontsize=9,
             style="italic", wrap=True)

    FIG_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {FIG_PATH}")

    # Quick sanity print
    for name, Z in [("linear", Z_lin), ("monotone", Z_mono)]:
        frac_above = (Z > 0.5).mean()
        print(f"  {name}: fraction of grid above 0.5 = {frac_above:.3f}")


if __name__ == "__main__":
    main()

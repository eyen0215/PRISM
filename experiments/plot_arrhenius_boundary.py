"""
Clean boundary figure: one large coupled heatmap + two 1-D baseline strips.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import torch

from validity_predicates.predicate import ValidityPredicate
from validity_predicates.monotone_predicate import MonotonePredicate

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, "data", "coupled_arrhenius")
SAVED_DIR = os.path.join(ROOT, "validity_predicates", "saved")
FIG_DIR   = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Physical constants (must match generate_arrhenius_data.py)
# ---------------------------------------------------------------------------

Ea              = 20000.0
R               = 8.314
T_ref           = 300.0
sigma_ref       = 1e6
alpha           = 5.0
epsilon_threshold = 0.05

def _shift(sigma, T):
    return Ea / R * (1.0 / T - 1.0 / T_ref) + alpha * sigma / sigma_ref

def pred_error_fn(sigma, T):
    return np.abs(np.exp(-_shift(sigma, T)) - 1.0)

def sigma_boundary_at_T(T_vals):
    """Upper sigma boundary where pred_error = epsilon_threshold (positive-shift side)."""
    _thr = np.log(1.0 + epsilon_threshold)
    T_arr = np.asarray(T_vals, dtype=float)
    temp_contrib = Ea / R * (1.0 / T_arr - 1.0 / T_ref)
    s_bd = (_thr - temp_contrib) * sigma_ref / alpha
    return np.where(s_bd < 0, np.nan, s_bd)

# ---------------------------------------------------------------------------
# Load training data (for normalization and region outline)
# ---------------------------------------------------------------------------

train_d        = np.load(os.path.join(DATA_DIR, "train_coupled.npz"))
train_features = train_d["features"].astype(np.float32)   # [sigma, T]

sigma_train = train_features[:, 0:1]
T_train     = train_features[:, 1:2]

T_train_lo,   T_train_hi   = float(T_train.min()),     float(T_train.max())
sig_train_lo, sig_train_hi = float(sigma_train.min()), float(sigma_train.max())

# ---------------------------------------------------------------------------
# Load predictors
# ---------------------------------------------------------------------------

def _load_sigma_only():
    p = ValidityPredicate(hidden_dims=(32, 16), n_features=1,
                          log_transform_cols=[], feature_cols=["sigma"])
    p.set_normalization(sigma_train.mean(axis=0), sigma_train.std(axis=0))
    p.load_state_dict(torch.load(os.path.join(SAVED_DIR, "arrhenius_sigma_only.pt"),
                                 weights_only=True))
    p.eval()
    return p

def _load_T_only():
    p = ValidityPredicate(hidden_dims=(32, 16), n_features=1,
                          log_transform_cols=[], feature_cols=["T"])
    p.set_normalization(T_train.mean(axis=0), T_train.std(axis=0))
    p.load_state_dict(torch.load(os.path.join(SAVED_DIR, "arrhenius_T_only.pt"),
                                 weights_only=True))
    p.eval()
    return p

def _load_coupled():
    feat = train_features.astype(np.float32)
    p = MonotonePredicate(n_features=2, signs=[-1, +1],
                          log_transform_cols=(), feature_cols=["sigma", "T"],
                          hidden_dims_mono=(32, 16), hidden_dims_mlp=(16, 8))
    p.set_normalization(feat.mean(axis=0), feat.std(axis=0))
    p.load_state_dict(torch.load(os.path.join(SAVED_DIR, "arrhenius_coupled.pt"),
                                 weights_only=True))
    p.eval()
    return p

pred_sigma   = _load_sigma_only()
pred_T       = _load_T_only()
pred_coupled = _load_coupled()

# ---------------------------------------------------------------------------
# Grid for main heatmap
# ---------------------------------------------------------------------------

T_LO, T_HI     = 270.0, 330.0
SIG_LO, SIG_HI = 0.0,   0.08e6     # Pa

N = 80
T_grid   = np.linspace(T_LO,   T_HI,   N)
sig_grid = np.linspace(SIG_LO, SIG_HI, N)
TT, SS   = np.meshgrid(T_grid, sig_grid)   # rows=sigma, cols=T

flat_T   = TT.ravel().astype(np.float32)
flat_sig = SS.ravel().astype(np.float32)

Z_coupled = pred_coupled.predict(
    np.stack([flat_sig, flat_T], axis=1)
).reshape(N, N)

# ---------------------------------------------------------------------------
# True boundary curve
# ---------------------------------------------------------------------------

T_curve       = np.linspace(T_LO, T_HI, 600)
sig_curve_MPa = sigma_boundary_at_T(T_curve) / 1e6
valid_bd      = np.isfinite(sig_curve_MPa) & (sig_curve_MPa >= 0) & \
                (sig_curve_MPa <= SIG_HI / 1e6)

# ---------------------------------------------------------------------------
# Scenario 3 test points
# ---------------------------------------------------------------------------

s3       = np.load(os.path.join(DATA_DIR, "test_scenario_3.npz"))
feat_s3  = s3["features"]
s3_T     = feat_s3[:, 1]
s3_s_MPa = feat_s3[:, 0] / 1e6

# ---------------------------------------------------------------------------
# 1-D sweep data for the strip plots
# ---------------------------------------------------------------------------

N1D = 200

# sigma-only strip: score vs sigma at T=T_ref
sig_sweep     = np.linspace(SIG_LO, SIG_HI, N1D).astype(np.float32)
scores_sigma  = pred_sigma.predict(sig_sweep[:, None])

# true sigma threshold (vertical reference line)
_thr_shift = np.log(1.0 + epsilon_threshold)
sig_true_threshold_MPa = _thr_shift * sigma_ref / (alpha * 1e6)

# T-only strip: score vs T at sigma=0
T_sweep    = np.linspace(T_LO, T_HI, N1D).astype(np.float32)
scores_T   = pred_T.predict(T_sweep[:, None])

# true T threshold (vertical reference line in T-strip)
T_true_threshold = 1.0 / (1.0 / T_ref + _thr_shift / (Ea / R))

# ---------------------------------------------------------------------------
# Figure layout: GridSpec with 2 cols (wide main | narrow strips)
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(13, 6))
gs  = gridspec.GridSpec(
    2, 2,
    width_ratios=[4, 1.1],
    height_ratios=[1, 1],
    hspace=0.55,
    wspace=0.32,
)

ax_main  = fig.add_subplot(gs[:, 0])    # left: spans both rows
ax_sigma = fig.add_subplot(gs[0, 1])   # top-right strip
ax_T     = fig.add_subplot(gs[1, 1])   # bottom-right strip

CMAP = "RdYlGn"

# ---- Main heatmap -------------------------------------------------------

pc = ax_main.pcolormesh(
    T_grid, sig_grid / 1e6, Z_coupled,
    cmap=CMAP, vmin=0.0, vmax=1.0, shading="auto",
)

# Learned decision boundary (score = 0.5)
try:
    cs = ax_main.contour(T_grid, sig_grid / 1e6, Z_coupled, levels=[0.5],
                         colors=["navy"], linewidths=2.0, linestyles="-")
    ax_main.clabel(cs, fmt={0.5: "score=0.5"}, fontsize=8, inline=True)
except Exception:
    pass

# True boundary (dashed black)
ax_main.plot(T_curve[valid_bd], sig_curve_MPa[valid_bd],
             color="black", linestyle="--", linewidth=1.8,
             label="True boundary (5% pred error)")

# Training region (dotted gray rectangle)
rect = mpatches.FancyBboxPatch(
    (T_train_lo, sig_train_lo / 1e6),
    T_train_hi - T_train_lo,
    (sig_train_hi - sig_train_lo) / 1e6,
    boxstyle="square,pad=0",
    linewidth=1.4, edgecolor="dimgray", facecolor="none", linestyle=":",
    label="Training region",
)
ax_main.add_patch(rect)

# Scenario 3 dots
ax_main.scatter(s3_T, s3_s_MPa, c="orange", s=14, zorder=6,
                alpha=0.85, linewidths=0.3, edgecolors="saddlebrown",
                label="Scenario 3 (joint breakdown)")

ax_main.set_xlim(T_LO, T_HI)
ax_main.set_ylim(SIG_LO / 1e6, SIG_HI / 1e6)
ax_main.set_xlabel("Temperature  T (K)", fontsize=11)
ax_main.set_ylabel("Stress  σ (MPa)", fontsize=11)
ax_main.set_title("Coupled monotone predicate", fontsize=12, fontweight="bold", pad=8)
ax_main.legend(loc="upper right", fontsize=8.5, framealpha=0.85)

cbar = fig.colorbar(pc, ax=ax_main, orientation="vertical",
                    fraction=0.03, pad=0.03, shrink=0.92)
cbar.set_label("Validity score", fontsize=10)
cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
cbar.set_ticklabels(["0.0\n(invalid)", "0.25", "0.5\n(boundary)", "0.75", "1.0\n(valid)"])
cbar.ax.axhline(0.5, color="navy", linewidth=1.4, linestyle="-")

# ---- σ-only strip -------------------------------------------------------

ax_sigma.plot(sig_sweep / 1e6, scores_sigma,
              color="steelblue", linewidth=2.0)
ax_sigma.axhline(0.5, color="navy", linewidth=1.2, linestyle="--", alpha=0.7)
ax_sigma.axvline(sig_true_threshold_MPa, color="black", linewidth=1.2,
                 linestyle="--", alpha=0.7, label="True threshold")
ax_sigma.fill_between(sig_sweep / 1e6, scores_sigma, 0.5,
                      where=(scores_sigma < 0.5),
                      alpha=0.18, color="red", label="Fires (score<0.5)")
# Mark scenario 3 sigma range
ax_sigma.axvspan(s3_s_MPa.min(), s3_s_MPa.max(),
                 alpha=0.22, color="orange", label="Scenario 3 σ range")
ax_sigma.set_xlim(SIG_LO / 1e6, SIG_HI / 1e6)
ax_sigma.set_ylim(-0.05, 1.08)
ax_sigma.set_xlabel("σ (MPa)", fontsize=9)
ax_sigma.set_ylabel("Score", fontsize=9)
ax_sigma.set_title("σ-only predictor\n(at T = T$_{ref}$)", fontsize=9,
                   fontweight="bold")
ax_sigma.tick_params(labelsize=8)
ax_sigma.legend(fontsize=7, loc="lower left", framealpha=0.8)

# ---- T-only strip -------------------------------------------------------

ax_T.plot(T_sweep, scores_T, color="firebrick", linewidth=2.0)
ax_T.axhline(0.5, color="navy", linewidth=1.2, linestyle="--", alpha=0.7)
ax_T.axvline(T_true_threshold, color="black", linewidth=1.2,
             linestyle="--", alpha=0.7, label="True threshold")
ax_T.fill_between(T_sweep, scores_T, 0.5,
                  where=(scores_T < 0.5),
                  alpha=0.18, color="red", label="Fires (score<0.5)")
# Mark scenario 3 T range
ax_T.axvspan(s3_T.min(), s3_T.max(),
             alpha=0.22, color="orange", label="Scenario 3 T range")
ax_T.set_xlim(T_LO, T_HI)
ax_T.set_ylim(-0.05, 1.08)
ax_T.set_xlabel("T (K)", fontsize=9)
ax_T.set_ylabel("Score", fontsize=9)
ax_T.set_title("T-only predictor\n(at σ = 0)", fontsize=9,
               fontweight="bold")
ax_T.tick_params(labelsize=8)
ax_T.legend(fontsize=7, loc="lower right", framealpha=0.8)

# ---- Overall title and caption ------------------------------------------

fig.suptitle(
    "Coupled boundary detection: single-variable predictors miss the joint effect",
    fontsize=13, fontweight="bold", y=1.01,
)
fig.text(
    0.5, -0.03,
    "Single-variable predictors (right) only see their own axis and cannot represent the diagonal boundary.\n"
    "The coupled predicate (left) correctly bends its decision boundary to follow the true joint threshold.",
    ha="center", va="top", fontsize=9, color="dimgray",
    wrap=True,
)

# ---- Save ---------------------------------------------------------------

out_path = os.path.join(FIG_DIR, "arrhenius_coupled_boundary_v2.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved: %s" % out_path)

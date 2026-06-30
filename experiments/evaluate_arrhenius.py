"""
Evaluate three Arrhenius validity predictors across three test scenarios.

Predictor 1 -- sigma-only  : ValidityPredicate, features=[sigma]
Predictor 2 -- T-only      : ValidityPredicate, features=[T]
Predictor 3 -- coupled     : MonotonePredicate, features=[sigma, T], signs=[-1, +1]

Scenarios:
  1 -- high-sigma drives breakdown (T near T_ref)
  2 -- low-T drives breakdown (sigma moderate)
  3 -- joint effect only (KEY TEST: neither alone should fire)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from validity_predicates.predicate import ValidityPredicate
from validity_predicates.monotone_predicate import MonotonePredicate

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "coupled_arrhenius")
SAVED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "validity_predicates", "saved")


# ---------------------------------------------------------------------------
# Load predictors
# ---------------------------------------------------------------------------

def load_sigma_only(train_features):
    """Rebuild ValidityPredicate and restore weights + normalization."""
    sigma_col = train_features[:, 0:1].astype(np.float32)
    pred = ValidityPredicate(
        hidden_dims=(32, 16),
        n_features=1,
        log_transform_cols=[],
        feature_cols=["sigma"],
    )
    pred.set_normalization(sigma_col.mean(axis=0), sigma_col.std(axis=0))
    pred.load_state_dict(torch.load(
        os.path.join(SAVED_DIR, "arrhenius_sigma_only.pt"), weights_only=True))
    pred.eval()
    return pred


def load_T_only(train_features):
    T_col = train_features[:, 1:2].astype(np.float32)
    pred = ValidityPredicate(
        hidden_dims=(32, 16),
        n_features=1,
        log_transform_cols=[],
        feature_cols=["T"],
    )
    pred.set_normalization(T_col.mean(axis=0), T_col.std(axis=0))
    pred.load_state_dict(torch.load(
        os.path.join(SAVED_DIR, "arrhenius_T_only.pt"), weights_only=True))
    pred.eval()
    return pred


def load_coupled(train_features):
    feat = train_features.astype(np.float32)
    pred = MonotonePredicate(
        n_features=2,
        signs=[-1, +1],
        log_transform_cols=(),
        feature_cols=["sigma", "T"],
        hidden_dims_mono=(32, 16),
        hidden_dims_mlp=(16, 8),
    )
    pred.set_normalization(feat.mean(axis=0), feat.std(axis=0))
    pred.load_state_dict(torch.load(
        os.path.join(SAVED_DIR, "arrhenius_coupled.pt"), weights_only=True))
    pred.eval()
    return pred


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

def score_sigma_only(pred, features):
    """Score using sigma column only."""
    sigma_col = features[:, 0:1].astype(np.float32)
    return pred.predict(sigma_col)        # shape (N,), values in (0,1)


def score_T_only(pred, features):
    T_col = features[:, 1:2].astype(np.float32)
    return pred.predict(T_col)


def score_coupled(pred, features):
    return pred.predict(features.astype(np.float32))


def fire_rate(scores):
    """Fraction of samples predicted invalid (score < 0.5)."""
    return float((scores < 0.5).mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ---- Load data ---------------------------------------------------------
    train_d = np.load(os.path.join(DATA_DIR, "train_coupled.npz"))
    train_features = train_d["features"].astype(np.float32)

    # Valid holdout: last 20% of training set
    n_holdout = int(0.2 * len(train_features))
    holdout_features = train_features[-n_holdout:]

    s1 = np.load(os.path.join(DATA_DIR, "test_scenario_1.npz"))
    s2 = np.load(os.path.join(DATA_DIR, "test_scenario_2.npz"))
    s3 = np.load(os.path.join(DATA_DIR, "test_scenario_3.npz"))

    feat_s1 = s1["features"].astype(np.float32)
    feat_s2 = s2["features"].astype(np.float32)
    feat_s3 = s3["features"].astype(np.float32)

    print("Data loaded:")
    print("  Train (total):   %d samples" % len(train_features))
    print("  Valid holdout:   %d samples (last 20%% of train)" % len(holdout_features))
    print("  Scenario 1:      %d samples" % len(feat_s1))
    print("  Scenario 2:      %d samples" % len(feat_s2))
    print("  Scenario 3:      %d samples" % len(feat_s3))
    print()

    # ---- Load predictors ---------------------------------------------------
    pred_sigma  = load_sigma_only(train_features)
    pred_T      = load_T_only(train_features)
    pred_coupled = load_coupled(train_features)

    # ---- Score all splits for all predictors --------------------------------
    # Holdout (valid, label=1)
    sc_hold_sigma   = score_sigma_only(pred_sigma,   holdout_features)
    sc_hold_T       = score_T_only(pred_T,           holdout_features)
    sc_hold_coupled = score_coupled(pred_coupled,    holdout_features)

    # Scenario 1
    sc1_sigma   = score_sigma_only(pred_sigma,   feat_s1)
    sc1_T       = score_T_only(pred_T,           feat_s1)
    sc1_coupled = score_coupled(pred_coupled,    feat_s1)

    # Scenario 2
    sc2_sigma   = score_sigma_only(pred_sigma,   feat_s2)
    sc2_T       = score_T_only(pred_T,           feat_s2)
    sc2_coupled = score_coupled(pred_coupled,    feat_s2)

    # Scenario 3
    sc3_sigma   = score_sigma_only(pred_sigma,   feat_s3)
    sc3_T       = score_T_only(pred_T,           feat_s3)
    sc3_coupled = score_coupled(pred_coupled,    feat_s3)

    # ---- Per-scenario fire rates -------------------------------------------
    print("=== Scenario 1: high-sigma drives breakdown ===")
    print("  sigma-only fire rate:   %.3f  (expect HIGH -- stress is extreme)"    % fire_rate(sc1_sigma))
    print("  T-only fire rate:       %.3f  (expect LOW  -- temperature is normal)" % fire_rate(sc1_T))
    print("  Coupled fire rate:      %.3f  (expect HIGH -- should detect)"         % fire_rate(sc1_coupled))
    print()

    print("=== Scenario 2: low-T drives breakdown ===")
    print("  sigma-only fire rate:   %.3f  (expect LOW  -- stress is moderate)"    % fire_rate(sc2_sigma))
    print("  T-only fire rate:       %.3f  (expect HIGH -- temperature is extreme)" % fire_rate(sc2_T))
    print("  Coupled fire rate:      %.3f  (expect HIGH -- should detect)"          % fire_rate(sc2_coupled))
    print()

    print("=== Scenario 3: joint effect -- KEY TEST ===")
    fr3_sigma   = fire_rate(sc3_sigma)
    fr3_T       = fire_rate(sc3_T)
    fr3_coupled = fire_rate(sc3_coupled)
    print("  sigma-only fire rate:   %.3f  (expect LOW  -- stress alone insufficient)"  % fr3_sigma)
    print("  T-only fire rate:       %.3f  (expect LOW  -- temperature alone insufficient)" % fr3_T)
    print("  Coupled fire rate:      %.3f  (expect HIGH -- only joint detection works)" % fr3_coupled)
    print()

    # ---- Success criteria from CLAUDE_COUPLED_ARRHENIUS.md -----------------
    print("=== Scenario 3 success criteria ===")
    p1 = "PASS" if fr3_sigma   < 0.20 else "FAIL"
    p2 = "PASS" if fr3_T       < 0.20 else "FAIL"
    p3 = "PASS" if fr3_coupled > 0.70 else "FAIL"
    print("  Scenario 3 sigma-only  < 0.20: %s  (%.3f)" % (p1, fr3_sigma))
    print("  Scenario 3 T-only      < 0.20: %s  (%.3f)" % (p2, fr3_T))
    print("  Scenario 3 coupled     > 0.70: %s  (%.3f)" % (p3, fr3_coupled))
    all_pass = (p1 == "PASS" and p2 == "PASS" and p3 == "PASS")
    print("  Overall: %s" % ("ALL PASS -- coupled boundary detection works" if all_pass
                              else "PARTIAL/FAIL -- see individual results"))
    print()

    # ---- AUROC (pooled across all three test scenarios vs holdout) ----------
    # Valid holdout label=1 (valid), test scenarios label=0 (invalid)
    n_hold = len(holdout_features)

    def pooled_auroc(sc_hold, sc_s1, sc_s2, sc_s3):
        scores = np.concatenate([sc_hold, sc_s1, sc_s2, sc_s3])
        labels = np.concatenate([
            np.ones(n_hold),
            np.zeros(len(sc_s1)),
            np.zeros(len(sc_s2)),
            np.zeros(len(sc_s3)),
        ])
        return roc_auc_score(labels, scores)

    auroc_sigma   = pooled_auroc(sc_hold_sigma,   sc1_sigma,   sc2_sigma,   sc3_sigma)
    auroc_T       = pooled_auroc(sc_hold_T,       sc1_T,       sc2_T,       sc3_T)
    auroc_coupled = pooled_auroc(sc_hold_coupled, sc1_coupled, sc2_coupled, sc3_coupled)

    print("=== AUROC (pooled across all three test scenarios vs valid holdout) ===")
    print("%-16s | %-6s | %s" % ("Predictor", "AUROC", "Notes"))
    print("-" * 52)
    print("%-16s | %.3f  | baseline" % ("sigma-only",    auroc_sigma))
    print("%-16s | %.3f  | baseline" % ("T-only",        auroc_T))
    print("%-16s | %.3f  | key predictor" % ("Coupled",  auroc_coupled))
    print()

    # ---- False positive rates on valid holdout ------------------------------
    print("=== False positive rates on valid holdout (fire rate on known-valid samples) ===")
    print("  sigma-only FPR:  %.3f" % fire_rate(sc_hold_sigma))
    print("  T-only FPR:      %.3f" % fire_rate(sc_hold_T))
    print("  Coupled FPR:     %.3f" % fire_rate(sc_hold_coupled))

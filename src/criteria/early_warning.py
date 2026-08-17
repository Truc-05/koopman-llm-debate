# src/criteria/early_warning.py
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from src.koopman.spectrum import spectral_decomposition, spectral_gap, eigenfunctions
from src.criteria.truth_alignment import truth_alignment_index


def early_warning_features(K_early, dictionary, Z_early, h_star_values):
    """Spectral features from an early-round operator fit (Thm 3).

    NOTE: fitting a fresh K from 2-3 rounds of ONE debate is data-starved
    (S << M). Prefer a K fit pooled across debates, then per-debate features
    via per_debate_mode_ratios below."""
    eigvals, V, W = spectral_decomposition(K_early)
    gap = spectral_gap(eigvals)
    Phi = eigenfunctions(dictionary, W, Z_early)
    phi1 = Phi[:, 0]
    phi2 = Phi[:, 1] if Phi.shape[1] > 1 else np.zeros_like(phi1)
    tau1 = truth_alignment_index(h_star_values, phi1)
    tau2 = truth_alignment_index(h_star_values, phi2)
    return {
        "lambda1": eigvals[0],
        "lambda2": eigvals[1] if len(eigvals) > 1 else None,
        "abs_lambda2": float(np.abs(eigvals[1])) if len(eigvals) > 1 else 0.0,
        "spectral_gap": gap,
        "tau1": tau1,
        "tau2": tau2,
    }


def per_debate_mode_ratios(W, dictionary, Z_debate, n_modes=3, eps=1e-9):
    """Per-debate effective decay of each GLOBAL Koopman mode.

    W comes from a shared K (fit on training debates). For debate rounds
    z_0..z_T, the ratio |phi_i(z_{t+1})| / |phi_i(z_t)| estimates the decay
    this debate actually exhibits along mode i (median over rounds).
    Herding debates keep the subdominant ratio near 1. Works from 2-3 rounds
    because no operator is re-fit per debate."""
    Phi = eigenfunctions(dictionary, W, np.atleast_2d(Z_debate))
    n = min(n_modes, Phi.shape[1])
    ratios = np.empty(n)
    for i in range(n):
        num = np.abs(Phi[1:, i])
        den = np.maximum(np.abs(Phi[:-1, i]), eps)
        ratios[i] = float(np.median(num / den))
    return ratios


def herding_classifier_score(features_list):
    """Doc heuristic: score = |lambda_2| - tau1 (high => herding likely)."""
    scores = []
    for f in features_list:
        gap = f["spectral_gap"] if f["spectral_gap"] is not None else 0.0
        scores.append((1.0 - gap) - f["tau1"])
    return np.array(scores)


def evaluate_auc(scores, labels):
    auc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return auc, fpr, tpr, thresholds

# tests/test_criteria.py
import numpy as np

from src.criteria.truth_alignment import truth_alignment_index, truth_vs_anchor
from src.criteria.consensus import check_consensus, consensus_value
from src.criteria.early_warning import (early_warning_features,
                                        herding_classifier_score, evaluate_auc,
                                        per_debate_mode_ratios)
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import build_snapshots, fit_edmd
from src.koopman.spectrum import spectral_decomposition


def test_truth_alignment_basic():
    h = np.array([1.0, 0.0, 1.0, 0.0])
    assert np.isclose(truth_alignment_index(h, h), 1.0)
    assert np.isclose(truth_alignment_index(h, 3.7 * h), 1.0)  # scale invariant
    orth = np.array([1.0, 0.0, -1.0, 0.0])
    assert np.isclose(truth_alignment_index(h, orth), 0.0, atol=1e-12)
    tau_t, tau_a = truth_vs_anchor(h, h, orth)
    assert tau_t > tau_a


def test_check_consensus():
    assert check_consensus(np.diag([1.0, 0.6, 0.2]))["converges"]
    # slow second mode sitting on the circle => herding-like, no unique consensus
    res = check_consensus(np.diag([1.0, 0.9999, 0.2]), tol=1e-3)
    assert not res["converges"]
    # unstable spectrum
    assert not check_consensus(np.diag([1.2, 1.0, 0.2]))["converges"]


def test_consensus_value_linear_system():
    # z' = A z with A = diag(0.5, 0.4): unique fixed point 0; the only persistent
    # observable is the constant 1, so any g = c0 + <b, z> converges to c0.
    A = np.diag([0.5, 0.4])
    rng = np.random.default_rng(0)
    Z_t = rng.normal(size=(400, 2))
    d = PolynomialDictionary(degree=1)
    Psi_X, Psi_Y = build_snapshots(d, Z_t, Z_t @ A.T)
    K, _, _ = fit_edmd(Psi_X, Psi_Y, reg=1e-10)
    z0 = np.array([2.0, -1.0])
    g = np.array([3.0, 1.0, -2.0])  # g(z) = 3 + z1 - 2 z2  ->  3
    val = consensus_value(K, d, z0, obs_coeffs=g)
    assert np.isclose(np.real(val), 3.0, atol=1e-6)


def _fit_regime(A, seed=0):
    rng = np.random.default_rng(seed)
    Z_t = rng.normal(size=(400, 2))
    d = PolynomialDictionary(degree=1)
    Psi_X, Psi_Y = build_snapshots(d, Z_t, Z_t @ A.T)
    K, _, _ = fit_edmd(Psi_X, Psi_Y, reg=1e-10)
    return K, d, Z_t


def test_early_warning_separates_slow_from_fast():
    # herding-like: subdominant mode 0.98 (slow, persists); truth-like: 0.3
    K_herd, d, Z = _fit_regime(np.diag([0.98, 0.30]))
    K_truth, _, _ = _fit_regime(np.diag([0.20, 0.30]))

    h_herd = Z[:, 0] - Z[:, 0].mean()      # truth uncorrelated with consensus dir
    h_truth = np.ones(Z.shape[0]) + 0.01 * Z[:, 1]  # tracks the invariant

    f_herd = early_warning_features(K_herd, d, Z, h_herd)
    f_truth = early_warning_features(K_truth, d, Z, h_truth)
    assert f_herd["abs_lambda2"] > f_truth["abs_lambda2"]

    scores = herding_classifier_score([f_herd, f_truth])
    assert scores[0] > scores[1]
    auc, _, _, _ = evaluate_auc(np.array([scores[0], scores[1]]), np.array([1, 0]))
    assert auc == 1.0


def test_per_debate_mode_ratios():
    A = np.diag([0.95, 0.4])
    K, d, _ = _fit_regime(A)
    _, _, W = spectral_decomposition(K)
    # one debate trajectory under the same dynamics
    z = np.array([1.0, 1.0])
    Z_debate = [z]
    for _ in range(4):
        z = A @ z
        Z_debate.append(z)
    ratios = per_debate_mode_ratios(W, d, np.array(Z_debate), n_modes=3)
    # sorted modes: lam = 1 (constant), 0.95, 0.4 -> observed decays match
    assert np.allclose(ratios, [1.0, 0.95, 0.4], atol=1e-6)

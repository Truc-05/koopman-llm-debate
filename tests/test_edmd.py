# tests/test_edmd.py
import numpy as np

from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import build_snapshots, fit_edmd, fit_edmdc, rollout_states
from src.koopman.dmd_reduced import fit_dmd_reduced, modes_from_reduced

A = np.array([[0.9, 0.1],
              [0.0, 0.5]])


def _linear_pairs(S=500, seed=0):
    rng = np.random.default_rng(seed)
    Z_t = rng.normal(size=(S, 2))
    Z_tp1 = Z_t @ A.T
    return Z_t, Z_tp1


def test_edmd_recovers_linear_spectrum():
    # With a degree-1 dictionary [1, z], the lifted dynamics of z' = Az is
    # exactly blkdiag(1, A), so EDMD must recover {1} U eig(A).
    Z_t, Z_tp1 = _linear_pairs()
    d = PolynomialDictionary(degree=1)
    Psi_X, Psi_Y = build_snapshots(d, Z_t, Z_tp1)
    K, _, _ = fit_edmd(Psi_X, Psi_Y, reg=1e-10)
    got = np.sort(np.abs(np.linalg.eigvals(K)))
    expected = np.sort(np.abs(np.concatenate([[1.0], np.linalg.eigvals(A)])))
    assert np.allclose(got, expected, atol=1e-6)


def test_rollout_matches_true_linear_dynamics():
    Z_t, Z_tp1 = _linear_pairs()
    d = PolynomialDictionary(degree=1)
    Psi_X, Psi_Y = build_snapshots(d, Z_t, Z_tp1)
    K, _, _ = fit_edmd(Psi_X, Psi_Y, reg=1e-10)
    z0 = np.array([1.0, -2.0])
    pred = rollout_states(K, d, z0, steps=5)
    true = [z0]
    for _ in range(5):
        true.append(A @ true[-1])
    assert np.allclose(np.real(pred), np.array(true), atol=1e-6)


def test_edmdc_recovers_input_matrix():
    rng = np.random.default_rng(1)
    B = np.array([[0.7], [-0.3]])
    S = 600
    Z_t = rng.normal(size=(S, 2))
    U = rng.normal(size=(1, S))
    Z_tp1 = Z_t @ A.T + (B @ U).T
    d = PolynomialDictionary(degree=1)
    Psi_X, Psi_Y = build_snapshots(d, Z_t, Z_tp1)
    K, B_lift = fit_edmdc(Psi_X, Psi_Y, U, reg=1e-10)
    assert np.allclose(K[1:, 1:], A, atol=1e-6)      # state block
    assert np.allclose(B_lift[1:, :], B, atol=1e-6)  # input enters state rows
    assert np.allclose(B_lift[0, :], 0.0, atol=1e-6)  # constant row untouched


def test_dmd_reduced_matches_edmd_spectrum():
    Z_t, Z_tp1 = _linear_pairs()
    d = PolynomialDictionary(degree=2)
    Psi_X, Psi_Y = build_snapshots(d, Z_t, Z_tp1)
    K_tilde, U, _, _ = fit_dmd_reduced(Psi_X, Psi_Y)
    eigvals, _ = modes_from_reduced(K_tilde, U)
    # degree-2 lift of a linear map is exactly closed: eigenvalues are all
    # products lam_i * lam_j of {1, 0.9, 0.5}
    base = np.array([1.0, 0.9, 0.5])
    expected = sorted({base[i] * base[j] for i in range(3) for j in range(i, 3)},
                      reverse=True)
    assert np.allclose(np.sort(np.abs(eigvals))[::-1], expected, atol=1e-5)

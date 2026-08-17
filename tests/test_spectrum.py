# tests/test_spectrum.py
import numpy as np

from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import build_snapshots, fit_edmd
from src.koopman.spectrum import (spectral_decomposition, eigenfunctions,
                                  spectral_gap, is_unique_consensus)

A = np.array([[0.9, 0.1],
              [0.0, 0.5]])


def test_eigenfunction_property():
    # The defining property phi_i(z_{t+1}) = lambda_i phi_i(z_t) must hold on
    # held-out data. This fails if right eigenvectors are used instead of left.
    rng = np.random.default_rng(0)
    Z_t = rng.normal(size=(500, 2))
    Z_tp1 = Z_t @ A.T
    d = PolynomialDictionary(degree=2)
    Psi_X, Psi_Y = build_snapshots(d, Z_t, Z_tp1)
    K, _, _ = fit_edmd(Psi_X, Psi_Y, reg=1e-10)
    eigvals, V, W = spectral_decomposition(K)

    Z_test = rng.normal(size=(200, 2))
    Phi_X = eigenfunctions(d, W, Z_test)
    Phi_Y = eigenfunctions(d, W, Z_test @ A.T)
    for i, lam in enumerate(eigvals):
        if np.abs(lam) < 0.05:
            continue
        resid = np.linalg.norm(Phi_Y[:, i] - lam * Phi_X[:, i])
        scale = max(np.linalg.norm(Phi_X[:, i]), 1e-12)
        assert resid / scale < 1e-5, f"mode {i} (lam={lam}) violates K phi = lam phi"


def test_spectral_gap():
    eigvals = np.array([1.0, 0.8, 0.3])
    assert np.isclose(spectral_gap(eigvals), 0.2)
    assert spectral_gap(np.array([1.0])) is None


def test_is_unique_consensus():
    assert is_unique_consensus(np.array([1.0, 0.5, -0.2]))
    # second eigenvalue on the unit circle (herding-like) -> not unique
    assert not is_unique_consensus(np.array([1.0, 0.9995, 0.3]), tol=1e-3)
    # eigenvalue at modulus 1 but NOT at 1 (rotation) -> not consensus
    assert not is_unique_consensus(np.array([1.0, np.exp(0.5j)]))
    assert not is_unique_consensus(np.array([1.0, -1.0, 0.2]))
    # no eigenvalue at 1 at all
    assert not is_unique_consensus(np.array([0.95, 0.5]))

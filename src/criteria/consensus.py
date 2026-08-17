# src/criteria/consensus.py
import numpy as np
from src.koopman.spectrum import spectral_decomposition, is_unique_consensus


def check_consensus(K, tol=1e-3):
    """Prop 1 check: one simple eigenvalue at 1, everything else strictly inside
    the unit disk (with numerical tolerance — fitted spectra hover around 1)."""
    eigvals, V, W = spectral_decomposition(K)
    unique = is_unique_consensus(eigvals, tol=tol)
    if len(eigvals) > 1:
        rest_stable = bool(np.all(np.abs(eigvals[1:]) < 1.0 - tol))
    else:
        rest_stable = True
    return {
        "eigvals": eigvals,
        "unique_consensus": unique,
        "rest_stable": rest_stable,
        "converges": bool(unique and rest_stable),
    }


def consensus_value(K, dictionary, z0, obs_coeffs=None):
    """Prop 1 payoff: the limit of an observable g = obs_coeffs @ Psi along the
    debate started at z0.

    Psi_t = V diag(lam^t) W Psi_0  --t->inf-->  v_1 * phi_1(z0)   (lam_1 = 1)
    so   g_inf = (obs_coeffs @ v_1) * phi_1(z0).

    obs_coeffs=None returns the full lifted fixed point Psi_inf (decode its
    state block for the consensus state)."""
    eigvals, V, W = spectral_decomposition(K)
    psi0 = dictionary.transform(np.asarray(z0).reshape(1, -1))[0]
    phi1_z0 = W[0] @ psi0
    psi_inf = V[:, 0] * phi1_z0
    if obs_coeffs is None:
        return psi_inf
    return obs_coeffs @ psi_inf

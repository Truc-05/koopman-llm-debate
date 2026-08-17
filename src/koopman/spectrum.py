# src/koopman/spectrum.py
import numpy as np


def eigendecompose(K):
    """Right eigen-decomposition of K, sorted by |lambda| descending."""
    eigvals, eigvecs = np.linalg.eig(K)
    order = np.argsort(-np.abs(eigvals))
    return eigvals[order], eigvecs[:, order]


def spectral_decomposition(K):
    """Full bi-orthogonal decomposition K = V diag(lam) W  (W = V^{-1}).

    With the convention psi_{t+1} = K psi_t, Koopman eigenfunctions are the
    LEFT eigenvectors:  phi_i(z) = W[i, :] @ Psi(z), since
        W Psi(z_{t+1}) = W K Psi(z_t) = diag(lam) W Psi(z_t).
    (Using right eigenvectors here is a common — and wrong — shortcut.)
    """
    eigvals, V = eigendecompose(K)
    try:
        W = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        W = np.linalg.pinv(V)
    return eigvals, V, W


def eigenfunctions(dictionary, W, Z):
    """Eigenfunction values at samples Z: (S, M); column i = phi_i(Z).

    W must be the left-eigenvector matrix from spectral_decomposition (rows)."""
    Psi = dictionary.transform(np.atleast_2d(Z))
    return Psi @ W.T


def leading_eigenvalue(eigvals):
    return eigvals[0]


def spectral_gap(eigvals):
    """1 - |lambda_2| (eigvals sorted by modulus descending)."""
    if len(eigvals) < 2:
        return None
    return 1.0 - float(np.abs(eigvals[1]))


def slow_subspace_indices(eigvals, delta=0.05):
    """Chỉ số các mode chậm/đồng thuận: |λ| ≥ 1 − delta (eigvals đã sort giảm dần)."""
    return np.where(np.abs(np.asarray(eigvals)) >= 1.0 - delta)[0]


def project_stable(K, cap=1.0):
    """stable-DMD: kéo mọi trị riêng |λ|>cap về đúng đường tròn |λ|=cap.

    Trả K thực đã ổn định (|λ|≤cap), dùng cho phổ/claim Prop 1 và rollout nhiều
    bước. pinv để chịu được V suy biến."""
    evals, V = np.linalg.eig(K)
    mods = np.abs(evals)
    evals_s = np.where(mods > cap, evals * (cap / mods), evals)
    return np.real(V @ np.diag(evals_s) @ np.linalg.pinv(V))


def is_unique_consensus(eigvals, tol=1e-3):
    """Prop 1 condition: exactly ONE simple eigenvalue AT 1 (not just modulus 1),
    and no other eigenvalue on the unit circle."""
    eigvals = np.asarray(eigvals)
    at_one = np.abs(eigvals - 1.0) < tol
    on_circle = np.abs(np.abs(eigvals) - 1.0) < tol
    return int(at_one.sum()) == 1 and int(on_circle.sum()) == 1

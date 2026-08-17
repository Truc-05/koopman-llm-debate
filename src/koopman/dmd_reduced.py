# src/koopman/dmd_reduced.py
import numpy as np


def fit_dmd_reduced(Psi_X, Psi_Y, rank=None, sv_tol=1e-10):
    """SVD-projected DMD (numerically stable EDMD for tall dictionaries).

    rank=None picks it automatically by discarding singular values below
    sv_tol * s_max — this is what prevents 1/S blow-ups on rank-deficient data.
    """
    U, S, Vt = np.linalg.svd(Psi_X, full_matrices=False)
    if rank is None:
        rank = int(np.sum(S > sv_tol * S[0]))
    rank = max(1, min(rank, len(S)))
    U = U[:, :rank]
    S = S[:rank]
    Vt = Vt[:rank, :]
    V = Vt.T
    K_tilde = U.T @ Psi_Y @ V @ np.diag(1.0 / S)
    return K_tilde, U, S, V


def full_operator(K_tilde, U):
    """Rank-truncated Koopman matrix back in dictionary space: U K~ U^T.
    Use this for stable multi-step rollouts (truncation kills the spurious
    |lambda|>1 noise modes a raw EDMD fit picks up)."""
    return U @ K_tilde @ U.T


def modes_from_reduced(K_tilde, U):
    """Eigenvalues + full-space modes lifted from the reduced operator."""
    eigvals, Wr = np.linalg.eig(K_tilde)
    order = np.argsort(-np.abs(eigvals))
    eigvals = eigvals[order]
    Phi = U @ Wr[:, order]
    return eigvals, Phi

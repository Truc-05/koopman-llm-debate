# src/koopman/edmd.py
import numpy as np


def build_snapshots(dictionary, Z_t, Z_tp1):
    """Z_t, Z_tp1: (S, D) sample matrices -> Psi_X, Psi_Y: (M, S) lifted snapshots."""
    Psi_X = dictionary.transform(Z_t).T
    Psi_Y = dictionary.transform(Z_tp1).T
    return Psi_X, Psi_Y


def fit_edmd(Psi_X, Psi_Y, reg=1e-8):
    """Least-squares Koopman matrix: min ||Psi_Y - K Psi_X||_F^2 + reg||K||^2.

    K = A G^{-1} with G = Psi_X Psi_X^T (Gram), A = Psi_Y Psi_X^T.
    Solved via np.linalg.solve (G symmetric PD after ridge) — never form G^{-1}.
    """
    M = Psi_X.shape[0]
    G = Psi_X @ Psi_X.T + reg * np.eye(M)
    A = Psi_Y @ Psi_X.T
    K = np.linalg.solve(G, A.T).T
    return K, G, A


def fit_edmdc(Psi_X, Psi_Y, U, reg=1e-8):
    """EDMD with control inputs (Koopman with inputs):

        Psi_Y ≈ K Psi_X + B U,   U: (u_dim, S) input snapshots.

    Returns (K, B). This is what Koopman-MPC/LQR need to get B from data.
    """
    M = Psi_X.shape[0]
    u_dim = U.shape[0]
    Omega = np.vstack([Psi_X, U])
    G = Omega @ Omega.T + reg * np.eye(M + u_dim)
    A = Psi_Y @ Omega.T
    KB = np.linalg.solve(G, A.T).T
    return KB[:, :M], KB[:, M:]


def lift(dictionary, Z):
    """(S, D) states -> (M, S) lifted columns."""
    return dictionary.transform(np.atleast_2d(Z)).T


def unlift(dictionary, Psi):
    """(M, S) lifted columns -> (S, D) states, via the dictionary's linear block.

    Requires the dictionary to expose `state_slice` (where the raw state z sits
    inside Psi(z)); all built-in dictionaries do.
    """
    sl = getattr(dictionary, "state_slice", None)
    if sl is None:
        raise ValueError("dictionary has no state_slice; cannot recover z from psi")
    return Psi[sl, :].T


def predict_next(K, dictionary, z_t):
    """One-step prediction in lifted space; column vector (M, 1)."""
    psi = lift(dictionary, np.asarray(z_t).reshape(1, -1))
    return K @ psi


def rollout(K, dictionary, z0, steps, relift=False):
    """Roll `steps` ahead in lifted space, returns (M, steps+1).

    relift=False: pure linear K^t psi0 — the object spectral analysis talks about.
    relift=True:  decode z each step and re-encode (keeps psi on the dictionary
                  manifold; use this variant for multi-step fidelity checks).
    """
    psi = lift(dictionary, np.asarray(z0).reshape(1, -1))
    traj = [psi]
    for _ in range(steps):
        psi = K @ psi
        if relift:
            psi = lift(dictionary, unlift(dictionary, psi))
        traj.append(psi)
    return np.concatenate(traj, axis=1)


def rollout_states(K, dictionary, z0, steps, relift=False):
    """Same as rollout but decoded back to state space: (steps+1, D)."""
    return unlift(dictionary, rollout(K, dictionary, z0, steps, relift=relift))

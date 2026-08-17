# src/koopman/stochastic.py
import numpy as np
from src.koopman.edmd import fit_edmd, build_snapshots


def ensemble_edmd(dictionary, trajectories, n_bootstrap=20, reg=1e-8, seed=0):
    """Bootstrap over debates (resample whole trajectories, not single pairs —
    pairs within a debate are dependent). Returns mean/std of K and all K's."""
    rng = np.random.default_rng(seed)
    Ks = []
    n_traj = len(trajectories)
    for _ in range(n_bootstrap):
        idx = rng.choice(n_traj, n_traj, replace=True)
        Z_t = np.concatenate([trajectories[i][0] for i in idx], axis=0)
        Z_tp1 = np.concatenate([trajectories[i][1] for i in idx], axis=0)
        Psi_X, Psi_Y = build_snapshots(dictionary, Z_t, Z_tp1)
        K, _, _ = fit_edmd(Psi_X, Psi_Y, reg=reg)
        Ks.append(K)
    K_mean = np.mean(Ks, axis=0)
    K_std = np.std(Ks, axis=0)
    return K_mean, K_std, Ks


def ensemble_spectrum(Ks):
    """Bootstrap uncertainty of the early-warning quantities |lambda_2| and gap."""
    from src.koopman.spectrum import eigendecompose
    lam2 = []
    for K in Ks:
        ev, _ = eigendecompose(K)
        lam2.append(np.abs(ev[1]) if len(ev) > 1 else np.nan)
    lam2 = np.asarray(lam2, dtype=float)
    return {
        "abs_lambda2_mean": float(np.nanmean(lam2)),
        "abs_lambda2_std": float(np.nanstd(lam2)),
        "gap_mean": float(1.0 - np.nanmean(lam2)),
    }

# src/baselines/degroot.py
import numpy as np


def degroot_update(x_t, W):
    return W @ x_t


def degroot_rollout(x0, W, steps):
    x_t = x0.copy()
    traj = [x_t.copy()]
    for _ in range(steps):
        x_t = degroot_update(x_t, W)
        traj.append(x_t.copy())
    return np.array(traj)


def fit_degroot(trajs, reg=1e-6, row_stochastic=False):
    """Least-squares W from data: min_W sum || x_{t+1} - W x_t ||_F^2.

    trajs: list of (T, N, K) logit trajectories. row_stochastic=True projects
    onto nonnegative rows summing to 1 (classic DeGroot constraint)."""
    N = trajs[0].shape[1]
    Xs = np.concatenate([X[:-1].transpose(1, 0, 2).reshape(N, -1) for X in trajs], axis=1)
    Ys = np.concatenate([X[1:].transpose(1, 0, 2).reshape(N, -1) for X in trajs], axis=1)
    G = Xs @ Xs.T + reg * np.eye(N)
    W = np.linalg.solve(G, Xs @ Ys.T).T
    if row_stochastic:
        W = np.clip(W, 0.0, None)
        rows = W.sum(axis=1, keepdims=True)
        rows[rows == 0] = 1.0
        W = W / rows
    return W


def degroot_consensus(W):
    eigvals, eigvecs = np.linalg.eig(W.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    v = np.real(eigvecs[:, idx])
    v = v / np.sum(v)
    return v

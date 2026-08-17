# src/baselines/friedkin_johnsen.py
import numpy as np


def fj_update(x0, x_t, W, lam):
    """x_{t+1} = lam W x_t + (1-lam) x0.  x: (N, K) logits, W: (N, N)."""
    return lam * (W @ x_t) + (1 - lam) * x0


def fj_rollout(x0, W, lam, steps):
    x_t = x0.copy()
    traj = [x_t.copy()]
    for _ in range(steps):
        x_t = fj_update(x0, x_t, W, lam)
        traj.append(x_t.copy())
    return np.array(traj)


def fit_fj(trajs, lam_grid=None, reg=1e-6):
    """Actually FIT the FJ model (both W and lam), unlike a fixed uniform W.

    trajs: list of (T, N, K) logit trajectories; x0 of each debate = trajs[i][0].
    For each lam on the grid, W has the closed-form ridge solution of
        min_W sum_t || (x_{t+1} - (1-lam) x0) - lam W x_t ||_F^2
    shared across debates; the best (lam, W) by SSE is returned.
    """
    if lam_grid is None:
        lam_grid = np.linspace(0.05, 0.95, 19)
    best_lam, best_W, best_err = None, None, np.inf
    N = trajs[0].shape[1]
    for lam in lam_grid:
        Xs, Ys = [], []
        for X in trajs:
            x0 = X[0]
            for t in range(X.shape[0] - 1):
                Xs.append(X[t])
                Ys.append(X[t + 1] - (1 - lam) * x0)
        Xcat = np.concatenate(Xs, axis=1)  # (N, K*S)
        Ycat = np.concatenate(Ys, axis=1)
        G = Xcat @ Xcat.T + reg * np.eye(N)
        W = np.linalg.solve(G, Xcat @ Ycat.T).T / lam
        err = float(np.sum((lam * (W @ Xcat) - Ycat) ** 2))
        if err < best_err:
            best_lam, best_W, best_err = float(lam), W, err
    return best_lam, best_W, best_err

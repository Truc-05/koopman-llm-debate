# src/baselines/hidden_anchors.py
import numpy as np

def anchor_update(x0, x_t, W, lam, anchor_idx, anchor_strength):
    base = lam * (W @ x_t) + (1 - lam) * x0
    base[anchor_idx] = anchor_strength * x0[anchor_idx] + (1 - anchor_strength) * base[anchor_idx]
    return base

def anchor_rollout(x0, W, lam, anchor_idx, anchor_strength, steps):
    x_t = x0.copy()
    traj = [x_t.copy()]
    for _ in range(steps):
        x_t = anchor_update(x0, x_t, W, lam, anchor_idx, anchor_strength)
        traj.append(x_t.copy())
    return np.array(traj)
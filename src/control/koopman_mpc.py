# src/control/koopman_mpc.py
import numpy as np
from scipy.optimize import minimize


def target_psi(dictionary, n_agents, n_answers, a_star, confidence=5.0):
    """Lifted target state: every agent confident in the true answer a_star."""
    x = np.full((n_agents, n_answers), -confidence, dtype=float)
    x[:, a_star] = confidence
    return dictionary.transform(x.reshape(1, -1))[0]


def mpc_step(K, B, psi_current, psi_target, horizon=5, R=1.0, Q=None):
    """Receding-horizon control on the LINEAR lifted model psi' = K psi + B u.

    K, B come from fit_edmdc (src/koopman/edmd.py). psi vectors are 1-D (M,).
    Returns the first input of the optimal sequence."""
    M = K.shape[0]
    u_dim = B.shape[1]
    if Q is None:
        Q = np.eye(M)

    def cost(u_seq):
        psi = psi_current.copy()
        total = 0.0
        for h in range(horizon):
            u = u_seq[h * u_dim:(h + 1) * u_dim]
            psi = K @ psi + B @ u
            err = psi - psi_target
            total += err @ Q @ err + R * (u @ u)
        return float(total)

    res = minimize(cost, np.zeros(horizon * u_dim), method="L-BFGS-B")
    return res.x[:u_dim], res


def apply_control(K, B, psi_current, u):
    return K @ psi_current + B @ u

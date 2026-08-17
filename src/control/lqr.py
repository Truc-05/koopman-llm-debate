# src/control/lqr.py
import numpy as np
from scipy.linalg import solve_discrete_are

def lqr_gain(K, B, Q, R):
    P = solve_discrete_are(K, B, Q, R)
    Gain = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ K)
    return Gain, P

def lqr_control(K, B, Gain, psi_current, psi_target):
    u = -Gain @ (psi_current - psi_target)
    psi_next = K @ psi_current + B @ u
    return u, psi_next
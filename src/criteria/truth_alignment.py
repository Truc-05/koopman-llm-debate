# src/criteria/truth_alignment.py
import numpy as np

def truth_alignment_index(h_star_values, phi1_values):
    h = np.asarray(h_star_values).reshape(-1)
    phi = np.asarray(phi1_values).reshape(-1)
    num = np.abs(np.vdot(h, phi))
    den = np.linalg.norm(h) * np.linalg.norm(phi)
    if den < 1e-12:
        return 0.0
    return num / den

def truth_vs_anchor(h_star_values, phi1_values, anchor_values):
    tau_truth = truth_alignment_index(h_star_values, phi1_values)
    tau_anchor = truth_alignment_index(anchor_values, phi1_values)
    return tau_truth, tau_anchor


def subspace_alignment_index(h_star_values, phi_subspace):
    """τ robust: ‖chiếu h lên span(các eigenfunction cột của phi_subspace)‖ / ‖h‖.

    Bất biến với cách chọn cơ sở của không-gian-con → không phụ thuộc eigenvector
    tùy ý khi λ=1 suy biến (khác cosine-với-φ₁ đơn lẻ, vốn fragile). Khi subspace
    1 chiều (λ=1 tách rời) thì bằng đúng truth_alignment_index.

    phi_subspace: (S, k) — S mẫu, k eigenfunction có |λ|≈1.
    """
    h = np.asarray(h_star_values).reshape(-1).astype(complex)
    B = np.atleast_2d(np.asarray(phi_subspace, dtype=complex))
    if B.shape[0] != h.shape[0] and B.shape[1] == h.shape[0]:
        B = B.T
    den = np.linalg.norm(h)
    if den < 1e-12 or B.size == 0:
        return 0.0
    coef, *_ = np.linalg.lstsq(B, h, rcond=None)   # chiếu trực giao lên span(B)
    return float(np.linalg.norm(B @ coef) / den)
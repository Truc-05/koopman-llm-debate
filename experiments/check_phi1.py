"""KIỂM QUYẾT ĐỊNH: φ₁ (index-0, λ=1) có phải hàm HẰNG SỐ tầm thường không?

Nếu φ₁ = const → τ = |mean(h★)|/rms(h★) = baseline không-Koopman, gần vòng vo.
So 3 thứ:
  - τ_φ1   : cosine(h★, eigenfunction index-0)   [đang dùng]
  - τ_const: |mean(h★)|/rms(h★)                   [operator-free, tầm thường]
  - τ_φ2   : cosine(h★, eigenfunction index-1, λ<1) [mode động lực thật]
+ đo φ₁ có phẳng dọc quỹ đạo không (std≈0 => hằng số).

Dùng: python experiments/check_phi1.py mmlu_clean6 [degree=1]
"""
import sys, os, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import build_snapshots, fit_edmd
from src.koopman.spectrum import spectral_decomposition, project_stable, eigenfunctions
from src.debate.observables_truth import h_star_soft, along_trajectory

OUT = os.path.join(ROOT, "experiments", "out")
tag = sys.argv[1] if len(sys.argv) > 1 else "mmlu_clean6"
degree = int(sys.argv[2]) if len(sys.argv) > 2 else 1
N = 4

npz = np.load(os.path.join(OUT, f"trajs_{tag}.npz"))
trajs = [np.asarray(npz[f"traj_{i}"]) for i in range(len(npz.files))]
tr = json.load(open(os.path.join(OUT, f"transcripts_{tag}.json")))
y = np.array([bool(r["correct"]) for r in tr])
a_star = [int(r["a_star"]) for r in tr]
K = trajs[0].shape[1] // N
n = len(trajs)

d = PolynomialDictionary(degree=degree)
Zt = np.concatenate([t[:-1] for t in trajs]); Ztp1 = np.concatenate([t[1:] for t in trajs])
Px, Py = build_snapshots(d, Zt, Ztp1)
Kop, _, _ = fit_edmd(Px, Py, reg=1e-6); Kop = project_stable(Kop)
ev, V, W = spectral_decomposition(Kop)


def auc(s):
    p, m = s[y], s[~y]
    return ((p[:, None] > m[None, :]).sum() + 0.5 * (p[:, None] == m[None, :]).sum()) / (len(p) * len(m))


def cos(a, b):
    return abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


tau_phi1 = np.zeros(n); tau_phi2 = np.zeros(n); tau_const = np.zeros(n)
phi1_flatness = []                       # std(φ₁)/|mean(φ₁)| dọc quỹ đạo (≈0 => hằng)
for i, t in enumerate(trajs):
    Zi = t[:-1]
    h = along_trajectory(h_star_soft, Zi, n_agents=N, n_answers=K, a_star=a_star[i])
    Phi = eigenfunctions(d, W, Zi)
    p1 = Phi[:, 0]; p2 = Phi[:, 1]
    tau_phi1[i] = cos(h, p1)
    tau_phi2[i] = cos(h, p2)
    tau_const[i] = abs(np.mean(h)) / (np.sqrt(np.mean(h ** 2)) + 1e-12)   # = cos(h, 1)
    phi1_flatness.append(np.std(np.abs(p1)) / (np.mean(np.abs(p1)) + 1e-12))

print(f"tag={tag} degree={degree} n={n}  |λ1|={abs(ev[0]):.4f} |λ2|={abs(ev[1]):.4f}")
print(f"\nφ₁ có phẳng (hằng số) không? std/|mean| dọc quỹ đạo = {np.mean(phi1_flatness):.4f} "
      f"(≈0 => φ₁ là HẰNG SỐ tầm thường)")
print(f"\n{'score':<26}{'AUC':>7}   ghi chú")
print(f"{'τ_φ1 (index-0, đang dùng)':<26}{auc(tau_phi1):>7.4f}")
print(f"{'τ_const = |mean h|/rms h':<26}{auc(tau_const):>7.4f}   operator-FREE, tầm thường")
print(f"{'τ_φ2 (index-1, λ<1)':<26}{auc(tau_phi2):>7.4f}   mode động lực thật")
print(f"\ncorr(τ_φ1, τ_const) = {np.corrcoef(tau_phi1, tau_const)[0,1]:.4f}")
print(f"max|τ_φ1 - τ_const|  = {np.max(np.abs(tau_phi1 - tau_const)):.4f}")
print("\nĐọc:")
print(" - φ₁ phẳng (std/|mean|≈0) & τ_φ1≈τ_const (corr≈1, sai khác≈0)")
print("   => φ₁ CHỈ là hàm hằng => τ KHÔNG dùng Koopman, chỉ là mean-truth-belief (VẤN ĐỀ).")
print(" - φ₁ KHÔNG phẳng & τ_φ1 khác τ_const => φ₁ là mode đồng thuận thật (ỔN).")
print(" - Nếu τ_φ2 mới cao/khác => tín hiệu Koopman thật nằm ở φ₂, nên dùng index-1 thay index-0.")

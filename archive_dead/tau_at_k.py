"""tau@k early-warning: du bao dung/sai chi tu k vong DAU.

φ₁ hoc offline tren train-fold (full trajectory, topic-disjoint cross-fit) — la
observable co dinh. Voi debate test, sau k vong: tinh τ@k tren k trang thai nguon
dau + cong chuyen dong disp_k=||z_k-z_0||. Neu k nho (2-3) van tach dung/sai ->
CANH BAO SOM that su, khong chi chan doan hau nghiem (nhip 3 pitch doc.md).

Dung: python experiments/tau_at_k.py mmlu_40_r5
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import build_snapshots, fit_edmd
from src.koopman.spectrum import spectral_decomposition, eigenfunctions
from src.criteria.truth_alignment import truth_alignment_index
from src.debate.observables_truth import h_star_soft, along_trajectory

OUT, tag = "experiments/out", sys.argv[1] if len(sys.argv) > 1 else "mmlu_40_r5"
N, DEG, REG, NFOLD = 4, 1, 1e-6, 5

tr = json.load(open(f"{OUT}/transcripts_{tag}.json"))
y = np.array([bool(r["correct"]) for r in tr])
a_star = [int(r["a_star"]) for r in tr]
npz = np.load(f"{OUT}/trajs_{tag}.npz")
trajs = [np.asarray(npz[f"traj_{i}"]) for i in range(len(tr))]
K, base, n = trajs[0].shape[1] // N, y.mean(), len(tr)
T = trajs[0].shape[0] - 1        # so vong (6)


def auc(s):
    p, m = s[y], s[~y]
    return ((p[:, None] > m[None, :]).sum() + 0.5 * (p[:, None] == m[None, :]).sum()) / (len(p) * len(m))


# ---- cross-fit: W (left-eigvec) per fold, fit tren full trajectory train ----
fold = np.arange(n) % NFOLD
d = PolynomialDictionary(degree=DEG)
W_of = {}
for f in range(NFOLD):
    idx = np.where(fold != f)[0]
    Zt = np.concatenate([trajs[i][:-1] for i in idx], 0)
    Ztp1 = np.concatenate([trajs[i][1:] for i in idx], 0)
    Px, Py = build_snapshots(d, Zt, Ztp1)
    Kop, _, _ = fit_edmd(Px, Py, reg=REG)
    _, _, W_of[f] = spectral_decomposition(Kop)

# ---- τ@k + cong, cho tung k ----
print(f"tag={tag}  n={n}  base={base:.3f}  T={T} vong  (φ₁ offline, chi cham k vong dau)")
print(f"{'k':>3} {'disp_k(med)':>12} {'AUC τ@k':>9} {'AUC gated@k':>12} {'band P':>8} {'band n':>7}")
for k in range(2, T + 1):
    tau_k = np.zeros(n)
    disp_k = np.array([np.linalg.norm(trajs[i][k] - trajs[i][0]) for i in range(n)])
    for i in range(n):
        Zi = trajs[i][:k]                        # k trang thai nguon dau (vong 0..k-1)
        h = along_trajectory(h_star_soft, Zi, n_agents=N, n_answers=K, a_star=a_star[i])
        phi1 = eigenfunctions(d, W_of[fold[i]], Zi)[:, 0]
        tau_k[i] = float(truth_alignment_index(h, phi1))
    med = np.median(disp_k)
    gated = tau_k * np.minimum(1.0, disp_k / med)
    # band [0.90,0.99) tren gated (g_clip giu thang τ khi disp>=med)
    bb = (gated >= 0.90) & (gated < 0.99)
    p_b = y[bb].mean() if bb.any() else float("nan")
    print(f"{k:>3} {med:>12.2f} {auc(tau_k):>9.3f} {auc(gated):>12.3f} "
          f"{p_b:>8.3f} {bb.sum():>7}")

print(f"\n(k={T} = dung ca quy dao = ket qua tau_gated day du: AUC~0.75, band~0.92)")
print("doc: neu AUC gated@k giu cao tu k=2-3 => canh bao som THAT SU.")

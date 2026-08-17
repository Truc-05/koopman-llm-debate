"""τ ROBUST cuối: chiếu-subspace + stable-DMD, OOS cross-fit, quét δ.

Trả lời dứt điểm: (a) subspace-τ có tách đúng/sai ổn định qua δ và out-of-sample
không? (b) phổ sau stable-DMD có sạch (|λ|≤1) không? (c) so với φ₁-τ cũ (fragile).

Dùng: python experiments/tau_robust.py mmlu_160 [degree=1] [ncut=0(=full)]
  ncut>0: cắt mỗi debate còn ncut+1 trạng thái đầu (transient). vd 7 = 6 vòng đầu.
"""
import sys, os, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import build_snapshots, fit_edmd
from src.koopman.spectrum import (spectral_decomposition, project_stable,
                                  slow_subspace_indices, spectral_gap,
                                  eigenfunctions)
from src.criteria.truth_alignment import (truth_alignment_index,
                                         subspace_alignment_index)
from src.debate.observables_truth import h_star_soft, along_trajectory

OUT = os.path.join(ROOT, "experiments", "out")
tag = sys.argv[1] if len(sys.argv) > 1 else "mmlu_160"
degree = int(sys.argv[2]) if len(sys.argv) > 2 else 1
ncut = int(sys.argv[3]) if len(sys.argv) > 3 else 0
N, REG, NFOLD = 4, 1e-6, 5

npz = np.load(os.path.join(OUT, f"trajs_{tag}.npz"))
trajs = [np.asarray(npz[f"traj_{i}"]) for i in range(len(npz.files))]
if ncut > 0:
    trajs = [t[:ncut + 1] for t in trajs]
tr = json.load(open(os.path.join(OUT, f"transcripts_{tag}.json")))
y = np.array([bool(r["correct"]) for r in tr])
a_star = [int(r["a_star"]) for r in tr]
Kans = trajs[0].shape[1] // N
n = len(trajs)
fold = np.arange(n) % NFOLD
d = PolynomialDictionary(degree=degree)


def auc(s):
    p, m = s[y], s[~y]
    return ((p[:, None] > m[None, :]).sum() + 0.5 * (p[:, None] == m[None, :]).sum()) / (len(p) * len(m))


def decile(s):
    return " ".join(f"{y[i].mean():.2f}" for i in np.array_split(np.argsort(s), 10))


def fit_fold(idx, stable):
    Zt = np.concatenate([trajs[i][:-1] for i in idx], 0)
    Ztp1 = np.concatenate([trajs[i][1:] for i in idx], 0)
    Px, Py = build_snapshots(d, Zt, Ztp1)
    K, _, _ = fit_edmd(Px, Py, reg=REG)
    if stable:
        K = project_stable(K)
    return spectral_decomposition(K)          # ev, V, W


# ---- phổ pooled (báo cáo cleanliness trước/sau stable-DMD) ----
ev_raw, _, _ = fit_fold(np.arange(n), stable=False)
ev_stb, _, _ = fit_fold(np.arange(n), stable=True)
print(f"tag={tag} degree={degree} ncut={ncut or 'full'}  n={n}  base={y.mean():.3f}  "
      f"{sum(t.shape[0]-1 for t in trajs)} transitions")
for nm, ev in [("raw", ev_raw), ("stable-DMD", ev_stb)]:
    mods = np.abs(ev)
    print(f"  phổ {nm:<11}: |λ1|={mods[0]:.3f} |λ2|={mods[1]:.3f} gap={spectral_gap(ev):+.3f} "
          f"|λ|>1={int((mods>1.001).sum())} λ≈1(δ=.02)={len(slow_subspace_indices(ev,0.02))}")


# ---- OOS cross-fit: φ₁-τ (cũ) vs subspace-τ (mới) qua δ ----
def oos_scores(stable, deltas):
    phi1 = np.zeros(n)
    subs = {dl: np.zeros(n) for dl in deltas}
    for f in range(NFOLD):
        ev, V, W = fit_fold(np.where(fold != f)[0], stable)
        for i in np.where(fold == f)[0]:
            Zi = trajs[i][:-1]
            h = along_trajectory(h_star_soft, Zi, n_agents=N, n_answers=Kans, a_star=a_star[i])
            Phi = eigenfunctions(d, W, Zi)
            phi1[i] = truth_alignment_index(h, Phi[:, 0])
            for dl in deltas:
                idx = slow_subspace_indices(ev, dl)
                subs[dl][i] = subspace_alignment_index(h, Phi[:, idx])
    return phi1, subs


deltas = [0.02, 0.05, 0.10]
phi1, subs = oos_scores(stable=True, deltas=deltas)
print("\nOOS cross-fit (stable-DMD):")
print(f"  φ₁-τ (cũ, fragile)     AUC={auc(phi1):.3f}  decile: {decile(phi1)}")
for dl in deltas:
    s = subs[dl]
    print(f"  subspace-τ δ={dl:<4}       AUC={auc(s):.3f}  decile: {decile(s)}")
# ổn định qua δ:
mat = np.array([subs[dl] for dl in deltas])
cc = np.corrcoef(mat)
print(f"  corr subspace-τ giữa các δ: min={cc[np.triu_indices(len(deltas),1)].min():.3f} "
      f"(≈1 => KHÔNG nhạy δ)")
print(f"\n  đúng/sai (δ=0.05): {subs[0.05][y].mean():.3f} vs {subs[0.05][~y].mean():.3f}")

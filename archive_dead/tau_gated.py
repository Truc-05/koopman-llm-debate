"""tau co cong chuyen dong: ha τ khi debate gan bat dong (displacement thap).

Co che (tu tau_oos.py): debate τ=1 that ra la DONG BANG (disp ~6 vs ~33), cho
cosine gia ~1. Displacement la label-free va tach sach -> gate g(disp) de giet
false-positive τ=1 va lay lai don dieu. So voi τ goc, tren ca in-sample & OOS.

Dung: python experiments/tau_gated.py mmlu_40_r5
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

res = json.load(open(f"{OUT}/results_{tag}.json"))
tau_in = np.array([r["tau"] for r in res["runs"]], float)
tr = json.load(open(f"{OUT}/transcripts_{tag}.json"))
y = np.array([bool(r["correct"]) for r in tr])
a_star = [int(r["a_star"]) for r in tr]
npz = np.load(f"{OUT}/trajs_{tag}.npz")
trajs = [np.asarray(npz[f"traj_{i}"]) for i in range(len(tr))]
K, base = trajs[0].shape[1] // N, y.mean()

disp = np.array([np.linalg.norm(t[-1] - t[0]) for t in trajs])   # label-free


def auc(s):
    p, n = s[y], s[~y]
    return ((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n))


def band(s, lo=0.90, hi=0.99):
    b = (s >= lo) & (s < hi)
    return (b.sum(), y[b].mean() if b.any() else float("nan"))


def show(s, name):
    dec = [y[idx].mean() for idx in np.array_split(np.argsort(s), 10)]
    mono = all(dec[i] <= dec[i + 1] + 0.06 for i in range(9))   # gan don dieu?
    n_b, p_b = band(s)
    print(f"[{name:<22}] AUC={auc(s):.3f}  decile: " + " ".join(f"{c:.2f}" for c in dec) +
          f"  {'~MONO' if mono else 'U-nguoc'}")

# ---- tau_oos qua cross-fit (giong tau_oos.py) ----
fold = np.arange(len(tr)) % NFOLD
tau_oos = np.zeros(len(tr))
d = PolynomialDictionary(degree=DEG)
for f in range(NFOLD):
    idx_tr = np.where(fold != f)[0]
    Zt = np.concatenate([trajs[i][:-1] for i in idx_tr], 0)
    Ztp1 = np.concatenate([trajs[i][1:] for i in idx_tr], 0)
    Px, Py = build_snapshots(d, Zt, Ztp1)
    Kop, _, _ = fit_edmd(Px, Py, reg=REG)
    _, _, W = spectral_decomposition(Kop)
    for i in np.where(fold == f)[0]:
        Zi = trajs[i][:-1]
        h = along_trajectory(h_star_soft, Zi, n_agents=N, n_answers=K, a_star=a_star[i])
        tau_oos[i] = float(truth_alignment_index(h, eigenfunctions(d, W, Zi)[:, 0]))

# ---- cac cong (label-free, tu chuan hoa bang median disp) ----
d0 = np.median(disp)
g_soft = disp / (disp + d0)            # muot, tu scale, in [0,1)
g_clip = np.minimum(1.0, disp / d0)    # tuyen tinh chan

print(f"tag={tag}  n={len(tr)}  base={base:.3f}  median_disp={d0:.2f}")
print("\n-- IN-SAMPLE --")
show(tau_in, "tau (goc)")
show(tau_in * g_soft, "tau*g_soft")
show(tau_in * g_clip, "tau*g_clip")
print("\n-- OUT-OF-SAMPLE --")
show(tau_oos, "tau_oos (goc)")
show(tau_oos * g_soft, "tau_oos*g_soft")
show(tau_oos * g_clip, "tau_oos*g_clip")
print("\n-- doi chung: displacement don le --")
show(disp, "disp")

print("\nband P(dung) trong [0.90,0.99):")
for nm, s in [("tau_oos", tau_oos), ("tau_oos*g_soft", tau_oos * g_soft),
              ("tau_oos*g_clip", tau_oos * g_clip)]:
    n_b, p_b = band(s)
    print(f"  {nm:<16} n={n_b:>3} P={p_b:.3f} lift={p_b/base:.2f}x")

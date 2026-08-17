"""Test make-or-break: tau OUT-OF-SAMPLE qua cross-fit 5-fold.

φ₁ hien fit tren ca 200 debate gom chinh debate duoc cham (in-sample). O day:
chia 5 fold, fit φ₁ tren 4 fold, cham τ cho fold giu lai -> moi τ deu
out-of-sample. Roi so AUC + dai [0.90,0.99) in-sample vs OOS. Neu dai va chu-U
song sot -> phat hien that; neu sup -> chi la artifact in-sample.

+ probe co che debate τ=1 that su la gi (dinh chinh "confident herding").

Dung: python experiments/tau_oos.py mmlu_40_r5
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import build_snapshots, fit_edmd
from src.koopman.spectrum import spectral_decomposition, eigenfunctions
from src.criteria.truth_alignment import truth_alignment_index
from src.debate.observables_truth import h_star_soft, along_trajectory
from src.debate.state import mean_belief

OUT, tag = "experiments/out", sys.argv[1] if len(sys.argv) > 1 else "mmlu_40_r5"
N, DEG, REG, NFOLD = 4, 1, 1e-6, 5

res = json.load(open(f"{OUT}/results_{tag}.json"))
tau_in = np.array([r["tau"] for r in res["runs"]], float)   # in-sample (co san)
tr = json.load(open(f"{OUT}/transcripts_{tag}.json"))
y = np.array([bool(r["correct"]) for r in tr])
a_star = [int(r["a_star"]) for r in tr]
npz = np.load(f"{OUT}/trajs_{tag}.npz")
trajs = [np.asarray(npz[f"traj_{i}"]) for i in range(len(tr))]
K = trajs[0].shape[1] // N
base = y.mean()


def auc(s, lab):
    p, n = s[lab], s[~lab]
    return ((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n))


def band_report(tau, name):
    lo, hi = 0.90, 0.99
    b = (tau >= lo) & (tau < hi)
    order = np.argsort(tau)
    dec = [y[idx].mean() for idx in np.array_split(order, 10)]
    print(f"\n[{name}]  AUC={auc(tau, y):.3f}")
    print("  decile P: " + " ".join(f"{c:.2f}" for c in dec))
    if b.any():
        print(f"  BAND[{lo},{hi}): n={b.sum()} P={y[b].mean():.3f} lift={y[b].mean()/base:.2f}x cov={b.mean():.0%}")
    sat = tau >= 0.99
    if sat.any():
        print(f"  tau>=0.99: n={sat.sum()} P={y[sat].mean():.3f} (base {base:.3f})")


# ---- cross-fit 5-fold -> tau out-of-sample ----
fold = np.arange(len(tr)) % NFOLD
tau_oos = np.zeros(len(tr))
d = PolynomialDictionary(degree=DEG)
for f in range(NFOLD):
    tr_idx = np.where(fold != f)[0]
    Z_t = np.concatenate([trajs[i][:-1] for i in tr_idx], 0)
    Z_tp1 = np.concatenate([trajs[i][1:] for i in tr_idx], 0)
    Psi_X, Psi_Y = build_snapshots(d, Z_t, Z_tp1)
    Kop, _, _ = fit_edmd(Psi_X, Psi_Y, reg=REG)
    _, _, W = spectral_decomposition(Kop)
    for i in np.where(fold == f)[0]:
        Zi = trajs[i][:-1]
        h = along_trajectory(h_star_soft, Zi, n_agents=N, n_answers=K, a_star=a_star[i])
        phi1 = eigenfunctions(d, W, Zi)[:, 0]
        tau_oos[i] = float(truth_alignment_index(h, phi1))

print(f"tag={tag}  n={len(tr)}  base={base:.3f}  cross-fit {NFOLD}-fold")
band_report(tau_in, "IN-SAMPLE (goc)")
band_report(tau_oos, "OUT-OF-SAMPLE (cross-fit)")
print(f"\ncorr(tau_in, tau_oos) = {np.corrcoef(tau_in, tau_oos)[0,1]:.3f}")

# ---- probe co che: debate τ=1 (in-sample) that su la gi? ----
print("\n== probe co che nhom tau_in >= 0.99 ==")
hi = np.where(tau_in >= 0.99)[0]
disp = np.array([np.linalg.norm(trajs[i][-1] - trajs[i][0]) for i in range(len(tr))])
conf = np.array([mean_belief(trajs[i][-1], N, K).max() for i in range(len(tr))])
conv_star = np.array([int(np.argmax(mean_belief(trajs[i][-1], N, K)) == a_star[i]) for i in range(len(tr))])
print(f"  displacement ||z_T-z_0||:  tau1={disp[hi].mean():.3f}  toan bo={disp.mean():.3f}")
print(f"  final confidence max p:    tau1={conf[hi].mean():.3f}  toan bo={conf.mean():.3f}")
print(f"  hoi tu ve a_star:          tau1={conv_star[hi].mean():.3f}  toan bo={conv_star.mean():.3f}")
print("  (displacement thap => quy dao gan phang => tuong quan voi phi1 gia cao)")

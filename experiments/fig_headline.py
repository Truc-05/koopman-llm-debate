"""Hai (ba) hinh chu luc cho paper, tu run da luu:
 A. Reliability: decile-precision cua τ_oos GOC vs GATED -> cong xoa cu sup dinh.
 B. Early-warning: AUC gated@k vs k -> actionable tu k=3.
 C. Scatter τ vs displacement to mau dung/sai -> lo cum DONG BANG (disp thap, τ cao, do).

Dung: python experiments/fig_headline.py mmlu_40_r5
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
K, base, n = trajs[0].shape[1] // N, y.mean(), len(tr)
T = trajs[0].shape[0] - 1
fold = np.arange(n) % NFOLD
d = PolynomialDictionary(degree=DEG)


def auc(s):
    p, m = s[y], s[~y]
    return ((p[:, None] > m[None, :]).sum() + 0.5 * (p[:, None] == m[None, :]).sum()) / (len(p) * len(m))


def W_per_fold():
    Ws = {}
    for f in range(NFOLD):
        idx = np.where(fold != f)[0]
        Zt = np.concatenate([trajs[i][:-1] for i in idx], 0)
        Ztp1 = np.concatenate([trajs[i][1:] for i in idx], 0)
        Px, Py = build_snapshots(d, Zt, Ztp1)
        Kop, _, _ = fit_edmd(Px, Py, reg=REG)
        _, _, Ws[f] = spectral_decomposition(Kop)
    return Ws

Ws = W_per_fold()


def tau_at(k):
    t = np.zeros(n)
    for i in range(n):
        Zi = trajs[i][:k]
        h = along_trajectory(h_star_soft, Zi, n_agents=N, n_answers=K, a_star=a_star[i])
        t[i] = float(truth_alignment_index(h, eigenfunctions(d, Ws[fold[i]], Zi)[:, 0]))
    return t

disp_full = np.array([np.linalg.norm(t[-1] - t[0]) for t in trajs])
tau_oos = tau_at(T)
gated_oos = tau_oos * np.minimum(1.0, disp_full / np.median(disp_full))


def decile_p(score):
    return [y[idx].mean() for idx in np.array_split(np.argsort(score), 10)]

# ---- early-warning theo k ----
ks = list(range(2, T + 1))
auc_raw, auc_gat = [], []
for k in ks:
    tk = tau_at(k)
    dk = np.array([np.linalg.norm(trajs[i][k] - trajs[i][0]) for i in range(n)])
    auc_raw.append(auc(tk))
    auc_gat.append(auc(tk * np.minimum(1.0, dk / np.median(dk))))

# ================= VE =================
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

# A. reliability decile
xd = np.arange(1, 11)
ax[0].axhline(base, ls="--", c="gray", lw=1, label=f"base {base:.2f}")
ax[0].plot(xd, decile_p(tau_oos), "o-", c="tab:orange", label="τ (goc)")
ax[0].plot(xd, decile_p(gated_oos), "s-", c="tab:blue", label="τ gated")
ax[0].annotate("cu sup herding\n(dong bang)", (10, decile_p(tau_oos)[-1]),
               (7.2, 0.12), fontsize=8, color="tab:orange",
               arrowprops=dict(arrowstyle="->", color="tab:orange"))
ax[0].set_xlabel("decile diem (thap→cao)"); ax[0].set_ylabel("P(debate dung)")
ax[0].set_title("A. Cong xoa cu sup o dinh"); ax[0].set_ylim(-.02, 1.05)
ax[0].legend(fontsize=8)

# B. early warning
ax[1].axhline(0.5, ls="--", c="gray", lw=1, label="ngau nhien")
ax[1].plot(ks, auc_raw, "o-", c="tab:orange", label="τ@k (goc)")
ax[1].plot(ks, auc_gat, "s-", c="tab:blue", label="τ gated@k")
ax[1].axvline(3, ls=":", c="green", lw=1)
ax[1].text(3.05, 0.52, "actionable\nk=3", color="green", fontsize=8)
ax[1].set_xlabel("so vong quan sat k (T=6)"); ax[1].set_ylabel("AUC du bao ket cuc")
ax[1].set_title("B. Canh bao som"); ax[1].set_xticks(ks); ax[1].legend(fontsize=8)

# C. co che cong: trong nhom τ cao, P(dung) theo displacement
hi = tau_in >= 0.9                       # nhom "τ noi tin duoc"
dh, yh = disp_full[hi], y[hi]
order = np.argsort(dh)
bins = np.array_split(order, 5)          # 5 nhom ngu phan theo disp
xb = [dh[b].mean() for b in bins]
pb = [yh[b].mean() for b in bins]
ax[2].axhline(base, ls="--", c="gray", lw=1, label=f"base {base:.2f}")
ax[2].plot(xb, pb, "D-", c="tab:purple", ms=7)
ax[2].scatter(dh[yh], np.full(yh.sum(), 1.03), marker="|", c="tab:green", s=40, alpha=.5)
ax[2].scatter(dh[~yh], np.full((~yh).sum(), -0.03), marker="|", c="tab:red", s=40, alpha=.5)
ax[2].annotate("dong bang\n(τ cao GIA)", (xb[0], pb[0]), (xb[0] + 3, 0.18),
               fontsize=8, color="tab:red",
               arrowprops=dict(arrowstyle="->", color="tab:red"))
ax[2].set_xlabel("displacement ‖z_T−z_0‖"); ax[2].set_ylabel("P(dung) | τ≥0.9")
ax[2].set_title("C. Trong nhom τ cao, chuyen dong tach dung/sai")
ax[2].set_ylim(-.08, 1.1); ax[2].legend(fontsize=8, loc="center right")

plt.tight_layout()
out = f"{OUT}/koopman_headline.png"
plt.savefig(out, dpi=150)
print(f"saved -> {out}")
print(f"A: τ decile top={decile_p(tau_oos)[-1]:.2f} -> gated top={decile_p(gated_oos)[-1]:.2f}")
print(f"B: AUC gated@3={auc_gat[1]:.3f}  @6={auc_gat[-1]:.3f}")

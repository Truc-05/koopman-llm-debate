"""Sinh 2 figure supplementary từ ckpt THẬT (KHÔNG gọi LLM, KHÔNG bịa số):
  fig_phasemap.png       — phase-map: 4 model trên (Koopman order-advantage, semigroup defect), 2 regime.
  fig_vulnsub.png        — (a) scree reachability (low-rank r@90%), (b) heatmap overlap subspace 4×4.
Cũng in số cho 2 table supplementary. Palette Okabe-Ito (CVD-safe), heatmap sequential 1-hue.
Dùng: python experiments/make_supp_figures.py
"""
import json, glob, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import fit_edmdc
from src.debate.state import mean_belief

N, K_DEF, T_DEF = 4, 2, 6
MODELS = ["mistral_7b", "llama3_1_8b", "gemma2_9b", "qwen2_5_7b"]
LABEL = {"mistral_7b": "Mistral", "llama3_1_8b": "Llama", "gemma2_9b": "Gemma", "qwen2_5_7b": "Qwen"}
OKABE = {"mistral_7b": "#0072B2", "llama3_1_8b": "#E69F00", "gemma2_9b": "#009E73", "qwen2_5_7b": "#D55E00"}
DIRS = [os.path.join(ROOT, "experiments", d) for d in ["out", "out_qwen"]]

def collects(m):
    tagcols = []
    for D in DIRS:
        for f in glob.glob(os.path.join(D, f"ckpt_twindef_collect_*_{m}.jsonl")):
            c = [json.loads(l) for l in open(f)]
            if np.asarray(c[0]["traj"]).shape[1] == N * 4:
                tagcols.append((os.path.basename(f), c))
    tagcols.sort()                                   # deterministic; arc < math < mmlu < truthfulqa
    return tagcols

def fit_AB(cols):
    d = PolynomialDictionary(degree=1); d.transform(np.asarray(cols[0][0]["traj"])[:1]); ss = d.state_slice
    X, Y, U = [], [], []
    for c in cols:
        for r in c:
            Z = np.asarray(r["traj"], float)
            for t in range(len(Z) - 1):
                u = np.zeros(4)
                if r["c"] >= 0 and t >= K_DEF: u[r["c"]] = 1.0
                X.append(Z[t]); Y.append(Z[t + 1]); U.append(u)
    K, B = fit_edmdc(d.transform(np.array(X)).T, d.transform(np.array(Y)).T, np.array(U).T, reg=1e-6)
    A = np.real(K[ss, ss]); Bs = np.real(B[ss, :])
    return A, Bs, K, B, d, ss

def reach_svals(A, B, H=4):
    Ak = np.eye(A.shape[0]); blocks = [B]
    for _ in range(1, H):
        Ak = A @ Ak; blocks.append(Ak @ B)
    return np.linalg.svd(np.hstack(blocks), full_matrices=False)  # U,s,Vt

def eff_rank(s, thr=0.9):
    e = np.cumsum(s ** 2) / np.sum(s ** 2); return int(np.searchsorted(e, thr) + 1)

def semigroup(cols):
    X1, Y1, X2, Y2 = [], [], [], []
    for c in cols:
        for r in c:
            Z = np.asarray(r["traj"], float); cc = r["c"]
            for t in range(len(Z) - 1):
                if cc == -1 or t < K_DEF: X1.append(Z[t]); Y1.append(Z[t + 1])
            for t in range(len(Z) - 2):
                if cc == -1 or t + 1 < K_DEF: X2.append(Z[t]); Y2.append(Z[t + 2])
    fit = lambda X, Y: np.linalg.solve(np.array(X).T @ np.array(X) + 1e-3 * np.eye(16), np.array(X).T @ np.array(Y))
    A1, A2 = fit(X1, Y1), fit(X2, Y2)
    return float(np.linalg.norm(A2 - A1 @ A1) / (np.linalg.norm(A2) + 1e-12))

def roll_fn(K, B, d, ss):
    def roll(z0, chans):
        psi = d.transform(np.asarray(z0).reshape(1, -1))[0]
        for c in chans:
            u = np.zeros(4)
            if c >= 0: u[c] = 1.0
            psi = K @ psi + B @ u
        return mean_belief(np.real(psi[ss]), N, 4)
    return roll

def order_corrs(m, roll):
    f = glob.glob(os.path.join(ROOT, "experiments", "out*", f"ckpt_composition_*_{m}.jsonl"))
    if not f: return np.nan, np.nan
    seen = {}
    for l in open(f[0]):
        r = json.loads(l); seen.setdefault(r["ti"], r)
    R = list(seen.values()); fb = lambda t: mean_belief(np.asarray(t)[-1], N, 4)
    zk = lambda t: np.asarray(t)[K_DEF]; half = (T_DEF - K_DEF) // 2
    oa, ok, rc = [], [], []
    for r in R:
        c1, c2, C = r["c1"], r["c2"], r["conds"]
        oa.append(np.linalg.norm(fb(C["seq12"]) - fb(C["seq21"])))
        ok.append(np.linalg.norm(roll(zk(C["seq12"]), [c1] * half + [c2] * (T_DEF - K_DEF - half))
                                 - roll(zk(C["seq21"]), [c2] * half + [c1] * (T_DEF - K_DEF - half))))
        rc.append(np.linalg.norm(fb(C["s2"]) - fb(C["s1"])))
    c = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-9 and np.std(b) > 1e-9 else 0.0
    return c(oa, ok), c(oa, rc)   # (koopman-order-corr, recency-order-corr)

# ---------- compute ----------
D = {}
scree = {}
for m in MODELS:
    tagcols = collects(m)
    if not tagcols: continue
    cols = [c for _, c in tagcols]
    A, Bs, K, B, dic, ss = fit_AB(cols)                 # pooled: semig, scree, rollout
    U, s, _ = reach_svals(A, Bs)
    roll = roll_fn(K, B, dic, ss)
    kcorr, rcorr = order_corrs(m, roll)
    Ar, Br, *_ = fit_AB([tagcols[0][1]])                # per-cell rep (arc) — khớp vulnerability_subspace.py
    Ur, _, _ = reach_svals(Ar, Br)
    D[m] = dict(semig=semigroup(cols), r90=eff_rank(s), U2=Ur[:, :2],
                kcorr=kcorr, rcorr=rcorr, adv=kcorr - rcorr)
    scree[m] = s / s[0]

def subsim(U1, U2):
    return float(np.sum((U1[:2 if U1.shape[1] > 2 else U1.shape[1]].T @ U2) ** 2))  # placeholder
def sim2(a, b):
    return float(np.sum((a[:, :2].T @ b[:, :2]) ** 2) / 2)

mods = [m for m in MODELS if m in D]
# ---------- FIG 1: phase-map ----------
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(5.2, 4.0))
ax.axvspan(-0.6, 0.30, color="#BBBBBB", alpha=0.16)
ax.axvspan(0.30, 0.9, color="#D55E00", alpha=0.10)
ax.axvline(0.30, color="#888888", lw=1, ls="--")
for m in mods:
    ax.scatter(D[m]["adv"], D[m]["semig"], s=170, color=OKABE[m], edgecolor="white", lw=1.5, zorder=3)
    ax.annotate(f"{LABEL[m]}\n(r$_{{90}}$={D[m]['r90']})", (D[m]["adv"], D[m]["semig"]),
                textcoords="offset points", xytext=(9, -4), fontsize=9.5)
ax.text(-0.45, 0.90, "static, recency-collapse", color="#555555", fontsize=9.5, style="italic")
ax.text(0.42, 0.90, "noncommutative", color="#A5490B", fontsize=9.5, style="italic")
ax.set_xlabel("Koopman order-advantage  (corr$_{\\rm koop}-$corr$_{\\rm recency}$)")
ax.set_ylabel("semigroup defect  $\\|\\widehat A_2-\\widehat A_1^2\\|/\\|\\widehat A_2\\|$")
ax.set_xlim(-0.6, 0.85); ax.set_ylim(0.35, 1.0)
ax.set_title("Phase map of intervention dynamics", fontsize=11.5)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "fig_phasemap.png"), dpi=200); plt.close(fig)

# ---------- FIG 2: vulnerability subspace ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.5))
for m in mods:
    a1.plot(range(1, len(scree[m]) + 1), scree[m], "-o", ms=4, color=OKABE[m], label=LABEL[m])
a1.axhline(0, color="#CCC", lw=.5)
a1.set_xlabel("singular value index"); a1.set_ylabel("normalized singular value")
a1.set_title("Reachability spectrum (low-rank)", fontsize=11)
a1.legend(frameon=False, fontsize=9); a1.set_xlim(1, 8)
# heatmap overlap
Mx = np.eye(len(mods))
for i in range(len(mods)):
    for j in range(len(mods)):
        if i != j: Mx[i, j] = sim2(D[mods[i]]["U2"], D[mods[j]]["U2"])
im = a2.imshow(Mx, cmap="Blues", vmin=0, vmax=1)
a2.set_xticks(range(len(mods))); a2.set_yticks(range(len(mods)))
a2.set_xticklabels([LABEL[m] for m in mods], rotation=30, ha="right", fontsize=9)
a2.set_yticklabels([LABEL[m] for m in mods], fontsize=9)
for i in range(len(mods)):
    for j in range(len(mods)):
        a2.text(j, i, f"{Mx[i,j]:.2f}", ha="center", va="center",
                color="white" if Mx[i, j] > 0.55 else "#333", fontsize=9)
a2.set_title("Vulnerability-subspace overlap\n(random baseline $\\approx0.13$)", fontsize=10.5)
fig.colorbar(im, ax=a2, fraction=0.046, pad=0.04)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "fig_vulnsub.png"), dpi=200); plt.close(fig)

# ---------- in số cho table ----------
print("=== phase-map / composition table ===")
print(f"{'model':8s} {'semig':>6s} {'r90':>4s} {'koop-ord':>9s} {'recency-ord':>11s} {'adv':>6s}")
for m in mods:
    e = D[m]; print(f"{LABEL[m]:8s} {e['semig']:6.2f} {e['r90']:>4d} {e['kcorr']:>9.2f} {e['rcorr']:>11.2f} {e['adv']:>6.2f}")
print("\n=== subspace overlap matrix (off-diag) ===")
offs = [Mx[i, j] for i in range(len(mods)) for j in range(i + 1, len(mods))]
print("pairwise:", [f"{v:.2f}" for v in offs], " range", f"{min(offs):.2f}-{max(offs):.2f}")
print("saved: fig_phasemap.png, fig_vulnsub.png")

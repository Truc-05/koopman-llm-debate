"""Vẽ lại TOÀN BỘ ảnh kết quả theo panel rời (a,b,c,d) để tự ghép.
Xuất PNG vào  koopman-debate/figs_v2/  — mỗi panel 1 file, cùng nhóm cùng figsize + dpi.

  Nhóm 1  g1_[a-d]  accuracy: no-def / twin(obs.) / oracle × 4 bench   (1 panel/model)
  Nhóm 2  g2_[a-d]  ASR: attack(no-def) vs defended × 4 bench          (1 panel/model)
  Nhóm 3  g3_[a-d]  Koopman spectrum (eig A) trong đĩa đơn vị × 4 bench (1 panel/model)
  Nhóm 4  g4_[a-d]  phổ controllability Gramian W_H × 4 bench           (1 panel/model)
  Nhóm 5  g5_[a-d]  heatmap reach(x): bench × đáp án                    (1 panel/model)
  Nhóm 6  g6_[a-d]  scatter chéo 16 cell: Δ~ρ, Δ~base, Δ~logκ, ASR duality
  Nhóm 7  g7_[a-c]  phụ lục: phase-map, scree reachability, overlap subspace

Palette model + benchmark cố định xuyên suốt → ghép lại đọc như 1 hệ thống.
Dùng:  /home/alex/venvs/env/bin/python experiments/make_figs_v2.py
"""
import os, sys, json, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import fit_edmdc
from src.debate.state import mean_belief

OUT = os.path.join(ROOT, "figs_v2"); os.makedirs(OUT, exist_ok=True)
OUT_MAIN, OUT_QWEN = os.path.join(ROOT, "experiments", "out"), os.path.join(ROOT, "experiments", "out_qwen")
N, K_DEF = 4, 2

# ---- thứ tự panel a,b,c,d = Qwen, Mistral, Llama, Gemma ----
MODELS = [("Qwen2.5-7B", "qwen2_5_7b", OUT_QWEN),
          ("Mistral-7B", "mistral_7b", OUT_MAIN),
          ("Llama-3.1-8B", "llama3_1_8b", OUT_MAIN),
          ("Gemma-2-9B", "gemma2_9b", OUT_MAIN)]
BENCH = [("MMLU-math", "mmlu_clean6"), ("MATH", "math"), ("TruthfulQA", "truthfulqa"), ("ARC", "arc")]
XSHORT = ["MMLU", "MATH", "TQA", "ARC"]
LETTERS = "abcdef"

# ---- palette modern, nhất quán ----
MODEL_COLOR = {"Qwen2.5-7B": "#2563EB", "Mistral-7B": "#EA580C",
               "Llama-3.1-8B": "#059669", "Gemma-2-9B": "#7C3AED"}
BENCH_COLOR = {"MMLU-math": "#2775C9", "MATH": "#C81E5B", "TruthfulQA": "#12A594", "ARC": "#B7791F"}
C_NODEF, C_TWIN = "#CBD5E1", "#64748B"          # accuracy: no-def / twin
C_ATK, C_DEF = "#DC2626", "#16A34A"             # asr: attack / defended

# ---- kích thước theo loại chart (cùng loại = bằng nhau) ----
SZ_BAR, SZ_DISK, SZ_GRAM, SZ_HEAT, SZ_SCAT = (3.5, 2.8), (3.0, 3.0), (3.5, 2.9), (3.0, 3.0), (3.3, 3.0)

plt.rcParams.update({
    "font.size": 9.5, "axes.grid": True, "grid.alpha": 0.28, "grid.linewidth": 0.6,
    "grid.color": "#B8C0CC", "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#5B6472", "axes.linewidth": 0.9,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "figure.facecolor": "white", "xtick.color": "#3A414C", "ytick.color": "#3A414C",
    "font.family": "DejaVu Sans"})

pn = {(c["model"], c["bench"]): c for c in json.load(open(os.path.join(ROOT, "experiments", "paper_numbers.json")))["cells"]}
twd = {OUT_MAIN: json.load(open(os.path.join(OUT_MAIN, "analysis_twindef.json"))),
       OUT_QWEN: json.load(open(os.path.join(OUT_QWEN, "analysis_twindef.json")))}


def twin_acc(od, btag, msuf):
    return twd[od][f"{btag}_{msuf}"]["obj"]["honest_margin"]["twin"]["acc"]


def refit_AB(od, tag):
    """(A, B_state, T) refit EDMDc từ ckpt collect — giống make_paper_figures."""
    col = [json.loads(l) for l in open(os.path.join(od, f"ckpt_twindef_collect_{tag}.jsonl")) if l.strip()]
    D = np.asarray(col[0]["traj"]).shape[1]; Kans = D // N
    d = PolynomialDictionary(degree=1); d.transform(np.asarray(col[0]["traj"])[:1]); ss = d.state_slice
    X, Y, U = [], [], []
    for r in col:
        Z = np.asarray(r["traj"], float)
        for t in range(len(Z) - 1):
            u = np.zeros(Kans)
            if r["c"] >= 0 and t >= K_DEF: u[r["c"]] = 1.0
            X.append(Z[t]); Y.append(Z[t + 1]); U.append(u)
    K, B = fit_edmdc(d.transform(np.array(X)).T, d.transform(np.array(Y)).T, np.array(U).T, reg=1e-6)
    A = np.real(K[ss, ss]); Bs = np.real(B[ss, :]); T = np.asarray(col[0]["traj"]).shape[0] - 1
    return A, Bs, T


def save(fig, name):
    p = os.path.join(OUT, name); fig.savefig(p); plt.close(fig); print("  wrote", os.path.relpath(p, ROOT))


def tag_letter(ax, L, extra=""):
    ax.set_title(f"({L}) {extra}", fontsize=10.5, fontweight="bold", loc="left", pad=6, color="#1F2530")


# ================= NHÓM 1 — accuracy (headroom) =================
def group1():
    for L, (ml, ms, od) in zip(LETTERS, MODELS):
        fig, ax = plt.subplots(figsize=SZ_BAR)
        x = np.arange(4); w = 0.26
        nd = [pn[(ml, bl)]["nodef"] for bl, _ in BENCH]
        tw = [twin_acc(od, bt, ms) for _, bt in BENCH]
        orc = [pn[(ml, bl)]["oracle"] for bl, _ in BENCH]
        lo = [pn[(ml, bl)]["nodef"] + pn[(ml, bl)]["ci"][0] for bl, _ in BENCH]
        hi = [pn[(ml, bl)]["nodef"] + pn[(ml, bl)]["ci"][1] for bl, _ in BENCH]
        ax.bar(x - w, nd, w, color=C_NODEF, label="no-def", edgecolor="white", linewidth=0.6)
        ax.bar(x, tw, w, color=C_TWIN, label="twin (obs.)", edgecolor="white", linewidth=0.6)
        ax.bar(x + w, orc, w, color=MODEL_COLOR[ml], label="oracle", edgecolor="white", linewidth=0.6)
        yerr = [np.array(orc) - np.array(lo), np.array(hi) - np.array(orc)]
        ax.errorbar(x + w, orc, yerr=yerr, fmt="none", ecolor="#2A2F38", elinewidth=0.9, capsize=2.2, zorder=5)
        for i, (bl, _) in enumerate(BENCH):
            if pn[(ml, bl)]["mcnemar_p"] < 0.05:
                ax.text(i + w, hi[i] + 0.03, "$*$", ha="center", fontsize=12, color="#1F2530")
        ax.set_xticks(x); ax.set_xticklabels(XSHORT, fontsize=8.5)
        ax.set_ylim(0, 0.9); ax.set_ylabel("accuracy")
        if L == "a":
            ax.legend(loc="upper left", fontsize=7.5, framealpha=0.92, edgecolor="#D0D5DD")
        tag_letter(ax, L, ml)
        save(fig, f"g1_{L}.png")


# ================= NHÓM 2 — ASR (attack↔defense) =================
def group2():
    for L, (ml, ms, od) in zip(LETTERS, MODELS):
        fig, ax = plt.subplots(figsize=SZ_BAR)
        x = np.arange(4); w = 0.36
        an = [pn[(ml, bl)]["asr_nd"] for bl, _ in BENCH]
        ad = [pn[(ml, bl)]["asr_def"] for bl, _ in BENCH]
        ax.bar(x - w / 2, an, w, color=C_ATK, label="attack (no-def)", edgecolor="white", linewidth=0.6)
        ax.bar(x + w / 2, ad, w, color=C_DEF, label="defended", edgecolor="white", linewidth=0.6)
        for i in range(4):
            ax.text(i + w / 2, ad[i] + 0.015, f"{ad[i]:.02f}".lstrip("0"), ha="center", fontsize=6.5, color="#14532D")
        ax.set_xticks(x); ax.set_xticklabels(XSHORT, fontsize=8.5)
        ax.set_ylim(0, 0.7); ax.set_ylabel("attack success rate")
        if L == "a":
            ax.legend(loc="upper right", fontsize=7.5, framealpha=0.92, edgecolor="#D0D5DD")
        tag_letter(ax, L, ml)
        save(fig, f"g2_{L}.png")


# ================= NHÓM 3 — Koopman spectrum =================
def group3():
    th = np.linspace(0, 2 * np.pi, 240)
    for L, (ml, ms, od) in zip(LETTERS, MODELS):
        fig, ax = plt.subplots(figsize=SZ_DISK)
        ax.fill(np.cos(th), np.sin(th), color="#EEF2F7", zorder=0)
        ax.plot(np.cos(th), np.sin(th), "-", color="#94A3B8", lw=1.0, zorder=1)
        ax.axhline(0, color="#CBD5E1", lw=0.6); ax.axvline(0, color="#CBD5E1", lw=0.6)
        for bl, bt in BENCH:
            try:
                A, _, _ = refit_AB(od, f"{bt}_{ms}")
            except Exception as e:
                print("   [skip]", ml, bl, repr(e)); continue
            lam = np.linalg.eigvals(A)
            ax.scatter(lam.real, lam.imag, s=26, color=BENCH_COLOR[bl], alpha=0.85,
                       edgecolor="white", linewidth=0.4, zorder=3,
                       label=f"{bl.replace('-math','')} ($\\rho${np.abs(lam).max():.2f})")
        ax.set_aspect("equal"); ax.set_xlim(-1.12, 1.12); ax.set_ylim(-1.12, 1.12)
        ax.set_xlabel(r"$\mathrm{Re}\,\lambda$"); ax.set_ylabel(r"$\mathrm{Im}\,\lambda$")
        ax.grid(False)
        if L == "a":
            ax.legend(loc="lower left", fontsize=6.3, framealpha=0.9, edgecolor="#D0D5DD", handletextpad=0.2)
        tag_letter(ax, L, ml)
        save(fig, f"g3_{L}.png")


# ================= NHÓM 4 — controllability Gramian =================
def group4():
    for L, (ml, ms, od) in zip(LETTERS, MODELS):
        fig, ax = plt.subplots(figsize=SZ_GRAM)
        for bl, bt in BENCH:
            try:
                A, Bs, T = refit_AB(od, f"{bt}_{ms}")
            except Exception as e:
                print("   [skip]", ml, bl, repr(e)); continue
            D = A.shape[0]; W = np.zeros((D, D)); Ak = np.eye(D)
            for _ in range(T):
                W += Ak @ Bs @ Bs.T @ Ak.T; Ak = A @ Ak
            w = np.sort(np.linalg.eigvalsh(W))[::-1]
            k = w[0] / max(w[-1], 1e-15)
            ax.semilogy(range(1, D + 1), np.maximum(w, 1e-16), "-o", ms=3.2, lw=1.3,
                        color=BENCH_COLOR[bl], label=f"{bl.replace('-math','')} ($\\log\\kappa${np.log10(k):.1f})")
        ax.set_xlabel("index $i$"); ax.set_ylabel(r"$\lambda_i(W_H)$")
        if L == "a":
            ax.legend(loc="upper right", fontsize=6.3, framealpha=0.9, edgecolor="#D0D5DD", handletextpad=0.3)
        tag_letter(ax, L, ml)
        save(fig, f"g4_{L}.png")


# ================= NHÓM 5 — reachability heatmap =================
def group5():
    cmap = plt.get_cmap("viridis")
    for L, (ml, ms, od) in zip(LETTERS, MODELS):
        M = np.array([pn[(ml, bl)]["reach"] for bl, _ in BENCH])
        fig, ax = plt.subplots(figsize=SZ_HEAT)
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(M.shape[1])); ax.set_xticklabels([f"$x_{j}$" for j in range(M.shape[1])], fontsize=8.5)
        ax.set_yticks(range(4)); ax.set_yticklabels(XSHORT, fontsize=8.5)
        ax.set_xlabel("target answer")
        ax.set_xticks(np.arange(-.5, M.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
        ax.grid(which="major", visible=False); ax.tick_params(which="minor", length=0)
        for i in range(4):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] < 0.55 else "#0B1020", fontweight="medium")
        tag_letter(ax, L, ml)
        save(fig, f"g5_{L}.png")
    # thanh màu dùng chung (ghép tuỳ ý)
    fig, ax = plt.subplots(figsize=(0.5, 3.0))
    import matplotlib as mpl
    mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=mpl.colors.Normalize(0, 1)).set_label("reach$(x)$")
    save(fig, "g5_cbar.png")


# ================= NHÓM 6 — scatter chéo 16 cell =================
def group6():
    def scatter(ax, fx, fy, xlab, ylab, fit=False, diag=False, legend=False):
        xs, ys = [], []
        for ml, ms, od in MODELS:
            X = [fx(pn[(ml, bl)]) for bl, _ in BENCH]; Y = [fy(pn[(ml, bl)]) for bl, _ in BENCH]
            ax.scatter(X, Y, s=46, color=MODEL_COLOR[ml], edgecolor="white", linewidth=0.7,
                       label=ml.split("-")[0], zorder=3)
            xs += X; ys += Y
        xs, ys = np.array(xs), np.array(ys)
        if fit:
            b, a = np.polyfit(xs, ys, 1); r = np.corrcoef(xs, ys)[0, 1]
            xf = np.linspace(xs.min(), xs.max(), 50)
            ax.plot(xf, a + b * xf, "--", color="#475569", lw=1.2, zorder=2)
            ax.text(0.04, 0.05, f"Pearson $r={r:.2f}$", transform=ax.transAxes, fontsize=8,
                    bbox=dict(boxstyle="round", fc="white", ec="#CBD5E1", alpha=0.92))
        if diag:
            lim = [0, max(xs.max(), ys.max()) * 1.05]
            ax.plot(lim, lim, "--", color="#94A3B8", lw=1.0, zorder=1)
            ax.set_xlim(0, lim[1]); ax.set_ylim(0, lim[1])
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        if legend:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.92, edgecolor="#D0D5DD")

    specs = [("a", lambda c: c["rho"], lambda c: c["delta"], r"spectral radius $\rho(A)$", r"oracle gain $\Delta$",
              dict(fit=True, legend=True)),
             ("b", lambda c: c["nodef"], lambda c: c["delta"], "base accuracy", r"oracle gain $\Delta$",
              dict(fit=True)),
             ("c", lambda c: c["log10kappa"], lambda c: c["delta"], r"$\log_{10}\kappa(W_H)$", r"oracle gain $\Delta$",
              dict(fit=True)),
             ("d", lambda c: c["asr_nd"], lambda c: c["asr_def"], "ASR (no-def)", "ASR (defended)",
              dict(diag=True, legend=True))]
    titles = {"a": "Gain vs contraction", "b": "Gain vs headroom", "c": "Gain vs anisotropy", "d": "Attack↔defense"}
    for L, fx, fy, xl, yl, kw in specs:
        fig, ax = plt.subplots(figsize=SZ_SCAT)
        scatter(ax, fx, fy, xl, yl, **kw)
        tag_letter(ax, L, titles[L])
        save(fig, f"g6_{L}.png")


# ================= NHÓM 7 — phụ lục cấu trúc =================
def _collects(m):
    tc = []
    for D in (OUT_MAIN, OUT_QWEN):
        for f in glob.glob(os.path.join(D, f"ckpt_twindef_collect_*_{m}.jsonl")):
            c = [json.loads(l) for l in open(f)]
            if np.asarray(c[0]["traj"]).shape[1] == N * 4: tc.append((os.path.basename(f), c))
    tc.sort(); return tc


def _fit_AB(cols):
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
    return np.real(K[ss, ss]), np.real(B[ss, :]), K, B, d, ss


def _reach_svals(A, B, H=4):
    Ak = np.eye(A.shape[0]); blocks = [B]
    for _ in range(1, H):
        Ak = A @ Ak; blocks.append(Ak @ B)
    return np.linalg.svd(np.hstack(blocks), full_matrices=False)


def _eff_rank(s, thr=0.9):
    e = np.cumsum(s ** 2) / np.sum(s ** 2); return int(np.searchsorted(e, thr) + 1)


def _semigroup(cols):
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


def _roll_fn(K, B, d, ss):
    def roll(z0, chans):
        psi = d.transform(np.asarray(z0).reshape(1, -1))[0]
        for c in chans:
            u = np.zeros(4)
            if c >= 0: u[c] = 1.0
            psi = K @ psi + B @ u
        return mean_belief(np.real(psi[ss]), N, 4)
    return roll


def _order_corrs(m, roll):
    f = glob.glob(os.path.join(ROOT, "experiments", "out*", f"ckpt_composition_*_{m}.jsonl"))
    if not f: return np.nan, np.nan
    seen = {}
    for l in open(f[0]):
        r = json.loads(l); seen.setdefault(r["ti"], r)
    R = list(seen.values()); fb = lambda t: mean_belief(np.asarray(t)[-1], N, 4)
    zk = lambda t: np.asarray(t)[K_DEF]; half = (6 - K_DEF) // 2
    oa, ok, rc = [], [], []
    for r in R:
        c1, c2, C = r["c1"], r["c2"], r["conds"]
        oa.append(np.linalg.norm(fb(C["seq12"]) - fb(C["seq21"])))
        ok.append(np.linalg.norm(roll(zk(C["seq12"]), [c1] * half + [c2] * (6 - K_DEF - half))
                                 - roll(zk(C["seq21"]), [c2] * half + [c1] * (6 - K_DEF - half))))
        rc.append(np.linalg.norm(fb(C["s2"]) - fb(C["s1"])))
    cc = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-9 and np.std(b) > 1e-9 else 0.0
    return cc(oa, ok), cc(oa, rc)


def _sim2(a, b):
    return float(np.sum((a[:, :2].T @ b[:, :2]) ** 2) / 2)


def group7():
    D, scree = {}, {}
    for ml, ms, od in MODELS:
        tc = _collects(ms)
        if not tc: continue
        cols = [c for _, c in tc]
        A, Bs, K, B, dic, ss = _fit_AB(cols)
        U, s, _ = _reach_svals(A, Bs)
        kcorr, rcorr = _order_corrs(ms, _roll_fn(K, B, dic, ss))
        Ar, Br, *_ = _fit_AB([tc[0][1]])
        Ur, _, _ = _reach_svals(Ar, Br)
        D[ml] = dict(semig=_semigroup(cols), r90=_eff_rank(s), U2=Ur[:, :2], adv=kcorr - rcorr)
        scree[ml] = s / s[0]
    mods = [ml for ml, _, _ in MODELS if ml in D]

    # (a) phase-map
    fig, ax = plt.subplots(figsize=(3.9, 3.3))
    ax.axvspan(-0.6, 0.30, color="#94A3B8", alpha=0.14); ax.axvspan(0.30, 0.9, color="#EA580C", alpha=0.10)
    ax.axvline(0.30, color="#64748B", lw=1, ls="--")
    for ml in mods:
        ax.scatter(D[ml]["adv"], D[ml]["semig"], s=180, color=MODEL_COLOR[ml], edgecolor="white", lw=1.6, zorder=3)
        ax.annotate(f"{ml.split('-')[0]}\n(r$_{{90}}$={D[ml]['r90']})", (D[ml]["adv"], D[ml]["semig"]),
                    textcoords="offset points", xytext=(9, -4), fontsize=8.5)
    ax.text(-0.45, 0.92, "static / recency", color="#475569", fontsize=8.5, style="italic")
    ax.text(0.40, 0.92, "noncommutative", color="#B34700", fontsize=8.5, style="italic")
    ax.set_xlabel(r"Koopman order-advantage (corr$_{\rm koop}-$corr$_{\rm rec}$)")
    ax.set_ylabel(r"semigroup defect $\|\widehat A_2-\widehat A_1^2\|/\|\widehat A_2\|$")
    ax.set_xlim(-0.6, 0.85); ax.set_ylim(0.35, 1.0); ax.grid(alpha=0.25)
    tag_letter(ax, "a", "Phase map")
    save(fig, "g7_a.png")

    # (b) scree reachability
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    for ml in mods:
        ax.plot(range(1, len(scree[ml]) + 1), scree[ml], "-o", ms=4, lw=1.4,
                color=MODEL_COLOR[ml], label=ml.split("-")[0])
    ax.set_xlabel("singular value index"); ax.set_ylabel("normalized singular value")
    ax.set_xlim(1, 8); ax.legend(fontsize=7.5, framealpha=0.9, edgecolor="#D0D5DD")
    tag_letter(ax, "b", "Reachability spectrum")
    save(fig, "g7_b.png")

    # (c) overlap subspace heatmap
    fig, ax = plt.subplots(figsize=(3.3, 3.1))
    Mx = np.eye(len(mods))
    for i in range(len(mods)):
        for j in range(len(mods)):
            if i != j: Mx[i, j] = _sim2(D[mods[i]]["U2"], D[mods[j]]["U2"])
    im = ax.imshow(Mx, cmap="viridis", vmin=0, vmax=1)
    lbl = [m.split("-")[0] for m in mods]
    ax.set_xticks(range(len(mods))); ax.set_yticks(range(len(mods)))
    ax.set_xticklabels(lbl, rotation=30, ha="right", fontsize=8.5); ax.set_yticklabels(lbl, fontsize=8.5)
    ax.grid(False)
    for i in range(len(mods)):
        for j in range(len(mods)):
            ax.text(j, i, f"{Mx[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if Mx[i, j] < 0.55 else "#0B1020")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    tag_letter(ax, "c", "Subspace overlap")
    save(fig, "g7_c.png")


def main():
    print("Sinh figs_v2 →", os.path.relpath(OUT, ROOT))
    for g in (group1, group2, group3, group4, group5, group6, group7):
        print(f"[{g.__name__}]")
        try:
            g()
        except Exception as e:
            print("  [SKIP GROUP]", g.__name__, "->", repr(e))
    print("Xong. Ghép các file cùng tiền tố (g1_*, g2_*, ...) thành 1 hình.")


if __name__ == "__main__":
    main()

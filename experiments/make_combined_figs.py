"""Ghep a,b,c,d thanh 1 hinh / 1 PDF, chu to ro, cung palette xuyen suot.
Xuat ra ROOT (de dung ten dang \\includegraphics trong nd.md):
  fig_spectrum.pdf      (Figure 2)  eigenvalues cua A trong dia don vi   — 2x2, 1 panel/model
  controllability.pdf   (Figure 3)  pho controllability Gramian W_H      — 2x2, 1 panel/model
  fig_headroom.pdf      (Figure 7)  quan he cheo 16 cell (a..d)          — 2x2

Nhat quan voi Table 3 (tab:struct):
  rho(A)            = max|eig(A)|                       (khop cot rho(A))
  log10 kappa(W_H)  = Gramian voi horizon H = n_x = 16  (khop cot log10 kappa(W_H),
                      DUNG chuan analyze_structure.py, KHONG dung horizon T=6 cua pn)
Cung xuat ban .png xem nhanh de kiem tra bang mat.
Dung: /home/alex/venvs/env/bin/python experiments/make_combined_figs.py
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import fit_edmdc

OUT_MAIN = os.path.join(ROOT, "experiments", "out")
OUT_QWEN = os.path.join(ROOT, "experiments", "out_qwen")
PREVIEW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "experiments", "_preview")
os.makedirs(PREVIEW, exist_ok=True)
N, K_DEF = 4, 2

# panel a,b,c,d = Qwen, Mistral, Llama, Gemma
MODELS = [("Qwen2.5-7B", "qwen2_5_7b", OUT_QWEN),
          ("Mistral-7B", "mistral_7b", OUT_MAIN),
          ("Llama-3.1-8B", "llama3_1_8b", OUT_MAIN),
          ("Gemma-2-9B", "gemma2_9b", OUT_MAIN)]
BENCH = [("MMLU-math", "mmlu_clean6", "MMLU"), ("MATH", "math", "MATH"),
         ("TruthfulQA", "truthfulqa", "TQA"), ("ARC", "arc", "ARC")]
LETTERS = "abcd"

MODEL_COLOR = {"Qwen2.5-7B": "#2563EB", "Mistral-7B": "#EA580C",
               "Llama-3.1-8B": "#059669", "Gemma-2-9B": "#7C3AED"}
BENCH_COLOR = {"MMLU-math": "#2775C9", "MATH": "#C81E5B", "TruthfulQA": "#12A594", "ARC": "#B7791F"}

plt.rcParams.update({
    "font.size": 13, "font.family": "DejaVu Sans",
    "axes.grid": True, "grid.alpha": 0.28, "grid.linewidth": 0.6, "grid.color": "#B8C0CC",
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#5B6472", "axes.linewidth": 1.0,
    "xtick.color": "#3A414C", "ytick.color": "#3A414C",
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.labelsize": 14, "legend.fontsize": 11.5,
    "savefig.bbox": "tight", "savefig.facecolor": "white", "figure.facecolor": "white",
})

pn = {(c["model"], c["bench"]): c
      for c in json.load(open(os.path.join(ROOT, "experiments", "paper_numbers.json")))["cells"]}


def refit_AB(od, tag):
    col = [json.loads(l) for l in open(os.path.join(od, f"ckpt_twindef_collect_{tag}.jsonl")) if l.strip()]
    D = np.asarray(col[0]["traj"]).shape[1]; Kans = D // N
    d = PolynomialDictionary(degree=1); d.transform(np.asarray(col[0]["traj"])[:1]); ss = d.state_slice
    X, Y, U = [], [], []
    for r in col:
        Z = np.asarray(r["traj"], float)
        for t in range(len(Z) - 1):
            u = np.zeros(Kans)
            if r["c"] >= 0 and t >= K_DEF:
                u[r["c"]] = 1.0
            X.append(Z[t]); Y.append(Z[t + 1]); U.append(u)
    K, B = fit_edmdc(d.transform(np.array(X)).T, d.transform(np.array(Y)).T, np.array(U).T, reg=1e-6)
    return np.real(K[ss, ss]), np.real(B[ss, :])


def build_cache():
    """Mot lan refit / cell -> eig(A), rho, pho Gramian W_H (H=n_x), log10 kappa (khop Table 3)."""
    cache = {}
    for ml, ms, od in MODELS:
        for bl, bt, _ in BENCH:
            A, Bz = refit_AB(od, f"{bt}_{ms}")
            H = A.shape[0]                                              # horizon = n_x = 16 (chuan Table 3)
            lam = np.linalg.eigvals(A)
            Wc = sum((np.linalg.matrix_power(A, t) @ Bz) @ (np.linalg.matrix_power(A, t) @ Bz).T
                     for t in range(H))
            gev = np.sort(np.real(np.linalg.eigvalsh((Wc + Wc.T) / 2)))[::-1]
            mn = max(float(gev[-1]), 0.0); mx = float(gev[0])
            log10kappa = float(np.log10(mx / mn)) if mn > 0 else np.inf
            cache[(ml, bl)] = dict(lam=lam, rho=float(np.abs(lam).max()),
                                   gev=gev, log10kappa=log10kappa)
    return cache


def fmt_rho(v):
    if v < 0.95:
        return f"{v:.2f}"
    if v < 0.9995:
        return f"{v:.3f}"
    return f"{v:.4f}"


def panel_title(ax, L, extra):
    ax.set_title(f"({L}) {extra}", fontsize=15, fontweight="bold", loc="left", pad=7, color="#1F2530")


def save_both(fig, name):
    fig.savefig(os.path.join(ROOT, name))                                     # PDF (vector) -> deliverable
    fig.savefig(os.path.join(PREVIEW, name.replace(".pdf", ".png")), dpi=150)  # PNG xem nhanh
    plt.close(fig)
    print("  wrote", name, "(+ preview png)")


# ============ Figure 2 — Koopman spectrum ============
def fig_spectrum(C):
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.6), constrained_layout=True)
    th = np.linspace(0, 2 * np.pi, 260)
    for L, (ml, ms, od), ax in zip(LETTERS, MODELS, axes.flat):
        ax.fill(np.cos(th), np.sin(th), color="#EEF2F7", zorder=0)
        ax.plot(np.cos(th), np.sin(th), "-", color="#94A3B8", lw=1.1, zorder=1)
        ax.axhline(0, color="#CBD5E1", lw=0.7); ax.axvline(0, color="#CBD5E1", lw=0.7)
        for bl, bt, short in BENCH:
            e = C[(ml, bl)]
            ax.scatter(e["lam"].real, e["lam"].imag, s=34, color=BENCH_COLOR[bl], alpha=0.85,
                       edgecolor="white", linewidth=0.5, zorder=3,
                       label=f"{short}  $\\rho$={fmt_rho(e['rho'])}")
        ax.set_aspect("equal"); ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
        ax.set_xlabel(r"$\mathrm{Re}\,\lambda$"); ax.set_ylabel(r"$\mathrm{Im}\,\lambda$")
        ax.grid(False)
        ax.legend(loc="lower left", fontsize=10, framealpha=0.92, edgecolor="#D0D5DD",
                  handletextpad=0.3, borderpad=0.35, labelspacing=0.25)
        panel_title(ax, L, ml)
    save_both(fig, "fig_spectrum.pdf")


# ============ Figure 3 — controllability Gramian spectrum (H = n_x = 16) ============
def fig_controllability(C):
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.8), constrained_layout=True)
    for L, (ml, ms, od), ax in zip(LETTERS, MODELS, axes.flat):
        D = None
        for bl, bt, short in BENCH:
            e = C[(ml, bl)]; w = np.maximum(e["gev"], 1e-16); D = len(w)
            ax.semilogy(range(1, D + 1), w, "-o", ms=4.2, lw=1.6, color=BENCH_COLOR[bl],
                        label=f"{short}  $\\log_{{10}}\\kappa$={e['log10kappa']:.1f}")
        ax.set_xlabel("index $i$"); ax.set_ylabel(r"$\lambda_i(W_H)$")
        ax.set_xlim(0.5, D + 0.5)
        ax.legend(loc="upper right", fontsize=10, framealpha=0.92, edgecolor="#D0D5DD",
                  handletextpad=0.4, borderpad=0.35, labelspacing=0.25)
        panel_title(ax, L, ml)
    save_both(fig, "controllability.pdf")


# ============ Figure 7 — cross-cell relationships ============
def fig_headroom(C):
    def scatter(ax, fx, fy, xlab, ylab, fit=False, diag=False, legend=False):
        xs, ys = [], []
        for ml, ms, od in MODELS:
            X = [fx(ml, bl) for bl, _, _ in BENCH]
            Y = [fy(ml, bl) for bl, _, _ in BENCH]
            ax.scatter(X, Y, s=70, color=MODEL_COLOR[ml], edgecolor="white", linewidth=0.9,
                       label=ml.split("-")[0], zorder=3)
            xs += X; ys += Y
        xs, ys = np.array(xs), np.array(ys)
        if fit:
            b, a = np.polyfit(xs, ys, 1); r = np.corrcoef(xs, ys)[0, 1]
            xf = np.linspace(xs.min(), xs.max(), 50)
            ax.plot(xf, a + b * xf, "--", color="#475569", lw=1.5, zorder=2)
            ax.text(0.04, 0.06, f"Pearson $r={r:.2f}$", transform=ax.transAxes, fontsize=12,
                    bbox=dict(boxstyle="round", fc="white", ec="#CBD5E1", alpha=0.92))
        if diag:
            hi = max(xs.max(), ys.max()) * 1.05
            ax.plot([0, hi], [0, hi], "--", color="#94A3B8", lw=1.2, zorder=1)
            ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        if legend:
            ax.legend(fontsize=10.5, loc="upper right", framealpha=0.92, edgecolor="#D0D5DD",
                      labelspacing=0.25, handletextpad=0.3)

    rho = lambda ml, bl: C[(ml, bl)]["rho"]
    kap = lambda ml, bl: C[(ml, bl)]["log10kappa"]         # H=16, khop Table 3
    delta = lambda ml, bl: pn[(ml, bl)]["delta"]
    base = lambda ml, bl: pn[(ml, bl)]["nodef"]
    asrn = lambda ml, bl: pn[(ml, bl)]["asr_nd"]
    asrd = lambda ml, bl: pn[(ml, bl)]["asr_def"]

    specs = [
        ("a", "Gain vs contraction", rho, delta,
         r"spectral radius $\rho(A)$", r"realized-best gain $\Delta$", dict(fit=True, legend=True)),
        ("b", "Gain vs baseline accuracy", base, delta,
         "baseline accuracy", r"realized-best gain $\Delta$", dict(fit=True)),
        ("c", "Gain vs anisotropy", kap, delta,
         r"$\log_{10}\kappa(W_H)$", r"realized-best gain $\Delta$", dict(fit=True)),
        ("d", "Attack vs defense", asrn, asrd,
         "attack-success rate (no defense)", "attack-success rate (defended)", dict(diag=True, legend=True)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 7.2), constrained_layout=True)
    for (L, title, fx, fy, xl, yl, kw), ax in zip(specs, axes.flat):
        scatter(ax, fx, fy, xl, yl, **kw)
        panel_title(ax, L, title)
    save_both(fig, "fig_headroom.pdf")


def main():
    print("Refit 16 cell + build cache ...")
    C = build_cache()
    print("Ghep hinh -> ROOT:", ROOT)
    fig_spectrum(C)
    fig_controllability(C)
    fig_headroom(C)
    print("Xong. Preview PNG:", os.path.relpath(PREVIEW, ROOT))


if __name__ == "__main__":
    main()

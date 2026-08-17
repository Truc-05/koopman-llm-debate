"""Sinh hình paper từ số ĐÃ verify (paper_numbers.json + analysis_twindef.json) + refit operator.
Xuất PNG vào ROOT (khớp \\includegraphics không-path trong paper.md):
  koopman_spectrum.png, controllability.png  (thay placeholder đang THIẾU, vẽ từ (A,B) thật)
  fig_accuracy.png   — no-def/twin/oracle × 4 bench × 4 model (+ McNemar sao)  [HEADLINE]
  fig_asr.png        — attack(no-def) vs defended ASR (attack–defense duality)
  fig_reach.png      — heatmap reach[x] (answer × cell): anisotropy/positional-bias
  fig_headroom.png   — scatter Δ(oracle−nodef) vs ρ(A), fit + r
Dùng:  python experiments/make_paper_figures.py    (offline, user tự chạy)
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
N, K_DEF = 4, 2

MODELS = [("Qwen2.5-7B", "qwen2_5_7b", OUT_QWEN, "#0072B2"),
          ("Mistral-7B", "mistral_7b", OUT_MAIN, "#D55E00"),
          ("Llama-3.1-8B", "llama3_1_8b", OUT_MAIN, "#009E73"),
          ("Gemma-2-9B", "gemma2_9b", OUT_MAIN, "#CC79A7")]
BENCH = [("MMLU-math", "mmlu_clean6"), ("MATH", "math"), ("TruthfulQA", "truthfulqa"), ("ARC", "arc")]

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "grid.linewidth": 0.5, "figure.dpi": 200, "savefig.bbox": "tight",
                     "axes.axisbelow": True, "figure.facecolor": "white"})

pn = {(c["model"], c["bench"]): c for c in json.load(open(os.path.join(ROOT, "experiments", "paper_numbers.json")))["cells"]}
twd = {OUT_MAIN: json.load(open(os.path.join(OUT_MAIN, "analysis_twindef.json"))),
       OUT_QWEN: json.load(open(os.path.join(OUT_QWEN, "analysis_twindef.json")))}


def twin_acc(mlabel, msuf, out_dir, btag):
    return twd[out_dir][f"{btag}_{msuf}"]["obj"]["honest_margin"]["twin"]["acc"]


def refit_AB(out_dir, tag):
    col = [json.loads(l) for l in open(os.path.join(out_dir, f"ckpt_twindef_collect_{tag}.jsonl")) if l.strip()]
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
    A = np.real(K[ss, ss]); Bs = np.real(B[ss, :]); T = np.asarray(col[0]["traj"]).shape[0] - 1
    return A, Bs, T


def save(fig, name):
    p = os.path.join(ROOT, name); fig.savefig(p); plt.close(fig); print("  wrote", name)


# ---------- FIG accuracy ----------
def fig_accuracy():
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.5), sharey=True)
    x = np.arange(4); w = 0.27
    for ax, (ml, ms, od, col) in zip(axes, MODELS):
        nd = [pn[(ml, bl)]["nodef"] for bl, _ in BENCH]
        tw = [twin_acc(ml, ms, od, bt) for _, bt in BENCH]
        orc = [pn[(ml, bl)]["oracle"] for bl, _ in BENCH]
        ax.bar(x - w, nd, w, color="#B0B0B0", label="no-def")
        ax.bar(x, tw, w, color="#88C0E0", label="twin (obs.)")
        ax.bar(x + w, orc, w, color=col, label="oracle")
        for i, (bl, _) in enumerate(BENCH):
            if pn[(ml, bl)]["mcnemar_p"] < 0.05:
                ax.text(i + w, orc[i] + 0.02, "$*$", ha="center", fontsize=11)
        ax.set_title(ml, fontsize=9); ax.set_xticks(x)
        ax.set_xticklabels(["MMLU", "MATH", "TQA", "ARC"], rotation=0, fontsize=7.5)
        ax.set_ylim(0, 0.85)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(loc="upper left", fontsize=7, framealpha=0.9)
    fig.suptitle("Accuracy: no-defense vs digital-twin (observation-based) vs oracle  ($*$: McNemar $p<.05$)",
                 fontsize=9, y=1.03)
    save(fig, "fig_accuracy.png")


# ---------- FIG asr ----------
def fig_asr():
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.4), sharey=True)
    x = np.arange(4); w = 0.38
    for ax, (ml, ms, od, col) in zip(axes, MODELS):
        an = [pn[(ml, bl)]["asr_nd"] for bl, _ in BENCH]
        ad = [pn[(ml, bl)]["asr_def"] for bl, _ in BENCH]
        ax.bar(x - w / 2, an, w, color="#D55E00", label="attack (no-def)")
        ax.bar(x + w / 2, ad, w, color="#0072B2", label="defended")
        ax.set_title(ml, fontsize=9); ax.set_xticks(x)
        ax.set_xticklabels(["MMLU", "MATH", "TQA", "ARC"], fontsize=7.5)
        ax.set_ylim(0, 0.7)
    axes[0].set_ylabel("attack success rate")
    axes[0].legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.suptitle("Single-adversary attack-success rate, label-free defense on the same channel $B$",
                 fontsize=9, y=1.03)
    save(fig, "fig_asr.png")


# ---------- FIG reach heatmap ----------
def fig_reach():
    rows, labels = [], []
    for ml, ms, od, _ in MODELS:
        for bl, bt in BENCH:
            rows.append(pn[(ml, bl)]["reach"]); labels.append(f"{ml.split('-')[0]}·{bl.replace('-math','')}")
    M = np.array(rows)
    fig, ax = plt.subplots(figsize=(3.4, 5.2))
    im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels([f"$x_{i}$" for i in range(M.shape[1])])
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("target answer"); ax.grid(False)
    ax.set_xticks(np.arange(-.5, M.shape[1], 1), minor=True)      # white separators between cells
    ax.set_yticks(np.arange(-.5, M.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if M[i, j] > 0.55 else "#08306b")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="reach$(x)$")
    ax.set_title("Per-answer reachability\n(anisotropy / positional bias)", fontsize=8.5)
    save(fig, "fig_reach.png")


# ---------- FIG headroom scatter ----------
def fig_headroom():
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    xs, ys = [], []
    for ml, ms, od, col in MODELS:
        rx = [pn[(ml, bl)]["rho"] for bl, _ in BENCH]; ry = [pn[(ml, bl)]["delta"] for bl, _ in BENCH]
        ax.scatter(rx, ry, s=34, color=col, label=ml.split("-")[0], zorder=3, edgecolor="k", linewidth=0.4)
        xs += rx; ys += ry
    xs, ys = np.array(xs), np.array(ys)
    b, a = np.polyfit(xs, ys, 1)
    r = np.corrcoef(xs, ys)[0, 1]
    xf = np.linspace(xs.min(), xs.max(), 50)
    ax.plot(xf, a + b * xf, "--", color="0.3", lw=1, zorder=2)
    ax.text(0.05, 0.06, f"Pearson $r={r:.2f}$", transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.set_xlabel(r"spectral radius $\rho(A)$"); ax.set_ylabel(r"oracle gain $\Delta$")
    ax.legend(fontsize=7, loc="upper right"); ax.set_title("Headroom: gain vs contraction", fontsize=8.5)
    save(fig, "fig_headroom.png")


# ---------- FIG spectrum (replaces missing koopman_spectrum.png) ----------
def fig_spectrum():
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), "-", color="0.5", lw=0.8)
    for ml, ms, od, col in MODELS:
        A, _, _ = refit_AB(od, f"mmlu_clean6_{ms}")
        lam = np.linalg.eigvals(A)
        ax.scatter(lam.real, lam.imag, s=22, color=col, label=f"{ml.split('-')[0]} ($\\rho$={np.abs(lam).max():.2f})",
                   alpha=0.85, edgecolor="k", linewidth=0.3, zorder=3)
    ax.axhline(0, color="0.8", lw=0.5); ax.axvline(0, color="0.8", lw=0.5)
    ax.set_xlabel(r"$\mathrm{Re}\,\lambda$"); ax.set_ylabel(r"$\mathrm{Im}\,\lambda$")
    ax.set_aspect("equal"); ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=6.5, loc="lower left"); ax.set_title("Koopman spectrum (MMLU-math)", fontsize=8.5)
    save(fig, "koopman_spectrum.png")


# ---------- FIG controllability (replaces missing controllability.png) ----------
def fig_controllability():
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    for ml, ms, od, col in MODELS:
        A, Bs, T = refit_AB(od, f"mmlu_clean6_{ms}")
        D = A.shape[0]; W = np.zeros((D, D)); Ak = np.eye(D)
        for _ in range(T):
            W += Ak @ Bs @ Bs.T @ Ak.T; Ak = A @ Ak
        w = np.sort(np.linalg.eigvalsh(W))[::-1]
        k = w[0] / max(w[-1], 1e-15)
        ax.semilogy(range(1, D + 1), w, "-o", ms=3, color=col, lw=1,
                    label=f"{ml.split('-')[0]} ($\\log_{{10}}\\kappa$={np.log10(k):.1f})")
    ax.set_xlabel("index $i$"); ax.set_ylabel(r"$\lambda_i(W_H)$  (reachable semi-axes$^2$)")
    ax.legend(fontsize=6.5, loc="upper right"); ax.set_title("Controllability Gramian spectrum (MMLU-math)", fontsize=8.5)
    save(fig, "controllability.png")


def main():
    print("Sinh hình →", ROOT)
    for f in (fig_accuracy, fig_asr, fig_reach, fig_headroom, fig_spectrum, fig_controllability):
        try:
            f()
        except Exception as e:
            print("  [SKIP]", f.__name__, "->", repr(e))
    print("Xong. (overall.png = sơ đồ framework, vẽ tay/tikz — chưa tự sinh.)")


if __name__ == "__main__":
    main()

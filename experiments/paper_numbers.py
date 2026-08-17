"""SỐ LIỆU PAPER (analysis-only, offline, KHÔNG debate mới) — hợp nhất mọi con số §Experiments cần cho
4 model × 4 benchmark LÕI (mmlu_clean6, math, truthfulqa, arc), gộp cả out/ (mistral/llama/gemma) và
out_qwen/ (qwen), theo ĐÚNG công thức của analyze_twindef (honest-margin oracle, McNemar exact).

In ra 4 khối:
  [A] PER-CELL   — tái lập Table 1 (no-def/oracle/Δ/p/ASR) + CI95 per-cell + ρ(A) + reach-index.
  [B] PER-MODEL  — pooled Δ trên 4 bench lõi + stratified bootstrap CI95 (CHƯA script nào in), sig-count
                   (McNemar & CI), ASR max no-def→def, P(oracle-pick==true) range, ρ(A)/reach range.
  [C] HEADROOM   — tương quan qua 16 cell: Δ vs no-def-acc và Δ vs ρ(A) (Pearson+Spearman, p hoán vị).
  [D] CLAIMS     — vài dòng đúc sẵn để dán thẳng vào paper.

Ghi experiments/paper_numbers.json. Dùng:  python experiments/paper_numbers.py    (user tự chạy)
Chỉnh OUT nếu cần:  OUT_MAIN=experiments/out  OUT_QWEN=experiments/out_qwen  python experiments/paper_numbers.py
"""
import os, sys, json
import numpy as np, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import fit_edmdc
from src.debate.state import mean_belief
from experiments.run_twin_mpc_defense import wilson, mcnemar_exact, honest_lead, K_DEF

cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "debate.yaml")))
N = cfg["n_agents"]
rng = np.random.default_rng(0)

OUT_MAIN = os.environ.get("OUT_MAIN", os.path.join(ROOT, "experiments", "out"))
OUT_QWEN = os.environ.get("OUT_QWEN", os.path.join(ROOT, "experiments", "out_qwen"))

# (label, tag-suffix, OUT-dir) — thứ tự = thứ tự trong paper (2 model 4/4 trước)
MODELS = [
    ("Qwen2.5-7B",  "qwen2_5_7b",  OUT_QWEN),
    ("Mistral-7B",  "mistral_7b",  OUT_MAIN),
    ("Llama-3.1-8B", "llama3_1_8b", OUT_MAIN),
    ("Gemma-2-9B",  "gemma2_9b",   OUT_MAIN),
]
BENCH = [("MMLU-math", "mmlu_clean6"), ("MATH", "math"), ("TruthfulQA", "truthfulqa"), ("ARC", "arc")]


def boot_ci(diffs, B=5000):
    d = np.asarray(diffs, float)
    if len(d) == 0:
        return (float("nan"), float("nan"))
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def outcomes(out_dir, tag):
    """Đọc eval jsonl → per-debate outcomes theo honest-margin oracle (khớp analyze_twindef)."""
    ev = [json.loads(l) for l in open(os.path.join(out_dir, f"ckpt_twindef_eval_{tag}.jsonl")) if l.strip()]
    Kans = len(next(iter(ev[0]["outs"].values()))["bel"]); cands = [-1] + list(range(Kans))
    nd_c, ex_c, nd_a, ex_a, exc_true = [], [], [], [], []
    for r in ev:
        x = r["x_adv"]; a = r["a_star"]; outs = r["outs"]; z = np.asarray(r["z_kdef"])
        ch = honest_lead(z, x, N, Kans)
        exc = max(cands, key=lambda c: np.asarray(outs[str(c)]["bel"])[ch] - np.asarray(outs[str(c)]["bel"])[x])
        nd, ex = outs["-1"], outs[str(exc)]
        nd_c.append(bool(nd["correct"])); ex_c.append(bool(ex["correct"]))
        nd_a.append(int(np.argmax(nd["bel"]) == x)); ex_a.append(int(np.argmax(ex["bel"]) == x))
        exc_true.append(int(exc == a))
    n = len(ev)
    diffs = [int(b) - int(c) for b, c in zip(ex_c, nd_c)]
    lo, hi = boot_ci(diffs)
    return {
        "n": n, "Kans": Kans, "diffs": diffs,
        "nodef": float(np.mean(nd_c)), "oracle": float(np.mean(ex_c)),
        "delta": float(np.mean(diffs)), "ci": [lo, hi],
        "mcnemar_p": mcnemar_exact(ex_c, nd_c)[2],
        "asr_nd": float(np.mean(nd_a)), "asr_def": float(np.mean(ex_a)),
        "p_pick_true": float(np.mean(exc_true)),
    }


def rho_reach(out_dir, tag):
    """Refit EDMDc degree-1 trên collect → ρ(A), reach-index, ctrb-rank, log10 κ(Wc) (khớp control_analysis)."""
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
    T = np.asarray(col[0]["traj"]).shape[0] - 1
    A = np.real(K[ss, ss]); Bs = np.real(B[ss, :])
    rho = float(np.abs(np.linalg.eigvals(A)).max())
    # ctrb rank + Gramian cond
    Ak = np.eye(D); blocks = [Bs]
    for _ in range(1, D):
        Ak = A @ Ak; blocks.append(Ak @ Bs)
    rank = int(np.linalg.matrix_rank(np.hstack(blocks), tol=1e-6))
    W = np.zeros((D, D)); Ak = np.eye(D)
    for _ in range(T):
        W += Ak @ Bs @ Bs.T @ Ak.T; Ak = A @ Ak
    w = np.linalg.eigvalsh(W); log10kappa = float(np.log10(w.max() / max(w.min(), 1e-15)))
    # reachability per target từ uniform
    z0 = np.zeros(D); reach = []
    for x in range(Kans):
        psi = d.transform(z0.reshape(1, -1))[0]; u = np.zeros(Kans); u[x] = 1.0
        for _ in range(T):
            psi = K @ psi + B @ u
        reach.append(float(mean_belief(np.real(psi[ss]), N, Kans)[x]))
    return {"rho": rho, "rank": rank, "dim": D, "log10kappa": log10kappa,
            "reach": reach, "reach_index": float(np.mean(reach))}


def perm_corr(x, y, B=20000, kind="pearson"):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if kind == "spearman":
        x = np.argsort(np.argsort(x)).astype(float); y = np.argsort(np.argsort(y)).astype(float)

    def r_(a, b):
        a = a - a.mean(); b = b - b.mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / den) if den > 0 else 0.0
    r0 = r_(x, y)
    cnt = sum(abs(r_(x, y[rng.permutation(len(y))])) >= abs(r0) - 1e-12 for _ in range(B))
    return r0, (cnt + 1) / (B + 1)


def main():
    cells, per_model = [], {}
    for label, suf, out_dir in MODELS:
        rows = []
        for bl, bt in BENCH:
            tag = f"{bt}_{suf}"
            o = outcomes(out_dir, tag); rr = rho_reach(out_dir, tag)
            row = {"model": label, "bench": bl, **o, **rr}
            rows.append(row); cells.append(row)
        per_model[label] = rows

    # ---------- [A] PER-CELL ----------
    print("\n===== [A] PER-CELL  (honest-margin oracle; reproduces Table 1 + CI95 + ρ(A) + reach) =====")
    h = f"{'model':12s} {'bench':10s} {'n':>3s} {'nodef':>6s} {'orac':>6s} {'Δ':>6s} {'p(McN)':>7s} " \
        f"{'CI95':>16s} {'ASR nd→def':>11s} {'P(pk=T)':>7s} {'ρ(A)':>5s} {'reach':>5s} {'rank':>6s}"
    print(h)
    for r in cells:
        print(f"{r['model']:12s} {r['bench']:10s} {r['n']:>3d} {r['nodef']:>6.3f} {r['oracle']:>6.3f} "
              f"{r['delta']:>+6.3f} {r['mcnemar_p']:>7.3f} [{r['ci'][0]:>+.3f},{r['ci'][1]:>+.3f}] "
              f"{r['asr_nd']:>4.2f}→{r['asr_def']:<4.2f} {r['p_pick_true']:>7.3f} {r['rho']:>5.2f} "
              f"{r['reach_index']:>5.2f} {str(r['rank'])+'/'+str(r['dim']):>6s}")

    # ---------- [B] PER-MODEL POOLED ----------
    print("\n===== [B] PER-MODEL POOLED over 4 core benchmarks (stratified bootstrap CI95) =====")
    print(f"{'model':12s} {'pooledΔ':>8s} {'CI95':>18s} {'sig McN':>8s} {'sig CI':>7s} "
          f"{'ASRmax nd→def':>13s} {'P(pk=T)rng':>13s} {'ρ(A) rng':>13s} {'reach rng':>13s}")
    summ = {}
    for label, rows in per_model.items():
        strata = [np.asarray(r["diffs"], float) for r in rows]
        pt = float(np.mean([s.mean() for s in strata]))
        bs = [np.mean([s[rng.integers(0, len(s), len(s))].mean() for s in strata]) for _ in range(5000)]
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        sig_mcn = sum(r["mcnemar_p"] < 0.05 for r in rows)
        sig_ci = sum(r["ci"][0] > 0 for r in rows)
        asr_nd = max(r["asr_nd"] for r in rows); asr_def = max(r["asr_def"] for r in rows)
        pk = [r["p_pick_true"] for r in rows]; rho = [r["rho"] for r in rows]; rch = [r["reach_index"] for r in rows]
        summ[label] = {"pooled": pt, "ci": [lo, hi], "sig_mcn": sig_mcn, "sig_ci": sig_ci,
                       "asr_nd_max": asr_nd, "asr_def_max": asr_def,
                       "pk_range": [min(pk), max(pk)], "rho_range": [min(rho), max(rho)],
                       "reach_range": [min(rch), max(rch)]}
        print(f"{label:12s} {pt:>+8.3f} [{lo:>+.3f},{hi:>+.3f}]{'*' if lo>0 else ' '} "
              f"{str(sig_mcn)+'/4':>8s} {str(sig_ci)+'/4':>7s} "
              f"{asr_nd:>4.2f}→{asr_def:<4.2f}   {min(pk):>.2f}–{max(pk):<.2f}   "
              f"{min(rho):>.2f}–{max(rho):<.2f}   {min(rch):>.2f}–{max(rch):<.2f}")

    # ---------- [C] HEADROOM CORRELATION ----------
    print("\n===== [C] HEADROOM CORRELATION over 16 cells (perm p, 20k) =====")
    nodef = [r["nodef"] for r in cells]; delta = [r["delta"] for r in cells]; rho = [r["rho"] for r in cells]
    for name, xv in [("no-def acc", nodef), ("ρ(A)", rho)]:
        rp, pp = perm_corr(xv, delta, kind="pearson")
        rs, ps = perm_corr(xv, delta, kind="spearman")
        print(f"  Δ vs {name:11s}: Pearson r={rp:>+.3f} (p={pp:.3f}) | Spearman ρ={rs:>+.3f} (p={ps:.3f})")
    print("  (âm với no-def acc = base thấp ⇒ Δ lớn = HEADROOM; âm với ρ(A) = càng ít co ⇒ Δ nhỏ)")

    # ---------- [D] CLAIMS ----------
    print("\n===== [D] PAPER-READY =====")
    n44 = [m for m, s in summ.items() if s["sig_mcn"] == 4]
    print(f"  • {len(n44)}/4 model đạt 4/4 sig (McNemar): {n44}")
    for m, s in summ.items():
        print(f"  • {m:12s}: pooled Δ={s['pooled']*100:+.0f} pts CI[{s['ci'][0]*100:+.0f},{s['ci'][1]*100:+.0f}]"
              f" | {s['sig_mcn']}/4 McN, {s['sig_ci']}/4 CI"
              f" | attack≤{s['asr_nd_max']*100:.0f}%→def≤{s['asr_def_max']*100:.0f}%"
              f" | ρ(A) {s['rho_range'][0]:.2f}–{s['rho_range'][1]:.2f}")

    json.dump({"cells": [{k: v for k, v in r.items() if k != "diffs"} for r in cells], "per_model": summ},
              open(os.path.join(ROOT, "experiments", "paper_numbers.json"), "w"), indent=1)
    print("\n→ experiments/paper_numbers.json")


if __name__ == "__main__":
    main()

"""Số liệu OFFLINE trả lời reviewer (KHÔNG gọi LLM). Chạy từ ckpt đã commit.
  M4    exact rho(A) 4 chữ số + có cell nào cần chiếu |lambda|>1 không
  STATS Holm + Benjamini-Hochberg trên 16 p McNemar (oracle vs no-def)
  M8    truth-agnosticity thực nghiệm: Δm(push đúng) vs Δm(push sai) + TOST
  M6    sample-stability: rho, r90, reach-index, subspace ổn định theo n∈{20,40,60}
  M7    overlap-metric giải tích (k/d) + transfer positional reach giữa model
  M9    baseline tĩnh: direct-linear, reduced-rank(2), channel-mean vs Koopman vs oracle
Dùng:  /home/alex/venvs/env/bin/python experiments/reviewer_analysis.py
"""
import os, sys, json, glob
from math import comb
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import fit_edmdc
from src.debate.state import mean_belief, split_state

N, K_DEF = 4, 2
OUT_MAIN, OUT_QWEN = os.path.join(ROOT, "experiments", "out"), os.path.join(ROOT, "experiments", "out_qwen")
MODELS = [("Qwen", "qwen2_5_7b", OUT_QWEN), ("Mistral", "mistral_7b", OUT_MAIN),
          ("Llama", "llama3_1_8b", OUT_MAIN), ("Gemma", "gemma2_9b", OUT_MAIN)]
BENCH = ["mmlu_clean6", "math", "truthfulqa", "arc"]
RNG = np.random.default_rng(0)


def load_collect(od, tag):
    return [json.loads(l) for l in open(os.path.join(od, f"ckpt_twindef_collect_{tag}.jsonl")) if l.strip()]


def load_eval(od, tag):
    p = os.path.join(od, f"ckpt_twindef_eval_{tag}.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def fit_AB(col):
    """(A, B_state, Kop_full, B_full, dict, ss, Kans, T) từ danh sách trajectory."""
    D = np.asarray(col[0]["traj"]).shape[1]; Kans = D // N; T = np.asarray(col[0]["traj"]).shape[0] - 1
    d = PolynomialDictionary(degree=1); d.transform(np.asarray(col[0]["traj"])[:1]); ss = d.state_slice
    X, Y, U = [], [], []
    for r in col:
        Z = np.asarray(r["traj"], float)
        for t in range(len(Z) - 1):
            u = np.zeros(Kans)
            if r["c"] >= 0 and t >= K_DEF: u[r["c"]] = 1.0
            X.append(Z[t]); Y.append(Z[t + 1]); U.append(u)
    K, B = fit_edmdc(d.transform(np.array(X)).T, d.transform(np.array(Y)).T, np.array(U).T, reg=1e-6)
    return np.real(K[ss, ss]), np.real(B[ss, :]), K, B, d, ss, Kans, T


def reach_svals(A, B, H=4):
    Ak = np.eye(A.shape[0]); blocks = [B]
    for _ in range(1, H):
        Ak = A @ Ak; blocks.append(Ak @ B)
    return np.linalg.svd(np.hstack(blocks), full_matrices=False)   # U, s, Vt


def eff_rank(s, thr=0.9):
    e = np.cumsum(s ** 2) / np.sum(s ** 2); return int(np.searchsorted(e, thr) + 1)


def subspace_overlap(Ua, Ub, k=2):
    return float(np.sum((Ua[:, :k].T @ Ub[:, :k]) ** 2) / k)


def mcnemar_p(a, b):
    """exact two-sided McNemar; a,b bool arrays (correct)."""
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    n01 = int(np.sum(a & ~b)); n10 = int(np.sum(~a & b))     # a-only, b-only
    n = n01 + n10
    if n == 0: return 1.0, n01, n10
    k = min(n01, n10)
    p = 2.0 * sum(comb(n, i) for i in range(0, k + 1)) * 0.5 ** n
    return min(1.0, p), n01, n10


def holm(ps):
    m = len(ps); order = np.argsort(ps); adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * ps[idx]; run = max(run, val); adj[idx] = min(1.0, run)
    return adj


def bh(ps):
    m = len(ps); order = np.argsort(ps); adj = np.empty(m); run = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]; val = ps[idx] * m / (rank + 1); run = min(run, val); adj[idx] = min(1.0, run)
    return adj


def honest_lead(z, x, Kans):
    _, p = split_state(np.asarray(z), N, Kans); h = p[1:].mean(0).astype(float); h[x] = -np.inf
    return int(np.argmax(h))


# ============================================================ M4 + M6 (per-cell fits)
print("=" * 70, "\n[M4] exact rho(A) và projection check\n", "=" * 70)
CELLS = {}
rows = []
for ml, ms, od in MODELS:
    for bt in BENCH:
        tag = f"{bt}_{ms}"
        try:
            col = load_collect(od, tag)
        except FileNotFoundError:
            continue
        A, Bs, K, B, d, ss, Kans, T = fit_AB(col)
        lam = np.linalg.eigvals(A); rho = float(np.abs(lam).max())
        ngt1 = int(np.sum(np.abs(lam) > 1.0 + 1e-9))
        Kl = np.linalg.eigvals(K); nKgt1 = int(np.sum(np.abs(Kl) > 1.0 + 1e-9))
        U, s, _ = reach_svals(A, Bs)
        CELLS[(ml, bt)] = dict(A=A, Bs=Bs, T=T, Kans=Kans, col=col, rho=rho, U=U, s=s,
                               r90=eff_rank(s), reach_idx=float("nan"))
        rows.append((ml, bt, rho, ngt1, nKgt1))
        print(f"  {ml:8s} {bt:12s} rho(A)={rho:.4f}  |lam(A)|>1: {ngt1}  |lam(K)|>1: {nKgt1}")
print(f"  --> cells needing projection on A: {sum(r[3] for r in rows)}/{len(rows)} ;"
      f" on full K: {sum(r[4] for r in rows)}/{len(rows)}")

print("\n" + "=" * 70, "\n[M6] sample-stability (n=20,40,60), MMLU cell per model, R=40 resamples\n", "=" * 70)
for ml, ms, od in MODELS:
    key = (ml, "mmlu_clean6")
    if key not in CELLS: continue
    col = CELLS[key]["col"]; ndeb = len(col)
    A0, B0, _, _, d0, ss0, Kans0, T0 = fit_AB(col)
    U0, _, _ = reach_svals(A0, B0)
    print(f"  {ml} (full n={ndeb}):  rho={np.abs(np.linalg.eigvals(A0)).max():.3f}  r90={eff_rank(reach_svals(A0,B0)[1])}")
    for nsub in (20, 40, 60):
        if nsub > ndeb: continue
        rr, r9, ov = [], [], []
        for _ in range(40):
            idx = RNG.choice(ndeb, size=nsub, replace=False)
            sub = [col[i] for i in idx]
            As, Bss, *_ = fit_AB(sub)
            rr.append(np.abs(np.linalg.eigvals(As)).max())
            Us, ss_, _ = reach_svals(As, Bss); r9.append(eff_rank(ss_)); ov.append(subspace_overlap(U0, Us))
        print(f"      n={nsub:3d}: rho={np.mean(rr):.3f}±{np.std(rr):.3f}  "
              f"r90={np.mean(r9):.2f}±{np.std(r9):.2f}  subspace-overlap-vs-full={np.mean(ov):.2f}±{np.std(ov):.2f}")

# ============================================================ STATS: McNemar + corrections
print("\n" + "=" * 70, "\n[STATS] McNemar oracle(honest-margin realized-best) vs no-def + Holm/BH\n", "=" * 70)
pvals, labels = [], []
for ml, ms, od in MODELS:
    A0 = None
    for bt in BENCH:
        tag = f"{bt}_{ms}"; ev = load_eval(od, tag)
        if not ev: continue
        nd, orc = [], []
        for r in ev:
            outs = r["outs"]; x = r["x_adv"]; z = np.asarray(r["z_kdef"])
            Ka = len([k for k in outs if int(k) >= 0]); c_hon = honest_lead(z, x, Ka)
            cands = [int(k) for k in outs]
            ex_c = max(cands, key=lambda c: outs[str(c)]["bel"][c_hon] - outs[str(c)]["bel"][x])
            orc.append(bool(outs[str(ex_c)]["correct"])); nd.append(bool(outs["-1"]["correct"]))
        p, b, c = mcnemar_p(orc, nd)
        pvals.append(p); labels.append(f"{ml}-{bt}")
        print(f"  {ml:8s} {bt:12s} nodef={np.mean(nd):.3f} oracle={np.mean(orc):.3f}  McNemar p={p:.4f}  (+{b}/-{c})")
pvals = np.array(pvals)
h = holm(pvals); q = bh(pvals)
print(f"\n  raw p<.05: {int(np.sum(pvals<.05))}/{len(pvals)}   Holm<.05: {int(np.sum(h<.05))}   BH-FDR<.05: {int(np.sum(q<.05))}")
for lab, p, hp, qp in sorted(zip(labels, pvals, h, q), key=lambda t: t[1]):
    print(f"    {lab:16s} p={p:.4f}  Holm={hp:.4f}  BH={qp:.4f}  {'*' if qp<.05 else ''}")

# ============================================================ M8: truth-agnosticity
print("\n" + "=" * 70, "\n[M8] truth-agnosticity: realized push-effect Δm(pushed) correct vs wrong target\n", "=" * 70)
for ml, ms, od in MODELS:
    dcorr, dwrong = [], []          # per-trajectory mean Δm on pushed coordinate
    for bt in BENCH:
        try:
            col = load_collect(od, f"{bt}_{ms}")
        except FileNotFoundError:
            continue
        Kans = np.asarray(col[0]["traj"]).shape[1] // N
        for r in col:
            c = r["c"]
            if c < 0: continue
            Z = np.asarray(r["traj"], float); eff = []
            for t in range(K_DEF, len(Z) - 1):
                eff.append(mean_belief(Z[t + 1], N, Kans)[c] - mean_belief(Z[t], N, Kans)[c])
            if not eff: continue
            (dcorr if c == r["a_star"] else dwrong).append(float(np.mean(eff)))
    dcorr, dwrong = np.array(dcorr), np.array(dwrong)
    diff = dcorr.mean() - dwrong.mean()
    # cluster (trajectory) bootstrap on the difference
    bs = []
    for _ in range(3000):
        a = dcorr[RNG.integers(0, len(dcorr), len(dcorr))]; b = dwrong[RNG.integers(0, len(dwrong), len(dwrong))]
        bs.append(a.mean() - b.mean())
    lo, hi = np.percentile(bs, [5, 95])
    tost = (lo > -0.05) and (hi < 0.05)
    print(f"  {ml:8s} Δm(correct)={dcorr.mean():+.3f} (n={len(dcorr)})  Δm(wrong)={dwrong.mean():+.3f} (n={len(dwrong)})  "
          f"diff={diff:+.3f} 90%CI[{lo:+.3f},{hi:+.3f}]  TOST±.05 {'EQUIV' if tost else 'no'}")

# ============================================================ M7: overlap metric + transfer
print("\n" + "=" * 70, "\n[M7] overlap-metric analytic + positional-reach transfer across models\n", "=" * 70)
d_state = 16; k_sub = 2
print(f"  analytic random overlap E = k/d = {k_sub}/{d_state} = {k_sub/d_state:.3f}  (matches reported 0.13)")
pn = {(c["model"], c["bench"]): c for c in json.load(open(os.path.join(ROOT, "experiments", "paper_numbers.json")))["cells"]}
MLAB = {"Qwen": "Qwen2.5-7B", "Mistral": "Mistral-7B", "Llama": "Llama-3.1-8B", "Gemma": "Gemma-2-9B"}
BLAB = {"mmlu_clean6": "MMLU-math", "math": "MATH", "truthfulqa": "TruthfulQA", "arc": "ARC"}
from itertools import combinations
def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])
per_bench = {}
for bt in BENCH:
    corrs = []
    for (m1, _, _), (m2, _, _) in combinations(MODELS, 2):
        r1 = pn.get((MLAB[m1], BLAB[bt], ), {}).get("reach"); r2 = pn.get((MLAB[m2], BLAB[bt]), {}).get("reach")
        if r1 and r2: corrs.append(spearman(r1, r2))
    if corrs:
        per_bench[bt] = np.mean(corrs)
        print(f"  {BLAB[bt]:12s} mean cross-model Spearman(reach vector) = {np.mean(corrs):+.2f}  (pairs={len(corrs)})")
allc = []
for bt in BENCH:
    for (m1, _, _), (m2, _, _) in combinations(MODELS, 2):
        r1 = pn.get((MLAB[m1], BLAB[bt]), {}).get("reach"); r2 = pn.get((MLAB[m2], BLAB[bt]), {}).get("reach")
        if r1 and r2: allc.append(spearman(r1, r2))
print(f"  POOLED positional-reach transfer Spearman = {np.mean(allc):+.2f} (n_pairs={len(allc)}, random≈0)")

# ============================================================ M9: static baselines vs Koopman vs oracle
print("\n" + "=" * 70, "\n[M9] response predictors: Koopman vs direct-linear vs reduced-rank(2) vs channel-mean\n", "=" * 70)
def feats(z, Kans):
    z = np.asarray(z, float); b = mean_belief(z, N, Kans); return np.concatenate([[1.0], z, b])   # 1+16+K
def onehot(c, Kans):
    u = np.zeros(Kans)
    if c >= 0: u[c] = 1.0
    return u

for ml, ms, od in MODELS:
    # gộp 4 benchmark
    per_pred = {p: dict(sel=[], nd=[], eff_pred=[], eff_real=[], orc=[]) for p in
                ("koop", "direct", "rrr2", "chanmean")}
    for bt in BENCH:
        tag = f"{bt}_{ms}"
        try:
            col = load_collect(od, tag)
        except FileNotFoundError:
            continue
        ev = load_eval(od, tag)
        if not ev: continue
        A, Bs, Kop, Bm, dct, ss, Kans, T = fit_AB(col)
        # ---- train direct/reduced-rank/channel-mean từ collect: (feats(z_kdef), onehot(c)) -> final belief
        Xtr, Utr, Ytr = [], [], []
        chan_bel = {c: [] for c in range(-1, Kans)}
        for r in col:
            Z = np.asarray(r["traj"], float); c = r["c"]
            zk = Z[K_DEF]; yb = mean_belief(Z[-1], N, Kans)
            Xtr.append(np.concatenate([feats(zk, Kans), onehot(c, Kans)])); Ytr.append(yb)
            chan_bel[c].append(yb)
        Xtr, Ytr = np.array(Xtr), np.array(Ytr)
        Wd = np.linalg.solve(Xtr.T @ Xtr + 1.0 * np.eye(Xtr.shape[1]), Xtr.T @ Ytr)   # direct ridge (F×K)
        Uw, sw, Vw = np.linalg.svd(Wd, full_matrices=False)                            # reduced-rank
        Wr = (Uw[:, :2] * sw[:2]) @ Vw[:2]
        cmean = {c: (np.mean(v, 0) if v else None) for c, v in chan_bel.items()}
        def koop_pred(zk, c):
            psi = dct.transform(np.asarray(zk).reshape(1, -1))[0]
            for _ in range(K_DEF, T):
                psi = Kop @ psi + Bm @ onehot(c, Kans)
            return mean_belief(np.real(psi[ss]), N, Kans)
        def direct_pred(zk, c, Wm):
            return np.asarray(np.concatenate([feats(zk, Kans), onehot(c, Kans)]) @ Wm)
        predictors = {"koop": lambda zk, c: koop_pred(zk, c),
                      "direct": lambda zk, c: direct_pred(zk, c, Wd),
                      "rrr2": lambda zk, c: direct_pred(zk, c, Wr),
                      "chanmean": lambda zk, c: (cmean[c] if cmean[c] is not None else cmean[-1])}
        cands = list(range(-1, Kans))
        for r in ev:
            outs = r["outs"]; x = r["x_adv"]; z = np.asarray(r["z_kdef"]); c_hon = honest_lead(z, x, Kans)
            real_nd = np.asarray(outs["-1"]["bel"])
            for pname, pf in predictors.items():
                sel = max([c for c in cands if str(c) in outs],
                          key=lambda c: (pf(z, c)[c_hon] - pf(z, c)[x]))
                per_pred[pname]["sel"].append(bool(outs[str(sel)]["correct"]))
                per_pred[pname]["nd"].append(bool(outs["-1"]["correct"]))
                # oracle realized best-of-K (chung cho mọi predictor)
                ex = max([c for c in cands if str(c) in outs],
                         key=lambda c: outs[str(c)]["bel"][c_hon] - outs[str(c)]["bel"][x])
                per_pred[pname]["orc"].append(bool(outs[str(ex)]["correct"]))
                # effect-corr (chỉ push>=0)
                for c in range(Kans):
                    if str(c) in outs:
                        per_pred[pname]["eff_pred"].append(float((pf(z, c) - pf(z, -1))[c]))
                        per_pred[pname]["eff_real"].append(float(np.asarray(outs[str(c)]["bel"])[c] - real_nd[c]))
    print(f"  --- {ml} (pooled 4 bench, n_q={len(per_pred['koop']['sel'])}) ---")
    orc_acc = np.mean(per_pred["koop"]["orc"]); nd_acc = np.mean(per_pred["koop"]["nd"])
    print(f"      no-def={nd_acc:.3f}  oracle(realized best-of-K)={orc_acc:.3f}")
    for pname in ("koop", "direct", "rrr2", "chanmean"):
        d = per_pred[pname]
        sel = np.mean(d["sel"]); ep = np.array(d["eff_pred"]); er = np.array(d["eff_real"])
        r = float(np.corrcoef(ep, er)[0, 1]) if ep.std() > 1e-9 else 0.0
        print(f"      {pname:9s} select-acc={sel:.3f}  effect-corr={r:+.2f}")
print("\nDONE.")

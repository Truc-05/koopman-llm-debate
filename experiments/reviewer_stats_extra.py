"""Inferential add-ons for reviewer (OFFLINE, no LLM): CI + permutation p for
  (1) cross-model vulnerability-subspace overlap (top-2 reachability subspace, ARC cell),
  (2) positional-reach transfer Spearman (per-answer reach vectors, all 16 cells).
Reuses the exact fit/reach/overlap conventions of paper_numbers.py + make_supp_figures.py.
Run: /home/alex/venvs/env/bin/python experiments/reviewer_stats_extra.py
"""
import os, sys, json
from itertools import combinations
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import fit_edmdc
from src.debate.state import mean_belief

N, K_DEF = 4, 2
OUT_MAIN = os.path.join(ROOT, "experiments", "out")
OUT_QWEN = os.path.join(ROOT, "experiments", "out_qwen")
MODELS = [("Qwen", "qwen2_5_7b", OUT_QWEN), ("Mistral", "mistral_7b", OUT_MAIN),
          ("Llama", "llama3_1_8b", OUT_MAIN), ("Gemma", "gemma2_9b", OUT_MAIN)]
BENCH = ["mmlu_clean6", "math", "truthfulqa", "arc"]
RNG = np.random.default_rng(0)


def load_collect(od, tag):
    return [json.loads(l) for l in open(os.path.join(od, f"ckpt_twindef_collect_{tag}.jsonl")) if l.strip()]


def fit_cell(col):
    """(K_full, B_full, A, Bs, dict, ss, Kans, T) from a list of trajectory dicts."""
    Z0 = np.asarray(col[0]["traj"], float)
    D = Z0.shape[1]; Kans = D // N; T = Z0.shape[0] - 1
    d = PolynomialDictionary(degree=1); d.transform(Z0[:1]); ss = d.state_slice
    X, Y, U = [], [], []
    for r in col:
        Z = np.asarray(r["traj"], float)
        for t in range(len(Z) - 1):
            u = np.zeros(Kans)
            if r["c"] >= 0 and t >= K_DEF:
                u[r["c"]] = 1.0
            X.append(Z[t]); Y.append(Z[t + 1]); U.append(u)
    K, B = fit_edmdc(d.transform(np.array(X)).T, d.transform(np.array(Y)).T, np.array(U).T, reg=1e-6)
    return K, B, np.real(K[ss, ss]), np.real(B[ss, :]), d, ss, Kans, T


def reach_vec(K, B, d, ss, Kans, T):
    """per-answer reachability from a uniform prior (matches paper_numbers.rho_reach)."""
    D = K[ss, ss].shape[0]; z0 = np.zeros(D); out = []
    for x in range(Kans):
        psi = d.transform(z0.reshape(1, -1))[0]; u = np.zeros(Kans); u[x] = 1.0
        for _ in range(T):
            psi = K @ psi + B @ u
        out.append(float(mean_belief(np.real(psi[ss]), N, Kans)[x]))
    return np.array(out)


def U2(A, Bs, H=4):
    Ak = np.eye(A.shape[0]); blocks = [Bs]
    for _ in range(1, H):
        Ak = A @ Ak; blocks.append(Ak @ Bs)
    Uu, _, _ = np.linalg.svd(np.hstack(blocks), full_matrices=False)
    return Uu[:, :2]


def sim2(a, b):
    """mean squared principal cosine of two top-2 subspaces in the shared R^16 basis."""
    return float(np.sum((a[:, :2].T @ b[:, :2]) ** 2) / 2)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def resample(col):
    idx = RNG.integers(0, len(col), len(col))
    return [col[i] for i in idx]


# ======================================================= load everything once
CELLS = {}                    # (model,bench) -> col
for ml, ms, od in MODELS:
    for bt in BENCH:
        try:
            CELLS[(ml, bt)] = load_collect(od, f"{bt}_{ms}")
        except FileNotFoundError:
            pass

# ======================================================= (1) OVERLAP: ARC cell
print("=" * 72, "\n(1) CROSS-MODEL VULNERABILITY-SUBSPACE OVERLAP (top-2, ARC cell)\n", "=" * 72)
arcU = {}
for ml, ms, od in MODELS:
    _, _, A, Bs, *_ = fit_cell(CELLS[(ml, "arc")])
    arcU[ml] = U2(A, Bs)
pairs = list(combinations([m[0] for m in MODELS], 2))
obs_ov = {p: sim2(arcU[p[0]], arcU[p[1]]) for p in pairs}
obs_vals = np.array(list(obs_ov.values()))
print("  observed cross-model overlaps (ARC top-2):")
for p in pairs:
    print(f"      {p[0]:8s} x {p[1]:8s} = {obs_ov[p]:.2f}")
print(f"  --> range [{obs_vals.min():.2f}, {obs_vals.max():.2f}]  pooled mean = {obs_vals.mean():.3f}")

# trajectory bootstrap on the pooled mean overlap
B_OV = 1000
boot_mean, boot_min = [], []
for _ in range(B_OV):
    Ub = {}
    for ml, ms, od in MODELS:
        _, _, A, Bs, *_ = fit_cell(resample(CELLS[(ml, "arc")]))
        Ub[ml] = U2(A, Bs)
    v = np.array([sim2(Ub[p[0]], Ub[p[1]]) for p in pairs])
    boot_mean.append(v.mean()); boot_min.append(v.min())
lo, hi = np.percentile(boot_mean, [2.5, 97.5])
mlo, mhi = np.percentile(boot_min, [2.5, 97.5])
print(f"  trajectory-bootstrap 95% CI pooled-mean overlap = [{lo:.3f}, {hi:.3f}]  (B={B_OV})")
print(f"  trajectory-bootstrap 95% CI of the MIN pair overlap = [{mlo:.3f}, {mhi:.3f}]")

# random-k-plane permutation null (analytic E = k/d = 0.125)
B_PERM, d_state = 20000, 16
null = np.empty(B_PERM)
for i in range(B_PERM):
    Q1, _ = np.linalg.qr(RNG.standard_normal((d_state, 2)))
    Q2, _ = np.linalg.qr(RNG.standard_normal((d_state, 2)))
    null[i] = sim2(Q1, Q2)
p_ov = (np.sum(null >= obs_vals.mean()) + 1) / (B_PERM + 1)
p_min = (np.sum(null >= obs_vals.min()) + 1) / (B_PERM + 1)
print(f"  random-k-plane null: mean={null.mean():.3f} (analytic 0.125), 95th pct={np.percentile(null,95):.3f}")
print(f"  permutation p(pooled-mean >= null) = {p_ov:.4g} ;  p(MIN pair >= null) = {p_min:.4g}")

# ======================================================= (2) TRANSFER SPEARMAN
print("\n" + "=" * 72, "\n(2) POSITIONAL-REACH TRANSFER SPEARMAN (per-answer reach, all cells)\n", "=" * 72)


def reach_all(cells):
    R = {}
    for (ml, bt), col in cells.items():
        K, B, A, Bs, d, ss, Kans, T = fit_cell(col)
        R[(ml, bt)] = reach_vec(K, B, d, ss, Kans, T)
    return R


def pooled_spearman(R):
    vals, per_b = [], {}
    for bt in BENCH:
        cb = []
        for (m1, _, _), (m2, _, _) in combinations(MODELS, 2):
            if (m1, bt) in R and (m2, bt) in R:
                cb.append(spearman(R[(m1, bt)], R[(m2, bt)]))
        if cb:
            per_b[bt] = np.mean(cb); vals += cb
    return float(np.mean(vals)), per_b, len(vals)


R0 = reach_all(CELLS)
obs_sp, per_b, npair = pooled_spearman(R0)
for bt in BENCH:
    print(f"      {bt:12s} mean cross-model Spearman = {per_b[bt]:+.2f}")
print(f"  --> POOLED transfer Spearman = {obs_sp:+.3f}  (n_pairs={npair})")

# trajectory bootstrap: resample debates per cell, refit, recompute reach, recompute pooled Spearman
B_SP = 1000
boot_sp = []
for _ in range(B_SP):
    Rb = {}
    for (ml, bt), col in CELLS.items():
        K, B, A, Bs, d, ss, Kans, T = fit_cell(resample(col))
        Rb[(ml, bt)] = reach_vec(K, B, d, ss, Kans, T)
    boot_sp.append(pooled_spearman(Rb)[0])
slo, shi = np.percentile(boot_sp, [2.5, 97.5])
print(f"  trajectory-bootstrap 95% CI pooled Spearman = [{slo:+.3f}, {shi:+.3f}]  (B={B_SP})")

# permutation null: independently shuffle answer order of each model's reach vector per benchmark
B_SPP = 20000
nullsp = np.empty(B_SPP)
keys = list(R0.keys())
for i in range(B_SPP):
    Rp = {k: R0[k][RNG.permutation(len(R0[k]))] for k in keys}
    nullsp[i] = pooled_spearman(Rp)[0]
p_sp = (np.sum(nullsp >= obs_sp) + 1) / (B_SPP + 1)
print(f"  answer-permutation null: mean={nullsp.mean():+.3f} (expected 0), 95th pct={np.percentile(nullsp,95):+.3f}")
print(f"  permutation p(pooled Spearman >= null) = {p_sp:.4g}")
print("\nDONE.")

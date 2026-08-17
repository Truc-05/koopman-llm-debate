"""Ổn định hóa EDMD + kiểm robust của τ (yêu cầu bắt buộc trước khi đi tiếp).

3 việc:
 (1) Ổn định phổ: sweep ridge γ, SVD-truncation (DMD-r), degree 1 vs 2 → mục tiêu
     |λ|≤1, gap>0, đúng MỘT λ=1.
 (2) Kiểm τ khi có nhiều λ≈1 suy biến: so τ(φ₁ đơn lẻ, tùy ý trong eigenspace) với
     τ_sub = chiếu h lên KHÔNG GIAN CON bất biến λ≈1 (spectral projector, bất biến cơ sở).
     Nếu hai cái khớp + đều tách đúng/sai → τ không ăn may.
 (3) Per-debate: 6 transitions/debate là quá ít → dùng pooled + regularize, nói rõ giới hạn.

Dùng: python experiments/stabilize_edmd.py mmlu_160
"""
import sys, os, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import build_snapshots, fit_edmd
from src.koopman.dmd_reduced import fit_dmd_reduced, full_operator
from src.koopman.spectrum import spectral_decomposition, eigenfunctions
from src.debate.observables_truth import h_star_soft, along_trajectory

OUT = os.path.join(ROOT, "experiments", "out")
tag = sys.argv[1] if len(sys.argv) > 1 else "mmlu_160"
N = 4

npz = np.load(os.path.join(OUT, f"trajs_{tag}.npz"))
trajs = [np.asarray(npz[f"traj_{i}"]) for i in range(len(npz.files))]
tr = json.load(open(os.path.join(OUT, f"transcripts_{tag}.json")))
y = np.array([bool(r["correct"]) for r in tr])
a_star = [int(r["a_star"]) for r in tr]
Kans = trajs[0].shape[1] // N
n_trans = sum(t.shape[0] - 1 for t in trajs)
print(f"tag={tag}  {len(trajs)} debate  {n_trans} transitions  (mỗi debate {trajs[0].shape[0]-1})")


def spec_summary(ev):
    ev = ev[np.argsort(-np.abs(ev))]
    mods = np.abs(ev)
    gap = 1 - mods[1] if len(mods) > 1 else np.nan
    n_gt1 = int((mods > 1.001).sum())
    n_at1 = int((np.abs(ev - 1.0) < 1e-2).sum())
    return dict(l1=mods[0], l2=mods[1] if len(mods) > 1 else np.nan,
                gap=gap, n_gt1=n_gt1, n_at1=n_at1)


def auc(s):
    p, m = s[y], s[~y]
    return ((p[:, None] > m[None, :]).sum() + 0.5 * (p[:, None] == m[None, :]).sum()) / (len(p) * len(m))


def tau_phi1(d, W, degree):
    """τ hiện tại: cosine với eigenfunction dẫn đầu (đơn lẻ)."""
    out = np.zeros(len(trajs))
    for i, t in enumerate(trajs):
        Zi = t[:-1]
        h = along_trajectory(h_star_soft, Zi, n_agents=N, n_answers=Kans, a_star=a_star[i])
        phi1 = eigenfunctions(d, W, Zi)[:, 0]
        num = np.abs(np.vdot(h, phi1)); den = np.linalg.norm(h) * np.linalg.norm(phi1)
        out[i] = num / den if den > 1e-12 else 0.0
    return out


def tau_subspace(d, W, ev, tol=0.02):
    """τ robust: chiếu h lên KHÔNG GIAN CON bất biến {|λ|≥1-tol}. Bất biến cơ sở."""
    idx = np.where(np.abs(ev) >= 1 - tol)[0]
    out = np.zeros(len(trajs))
    for i, t in enumerate(trajs):
        Zi = t[:-1]
        h = along_trajectory(h_star_soft, Zi, n_agents=N, n_answers=Kans, a_star=a_star[i]).astype(complex)
        B = eigenfunctions(d, W, Zi)[:, idx]                 # cột = các eigenfunction λ≈1
        coef, *_ = np.linalg.lstsq(B, h, rcond=None)         # chiếu trực giao lên span(B)
        proj = B @ coef
        den = np.linalg.norm(h)
        out[i] = np.linalg.norm(proj) / den if den > 1e-12 else 0.0
    return np.real(out), len(idx)


def fit_and_report(name, Psi_X, Psi_Y, d, degree, reg=None, rank=None):
    if rank is not None:
        Kt, U, S, V = fit_dmd_reduced(Psi_X, Psi_Y, rank=rank)
        K = full_operator(Kt, U)
    else:
        K, _, _ = fit_edmd(Psi_X, Psi_Y, reg=reg)
    ev, Vec, W = spectral_decomposition(K)
    s = spec_summary(ev)
    tau = tau_phi1(d, W, degree)
    print(f"  {name:<26} |λ1|={s['l1']:.3f} |λ2|={s['l2']:.3f} gap={s['gap']:+.3f} "
          f"|λ|>1:{s['n_gt1']:>2} λ≈1:{s['n_at1']:>2} | τ AUC={auc(tau):.3f} "
          f"(đúng {tau[y].mean():.3f}/sai {tau[~y].mean():.3f})")
    return K, ev, W, tau


# ===== (1) sweep ổn định hóa =====
d2 = PolynomialDictionary(degree=2)
X2, Y2 = build_snapshots(d2, np.concatenate([t[:-1] for t in trajs]),
                         np.concatenate([t[1:] for t in trajs]))
print(f"\ndegree=2: {X2.shape[0]} features, {X2.shape[1]} snapshots "
      f"({'ĐỦ' if X2.shape[1] > 2*X2.shape[0] else 'THIẾU'} 2×features)")
print("== ridge sweep (degree 2) ==")
res = {}
for g in [1e-6, 1e-3, 1e-1, 1e0, 1e1, 1e2]:
    res[f"ridge γ={g:g}"] = fit_and_report(f"ridge γ={g:g}", X2, Y2, d2, 2, reg=g)
print("== SVD-truncation DMD-r (degree 2) ==")
for r in [4, 8, 16, 32]:
    res[f"dmd-r={r}"] = fit_and_report(f"dmd-r={r}", X2, Y2, d2, 2, rank=r)

d1 = PolynomialDictionary(degree=1)
X1, Y1 = build_snapshots(d1, np.concatenate([t[:-1] for t in trajs]),
                         np.concatenate([t[1:] for t in trajs]))
print(f"\n== degree=1 (linear, {X1.shape[0]} features) ==")
for g in [1e-6, 1e-3, 1e-1]:
    res[f"deg1 γ={g:g}"] = fit_and_report(f"deg1 γ={g:g}", X1, Y1, d1, 1, reg=g)
for r in [4, 8, 16]:
    res[f"deg1 dmd-r={r}"] = fit_and_report(f"deg1 dmd-r={r}", X1, Y1, d1, 1, rank=r)

# ===== (2) robust τ trên config SẠCH nhất =====
# chọn config: |λ|>1 == 0, gap>0, τ AUC còn cao — ưu tiên degree 2 nếu có
print("\n== (2) robust τ: φ₁ đơn lẻ vs chiếu không-gian-con λ≈1 ==")
print("(chọn vài config sạch để so; nếu τ_sub ≈ τ_phi1 và đều tách => KHÔNG ăn may)")
for name in list(res.keys()):
    K, ev, W, tau = res[name]
    if spec_summary(ev)["n_gt1"] == 0 and spec_summary(ev)["gap"] > 0:
        d = d2 if not name.startswith("deg1") else d1
        tau_s, ndim = tau_subspace(d, W, ev)
        corr = np.corrcoef(tau, tau_s)[0, 1]
        print(f"  {name:<26} dim(λ≈1)={ndim:>2} | τ_phi1 AUC={auc(tau):.3f}  "
              f"τ_sub AUC={auc(tau_s):.3f}  corr(phi1,sub)={corr:.3f}")

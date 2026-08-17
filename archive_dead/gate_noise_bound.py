"""Gate cho EDMDc-noise-bound (giá-vào-cửa TCyb) — OFFLINE, đọc collect ckpt, KHÔNG debate/API mới.
Test 3 CỬA TỬ (xem edmdc_noise_bound_sketch.md §5) TRƯỚC khi cam kết 2-4 tuần chứng minh:
  Cửa 1: misspecification bias có nuốt variance không (bias-floor ≫ noise)?
  Cửa 2: dynamics quasi-static / control-SNR ≤ 1 (bound đúng-mà-vacuous)?
  Cửa 3: variance-structure của bound có thật (Var_i ∝ 1/λ_i, đồng thời validate N3)?
Xanh cả 3 ⇒ derive full. Bất kỳ đỏ ⇒ bound decorative ⇒ đừng derive, xét lại venue.

Dùng:  python experiments/gate_noise_bound.py [tag]     (KHÔNG tự chạy — user tự chạy)
       tag mặc định = mmlu_clean6_llama3_1_8b  (đổi model qua tag; OUT=out_qwen cho qwen)
"""
import sys, os, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import fit_edmdc
from src.debate.state import mean_belief

OUT = os.environ.get("OUT", os.path.join(ROOT, "experiments", "out"))
tag = sys.argv[1] if len(sys.argv) > 1 else "mmlu_clean6_llama3_1_8b"
K_DEF = 2
SEED = 0
N_BOOT = 300
rng = np.random.default_rng(SEED)

col = [json.loads(l) for l in open(os.path.join(OUT, f"ckpt_twindef_collect_{tag}.jsonl"))]
trajs = [np.asarray(r["traj"], float) for r in col]
cs = [int(r["c"]) for r in col]
N = 4
D = trajs[0].shape[1]
Kans = D // N
T = trajs[0].shape[0] - 1
print(f"\n===== GATE noise-bound (tag={tag}) — {len(col)} traj, D={D}, N={N}, Kans={Kans}, T={T} =====")


def u_of(c, t):
    u = np.zeros(Kans)
    if c >= 0 and t >= K_DEF:
        u[c] = 1.0
    return u


def snapshots(idx, degree=1):
    d = PolynomialDictionary(degree=degree); d.transform(trajs[0][:1]); ss = d.state_slice
    X, Y, U = [], [], []
    for i in idx:
        Z = trajs[i]
        for t in range(len(Z) - 1):
            X.append(Z[t]); Y.append(Z[t + 1]); U.append(u_of(cs[i], t))
    return d, ss, np.array(X), np.array(Y), np.array(U)


def fit(d, X, Y, U, reg=1e-6):
    K, B = fit_edmdc(d.transform(X).T, d.transform(Y).T, U.T, reg=reg)
    return K, B


def pred_state(d, ss, K, B, X, U):
    P = (K @ d.transform(X).T + B @ U.T)     # (M, S) lifted prediction
    return P[ss, :].T                        # (S, D) state prediction


all_idx = np.arange(len(trajs))
d1, ss1, X, Y, U = snapshots(all_idx, degree=1)
K1, B1 = fit(d1, X, Y, U)
Yhat = pred_state(d1, ss1, K1, B1, X, U)
R = Y - Yhat                                 # residual state (S, D)
rms = np.sqrt((R ** 2).mean())
z_scale = np.sqrt((X ** 2).mean())


# ---------- CỬA 1: misspecification bias vs noise ----------
# (i) residual có bị GIẢI THÍCH bởi state không (structured ⇒ bias)?
Xc = X - X.mean(0); Rc = R - R.mean(0)
beta, *_ = np.linalg.lstsq(Xc, Rc, rcond=None)         # regress residual trên state
R_from_state = Xc @ beta
r2_bias = 1 - ((Rc - R_from_state) ** 2).sum() / max((Rc ** 2).sum(), 1e-12)
# (ii) deg-2 có giảm residual mạnh không (linear misspecified)?
d2, ss2, X2, Y2, U2 = snapshots(all_idx, degree=2)
K2, B2 = fit(d2, X2, Y2, U2, reg=1e-6)
Yhat2 = pred_state(d2, ss2, K2, B2, X2, U2)
sse1 = ((Y - Yhat) ** 2).sum(); sst = ((Y - Y.mean(0)) ** 2).sum()
sse2 = ((Y2 - Yhat2) ** 2).sum()
r2_deg1 = 1 - sse1 / sst; r2_deg2 = 1 - sse2 / sst
g1_green = (r2_bias < 0.15) and (r2_deg2 - r2_deg1 < 0.10)
print(f"\n[Cửa 1] MISSPEC BIAS  residual-RMS={rms:.4f} (z-scale={z_scale:.3f}, rel={rms/z_scale:.3f})")
print(f"    R²(residual←state)={r2_bias:.3f}  (cao ⇒ residual structured = BIAS)")
print(f"    R²(deg-1)={r2_deg1:.3f}  R²(deg-2)={r2_deg2:.3f}  Δ={r2_deg2-r2_deg1:+.3f}  (Δ lớn ⇒ linear misspecified)")
print(f"    ⇒ {'🟢 XANH (bias nhỏ)' if g1_green else '🔴 ĐỎ (bias-floor thống trị → bound decorative)'}")


# ---------- CỬA 2: quasi-static / control-SNR ----------
A = np.real(K1[ss1][:, ss1]); Bs = np.real(B1[ss1, :])
rho = np.abs(np.linalg.eigvals(A)); a_minus_I = np.linalg.norm(A - np.eye(D), 2)
move = np.sqrt(((Y - X) ** 2).mean()) / z_scale            # z có nhúc nhích không
u_rows = U[(U.sum(1) > 0)]
ctrl_sig = np.sqrt(((U @ Bs.T) ** 2).mean(1))               # ‖B u_t‖ per snapshot
ctrl_snr = float(ctrl_sig[U.sum(1) > 0].mean()) / max(rms, 1e-9) if len(u_rows) else 0.0
g2_green = (ctrl_snr > 1.0) and (move > 0.05)
print(f"\n[Cửa 2] QUASI-STATIC  ρ(A)max={rho.max():.3f}  ‖A−I‖₂={a_minus_I:.3f}  move‖Δz‖/‖z‖={move:.3f}")
print(f"    ‖B‖₂={np.linalg.norm(Bs,2):.3f}  control-SNR=‖Bu‖/‖ξ‖={ctrl_snr:.2f}  (frac u≠0={len(u_rows)/len(U):.2f})")
print(f"    ⇒ {'🟢 XANH (có dynamics + control nhận-dạng-được)' if g2_green else '🔴 ĐỎ (quasi-static / control chìm dưới noise → bound vacuous)'}")


# ---------- CỬA 3: Var_i ∝ 1/λ_i  (validate N3) ----------
d, ss, Xf, Yf, Uf = snapshots(all_idx, degree=1)
PsiX = d.transform(Xf); Omega = np.hstack([PsiX, Uf]).T     # (M+Kans, S)
Vn = Omega @ Omega.T + 1e-6 * np.eye(Omega.shape[0])
lam, Vec = np.linalg.eigh(Vn)                              # tăng dần
order = np.argsort(lam)[::-1]; lam = lam[order]; Vec = Vec[:, order]
# bootstrap refit [K B], chiếu lên eigenvector của Vn, đo variance per hướng
Theta_b = []
nt = len(trajs)
for _ in range(N_BOOT):
    bi = rng.integers(0, nt, nt)
    db, _, Xb, Yb, Ub = snapshots(bi, degree=1)
    Kb, Bb = fit(db, Xb, Yb, Ub)
    Theta_b.append(np.hstack([np.real(Kb), np.real(Bb)]))  # (M, M+Kans)
Theta_b = np.stack(Theta_b)                                # (N_BOOT, M, M+Kans)
proj = Theta_b @ Vec                                       # (N_BOOT, M, ndir)
var_i = proj.var(axis=0).sum(axis=0)                       # Var per eigendirection (trace)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float(ra @ rb / np.sqrt((ra @ ra) * (rb @ rb)))


sp = spearman(np.log(var_i + 1e-30), -np.log(lam + 1e-30))
g3_green = sp > 0.6
print(f"\n[Cửa 3] VARIANCE-STRUCTURE  Spearman(log Var_i, −log λ_i)={sp:+.3f}  (bound nói ≈+1)")
print(f"    cond(Vn)=λmax/λmin={lam.max()/max(lam.min(),1e-12):.1e}   (anisotropy — nối positional-bias)")
print("    λ_i (top→bottom) vs Var_i:")
show = list(range(3)) + [None] + list(range(len(lam) - 3, len(lam)))
for i in show:
    if i is None:
        print("      ..."); continue
    print(f"      λ={lam[i]:.3e}   Var={var_i[i]:.3e}")
print(f"    ⇒ {'🟢 XANH (cấu trúc bound thật → N3 khả thi)' if g3_green else '🔴 ĐỎ (variance không theo 1/λ → decorative, nguy cơ chết-lần-4)'}")


# ---------- TỔNG ----------
greens = [g1_green, g2_green, g3_green]
print("\n" + "=" * 60)
print(f"KẾT: Cửa1={'🟢' if g1_green else '🔴'}  Cửa2={'🟢' if g2_green else '🔴'}  Cửa3={'🟢' if g3_green else '🔴'}")
if all(greens):
    print("⇒ 3/3 XANH: derive full bound (§1–4), TCyb-plausible. Killer fig = predicted-vs-actual per-direction error.")
else:
    print("⇒ CÓ CỬA ĐỎ: bound decorative/vacuous = dấu-hiệu-4 operator không có răng ⇒ ĐỪNG derive, xét lại venue (TNNLS/TAI).")
print("=" * 60)

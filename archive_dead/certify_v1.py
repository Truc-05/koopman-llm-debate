"""Exp#4 v1 — CERTIFICATE reformulation (RẺ, analysis-only, KHÔNG debate mới).
Vá 2 lỗi của v0 (ρ≈0):
  (1) v0 roll từ z0=UNIFORM (phi thực tế) + difference push-vs-nopush.
      → v1 roll từ STATE k_def THẬT (mean-start & per-debate), gain thô như EmpScore.
  (2) thêm certificate TUYẾN TÍNH thuần: step-response c_xᵀ(Σ_{t<h} Kᵗ)b_x
      = controllability answer-x qua input-x (bỏ softmax; Spearman rank nên khác scale vô hại).
So ρ(Cert, Emp) từng biến thể. Biến thể nào ρ>0.7 (& per-target hợp lý) → certificate CỨU được.
Dùng: python experiments/certify_v1.py [tag]   (mặc định mmlu_clean6; đọc ckpt_edmdc_<tag>.jsonl)
"""
import sys, os, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import fit_edmdc
from src.debate.state import mean_belief

OUT = os.path.join(ROOT, "experiments", "out")
tag = sys.argv[1] if len(sys.argv) > 1 else "mmlu_clean6"
N = Kans = 4
recs = [json.loads(l) for l in open(os.path.join(OUT, f"ckpt_edmdc_{tag}.jsonl"))]
trajs = [np.asarray(r["traj"], float) for r in recs]
kint = [int(r["k_int"]) for r in recs]; xpush = [int(r["x_push"]) for r in recs]
n = len(trajs); T = trajs[0].shape[0] - 1


def u_of(i, t):
    u = np.zeros(Kans)
    if t >= kint[i]:
        u[xpush[i]] = 1.0
    return u


def fit(idx):
    d = PolynomialDictionary(degree=1); d.transform(trajs[0][:1]); ss = d.state_slice
    X, Y, U = [], [], []
    for i in idx:
        Z = trajs[i]
        for t in range(len(Z) - 1):
            X.append(Z[t]); Y.append(Z[t + 1]); U.append(u_of(i, t))
    K, B = fit_edmdc(d.transform(np.array(X)).T, d.transform(np.array(Y)).T, np.array(U).T, reg=1e-6)
    return d, ss, K, B


d, ss, K, B = fit(np.arange(n))
L = K.shape[0]
ss_idx = np.arange(L)[ss]                              # chỉ số state (logit) trong không gian lift


def lift(z):
    return d.transform(z.reshape(1, -1))[0]


def roll_gain(z_start, x, h):
    """twin roll từ z_start, push x giữ h vòng → GAIN belief[x] (softmax, khớp EmpScore)."""
    psi = lift(z_start); u = np.zeros(Kans); u[x] = 1.0
    b0 = mean_belief(np.real(psi[ss]), N, Kans)[x]
    for _ in range(h):
        psi = K @ psi + B @ u
    return mean_belief(np.real(psi[ss]), N, Kans)[x] - b0


def emp(x, h):                                         # thực nghiệm: gain belief[x] qua h vòng, TB debate push x
    vals = [mean_belief(trajs[i][min(kint[i] + h, T)], N, Kans)[x] - mean_belief(trajs[i][kint[i]], N, Kans)[x]
            for i in range(n) if xpush[i] == x]
    return float(np.mean(vals)) if vals else np.nan


def mean_start(x):
    S = [trajs[i][kint[i]] for i in range(n) if xpush[i] == x]
    return np.mean(S, axis=0) if S else None


def cert_realmean(x, h):                               # roll từ MEAN state k_def thật của target x
    z = mean_start(x); return roll_gain(z, x, h) if z is not None else np.nan


def cert_perdebate(x, h):                              # roll từ state k_def RIÊNG mỗi debate rồi TB (khớp emp nhất)
    vals = [roll_gain(trajs[i][kint[i]], x, h) for i in range(n) if xpush[i] == x]
    return float(np.mean(vals)) if vals else np.nan


def readout(x):                                        # c_x: lấy logit[x] TB các agent (đọc tuyến tính)
    c = np.zeros(L)
    for a in range(N):
        c[ss_idx[a * Kans + x]] = 1.0 / N
    return c


def cert_linstep(x, h):                                # step-response tuyến tính: c_xᵀ(Σ_{t<h}Kᵗ)b_x
    c = readout(x); b = B[:, x]; acc = np.zeros(L); Kt = np.eye(L)
    for _ in range(h):
        acc = acc + Kt @ b; Kt = K @ Kt
    return float(np.real(c @ acc))


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b)); a, b = a[ok], b[ok]
    if len(a) < 3:
        return np.nan
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean(); dd = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / dd) if dd > 0 else np.nan


targets = sorted(set(xpush)); horizons = [2, 3, 4]; h0 = 3
pts = [(x, h) for x in targets for h in horizons]
E = [emp(x, h) for x, h in pts]
Et = [emp(x, h0) for x in targets]

variants = [("realmean-start   ", cert_realmean),
            ("perdebate-start  ", cert_perdebate),
            ("linear-step(logit)", cert_linstep)]

print(f"\n===== EXP#4 v1 — CERTIFICATE REFORMULATION (tag={tag}, N=4, n={n}) =====")
print(f"(v0 tham chiếu: ρ_full=−0.105, ρ_pertarget=−0.80 → chết)\n")
print(f"{'biến thể':20s} {'ρ_full(12đ)':>12s} {'ρ_pertarget(h=3,4đ)':>20s}")
best = -2.0
for name, fn in variants:
    C = [fn(x, h) for x, h in pts]; Ct = [fn(x, h0) for x in targets]
    rf, rt = spearman(C, E), spearman(Ct, Et)
    best = max(best, rf)
    print(f"{name:20s} {rf:>+12.3f} {rt:>+20.3f}")

print("\n---- điểm chi tiết biến thể perdebate-start (khớp emp nhất) ----")
print(f"    {'x':>2} {'h':>2} {'Cert':>8} {'Emp':>8}")
for (x, h) in pts:
    print(f"    {x:>2} {h:>2} {cert_perdebate(x, h):>8.3f} {emp(x, h):>8.3f}")

verdict = ("CỨU ĐƯỢC ✓ → viết certify_manipulability.py, chạy N-sweep" if best >= 0.7 else
           "BIÊN → cân nhắc thêm điểm (config×target)" if best >= 0.5 else
           "XÁC NHẬN CHẾT → certificate về future-work, chốt Koopman-as-structural")
print(f"\n⇒ best ρ_full = {best:+.3f}  ⇒ {verdict}")

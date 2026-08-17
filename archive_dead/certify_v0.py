"""Exp#4 v0 — CERTIFICATE-CHECK (RẺ, analysis-only, KHÔNG debate mới).
Test cơ chế certificate trên data adversary N=4 đã có (ckpt_edmdc): CertScore từ operator (K,B)
có DỰ ĐOÁN EmpScore (steering THẬT) qua các điểm post-hoc (target x_push × horizon h) không?
  ρ>0.7 → cơ chế certificate SỐNG → đáng chạy N-sweep 240 debate.
  ρ chết → certificate về future-work, khỏi phí 4–8h.
Sample-efficiency: fit (K,B) trên k=15 debate, xem còn dự đoán EmpScore đo trên cả 80 không.
Dùng: python experiments/certify_v0.py [tag]
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


def cert(model, x, h):                                 # model belief[x] sau h bước push (từ uniform) − no-push
    d, ss, K, B = model; z0 = np.zeros(N * Kans)
    def roll(push):
        psi = d.transform(z0.reshape(1, -1))[0]; u = np.zeros(Kans)
        if push:
            u[x] = 1.0
        for _ in range(h):
            psi = K @ psi + B @ u
        return mean_belief(np.real(psi[ss]), N, Kans)[x]
    return roll(True) - roll(False)


def emp(x, h):                                          # thực nghiệm: gain belief[x] qua h vòng push, TB debate push x
    vals = [mean_belief(trajs[i][min(kint[i] + h, T)], N, Kans)[x] - mean_belief(trajs[i][kint[i]], N, Kans)[x]
            for i in range(n) if xpush[i] == x]
    return float(np.mean(vals)) if vals else np.nan


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b)); a, b = a[ok], b[ok]
    if len(a) < 3:
        return np.nan
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean(); dd = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / dd) if dd > 0 else np.nan


targets = sorted(set(xpush)); horizons = [2, 3, 4]
pts = [(x, h) for x in targets for h in horizons]
E = [emp(x, h) for x, h in pts]

mfull = fit(np.arange(n))
C = [cert(mfull, x, h) for x, h in pts]
m15 = fit(np.arange(15))
C15 = [cert(m15, x, h) for x, h in pts]

print(f"\n===== EXP#4 v0 — CERTIFICATE-CHECK (tag={tag}, adversary N=4, n={n}) =====")
print(" điểm = (target x_push × horizon h). CertScore=từ (K,B). EmpScore=steering thật (gain belief[x]).\n")
print(f"    {'x':>2} {'h':>2} {'CertScore':>10} {'EmpScore':>9}")
for (x, h), c, e in zip(pts, C, E):
    print(f"    {x:>2} {h:>2} {c:>10.3f} {e:>9.3f}")

rho = spearman(C, E); rho15 = spearman(C15, E)
# per-target ρ tại h=3 (thuần ranking target khó/dễ đẩy)
h0 = 3
Ct = [cert(mfull, x, h0) for x in targets]; Et = [emp(x, h0) for x in targets]
rho_t = spearman(Ct, Et)

print("\n---- VERDICT ----")
print(f" ρ(Cert, Emp) toàn {len(pts)} điểm (fit đủ {n})   = {rho:+.3f}")
print(f" ρ per-target (h=3, {len(targets)} điểm, rank khó/dễ) = {rho_t:+.3f}")
print(f" ρ SAMPLE-EFFICIENCY (fit chỉ 15 debate)      = {rho15:+.3f}   (điểm bán mạnh nếu ≈ ρ đủ)")
verdict = ("SỐNG ✓ → đáng chạy N-sweep 240 debate" if rho >= 0.7 else
           "YẾU/CHẾT → certificate về future-work, KHỎI phí 4–8h" if rho < 0.4 else
           "BIÊN → cân nhắc; có thể cần thêm điểm (config × target)")
print(f" ⇒ cơ chế certificate: {verdict}")

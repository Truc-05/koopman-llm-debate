"""Gate-v2 cho đường-sống impossibility-theorem — OFFLINE, KHÔNG debate/API mới.
Quyết định: misspec floor (foundation của impossibility-thm) là THẬT hay in-sample overfit-artifact?
  (A) CV deg-1 vs deg-2 OUT-OF-SAMPLE R²  — floor thật ⇔ deg-2 thắng deg-1 trên held-out.
  (B) sample-complexity của B — rel-std(B̂) theo n' (B có ổn định hay noise mãi?).
Xanh (floor thật) ⇒ impossibility-thm có nền → derive limit result cho TCyb.
Đỏ (deg-2 overfit) ⇒ floor ảo → picture đổi, chỉ còn noise → xét lại.

Dùng:  python experiments/gate_noise_bound_v2.py [tag]     (KHÔNG tự chạy — user tự chạy)
"""
import sys, os, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import fit_edmdc

OUT = os.environ.get("OUT", os.path.join(ROOT, "experiments", "out"))
tag = sys.argv[1] if len(sys.argv) > 1 else "mmlu_clean6_llama3_1_8b"
K_DEF = 2
rng = np.random.default_rng(0)
col = [json.loads(l) for l in open(os.path.join(OUT, f"ckpt_twindef_collect_{tag}.jsonl"))]
trajs = [np.asarray(r["traj"], float) for r in col]
cs = [int(r["c"]) for r in col]
N = 4
D = trajs[0].shape[1]; Kans = D // N; ntraj = len(trajs)
print(f"\n===== GATE-v2 impossibility-foundation (tag={tag}) — {ntraj} traj, D={D} =====")


def u_of(c, t):
    u = np.zeros(Kans)
    if c >= 0 and t >= K_DEF:
        u[c] = 1.0
    return u


def snaps(idx, degree):
    d = PolynomialDictionary(degree=degree); d.transform(trajs[0][:1]); ss = d.state_slice
    X, Y, U = [], [], []
    for i in idx:
        Z = trajs[i]
        for t in range(len(Z) - 1):
            X.append(Z[t]); Y.append(Z[t + 1]); U.append(u_of(cs[i], t))
    return d, ss, np.array(X), np.array(Y), np.array(U)


def fit_pred_r2(train_idx, test_idx, degree):
    d, ss, Xtr, Ytr, Utr = snaps(train_idx, degree)
    K, B = fit_edmdc(d.transform(Xtr).T, d.transform(Ytr).T, Utr.T, reg=1e-6)
    _, _, Xte, Yte, Ute = snaps(test_idx, degree)
    Yhat = (np.real(K) @ d.transform(Xte).T + np.real(B) @ Ute.T)[ss, :].T
    sse = ((Yte - Yhat) ** 2).sum()
    sst = ((Yte - Yte.mean(0)) ** 2).sum()
    return 1 - sse / sst


# ---------- (A) CV deg-1 vs deg-2 OUT-OF-SAMPLE ----------
KF = 6
perm = rng.permutation(ntraj); folds = np.array_split(perm, KF)
r1, r2 = [], []
for f in range(KF):
    te = folds[f]; tr = np.concatenate([folds[g] for g in range(KF) if g != f])
    r1.append(fit_pred_r2(tr, te, 1)); r2.append(fit_pred_r2(tr, te, 2))
r1, r2 = np.array(r1), np.array(r2)
delta = r2 - r1
floor_real = (delta.mean() > 0.05) and (delta.mean() - delta.std() / np.sqrt(KF) > 0)
print(f"\n[A] CV OUT-OF-SAMPLE (K={KF} fold theo trajectory):")
print(f"    R²(deg-1) OOS = {r1.mean():.3f} ± {r1.std():.3f}")
print(f"    R²(deg-2) OOS = {r2.mean():.3f} ± {r2.std():.3f}")
print(f"    Δ = {delta.mean():+.3f} ± {delta.std()/np.sqrt(KF):.3f}  (in-sample v1 Δ≈+0.18..0.24)")
print(f"    ⇒ {'🟢 FLOOR THẬT (deg-2 thắng OOS) → impossibility-thm có nền' if floor_real else '🔴 FLOOR ẢO (deg-2 overfit, không thắng OOS) → cửa-1 v1 là artifact, picture đổi'}")


# ---------- (B) sample-complexity của B ----------
print(f"\n[B] SAMPLE-COMPLEXITY của B̂ (rel-std qua bootstrap, theo số trajectory n'):")
print(f"    {'n_traj':>7} {'rel-std(B̂)':>12} {'rel-std(Â)':>12}")
for nprime in [10, 20, 30, 40, 50, ntraj]:
    Bs_bt, As_bt = [], []
    for _ in range(120):
        bi = rng.integers(0, ntraj, nprime)
        d, ss, Xb, Yb, Ub = snaps(bi, 1)
        K, B = fit_edmdc(d.transform(Xb).T, d.transform(Yb).T, Ub.T, reg=1e-6)
        Bs_bt.append(np.real(B[ss, :])); As_bt.append(np.real(K[ss][:, ss]))
    Bs_bt = np.stack(Bs_bt); As_bt = np.stack(As_bt)
    rsB = np.linalg.norm(Bs_bt.std(0)) / max(np.linalg.norm(Bs_bt.mean(0)), 1e-9)
    rsA = np.linalg.norm(As_bt.std(0)) / max(np.linalg.norm(As_bt.mean(0)), 1e-9)
    print(f"    {nprime:>7} {rsB:>12.3f} {rsA:>12.3f}")
print("    (rel-std giảm chậm / cao ở n'=max ⇒ B đói dữ liệu; giảm nhanh ⇒ B nhận-dạng-được đã ổn)")

print("\n" + "=" * 60)
print("Đọc: [A]🟢 ⇒ đường impossibility-thm sống (floor thật, dùng data có sẵn).")
print("     [A]🔴 ⇒ floor là overfit → KHÔNG có impossibility từ misspec → fold về TNNLS.")
print("=" * 60)

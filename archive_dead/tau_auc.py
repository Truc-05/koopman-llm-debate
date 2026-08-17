"""Tính AUC + threshold tối ưu cho tau trên 1 run results_*.json.
Dùng: python experiments/tau_auc.py experiments/out/results_mmlu_40_r5.json
"""
import sys, json
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "experiments/out/results_mmlu_40_r5.json"
runs = json.load(open(path))["runs"]
tau = np.array([r["tau"] for r in runs], float)
y = np.array([bool(r["correct"]) for r in runs])          # True = debate đúng
pos, neg = tau[y], tau[~y]
n1, n0 = len(pos), len(neg)

# AUC = P(tau_dung > tau_sai) qua thống kê Mann-Whitney U (xử lý ties = 0.5)
gt = (pos[:, None] > neg[None, :]).sum()
eq = (pos[:, None] == neg[None, :]).sum()
auc = (gt + 0.5 * eq) / (n1 * n0)

# Threshold tối ưu theo Youden J = TPR - FPR
thr = np.unique(tau)
best = max(thr, key=lambda t: (pos >= t).mean() - (neg >= t).mean())
tpr, fpr = (pos >= best).mean(), (neg >= best).mean()
acc = ((tau >= best) == y).mean()   # coi tau>=thr là "dự đoán đúng"

print(f"n = {len(tau)}  (dung={n1}, sai={n0})")
print(f"tau|dung = {pos.mean():.3f} ± {pos.std():.3f}")
print(f"tau|sai  = {neg.mean():.3f} ± {neg.std():.3f}")
print(f"AUC          = {auc:.3f}")
print(f"tau* (Youden)= {best:.3f}  | TPR={tpr:.3f} FPR={fpr:.3f}  bal-acc={(tpr+(1-fpr))/2:.3f}")
print(f"acc @ tau*   = {acc:.3f}")

# Precision@top-k: lấy k% debate có tau cao nhất, xem bao nhiêu % thực sự đúng
base = y.mean()
order = np.argsort(-tau)                       # tau giảm dần
print(f"\nbase rate (P debate dung) = {base:.3f}")
for frac in (0.10, 0.20, 0.30, 0.50):
    k = max(1, int(round(frac * len(tau))))
    top = order[:k]
    prec = y[top].mean()
    print(f"P@top-{int(frac*100):>2}%  (n={k:>3}) = {prec:.3f}   "
          f"(lift {prec/base:.2f}x, tau>={tau[top].min():.3f})")

# Precision theo từng decile của tau (xem đường cong có đơn điệu không)
print("\nprecision theo decile tau (thap -> cao):")
dec = np.array_split(order[::-1], 10)          # order[::-1] = tau tang dan
for i, idx in enumerate(dec, 1):
    lo, hi = tau[idx].min(), tau[idx].max()
    print(f"  D{i:>2} [{lo:.3f},{hi:.3f}] n={len(idx):>2}  P(dung)={y[idx].mean():.3f}")

# Soi rieng nhom dong thuan tuyet doi tau == 1.0 (herding?)
sat = np.isclose(tau, 1.0)
if sat.any():
    print(f"\ntau == 1.000 : n={sat.sum()}  P(dung)={y[sat].mean():.3f}  "
          f"(base {base:.3f}) -> {'HERDING' if y[sat].mean() < base else 'ok'}")

# Dai tin cay [lo, hi): loai bo boi roi (tau thap) va herding (tau bao hoa)
LO, HI = 0.90, 0.99
band = (tau >= LO) & (tau < HI)
if band.any():
    p = y[band].mean()
    print(f"\nBAND tau in [{LO},{HI}) : n={band.sum()}  P(dung)={p:.3f}  "
          f"lift={p/base:.2f}x  (coverage {band.mean():.0%})")

# Ve duong cong precision-vs-tau (10 decile) -> hinh chu luc
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    centers = [tau[idx].mean() for idx in dec]
    precs = [y[idx].mean() for idx in dec]
    plt.figure(figsize=(6, 4))
    plt.axhline(base, ls="--", c="gray", lw=1, label=f"base rate {base:.2f}")
    plt.axvspan(LO, HI, color="tab:green", alpha=0.12, label=f"reliability band [{LO},{HI})")
    plt.plot(centers, precs, "o-", c="tab:blue")
    plt.xlabel(r"consensus $\tau$ (decile center)")
    plt.ylabel("P(debate correct)")
    plt.title("Precision is non-monotonic in $\\tau$ (herding at $\\tau\\!\\to\\!1$)")
    plt.ylim(-0.02, 1.02); plt.legend(fontsize=8); plt.tight_layout()
    out = path.rsplit("/", 1)[0] + "/precision_vs_tau.png"
    plt.savefig(out, dpi=150)
    print(f"\nfigure -> {out}")
except Exception as e:
    print(f"\n(bo qua ve hinh: {e})")


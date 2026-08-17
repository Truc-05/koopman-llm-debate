"""tau hieu chinh chong-herding qua pho per-debate.

Y tuong: tau don le khong don dieu (tau->1 = herding). Ta fit reduced-DMD
RIENG cho tung trajectory de lay |lambda_2| per-debate. Prop 1: debate herding
co |lambda_2| gan 1 (nhieu mode cham => nhieu lop dong thuan), hoi tu lanh manh
thi |lambda_2| thap. -> gap_i = 1-|lambda_2| la tin hieu chong-herding DOC LAP
voi tau.

Dung: python experiments/tau_corrected.py mmlu_40_r5
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.koopman.dmd_reduced import fit_dmd_reduced, modes_from_reduced

OUT = "experiments/out"
tag = sys.argv[1] if len(sys.argv) > 1 else "mmlu_40_r5"
GAMMA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

res = json.load(open(f"{OUT}/results_{tag}.json"))
runs = res["runs"]
tau = np.array([r["tau"] for r in runs], float)
y = np.array([bool(r["correct"]) for r in runs])
npz = np.load(f"{OUT}/trajs_{tag}.npz")


def auc(score, lab):
    p, n = score[lab], score[~lab]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    gt = (p[:, None] > n[None, :]).sum()
    eq = (p[:, None] == n[None, :]).sum()
    return (gt + 0.5 * eq) / (len(p) * len(n))


# ---- pho per-debate qua reduced-DMD tren raw-state ----
lam2, n_slow = np.zeros(len(runs)), np.zeros(len(runs), int)
for i in range(len(runs)):
    Z = np.asarray(npz[f"traj_{i}"])            # (T+1, NK)
    Psi_X, Psi_Y = Z[:-1].T, Z[1:].T            # (NK, T) raw-state, khong lift
    Kt, U, S, V = fit_dmd_reduced(Psi_X, Psi_Y)
    ev, _ = modes_from_reduced(Kt, U)           # da sort theo |lambda| giam
    mods = np.abs(ev)
    lam2[i] = mods[1] if len(mods) > 1 else 0.0
    n_slow[i] = int((mods > 0.9).sum())         # so mode cham (gan don vi tron)

gap = 1.0 - lam2
base = y.mean()

# ---- cac diem so ----
s_gap = tau * gap                 # Koopman-grounded
s_pen = tau * (1 - tau) ** GAMMA  # phat bao hoa (doi chung)

print(f"tag={tag}  n={len(runs)}  base={base:.3f}  gamma={GAMMA}")
print(f"{'score':<16}{'AUC':>7}")
for name, s in [("tau (goc)", tau), ("gap_i", gap),
                (f"s_pen=tau(1-tau)^{GAMMA}", s_pen), ("s_gap=tau*gap", s_gap)]:
    print(f"{name:<16}{auc(s, y):>7.3f}")

# ---- monotonicity: decile precision cua diem tot nhat ----
def decile_curve(score, name):
    order = np.argsort(score)
    print(f"\ndecile {name} (thap->cao):")
    for j, idx in enumerate(np.array_split(order, 10), 1):
        print(f"  D{j:>2} [{score[idx].min():.3f},{score[idx].max():.3f}] "
              f"n={len(idx):>2} P={y[idx].mean():.3f}")

decile_curve(s_gap, "s_gap")

# ---- TEST QUYET DINH: trong o herding tau>=0.99, gap_i co tach dung/sai? ----
hi = tau >= 0.99
if hi.sum() >= 4:
    gh, yh = gap[hi], y[hi]
    print(f"\n== vung herding tau>=0.99 (n={hi.sum()}, P={yh.mean():.3f}) ==")
    print(f"  gap_i | dung = {gh[yh].mean():.3f} ± {gh[yh].std():.3f}  (n={yh.sum()})")
    print(f"  gap_i | sai  = {gh[~yh].mean():.3f} ± {gh[~yh].std():.3f}  (n={(~yh).sum()})")
    print(f"  |lam2||dung = {lam2[hi][yh].mean():.3f}   |lam2||sai = {lam2[hi][~yh].mean():.3f}")
    print(f"  n_slow|dung = {n_slow[hi][yh].mean():.2f}   n_slow|sai = {n_slow[hi][~yh].mean():.2f}")
    print(f"  AUC(gap_i trong vung herding) = {auc(gh, yh):.3f}  "
          f"<- >0.6 => Prop1 duoc data ung ho")

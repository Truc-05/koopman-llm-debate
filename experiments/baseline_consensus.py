"""Baseline: dai τ Koopman co hon chi so dong thuan NGAY THO khong?

Cau hoi song con cho paper: hien tuong chu-U-nguoc (herding o dong thuan bao hoa)
co RIENG cua τ Koopman, hay mot thuoc do dong thuan tam thuong (agreement,
confidence, entropy o round cuoi) cung cho ra U nguoc? Neu baseline cung co U,
thi τ phai tach SACH HON (AUC cao hon) moi bien minh duoc bo may Koopman.

Dung: python experiments/baseline_consensus.py mmlu_40_r5
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.debate.state import mean_belief, split_state
from src.debate.observables_truth import agreement_index

OUT = "experiments/out"
tag = sys.argv[1] if len(sys.argv) > 1 else "mmlu_40_r5"
N_AGENTS = 4

res = json.load(open(f"{OUT}/results_{tag}.json"))
runs = res["runs"]
tau = np.array([r["tau"] for r in runs], float)
y = np.array([bool(r["correct"]) for r in runs])
npz = np.load(f"{OUT}/trajs_{tag}.npz")
base = y.mean()


def auc(score, lab):
    p, n = score[lab], score[~lab]
    gt = (p[:, None] > n[None, :]).sum()
    eq = (p[:, None] == n[None, :]).sum()
    return (gt + 0.5 * eq) / (len(p) * len(n))


# ---- thuoc do dong thuan ngay tho o round cuoi (tu trajs) ----
NK = np.asarray(npz["traj_0"]).shape[1]
K = NK // N_AGENTS
agree, conf, concen = np.zeros(len(runs)), np.zeros(len(runs)), np.zeros(len(runs))
for i in range(len(runs)):
    z = np.asarray(npz[f"traj_{i}"])[-1]                 # round cuoi
    p_bar = mean_belief(z, N_AGENTS, K)                  # (K,) niem tin trung binh
    agree[i] = agreement_index(z, N_AGENTS, K)           # ty le agent theo phe da so
    conf[i] = float(p_bar.max())                         # khoi luong dinh
    concen[i] = float(1.0 + np.sum(p_bar * np.log(p_bar + 1e-12)) / np.log(K))  # 1 - H_norm

scores = {"tau (Koopman)": tau, "agreement": agree,
          "confidence": conf, "concentration": concen}

# ---- (1) AUC: τ co thang baseline khong? ----
print(f"tag={tag}  n={len(runs)}  base={base:.3f}  (K={K}, N={N_AGENTS})")
print("AUC vs correctness:")
for name, s in scores.items():
    print(f"  {name:<16} {auc(s, y):.3f}")

# ---- (2) U nguoc co RIENG cua τ? decile precision cho tung thuoc do ----
def decile(score, name):
    order = np.argsort(score)
    cells = []
    for idx in np.array_split(order, 10):
        cells.append(y[idx].mean())
    top = order[np.argsort(score[order])][-max(1, len(order)//10):]  # top decile
    print(f"  {name:<16} " + " ".join(f"{c:.2f}" for c in cells) +
          f"   | P@top10%={y[order[-len(order)//10:]].mean():.3f}")

print("\ndecile precision (thap->cao, 10 o) — tim U nguoc (dinh giua, sut o cuoi):")
for name, s in scores.items():
    decile(s, name)

# ---- (3) τ co gi hon dong thuan tho? tuong quan + AUC rieng phan du ----
print("\ntuong quan Pearson voi τ:")
for name in ("agreement", "confidence", "concentration"):
    r = np.corrcoef(tau, scores[name])[0, 1]
    print(f"  τ ~ {name:<14} r={r:.3f}")

# ---- (4) trong o bao hoa cua MOI thuoc do, precision co sut khong? (herding chung) ----
print("\nP(dung) trong o bao hoa (top-5% cao nhat) tung thuoc do:")
for name, s in scores.items():
    k = max(1, len(s)//20)
    top = np.argsort(-s)[:k]
    print(f"  {name:<16} n={k:>2}  P={y[top].mean():.3f}  (base {base:.3f})"
          f"  {'-> herding' if y[top].mean() < base else ''}")

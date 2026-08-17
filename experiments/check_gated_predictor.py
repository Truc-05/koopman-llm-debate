"""Gated predictor (nested-CV): entropy-gate có cứu được persistence? Koopman có hơn á-quân?

Ở vòng k: nếu entropy(z_k) < θ → giữ phe hiện tại (persistence). Nếu ≥ θ (mong manh) → LẬT:
  - variant "runner-up": lật về á quân (chỉ dùng belief, KHÔNG operator)
  - variant "koopman"  : lật về phe Koopman rollout K^(T-k)Ψ(z_k) đoán
So với persistence thuần. θ chọn NESTED-CV (train fold chọn θ, test fold đo) — không p-hack.

Mục tiêu = đoán đáp án CUỐI + đoán ĐÚNG/SAI. Nếu koopman > runner-up => operator đóng góp
HƯỚNG lật (giá trị Koopman thật). Nếu bằng => value chỉ ở entropy-gate (observable ②).

Dùng: python experiments/check_gated_predictor.py mmlu_clean6 [k=3]
"""
import sys, os, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import build_snapshots, fit_edmd
from src.koopman.spectrum import project_stable
from src.debate.state import split_state, mean_belief

OUT = os.path.join(ROOT, "experiments", "out")
tag = sys.argv[1] if len(sys.argv) > 1 else "mmlu_clean6"
kfix = int(sys.argv[2]) if len(sys.argv) > 2 else 3
N, REG, NFOLD = 4, 1e-6, 5

npz = np.load(os.path.join(OUT, f"trajs_{tag}.npz"))
trajs = [np.asarray(npz[f"traj_{i}"]) for i in range(len(npz.files))]
tr = json.load(open(os.path.join(OUT, f"transcripts_{tag}.json")))
a_star = np.array([int(r["a_star"]) for r in tr])
K = trajs[0].shape[1] // N
n = len(trajs); T = trajs[0].shape[0] - 1
fold = np.arange(n) % NFOLD
d = PolynomialDictionary(degree=1); d.transform(trajs[0][:1])


def fit_fold(idx):
    Zt = np.concatenate([trajs[i][:-1] for i in idx], 0)
    Ztp1 = np.concatenate([trajs[i][1:] for i in idx], 0)
    Px, Py = build_snapshots(d, Zt, Ztp1)
    Kop, _, _ = fit_edmd(Px, Py, reg=REG)
    return project_stable(Kop)


# đặc trưng per debate tại k (operator lấy theo fold của debate đó)
Kops = {f: fit_fold(np.where(fold != f)[0]) for f in range(NFOLD)}
ent = np.zeros(n); cur = np.zeros(n, int); run = np.zeros(n, int)
koo = np.zeros(n, int); fin = np.zeros(n, int)
for i, t in enumerate(trajs):
    pbar = mean_belief(t[kfix], N, K)
    order = np.argsort(-pbar)
    cur[i] = order[0]; run[i] = order[1]
    ent[i] = -(pbar * np.log(pbar + 1e-12)).sum()
    psi = d.transform(t[kfix].reshape(1, -1))[0]
    for _ in range(T - kfix):
        psi = Kops[fold[i]] @ psi
    koo[i] = int(np.argmax(mean_belief(np.real(psi[d.state_slice]), N, K)))
    fin[i] = int(np.argmax(mean_belief(t[-1], N, K)))
y = np.array([bool(r["correct"]) for r in tr])       # = (fin == a_star)


def gated_pred(mask, theta, dest):
    """dest: 'run' hoặc 'koo'. Trả mảng phe dự đoán cho các debate trong mask."""
    flip = ent[mask] >= theta
    d2 = {"run": run, "koo": koo}[dest][mask]
    return np.where(flip, d2, cur[mask])


def acc_final(pred, mask):  # đoán đáp án cuối
    return np.mean(pred == fin[mask])


def acc_correct(pred, mask):  # đoán ĐÚNG/SAI (pred==a_star có khớp fin==a_star)
    return np.mean((pred == a_star[mask]) == y[mask])


thetas = np.quantile(ent, np.linspace(0.3, 0.95, 20))
res = {"persist": [[], []], "gate→run": [[], []], "gate→koo": [[], []]}
for f in range(NFOLD):                                # nested-CV: chọn θ trên train, đo test
    te = np.where(fold == f)[0]; trn = np.where(fold != f)[0]
    # persistence (không θ)
    res["persist"][0].append(acc_final(cur[te], te)); res["persist"][1].append(acc_correct(cur[te], te))
    for dest, name in [("run", "gate→run"), ("koo", "gate→koo")]:
        best_th = max(thetas, key=lambda th: acc_final(gated_pred(trn, th, dest), trn))
        p = gated_pred(te, best_th, dest)
        res[name][0].append(acc_final(p, te)); res[name][1].append(acc_correct(p, te))

print(f"tag={tag} n={n} k={kfix}  #lật={int((cur!=fin).sum())}  (nested-CV θ theo entropy)")
print(f"{'predictor':<12}{'acc→đáp án cuối':>16}{'acc→đúng/sai':>14}")
for name in ["persist", "gate→run", "gate→koo"]:
    a0 = np.mean(res[name][0]); a1 = np.mean(res[name][1])
    print(f"{name:<12}{a0:>16.3f}{a1:>14.3f}")
print("\nĐọc:")
print(" - gate→koo > persist => Koopman+entropy-gate THÊM giá trị (vượt baseline tầm thường).")
print(" - gate→koo > gate→run => operator đóng góp HƯỚNG lật (không chỉ 'về á quân').")
print(" - gate→run ≈ gate→koo > persist => value ở entropy-observable (②), Koopman thừa cho hướng.")
print(" - cả hai ≈ persist => gate không cứu; Koopman-tự-trị hết cửa -> ① EDMDc.")

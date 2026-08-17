"""GATE cho Digital-Twin / Adaptive-MPC: (K,B) cuộn với CONTROL THẬT có đoán trạng thái
tương lai đúng hơn persistence không? + Part 2: reachability/controllability (cho #1/#2/#4).

Twin chỉ có giá trị nếu dự đoán đủ đúng. Ta có lợi thế: cho twin biết luôn chuỗi control thật
(u_t = đáp án moderator đẩy), nên đây là bài KHÁ THUẬN cho twin — nếu vẫn thua persistence thì
twin vô vọng.

Dùng: python experiments/check_twin_fidelity.py mmlu_clean6
"""
import sys, os, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import build_snapshots, fit_edmdc
from src.debate.state import mean_belief

OUT = os.path.join(ROOT, "experiments", "out")
tag = sys.argv[1] if len(sys.argv) > 1 else "mmlu_clean6"
N, REG, NFOLD, START = 4, 1e-6, 5, 1
recs = [json.loads(l) for l in open(os.path.join(OUT, f"ckpt_edmdc_{tag}.jsonl"))]
trajs = [np.asarray(r["traj"]) for r in recs]
kint = [r["k_int"] for r in recs]; xpush = [r["x_push"] for r in recs]
K = trajs[0].shape[1] // N; T = trajs[0].shape[0] - 1; n = len(trajs)
d = PolynomialDictionary(degree=1); d.transform(trajs[0][:1]); ss = d.state_slice
fold = np.arange(n) % NFOLD
print(f"tag={tag} n={n} T={T}  (twin biết control thật; cross-fit {NFOLD}-fold)")


def u_of(i, t):
    u = np.zeros(K)
    if t >= kint[i]:
        u[xpush[i]] = 1.0
    return u


def fit_fold(idx):
    Xc, Yc, Uc = [], [], []
    for i in idx:
        Z = trajs[i]
        for t in range(len(Z) - 1):
            Xc.append(Z[t]); Yc.append(Z[t + 1]); Uc.append(u_of(i, t))
    Px = d.transform(np.array(Xc)).T; Py = d.transform(np.array(Yc)).T
    return fit_edmdc(Px, Py, np.array(Uc).T, reg=REG)


# ---- Part 1: fidelity OOS ----
tw_mse, ps_mse, tw_lead, ps_lead = [], [], [], []
for f in range(NFOLD):
    Kop, B = fit_fold(np.where(fold != f)[0])
    for i in np.where(fold == f)[0]:
        Z = trajs[i]
        psi = d.transform(Z[START].reshape(1, -1))[0]
        for t in range(START, T):
            psi = Kop @ psi + B @ u_of(i, t)
        pred_bel = mean_belief(np.real(psi[ss]), N, K)
        act_bel = mean_belief(Z[-1], N, K)
        per_bel = mean_belief(Z[START], N, K)          # persistence: giữ nguyên
        tw_mse.append(np.mean((pred_bel - act_bel) ** 2))
        ps_mse.append(np.mean((per_bel - act_bel) ** 2))
        tw_lead.append(int(np.argmax(pred_bel)) == int(np.argmax(act_bel)))
        ps_lead.append(int(np.argmax(per_bel)) == int(np.argmax(act_bel)))
print("\n== Part 1: FIDELITY (cuộn từ vòng 1 tới cuối, có control thật) ==")
print(f"  belief-MSE cuối:  twin={np.mean(tw_mse):.4f}   persistence={np.mean(ps_mse):.4f}  "
      f"({'TWIN TỐT HƠN' if np.mean(tw_mse) < np.mean(ps_mse) else 'twin TỆ HƠN'})")
print(f"  đoán phe cuối:    twin={np.mean(tw_lead):.3f}   persistence={np.mean(ps_lead):.3f}  "
      f"({'TWIN TỐT HƠN' if np.mean(tw_lead) > np.mean(ps_lead) else 'twin TỆ/BẰNG'})")
print("  => twin >> persistence (cả 2) => Twin/MPC SỐNG. twin ≤ persistence => ảo ảnh, bỏ #6/#7.")

# ---- Part 2: reachability/controllability (K,B trên toàn data) ----
Kop, B = fit_fold(np.arange(n))
print("\n== Part 2: REACHABILITY (đẩy X từ uniform, belief mô hình cuối) — cho #1/#2/#4 ==")
z0 = np.zeros(N * K); reach = []
for x in range(K):
    psi = d.transform(z0.reshape(1, -1))[0]; u = np.zeros(K); u[x] = 1.0
    for _ in range(T):
        psi = Kop @ psi + B @ u
    bx = mean_belief(np.real(psi[ss]), N, K)[x]
    reach.append(bx)
    print(f"  target {x}: reachable belief[{x}] = {bx:.2f}")
print(f"  controllability index (TB reachable) = {np.mean(reach):.2f}  "
      f"(gần 1 => mọi đáp án reachable => debate cực điều-khiển-được)")
# reachability CÓ ĐIỀU KIỆN: đáp án đúng có nằm trong reachable set không (từ state thật)
print("\n  (mở rộng: reachable-set từ state THẬT mỗi debate -> 'đáp án đúng có tới được không'")
print("   = ý #1 Reachability; tính được nếu Part 1 fidelity đủ, hoặc như control-property thuần.)")

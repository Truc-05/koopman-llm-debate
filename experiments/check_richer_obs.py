"""② Test rẻ: quan sát BẤT ĐỒNG/entropy có tín hiệu mà belief-trung-bình (tầm thường) không có?

Ở vòng k, mỗi debate tính:
  - belief thường (tầm thường): truth_mass tích lũy, confidence phe dẫn
  - tương tác: phương sai belief giữa agent, entropy, spread, pairwise-TV, movement
Đo AUC dự đoán 2 thứ:
  (a) ĐÚNG/SAI cuối (so a_star)
  (b) LẬT (phe cuối != phe tại k)  <- chỗ Koopman-tự-trị chết; nếu tương tác đoán được
      lật => có đất để đưa vào operator; nếu ~0.5 => bất-đồng cũng vô dụng.

Dùng: python experiments/check_richer_obs.py mmlu_clean6 [k=3]
"""
import sys, os, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.debate.state import split_state, mean_belief

OUT = os.path.join(ROOT, "experiments", "out")
tag = sys.argv[1] if len(sys.argv) > 1 else "mmlu_clean6"
kfix = int(sys.argv[2]) if len(sys.argv) > 2 else 3
N = 4

npz = np.load(os.path.join(OUT, f"trajs_{tag}.npz"))
trajs = [np.asarray(npz[f"traj_{i}"]) for i in range(len(npz.files))]
tr = json.load(open(os.path.join(OUT, f"transcripts_{tag}.json")))
a_star = np.array([int(r["a_star"]) for r in tr])
y = np.array([bool(r["correct"]) for r in tr])
K = trajs[0].shape[1] // N
n = len(trajs); T = trajs[0].shape[0] - 1


def feats_at(traj, k, a_star_i):
    p = np.stack([split_state(traj[t], N, K)[1] for t in range(k + 1)])  # (k+1, N, K)
    pbar = p.mean(1)                                    # (k+1, K)
    pk = p[k]; pbark = pbar[k]                          # tại vòng k
    leader = int(np.argmax(pbark))
    return {
        # tầm thường (belief)
        "truth_mass": float(pbar[:, a_star_i].mean()),          # ~ τ_const
        "leader_conf": float(pbark[leader]),
        # tương tác
        "agent_var": float(pk[:, leader].var()),                # bất đồng về phe dẫn
        "entropy": float(-(pbark * np.log(pbark + 1e-12)).sum()),
        "spread": float(pk[:, leader].max() - pk[:, leader].min()),
        "pairwise_tv": float(np.mean([0.5 * np.abs(pk[i] - pk[j]).sum()
                                      for i in range(N) for j in range(i + 1, N)])),
        "movement": float(np.linalg.norm(traj[k] - traj[0])),
        "recent_move": float(np.linalg.norm(traj[k] - traj[k - 1])) if k >= 1 else 0.0,
    }, leader


def auc(score, lab):
    lab = np.asarray(lab, bool)
    if lab.all() or (~lab).any() == 0:
        return float("nan")
    p, m = score[lab], score[~lab]
    return ((p[:, None] > m[None, :]).sum() + 0.5 * (p[:, None] == m[None, :]).sum()) / (len(p) * len(m))


rows = []
flip = np.zeros(n, bool)
F = {}
for i, t in enumerate(trajs):
    f, leader = feats_at(t, kfix, a_star[i])
    final_leader = int(np.argmax(mean_belief(t[-1], N, K)))
    flip[i] = (leader != final_leader)
    for kk, vv in f.items():
        F.setdefault(kk, np.zeros(n))
        F[kk][i] = vv

print(f"tag={tag}  n={n}  k={kfix}  base(đúng)={y.mean():.3f}  #lật={flip.sum()} ({flip.mean():.2f})")
print(f"{'feature':<14}{'AUC→đúng':>10}{'AUC→lật':>10}   nhóm")
groups = {"truth_mass": "belief", "leader_conf": "belief",
          "agent_var": "tương tác", "entropy": "tương tác", "spread": "tương tác",
          "pairwise_tv": "tương tác", "movement": "động", "recent_move": "động"}
for kk in F:
    s = F[kk]
    # AUC→lật: đảo dấu nếu cần để đọc |.−0.5|; in cả 2 chiều bằng max
    a_corr = auc(s, y)
    a_flip = auc(s, flip)
    print(f"{kk:<14}{a_corr:>10.3f}{a_flip:>10.3f}   {groups[kk]}")

print("\nĐọc:")
print(" - Cột AUC→lật của nhóm 'tương tác' (agent_var/entropy/spread/pairwise_tv):")
print("   xa 0.5 (>0.6 hoặc <0.4) => bất đồng ĐOÁN được lật => có đất đưa vào operator (②/①).")
print("   ~0.5 hết => tương tác vô dụng cho lật => Koopman-tự-trị hết cửa, nhảy ① EDMDc (arguments).")
print(" - So AUC→đúng của 'tương tác' vs 'truth_mass' (baseline tầm thường ~0.87 full,")
print("   thấp hơn ở k nhỏ): tương tác có THÊM gì ngoài belief không.")

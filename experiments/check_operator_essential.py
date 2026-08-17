"""OPERATOR-ESSENTIAL test (quyết định paper Koopman top-tier có tồn tại):
Phát hiện thao túng bằng OPERATOR (residual so động lực bình thường học được) có HƠN
detector NGÂY THƠ (biên độ nhảy/displacement/entropy) không?

- Âm class (0): debate SẠCH (trajs_<tag>.npz, không can thiệp).
- Dương class (1): debate BỊ CAN THIỆP (ckpt_edmdc_<tag>.jsonl, có moderator đẩy 1 đáp án).
Fit K tự trị trên clean-train; residual ||Ψ(z_{t+1}) − K·Ψ(z_t)|| lớn = lệch động-lực-bình-thường
= dấu bị lái. So AUC với baseline không-operator.

Nếu operator-residual AUC >> naive => operator THIẾT YẾU (phát hiện được cái naive bỏ sót)
=> có đóng góp Koopman thật. Nếu ≈ naive => Koopman trang trí cho detection.

Dùng: python experiments/check_operator_essential.py mmlu_clean6
"""
import sys, os, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import build_snapshots, fit_edmd
from src.koopman.spectrum import project_stable
from src.debate.state import mean_belief

OUT = os.path.join(ROOT, "experiments", "out")
tag = sys.argv[1] if len(sys.argv) > 1 else "mmlu_clean6"
N, REG = 4, 1e-6
rng = np.random.default_rng(0)

# clean (0)
npz = np.load(os.path.join(OUT, f"trajs_{tag}.npz"))
clean = [np.asarray(npz[f"traj_{i}"]) for i in range(len(npz.files))]
# intervened (1)
interv = [np.asarray(r["traj"]) for r in
          (json.loads(l) for l in open(os.path.join(OUT, f"ckpt_edmdc_{tag}.jsonl")))]
K = clean[0].shape[1] // N
print(f"tag={tag}  clean={len(clean)}  intervened={len(interv)}")

d = PolynomialDictionary(degree=1); d.transform(clean[0][:1])

# fit K tự trị trên 70% clean (train) — detector là OOS
idx = rng.permutation(len(clean)); ntr = int(0.7 * len(clean))
tr_idx, te_idx = idx[:ntr], idx[ntr:]
Zt = np.concatenate([clean[i][:-1] for i in tr_idx]); Ztp1 = np.concatenate([clean[i][1:] for i in tr_idx])
Px, Py = build_snapshots(d, Zt, Ztp1)
Kop, _, _ = fit_edmd(Px, Py, reg=REG); Kop = project_stable(Kop)


def op_residual(traj):
    """max_t ||Ψ(z_{t+1}) − K·Ψ(z_t)|| — lệch động lực bình thường."""
    P = d.transform(traj)                      # (T+1, M)
    res = [np.linalg.norm(P[t + 1] - Kop @ P[t]) for t in range(len(traj) - 1)]
    return max(res)


def naive(traj):
    jumps = [np.linalg.norm(traj[t + 1] - traj[t]) for t in range(len(traj) - 1)]
    disp = np.linalg.norm(traj[-1] - traj[0])
    bel = np.stack([mean_belief(z, N, K) for z in traj])
    bjump = max(np.abs(bel[t + 1] - bel[t]).sum() for t in range(len(traj) - 1))
    ent_final = -(bel[-1] * np.log(bel[-1] + 1e-12)).sum()
    return {"max_jump": max(jumps), "displacement": disp, "max_belief_jump": bjump,
            "neg_final_entropy": -ent_final}


# tập test: clean held-out (0) + intervened (1)
test = [(clean[i], 0) for i in te_idx] + [(t, 1) for t in interv]
y = np.array([lab for _, lab in test], bool)
scores = {"OPERATOR-residual": np.array([op_residual(t) for t, _ in test])}
for kk in ["max_jump", "displacement", "max_belief_jump", "neg_final_entropy"]:
    scores[kk] = np.array([naive(t)[kk] for t, _ in test])


def auc(s):
    p, m = s[y], s[~y]
    return ((p[:, None] > m[None, :]).sum() + 0.5 * (p[:, None] == m[None, :]).sum()) / (len(p) * len(m))


print(f"\nphát hiện thao túng (clean {int((~y).sum())} vs intervened {int(y.sum())}):")
print(f"{'detector':<22}{'AUC':>7}   nhóm")
print(f"{'OPERATOR-residual':<22}{auc(scores['OPERATOR-residual']):>7.3f}   operator")
best_naive = 0
for kk in ["max_jump", "displacement", "max_belief_jump", "neg_final_entropy"]:
    a = auc(scores[kk]); best_naive = max(best_naive, a)
    print(f"{kk:<22}{a:>7.3f}   naive")

op = auc(scores['OPERATOR-residual'])
print(f"\nOPERATOR {op:.3f} vs naive tốt nhất {best_naive:.3f}  (Δ={op-best_naive:+.3f})")
print("Đọc:")
print(" - OPERATOR >> naive (Δ>~0.05) => operator THIẾT YẾU cho phát hiện => đóng góp Koopman thật => top-tier có cửa.")
print(" - OPERATOR ≈ naive => detector naive đủ => Koopman TRANG TRÍ cho detection => top-tier Koopman lung lay.")
print(" - (test này chỉ là detection; steering-MPC-vs-naive cần chạy debate mới.)")

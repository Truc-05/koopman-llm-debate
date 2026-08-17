"""Probe: HỘI TỤ SỚM (sụp đa dạng nhanh) có dự báo debate SAI không — và có PHẢI tín hiệu
thật (không phải confidence/mean-belief trá hình)?

Giả thuyết: agents đồng thuận quá sớm (đa dạng sụp nhanh) = khám phá thiếu = dễ chốt nhầm.
Nếu ĐÚNG + KHÔNG trivial => động cơ cho can thiệp Koopman-control chống-sụp (win không cần nhãn).

Đa dạng vòng t = TB pairwise total-variation giữa agents (cao = bất đồng).
Kiểm non-trivial: sau khi bỏ ảnh hưởng confidence (residualize), collapse còn dự báo không?

Dùng: python experiments/check_premature_convergence.py mmlu_clean6
"""
import sys, os, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.debate.state import split_state, mean_belief

OUT = os.path.join(ROOT, "experiments", "out")
tag = sys.argv[1] if len(sys.argv) > 1 else "mmlu_clean6"
N = 4
npz = np.load(os.path.join(OUT, f"trajs_{tag}.npz"))
trajs = [np.asarray(npz[f"traj_{i}"]) for i in range(len(npz.files))]
tr = json.load(open(os.path.join(OUT, f"transcripts_{tag}.json")))
y = np.array([bool(r["correct"]) for r in tr])
K = trajs[0].shape[1] // N
T = trajs[0].shape[0] - 1


def diversity(z):
    _, p = split_state(z, N, K)                       # (N,K)
    return float(np.mean([0.5 * np.abs(p[i] - p[j]).sum()
                          for i in range(N) for j in range(i + 1, N)]))


F = {k: np.zeros(len(trajs)) for k in
     ["div_mean", "div_final", "collapse_speed", "collapse_round", "final_conf", "final_entropy"]}
for i, t in enumerate(trajs):
    div = np.array([diversity(z) for z in t])         # (T+1,)
    pbar = mean_belief(t[-1], N, K)
    F["div_mean"][i] = div.mean()
    F["div_final"][i] = div[-1]
    F["collapse_speed"][i] = div[1] - div[-1] if len(div) > 1 else 0.0   # sụp nhanh = lớn
    below = np.where(div < 0.15)[0]
    F["collapse_round"][i] = below[0] if len(below) else T + 1           # sụp sớm = nhỏ
    F["final_conf"][i] = pbar.max()
    F["final_entropy"][i] = -(pbar * np.log(pbar + 1e-12)).sum()


def auc(s, lab=y):
    p, m = s[lab], s[~lab]
    if len(p) == 0 or len(m) == 0:
        return float("nan")
    return ((p[:, None] > m[None, :]).sum() + 0.5 * (p[:, None] == m[None, :]).sum()) / (len(p) * len(m))


print(f"tag={tag} n={len(trajs)} base(đúng)={y.mean():.3f}")
print(f"{'feature':<16}{'AUC→đúng':>10}   (xa 0.5 = có tín hiệu; đa dạng cao kỳ vọng →đúng)")
for k in ["div_mean", "div_final", "collapse_speed", "collapse_round", "final_conf", "final_entropy"]:
    print(f"{k:<16}{auc(F[k]):>10.3f}")

# --- non-trivial: bỏ ảnh hưởng confidence khỏi collapse, residual còn dự báo? ---
conf = F["final_conf"]
print("\n== non-trivial check: collapse có THÊM gì ngoài confidence? ==")
for k in ["div_mean", "collapse_speed", "collapse_round"]:
    s = F[k]
    c = np.corrcoef(s, conf)[0, 1]
    # residualize s trên conf (bậc 1)
    b = np.polyfit(conf, s, 1)
    resid = s - np.polyval(b, conf)
    print(f"  {k:<16} corr(.,conf)={c:+.3f}  AUC(residual→đúng)={auc(resid):.3f}  "
          f"(≈0.5 => chỉ là confidence trá hình; xa 0.5 => tín hiệu THẬT ngoài confidence)")

print("\nĐọc:")
print(" - div_mean/collapse_* AUC xa 0.5 => hội tụ sớm có liên hệ đúng/sai.")
print(" - AUC(residual) vẫn xa 0.5 sau khi bỏ confidence => tín hiệu THẬT (không trivial)")
print("   => có đất cho can thiệp Koopman chống-sụp. Nếu residual≈0.5 => lại là confidence trá hình.")

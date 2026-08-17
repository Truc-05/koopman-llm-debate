"""TEST QUYẾT ĐỊNH: operator Koopman có GIÁ TRỊ DỰ ĐOÁN mà baseline tầm thường không có?

Ở mỗi vòng k, cuộn K^(T-k)·Ψ(z_k) -> dự đoán đáp án cuối, so 2 baseline:
  - persistence: "debate giữ nguyên phe hiện tại" = argmax belief(z_k)  [TẦM THƯỜNG]
  - koopman    : argmax belief( decode(K^(T-k) Ψ(z_k)) )               [DÙNG OPERATOR]
Mục tiêu = đáp án CUỐI thực tế (argmax belief(z_T)). Out-of-sample (cross-fit theo topic).

Phép sắc nhất = các debate CÓ LẬT (phe cuối != phe tại k): persistence sai 100% theo định
nghĩa; nếu koopman đoán trúng lật => operator bắt được ĐỘNG LỰC thật (giá trị Koopman).

+ Bonus: dự đoán ĐÚNG/SAI (so a_star) — operator có hơn "phe hiện tại có phải đáp án đúng".

Dùng: python experiments/check_operator_value.py mmlu_clean6 [degree=1]
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
degree = int(sys.argv[2]) if len(sys.argv) > 2 else 1
N, REG, NFOLD = 4, 1e-6, 5

npz = np.load(os.path.join(OUT, f"trajs_{tag}.npz"))
trajs = [np.asarray(npz[f"traj_{i}"]) for i in range(len(npz.files))]
tr = json.load(open(os.path.join(OUT, f"transcripts_{tag}.json")))
a_star = [int(r["a_star"]) for r in tr]
K = trajs[0].shape[1] // N
n = len(trajs); T = trajs[0].shape[0] - 1
fold = np.arange(n) % NFOLD
d = PolynomialDictionary(degree=degree)
d.transform(trajs[0][:1])                      # set state_slice


def fit_fold(idx):
    Zt = np.concatenate([trajs[i][:-1] for i in idx], 0)
    Ztp1 = np.concatenate([trajs[i][1:] for i in idx], 0)
    Px, Py = build_snapshots(d, Zt, Ztp1)
    Kop, _, _ = fit_edmd(Px, Py, reg=REG)
    return project_stable(Kop)


Kops = {f: fit_fold(np.where(fold != f)[0]) for f in range(NFOLD)}


def koop_pred_leader(Kop, z_k, steps):
    psi = d.transform(z_k.reshape(1, -1))[0]
    for _ in range(steps):
        psi = Kop @ psi
    z_pred = np.real(psi[d.state_slice])
    return int(np.argmax(mean_belief(z_pred, N, K)))


print(f"tag={tag} degree={degree} n={n} T={T}  (out-of-sample cross-fit)")
print(f"{'k':>2} {'persist→final':>13} {'koopman→final':>14} | "
      f"{'#lật':>5} {'koop trúng lật':>14} | {'persist→đúng':>12} {'koop→đúng':>10}")
for k in range(1, T):
    pf = kf = nflip = koop_flip = pc = kc = 0
    for i in range(n):
        Kop = Kops[fold[i]]
        z_k = trajs[i][k]; z_T = trajs[i][-1]
        cur = int(np.argmax(mean_belief(z_k, N, K)))
        fin = int(np.argmax(mean_belief(z_T, N, K)))
        koo = koop_pred_leader(Kop, z_k, T - k)
        pf += (cur == fin); kf += (koo == fin)
        if cur != fin:
            nflip += 1; koop_flip += (koo == fin)
        pc += (cur == a_star[i]); kc += (koo == a_star[i])   # dự đoán ĐÚNG/SAI
    print(f"{k:>2} {pf/n:>13.3f} {kf/n:>14.3f} | {nflip:>5} "
          f"{(koop_flip/nflip if nflip else 0):>14.3f} | {pc/n:>12.3f} {kc/n:>10.3f}")

print("\nĐọc:")
print(" - koopman→final > persist→final (nhất là k nhỏ) => operator đoán hướng tốt hơn baseline giữ-nguyên.")
print(" - koop trúng lật > 0 (persist = 0 trên lật theo định nghĩa) => operator BẮT ĐƯỢC cú lật = giá trị THẬT.")
print(" - koop→đúng > persist→đúng => operator dự đoán ĐÚNG/SAI hơn 'phe hiện tại'.")
print(" - Nếu koopman ~= hoặc < persist ở mọi cột => operator KHÔNG thêm giá trị -> Koopman vô ích ở đây.")

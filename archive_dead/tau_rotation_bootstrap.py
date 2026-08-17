"""Phép thử XOAY-CƠ-SỞ + BOOTSTRAP cho φ₁-τ (kiểm nỗi lo suy biến).

Ý tưởng: nếu không-gian riêng λ≈1 suy biến m chiều, "φ₁" chỉ xác định tới một
phép xoay unitary trong không-gian đó → mọi hướng đơn vị trong span là eigenfunction
hợp lệ như nhau. Ta lấy R hướng ngẫu nhiên trong span top-m mode và tính AUC của
τ theo từng hướng:
  - m=1 (λ=1 TÁCH RỜI): chỉ 1 hướng → 1 AUC, KHÔNG có tự do xoay → φ₁-τ well-defined.
  - m>1 (SUY BIẾN): AUC trải rộng => φ₁ (chọn index-0) chỉ là 1 hướng may rủi.
Độ trải AUC khi xoay = thước đo trực tiếp "ăn may hay không".

Bootstrap: resample debate (có hoàn lại) B lần, refit operator, tính lại φ₁-τ AUC
→ phân phối AUC (ổn định theo mẫu?).

Dùng: python experiments/tau_rotation_bootstrap.py mmlu_160 [degree=1] [R=200] [B=200]
Chạy thêm mmlu_200 (12v, suy biến) để đối chứng.
"""
import sys, os, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import build_snapshots, fit_edmd
from src.koopman.spectrum import (spectral_decomposition, project_stable,
                                  eigenfunctions, spectral_gap)
from src.debate.observables_truth import h_star_soft, along_trajectory

OUT = os.path.join(ROOT, "experiments", "out")
tag = sys.argv[1] if len(sys.argv) > 1 else "mmlu_160"
degree = int(sys.argv[2]) if len(sys.argv) > 2 else 1
R = int(sys.argv[3]) if len(sys.argv) > 3 else 200
B = int(sys.argv[4]) if len(sys.argv) > 4 else 200
N, REG = 4, 1e-6
rng = np.random.default_rng(0)

npz = np.load(os.path.join(OUT, f"trajs_{tag}.npz"))
trajs = [np.asarray(npz[f"traj_{i}"]) for i in range(len(npz.files))]
tr = json.load(open(os.path.join(OUT, f"transcripts_{tag}.json")))
y = np.array([bool(r["correct"]) for r in tr])
a_star = [int(r["a_star"]) for r in tr]
Kans = trajs[0].shape[1] // N
n = len(trajs)
d = PolynomialDictionary(degree=degree)


def auc(s):
    p, m = s[y], s[~y]
    return ((p[:, None] > m[None, :]).sum() + 0.5 * (p[:, None] == m[None, :]).sum()) / (len(p) * len(m))


def fit(idx_debates, stable=True):
    Zt = np.concatenate([trajs[i][:-1] for i in idx_debates], 0)
    Ztp1 = np.concatenate([trajs[i][1:] for i in idx_debates], 0)
    Px, Py = build_snapshots(d, Zt, Ztp1)
    K, _, _ = fit_edmd(Px, Py, reg=REG)
    if stable:
        K = project_stable(K)
    return spectral_decomposition(K)


# per-debate h + full eigenfunction matrix (pooled operator, in-sample cho test hình học)
ev, V, W = fit(np.arange(n), stable=True)
H, PHI = [], []
for i in range(n):
    Zi = trajs[i][:-1]
    H.append(along_trajectory(h_star_soft, Zi, n_agents=N, n_answers=Kans, a_star=a_star[i]).astype(complex))
    PHI.append(eigenfunctions(d, W, Zi))       # (S_i, M), cột sort |λ| giảm

mods = np.abs(ev)
print(f"tag={tag} degree={degree}  n={n}  base={y.mean():.3f}")
print(f"phổ (stable-DMD): |λ1|={mods[0]:.4f} |λ2|={mods[1]:.4f} |λ3|={mods[2]:.4f} "
      f"gap={spectral_gap(ev):+.4f}")
print(f"φ₁-τ (hướng index-0) AUC = {auc(np.array([abs(np.vdot(H[i],PHI[i][:,0]))/(np.linalg.norm(H[i])*np.linalg.norm(PHI[i][:,0])+1e-12) for i in range(n)])):.3f}")

# ---- XOAY-CƠ-SỞ: AUC khi τ dùng hướng ngẫu nhiên trong span top-m mode ----
print(f"\n== xoay-cơ-sở: AUC của τ qua {R} hướng ngẫu nhiên trong span top-m ==")
print(f"{'m':>2} {'AUC mean':>9} {'std':>7} {'min':>7} {'max':>7}   (std nhỏ => KHÔNG ăn may)")
for m in [1, 2, 3, 5]:
    aucs = []
    for _ in range(R):
        c = rng.standard_normal(m) + 1j * rng.standard_normal(m)
        c /= np.linalg.norm(c)
        tau = np.empty(n)
        for i in range(n):
            v = PHI[i][:, :m] @ c
            tau[i] = abs(np.vdot(H[i], v)) / (np.linalg.norm(H[i]) * np.linalg.norm(v) + 1e-12)
        aucs.append(auc(tau))
    aucs = np.array(aucs)
    tag_m = " <- λ=1 tách rời (1 chiều)" if m == 1 else ""
    print(f"{m:>2} {aucs.mean():>9.3f} {aucs.std():>7.3f} {aucs.min():>7.3f} {aucs.max():>7.3f}{tag_m}")

# ---- BOOTSTRAP: refit trên mẫu có hoàn lại, φ₁-τ AUC ----
print(f"\n== bootstrap {B} lần (resample debate, refit, φ₁-τ) ==")
baucs, bl2, bidx_uniq = [], [], []
for b in range(B):
    idx = rng.integers(0, n, n)
    bidx_uniq.append(len(np.unique(idx)))          # số debate distinct trong mẫu (phải ~0.63n)
    ev_b, _, Wb = fit(idx, stable=True)
    bl2.append(float(np.abs(ev_b[1])))             # |λ2| mỗi lần refit
    tau = np.empty(n)
    for i in range(n):
        Zi = trajs[i][:-1]
        p = eigenfunctions(d, Wb, Zi)[:, 0]
        tau[i] = abs(np.vdot(H[i], p)) / (np.linalg.norm(H[i]) * np.linalg.norm(p) + 1e-12)
    baucs.append(auc(tau))
baucs = np.array(baucs)
print(f"AUC = {baucs.mean():.5f} ± {baucs.std():.5f}  "
      f"[min={baucs.min():.5f}, max={baucs.max():.5f}]  #AUC distinct={len(np.unique(np.round(baucs,5)))}/{B}")
# --- kiểm refit CÓ thực sự đổi không (nếu std(|λ2|)≈0 & distinct debate cố định => nghi bug) ---
print(f"chẩn đoán refit: |λ2| qua bootstrap = {np.mean(bl2):.4f} ± {np.std(bl2):.4f} "
      f"(std>0 => operator CÓ đổi qua mẫu); #debate distinct/mẫu = {np.mean(bidx_uniq):.0f} (kỳ vọng ~{0.63*n:.0f})")
print("Đọc: std(AUC)≈0 NHƯNG std(|λ2|)>0 => AUC bất biến thật (rank ổn định, KHÔNG bug). "
      "std(|λ2|)≈0 => refit không đổi => BUG resample.")

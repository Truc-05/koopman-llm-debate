# Nhịp 4 v2 — Trigger cập nhật cho τ SẠCH (đơn điệu), độc lập với bản cũ

*File nháp độc lập (2026-07-19). KHÔNG sửa `run_intervention.py` / `beat4_intervention_spec.md`
hiện có — đây là bản thiết kế trigger mới, khi triển khai thì tạo `run_intervention_v2.py`.*

## 1. Vì sao đổi trigger

Bản cũ (`run_intervention.py`) dùng **gated-τ = τ·min(1, disp/median)**. Cái "cổng chuyển động"
đó sinh ra để khử cụm **τ≈1 giả** — mà cụm đó hóa ra là **artifact parse** (model xuất JSON hỏng
→ uniform → quỹ đạo phẳng → τ giả cao). Sau khi sửa parse + dùng model ≥7B, cụm giả biến mất và
**τ trở nên đơn điệu** (τ thấp = sẽ sai, τ cao = sẽ đúng; xác nhận trên gpt-4o-mini & qwen).

⇒ **Bỏ cổng. Trigger mới = φ₁-τ@k thấp** (không nhân displacement, không band). Đơn giản, khớp
với bản chất đơn điệu đã kiểm.

## 2. Định nghĩa trigger v2

Detector đóng băng từ một **calibration set SẠCH** (vd `mmlu_clean6` — qwen2.5:7b, degree-1):
- Fit pooled EDMD (degree 1) → `W` (left-eigvec) → `φ₁ = eigenfunctions(d, W, ·)[:,0]`.
- **Không** cần median displacement nữa.
- `τ@k(debate) = |⟨h★, φ₁⟩| / (‖h★‖‖φ₁‖)` tính trên **k vòng đầu** (early-warning) hoặc full.
- Ngưỡng **θ**: chọn trên ROC OOS của calibration sao cho **P(sai | τ@k < θ) ≥ 0.70**
  (trigger thận trọng — chỉ can thiệp khi khá chắc debate sẽ hỏng).
- **Trigger ⇔ τ@k < θ.**

*Lưu ý regime*: giữ **degree-1 + ~6 vòng** để λ=1 tách rời (φ₁ well-defined). Nếu calibration
mới có |λ|>1 do nhiễu, dùng `project_stable` trước khi lấy phổ.

## 3. Snippet detector v2 (thả vào `run_intervention_v2.py`)

```python
import numpy as np
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import build_snapshots, fit_edmd
from src.koopman.spectrum import spectral_decomposition, project_stable, eigenfunctions
from src.criteria.truth_alignment import truth_alignment_index
from src.debate.observables_truth import h_star_soft, along_trajectory

def build_detector_v2(trajs, y, a_star, N, K, degree=1, reg=1e-6, precision=0.70):
    """Trả (d, W, θ). trajs/y/a_star: calibration SẠCH (vd mmlu_clean6)."""
    d = PolynomialDictionary(degree=degree)
    Zt  = np.concatenate([t[:-1] for t in trajs]);  Ztp1 = np.concatenate([t[1:] for t in trajs])
    Px, Py = build_snapshots(d, Zt, Ztp1)
    K_op, _, _ = fit_edmd(Px, Py, reg=reg)
    K_op = project_stable(K_op)                       # ép |λ|≤1 cho chắc
    _, _, W = spectral_decomposition(K_op)
    tau = np.array([_tau_full(d, W, t, N, K, a_star[i]) for i, t in enumerate(trajs)])
    theta = _choose_theta(tau, y, precision)          # θ để P(đúng|τ<θ) ≤ 1-precision
    return d, W, theta

def tau_at_k_v2(d, W, traj, k, N, K, a_star):
    Zi = traj[:k]                                     # k vòng đầu (early-warning); k=T = full
    h  = along_trajectory(h_star_soft, Zi, n_agents=N, n_answers=K, a_star=a_star)
    return float(truth_alignment_index(h, eigenfunctions(d, W, Zi)[:, 0]))

def _tau_full(d, W, traj, N, K, a_star):
    return tau_at_k_v2(d, W, traj, traj.shape[0]-1, N, K, a_star)

def _choose_theta(tau, y, precision):
    max_correct = 1.0 - precision
    best = float(np.quantile(tau, 0.40))
    for th in np.unique(tau):
        below = tau < th
        if below.sum() >= 5 and y[below].mean() <= max_correct:
            best = float(th)
    return best
# trigger: tau_at_k_v2(...) < theta
```

## 4. Phần GIỮ NGUYÊN từ `beat4_intervention_spec.md`

Toàn bộ khung thí nghiệm không đổi, chỉ thay detector:
- **Fork ghép cặp** tại vòng k (chung vòng 0..k-1, rẽ 4 nhánh).
- **4 arm**: A_koopman (điều hướng φ₁), A_generic, A_sham, A_none.
- **Test**: Wilcoxon signed-rank ghép cặp + McNemar + bootstrap CI, Holm cho H1/H2.
- **Power/cỡ mẫu**, **checkpoint/resume**, **fail-fast**: y như bản cũ (§4–§7 file đó).
- Chạy trên **cùng model + calibration sạch** (qwen2.5:7b local — free, không quota).

## 5. Điều cần kiểm lại trước khi chạy campaign

- [ ] Xác nhận τ@k đơn điệu trên **calibration mới** (chạy `tau_at_k.py` — cột AUC τ@k thô, KHÔNG
      cần gated). Nếu k=3 đã AUC cao → early-warning; nếu chỉ full-τ mới cao → chẩn đoán hậu nghiệm.
- [ ] θ cho precision-fail 0.70: log trigger-rate + P(sai|trigger) trên calibration.
- [ ] A_koopman: hàm chọn "challenger" theo loading φ₁ (đã có `koopman_challenger` trong
      `run_intervention.py` — tái dùng, không đổi).
- [ ] Vì degree-1 tuyến tính, "Koopman-MPC" đóng vòng (cần B từ EDMDc) vẫn là Stage 3 future;
      Stage 1-2 dùng can thiệp cố định như cũ.

## 6. Khác biệt tóm tắt so với bản cũ

| | v1 (cũ) | **v2 (mới)** |
|---|---|---|
| Trigger | gated-τ = τ·min(1,disp/med) < θ | **φ₁-τ@k < θ** (bỏ cổng) |
| Lý do cổng | khử cụm τ≈1 giả (parse artifact) | **hết artifact → không cần cổng** |
| Calibration | mmlu_40_r5 (8B, nhiễm parse) | **mmlu_clean6 (qwen, sạch)** |
| Phần còn lại | — | **giữ nguyên** (fork, 4 arm, stats) |

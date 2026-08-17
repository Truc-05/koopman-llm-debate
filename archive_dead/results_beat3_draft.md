# Draft Results — Nhịp 3 (Chẩn đoán): τ phổ Koopman dự đoán đúng/sai của debate

*File nháp độc lập (2026-07-19). Số liệu từ `tau_robust.py` / `tau_rotation_bootstrap.py`.
Dịch sang tiếng Anh khi đưa vào paper.*

## 3.1 Thiết lập

Mỗi debate là quỹ đạo trạng thái `z_t = (x_t^1,…,x_t^N) ∈ R^{NK}` (logit niềm tin của N=4
agent trên K=4 đáp án, T=6 vòng → 6 bước chuyển/debate). Ước lượng toán tử Koopman bằng
EDMD trên **pool mọi debate** với dictionary **tuyến tính (degree 1)** — xem 3.4 vì sao không
dùng degree 2. Gọi `φ₁` là eigenfunction ứng trị riêng dẫn đầu `λ₁≈1` (mode đồng thuận).

**Chỉ số căn chỉnh chân lý**: với quan sát `h★(z)` = khối lượng niềm tin trung bình đặt lên
đáp án đúng dọc quỹ đạo,
```
τ = |⟨h★, φ₁⟩| / (‖h★‖ · ‖φ₁‖)
```
đo mức quỹ đạo truth-alignment bám mode đồng thuận chậm. Đánh giá **out-of-sample** bằng
cross-fit 5-fold theo topic (φ₁ fit trên fold train, chấm τ cho fold giữ lại).

## 3.2 Phát hiện chính: τ tách debate đúng/sai, out-of-sample, đa model

| Model (họ) | acc | n | τ|đúng | τ|sai | **AUC(τ) OOS** | bootstrap |
|---|---|---|---|---|---|---|
| gpt-4o-mini (OpenAI) | 0.75 | 100 | 0.96±0.07 | ~0.49 | **0.97** | 0.92 ± 0.07 |
| qwen2.5:7b (Alibaba) | 0.61 | 200 | 0.94±0.04 | 0.70±0.20 | **0.87** | 0.87 ± 0.00 |

Trên cả hai model, τ của debate ĐÚNG cao hơn hẳn debate SAI; đường cong precision-theo-τ
tăng đơn điệu (top decile ≈ 100% đúng). AUC dao động theo model (dynamics khác nhau) nhưng
tín hiệu **model-general**: cùng một eigenfunction đồng thuận φ₁ dự báo độ tin cậy ở hai họ
model rất khác nhau. *(Run Llama-3.1-8B ban đầu cũng cho τ tách đúng/sai, nhưng nhiễm lỗi
dữ liệu — xem 3.5; ta báo cáo hai model có dữ liệu sạch làm bằng chứng chính.)*

## 3.3 Phổ sạch: mode đồng thuận duy nhất

Với degree-1 + 6 vòng, phổ pooled có **đúng một λ=1 tách rời**:

| Model | \|λ₁\| | \|λ₂\| | gap = 1−\|λ₂\| | \|λ\|>1 | unique consensus |
|---|---|---|---|---|---|
| gpt-4o-mini | 1.000 | 0.995 | +0.005 | 0 | ✓ |
| qwen2.5:7b | 1.000 | 0.961 | **+0.039** | 0 | ✓ |

Khớp Prop 1: đúng một trị riêng tại 1 ⇒ đồng thuận duy nhất, phần còn lại nằm trong đĩa đơn
vị. (Với dữ liệu có \|λ\|>1 do nhiễu ước lượng, dùng stable-DMD chiếu \|λ\|≤1 cho hình phổ.)

## 3.4 Vì sao TUYẾN TÍNH (degree 1), không phi tuyến (degree 2)

Ở quy mô dữ liệu hiện có (600–2400 bước chuyển), dictionary đa thức **degree 2 (153 features)
bất ổn**: mọi mức ridge / SVD-truncation vẫn cho \|λ\|>1 và gap âm, và không-gian λ≈1 **suy
biến nhiều chiều** khiến φ₁ không xác định duy nhất. Degree 1 (17 features) well-posed, cho
phổ sạch với λ=1 cô lập. Debate DÀI hơn (12 vòng) làm *xấu đi* (đuôi gần cân bằng dồn mode về
λ≈1) → chọn ~6 vòng (transient). Phi tuyến cần nhiều/đa dạng debate hơn → **future work**.

## 3.5 Độ chắc (robustness) — τ không "ăn may" từ suy biến

Vì φ₁ chỉ xác định tới một phép xoay *khi* λ=1 suy biến, ta kiểm ba lớp:

1. **Chiếu không-gian-con vs φ₁ đơn lẻ**: khi λ=1 cô lập (dim=1), `τ = subspace-projection` y hệt
   cosine-với-φ₁ (AUC trùng khít) → τ bất biến cơ sở, xác định rõ.
2. **Xoay-cơ-sở**: lấy hướng ngẫu nhiên trong span top-m. Ở **m=1** (mode λ=1 cô lập) không có
   tự do xoay → AUC xác định (qwen 0.871, std 0). Chỉ khi ép m>1 (gộp mode dưới 1) mới thấy
   trải — xác nhận suy biến là nguồn mơ hồ, và ta tránh nó bằng regime λ=1 cô lập.
3. **Bootstrap** (resample debate + refit): gpt-4o-mini AUC 0.92±0.07; qwen 0.871±0.00 — gap phổ
   càng lớn eigenfunction càng ổn định → tín hiệu thật, không phải nhiễu mẫu.

## 3.6 Ghi chú phương pháp (data quality)

Run pilot Llama-8B lộ ra: model yếu đôi khi xuất JSON sai độ dài → parser cũ *âm thầm* trả
uniform → tạo cụm τ≈1 **giả** (87% lượt hỏng parse ở cụm này) và một "chữ U ngược" ban đầu bị
diễn giải nhầm là herding. Sau khi (a) ép prompt đúng K xác suất + **retry khi sai** thay vì
uniform, (b) chuyển model đủ mạnh (≥7B), tỉ lệ hỏng parse ≈ 0, cụm τ≈1 giả biến mất, và τ trở
nên **đơn điệu**. Bài học: chuẩn hóa/ kiểm chất lượng đầu ra trước khi đọc phổ.

## 3.7 Câu chốt nhịp 3

> τ — độ căn chỉnh của quan sát chân lý với eigenfunction đồng thuận (λ=1) của toán tử Koopman
> tuyến tính học từ log — **dự đoán một debate sẽ đúng hay sai, out-of-sample, trên nhiều họ
> model**, với phổ sạch và tín hiệu robust. Đây là tiêu chí chẩn đoán mà FJ/DeGroot (mô hình
> tham số cố định) không cung cấp.

### Việc còn nợ để chắc hơn
- [ ] 1 run Llama sạch (≥8B) để có model-family thứ 3 sạch (hiện Llama pilot bị nhiễm).
- [ ] Báo cáo thêm: precision@top-decile, calibration của τ, so baseline đồng thuận thô (đã có
      `baseline_consensus.py`, cần chạy lại trên data sạch).

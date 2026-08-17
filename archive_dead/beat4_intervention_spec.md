# Đặc tả thí nghiệm Nhịp 4 — Can thiệp Koopman-điều-hướng (pre-registration)

Mục tiêu: chứng minh **can thiệp kích hoạt bởi tín hiệu phổ Koopman làm tăng accuracy** so với
không can thiệp, một cách **nhân quả** và **có ý nghĩa thống kê**, với chi phí API tối thiểu.

Neo vào kết quả đã có (run `mmlu_40_r5`, 200 debate, 4 agent × 6 vòng, K=4):
gated-τ@3 dự báo kết cục ở AUC 0.70 / band-precision 0.79 (out-of-sample, cấp topic).

---

## 0. Nguyên tắc "mỗi call đều tính"

1. **Thiết kế fork (paired)**: chạy chung vòng 0–k MỘT LẦN, rồi *rẽ nhánh* tại vòng trigger.
   Mọi nhánh chia sẻ y hệt lịch sử trước trigger ⇒ khử phương sai lớn nhất (độ khó câu hỏi +
   quỹ đạo đầu). Chỉ vòng k→T là tốn call nhân theo số nhánh.
2. **Chỉ can thiệp lên debate ĐƯỢC TRIGGER** (dự báo sẽ hỏng). Không đụng debate lành → tiết
   kiệm + tránh gây hại.
3. **Nhiều continuation/nhánh** để trung bình hóa nhiễu sampling (temperature>0), biến outcome
   nhị phân thành accuracy liên tục ⇒ test mạnh hơn.
4. **Thiết kế 2 giai đoạn**: pilot nhỏ ước lượng phương sai + hiệu ứng, rồi mới định N chính thức.

---

## 1. Định nghĩa can thiệp (intervention)

Can thiệp = **một lượt "moderator" chèn vào sau vòng trigger k**, hiển thị cho mọi agent trước
vòng k+1. Bốn nhánh (arms):

| Arm | Nội dung moderator | Vai trò |
|---|---|---|
| **A_koopman** (treatment chính) | Nêu phản biện MẠNH NHẤT nhằm vào đáp án có **loading φ₁-truth-aligned cao nhất theo operator** (không phải đáp án đang dẫn), + yêu cầu mỗi agent **suy luận lại độc lập từ đầu, bỏ qua đồng thuận trước**. | Điều hướng theo phổ Koopman |
| **A_generic** (active control) | Cùng format/độ dài, nhưng phản biện **chung chung** nhắm đáp án đang dẫn + "reconsider from scratch". KHÔNG dùng phổ. | Tách giá trị của điều hướng Koopman |
| **A_sham** (placebo) | Thông điệp **trung tính** ("hãy tiếp tục cẩn thận"), cùng vị trí/độ dài. | Tách hiệu ứng "bị ngắt nhịp/thêm token" |
| **A_none** (C0) | Không chèn gì. | Baseline thực tế |

Cùng một can thiệp phải xử lý được **cả hai chế độ hỏng**: đóng băng (bơm năng lượng/thông tin
mới) và trôi-về-sai (buộc xét lại) — nên format "phản biện + tái suy luận độc lập" chọn có chủ đích.

**Lưu ý MPC thật (đóng vòng)**: Koopman-MPC đầy đủ cần ma trận điều khiển B (từ `fit_edmdc`), mà B
chỉ học được TỪ dữ liệu can thiệp. Vì vậy:
- Giai đoạn 1–2 = can thiệp **cố định, mở-vòng** (A_koopman ở trên) — đủ để chứng minh "trigger
  phổ + nudge có ích". Đồng thời **thu thập (state, action, next-state)** để về sau fit B.
- Giai đoạn 3 (future work) = đóng vòng: dùng B đã học, MPC chọn action tối đa hóa mode-chân-lý dự báo.

---

## 2. Nhóm đối chứng & giả thuyết (pre-registered)

Đơn vị phân tích = **một topic được trigger** (fork chia sẻ vòng 0–k). Mỗi topic chạy đủ 4 arm
(hoặc tập rút gọn, xem §5), mỗi arm R continuation.

Giả thuyết chính (khóa trước khi chạy):
- **H1 (hiệu ứng tổng)**: acc(A_koopman) > acc(A_none) trên tập trigger.
- **H2 (luận điểm cốt lõi)**: acc(A_koopman) > acc(A_generic) — điều hướng Koopman hơn nudge chung.
- **H3 (đặc hiệu)**: can thiệp KHÔNG cải thiện (hoặc làm hại) debate **không-trigger** → targeting là cần.
  Kiểm bằng arm C2: can thiệp lên mẫu ngẫu nhiên debate lành.

Đa so sánh: khai báo H1 là primary; H2, H3 secondary; hiệu chỉnh **Holm**. Outcome chấm tự động
(argmax mean-belief == a_star) nên không cần blind.

---

## 3. Tiêu chí trigger (dựa τ, λ₂)

**Trigger chính = gated-τ@3 < θ** với `gated-τ@k = τ@k · min(1, disp_k/median_disp_k)`,
`disp_k = ‖z_k − z_0‖`, `τ@k = |⟨h_★, φ₁⟩|/(‖h_★‖‖φ₁‖)` trên k vòng đầu.
- φ₁ = eigenfunction dẫn đầu của operator EDMD **đóng băng từ calibration set** (200 debate r5).
- median_disp_3 cũng đóng băng từ calibration set. **Không refit trên campaign** (tránh rò rỉ).
- θ chọn trên đường ROC OOS để đạt **precision(sẽ-hỏng | dưới θ) ≥ 0.70** (trigger thận trọng,
  chỉ can thiệp khi khá chắc at-risk). Ghi lại θ, trigger-rate, và P(hỏng|trigger) trước khi chạy.

**Về λ₂ — trung thực**: τ@3 CHÍNH LÀ tiêu chí phổ (căn chỉnh với eigenfunction dẫn đầu φ₁), và
cổng chuyển động = "có động theo mode nào không". Còn `|λ₂|` **per-debate ở k=3 quá nhiễu** (6 cặp
chuyển tiếp/16 chiều — đã kiểm: AUC herding-zone chỉ 0.552). Do đó:
- Primary trigger = gated-τ@3 (đã validate).
- **Ablation (pre-registered, để báo cáo)**: trigger phổ thay thế `s = |λ₂|_pooled·(1−τ@3)` (herding
  score kiểu doc). Kỳ vọng YẾU hơn gated-τ → chính là bằng chứng "vì sao cần cổng chuyển động".

---

## 4. Đo Δaccuracy & cỡ mẫu

### Estimand
Δacc = acc(treatment) − acc(control) **trên tập trigger**. Với R continuation/arm/topic, mỗi topic
cho acc_arm,i ∈ [0,1]; hiệu ghép cặp δ_i = acc_T,i − acc_C,i.

### Test
- **Chính**: Wilcoxon signed-rank trên {δ_i} (hoặc paired t nếu δ xấp xỉ chuẩn). Ghép cặp qua fork
  ⇒ khử biến thiên topic ⇒ mạnh hơn nhiều so với so sánh không ghép.
- **Phụ (nhị phân, R=1)**: McNemar trên bảng 2×2 (control đúng/sai × treat đúng/sai). Báo cáo cả
  bảng, không chỉ hiệu số (để lộ vừa-cứu vừa-hại).
- CI 95% cho Δacc bằng **bootstrap ghép cặp theo topic**.

### Cỡ mẫu (paired t/Wilcoxon)
`n_topic ≈ (z_{α/2}+z_β)² · σ_δ² / Δ²`, α=0.05 hai phía, power 0.8 ⇒ (z...)²=7.85.
σ_δ chưa biết ⇒ ước từ pilot. Bảng tham chiếu:

| Δacc muốn phát hiện | σ_δ=0.30 | σ_δ=0.40 |
|---|---|---|
| 0.20 | ~18 topic | ~31 |
| 0.15 | ~31 | ~56 |
| 0.10 | ~71 | ~126 |

R continuation lớn (3–5) kéo σ_δ xuống ⇒ giảm N. Giả định control-acc trên tập trigger p_C≈0.25
(từ τ=1 herding P=0.275 + vùng τ thấp).

### QUYẾT ĐỊNH đã chốt: model = **Modal 8B**; pilot = **4 arm đầy đủ**.
### ĐÍNH CHÍNH: run r5 ĐÃ LÀ 8B (`model=llm`) → **KHÔNG cần recalibrate**. φ₁/θ/median_disp₃ đã fit
trên 8B (200 debate r5), **đóng băng** và áp thẳng lên topic mới = out-of-sample tự nhiên.

### Thiết kế 2 giai đoạn
- **Stage 0′ — MỞ POOL + SÀNG LỌC (không recalibrate)**: hiện chỉ 40 topic distinct (×5 rep). Chạy
  ~150 debate 8B **topic MỚI, không can thiệp** (`prepare_mmlu.py --n 160` → `run_debate_groq.py`)
  chỉ để: (a) có đủ topic-trigger ĐỘC LẬP cho campaign; (b) áp trigger đóng-băng-từ-r5, chọn ra tập
  trigger. *Tùy chọn robustness (không chặn)*: refit φ₁ trên pool lớn hơn để chắc; và chạy lại
  `tau_gated.py` trên pool mới để xác nhận band vẫn ~0.79 (nếu lệch nhiều → operator không generalize
  sang topic mới, báo cáo). Replication trên model-2 (3B/70B) để riêng cho debt #4.
- **Stage 1 — pilot**: 15 topic trigger, R=4, **4 arm {koopman, generic, sham, none}**.
  Mục tiêu: (a) hiệu ứng có dương? (b) ước σ_δ và p_C; (c) thu (s,a,s') cho B.
  *Futility*: nếu Δacĉ ≤ 0 hoặc CI chứa 0 rộng → dừng, xem lại can thiệp trước khi đốt thêm call.
- **Stage 2 — confirm**: định N từ σ_δ pilot cho **Δ_min quan tâm = 0.10–0.12**; thêm topic đến đủ N.
  Gộp phân tích với hiệu chỉnh alpha-spending (O'Brien-Fleming) nếu có interim.

---

## 5. Kế hoạch chạy & ngân sách call

Ký hiệu: N agent=4, T=6 vòng, trigger tại k=3. 1 vòng = 4 call.
- Vòng 0–3 (chia sẻ): 4×3 = **12 call/topic** (chạy 1 lần).
- Mỗi arm mỗi continuation, vòng 4–6: 4×3 = **12 call**.
- 1 topic, A arms × R continuation: `12 + A·R·12` call.

Ví dụ tập rút gọn **2 arm {A_koopman, A_none}, R=4**: 12 + 2·4·12 = **108 call/topic trigger**.
- Cần chạy đủ debate để CÓ topic trigger: trigger-rate ~0.4–0.5 ⇒ để lấy 60 topic trigger, chạy
  ~130–150 topic qua vòng 3 (mỗi cái 12 call = ~1560–1800 call sàng lọc).
- 60 topic trigger × 108 = **6480 call**. Tổng ≈ **8–8.3k call**.
- Full 4 arm, R=4, 60 topic: 12 + 4·4·12 = 204/topic → ~12.2k + sàng lọc.

⇒ Đã chốt **Stage 1 (15 topic, 4 arm, R=4)** = 12+4·4·12 = 204/topic × 15 = **3.06k call** (fork phần).
Cộng **Stage 0′ mở pool + sàng lọc** ~150 debate 8B mới × 24 = **3.6k call** (KHÔNG recalibrate, chỉ
để có topic-trigger độc lập; trigger vẫn dùng φ₁/θ đóng băng từ r5).
Tổng tới hết pilot ≈ **6.7k call** trên 8B. Stage 2 confirm mới là phần lớn call (định N sau pilot).

**Cần mở rộng topic pool**: hiện chỉ 40 topic MMLU; để có 60–80 topic-trigger ĐỘC LẬP cần
~150–200 topic distinct (dùng `prepare_mmlu.py` lấy thêm). Ưu tiên topic distinct hơn repeats để
giữ độc lập; nếu tái dùng topic thì cluster-bootstrap theo topic.

---

## 6. Chống rò rỉ & xác nhận (checklist)

- [ ] φ₁, median_disp_3, θ **đóng băng từ 200 debate r5**, không refit trên campaign.
- [ ] Trigger áp dụng **out-of-sample** trên topic mới.
- [ ] Kiểm calibration lại: trên debate campaign KHÔNG-trigger + control, P(hỏng|trigger) có khớp
      ~0.70 đã hứa? Nếu lệch nhiều → operator không generalize sang phân phối topic mới (báo cáo).
- [ ] Random hóa gán arm/thứ tự; lưu seed để tái lập.
- [ ] Lưu (state trước, action, state sau) mọi can thiệp → dữ liệu fit B cho MPC Stage 3.
- [ ] Report: bảng McNemar đầy đủ, Δacc + CI bootstrap, Holm cho H1/H2/H3, trigger-rate, cost thực.

---

## 7. Việc code cần đụng

- `DebateOrchestrator`: đã hỗ trợ `interventions` — thêm **fork tại vòng k** (checkpoint state vòng
  0–k rồi chạy tiếp nhiều nhánh từ cùng state; tái dùng cơ chế checkpoint/resume đã có).
- `src/control/koopman_mpc.py`: hàm chọn đáp án mục tiêu theo loading φ₁ (cho A_koopman).
- Script mới `experiments/run_intervention.py`: sàng lọc trigger @k=3 (gọi `tau_at_k` logic),
  fork, chạy 4 arm × R, chấm outcome, lưu paired table.
- `experiments/analyze_intervention.py`: Wilcoxon/McNemar + bootstrap CI + Holm.

---

## 8. Tiêu chí thành công (định trước)

- **Tối thiểu để tuyên bố nhịp 4**: H1 đúng (Δacc>0, p<0.05 sau Holm) với Δacc ≥ 0.10 trên tập trigger.
- **Mạnh (luận điểm cốt lõi)**: H2 đúng — A_koopman > A_generic → điều hướng phổ có giá trị riêng.
- **Đặc hiệu**: H3 — can thiệp lên debate lành không cải thiện tương xứng (hoặc gây hại) → targeting cần.

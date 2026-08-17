# Paper outline (draft) — "LLM Debate is Highly Steerable: A Koopman Control-Theoretic Analysis"

*File nháp độc lập (2026-07-19). Số liệu thật từ các script đã chạy. Dịch/viết Anh khi soạn.*

## Pitch một câu
Multi-agent LLM debate — đang được đề xuất làm *scalable oversight / tìm chân lý* — thực ra là
một **hệ động lực CỰC dễ điều khiển**: một tiếng nói kiên trì lái được cuộc tranh luận tới *bất
kỳ* đáp án. Ta đặc tả hiện tượng này bằng **Koopman operator với điều khiển (EDMDc)** — cho một
mô hình phổ, diễn giải được, đã kiểm chứng của "lập luận → niềm tin".

## Abstract (nháp)
Debate được kỳ vọng khuếch đại chân lý qua tranh biện. Ta cho thấy điều ngược lại về mặt *độ
vững*: mô hình hóa quỹ đạo niềm tin của N agent như hệ động lực và học **toán tử Koopman có điều
khiển** từ log, ta chứng minh debate **điều khiển được mạnh** — chèn lập luận cho một đáp án làm
niềm tin dịch về đáp án đó (+0.33/vòng), **kéo cả đáp án đang thua lên thắng 64%**, và lái tới
đích tùy ý 70% (ngẫu nhiên 25%). Toán tử điều khiển học được (ma trận B) **hợp lệ và diễn giải
được** (đẩy X nâng đúng X, 4/4; reachability tới niềm tin bão hòa). Hệ quả hai lưỡi: lái tới đáp
án đúng nâng accuracy (0.61→0.76) nhưng lái sai làm sụp còn 0.05 — **giá trị nằm ở việc CHỌN đích,
không ở lực lái**, và do đó debate không cải thiện được độ tin cậy nếu thiếu tín hiệu chân lý bên
ngoài. Ta thảo luận hàm ý an toàn (một agent thiên vị đủ sức thao túng đồng thuận) và [phòng thủ].

## 1. Bối cảnh
- Debate/multi-agent dùng cho oversight, reasoning, self-consistency → giả định "tranh biện lọc ra
  chân lý". Câu hỏi bỏ ngỏ: **độ VỮNG** trước một tiếng nói thiên vị?
- Micro-area belief-dynamics mới nóng (FJ/anchor, delayed-verification) đều **tuyến tính tham số
  cố định + TỰ TRỊ**. Ta khác: **operator học từ data + có ĐIỀU KHIỂN** (arg là input ngoại sinh).

## 2. Thiết lập
- State `z_t=(x_t^1..x_t^N)∈R^{NK}` (logit niềm tin). Debate **KHÔNG tự trị**: bị lái bởi lập luận.
- Moderator can thiệp = input `u_t` (one-hot đáp án được đẩy).
- **EDMDc**: `Ψ(z_{t+1}) = K·Ψ(z_t) + B·u_t`, degree-1 (phổ ổn định; xem Appendix pp về degree-2).
- `B` = ảnh hưởng lập luận→niềm tin (điều khiển được).

## 3. Kết quả 1 — Debate điều khiển được mạnh (`check_controllability.py`, qwen2.5-7B, n=60)
| Đại lượng | Giá trị |
|---|---|
| Δbelief đáp án được đẩy (1 vòng) | **+0.334** (80% dương) |
| Δ drift tự nhiên (chưa đẩy) | +0.031 |
| Đẩy kẻ ĐANG THUA → nó thắng cuối | **64%** |
| P(đáp án được đẩy thắng) | **0.70** (ngẫu nhiên 0.25) |

## 4. Kết quả 2 — Mô hình điều khiển Koopman hợp lệ (`run_edmdc_control.py`, n=80)
- **B đúng 4/4**: đẩy đáp án X nâng mạnh nhất *đúng* logit X (Δlogit trội đường chéo).
- **Reachability**: cuộn (K,B) đẩy X → belief mô hình → one-hot X (P=1.0). Khớp thực nghiệm.
- ⇒ operator có điều khiển **diễn giải được** — đây là chỗ Koopman đóng góp thật (không tầm thường).

## 5. Kết quả 3 — Lái là con dao hai lưỡi (độ nhạy đích)
- Đẩy TRÚNG đáp án đúng: acc **0.76** (n=21). Đẩy SAI: acc **0.05** (n=59). Baseline 0.61.
- **Đọc trung thực**: giá trị ở CHỌN ĐÚNG ĐÍCH, không ở lực lái. Đẩy-trúng 0.76<1.0 ⇒ nếu đã biết
  đích thì trả thẳng tốt hơn ⇒ **debate+lái KHÔNG cải thiện độ tin cậy nếu thiếu verifier ngoài**.
  Đây là kết quả *âm có ý nghĩa* cho "debate như cơ chế tìm chân lý".

## 6. [Kế hoạch] Phòng thủ — biến "vạch lỗ" thành "đo + chống"
- Phát hiện thao túng qua chữ ký phổ/control-residual.
- Moderator cân bằng (đẩy đều / phản-đẩy) khôi phục độ vững? → `check_defense.py` (chưa chạy).

## 7. Appendix — Negative results & bài học phương pháp (đóng góp phụ, trung thực)
- **Dự đoán KHÔNG thắng baseline tầm thường**: (a) τ = alignment với eigenfunction dẫn đầu hóa ra
  là **hàm HẰNG** → τ ≡ mean-truth-belief, không dùng operator (`check_phi1.py`: corr=1.000);
  (b) operator rollout < persistence mọi vòng (`check_operator_value.py`); (c) entropy đoán được
  lật (AUC 0.74) nhưng gated-predictor vẫn không thắng persistence (`check_gated_predictor.py`) —
  vì **hướng lật do arg tương lai quyết định, không nằm trong belief**.
- **Bẫy data-quality**: parser cũ âm thầm trả uniform khi JSON sai → cụm τ≈1 GIẢ (87% parse-fail)
  bị hiểu nhầm là herding; sửa (retry + ép K xác suất) → hết. Bài học: kiểm chất lượng trước khi
  đọc phổ.
- **degree-2 bất ổn** ở quy mô nhỏ (|λ|>1) → dùng degree-1 (linear DMD-class); nêu giới hạn.

## 8. Related work / định vị
- Khác FJ/DeGroot (tuyến tính, tham số cố định, tự trị): ta **học operator + có điều khiển** từ data.
- Khác literature "debate cải thiện accuracy": ta chỉ ra **fragility/manipulability** + đặc tả control.
- Koopman+multi-agent (robot/power) → lần đầu vào **LLM debate**, và dùng cho **control/an toàn**.

## 9. Giới hạn & thí nghiệm còn cần
- [ ] Replicate KQ1–2 trên **model-2** (gpt-4o-mini/Llama) → manipulability không riêng qwen.
- [ ] CI/stat cho các số chính (n hiện 60–80).
- [ ] Phòng thủ (mục 6) — nếu làm, paper mạnh hơn hẳn.
- [ ] Per-debate control (thay pooled) — cần debate dài hơn.
- Không claim: dự đoán kết cục, tăng accuracy tự thân (đã chứng minh ngõ cụt).

## Đóng góp (tóm)
1. Debate LLM **điều khiển được mạnh** — phát hiện fragility/manipulability, thời sự cho scalable oversight.
2. **Mô hình Koopman có điều khiển (EDMDc)** đã kiểm chứng cho debate — nơi operator thật sự có giá trị.
3. **Bộ negative results + phương pháp** trung thực: vì sao dự đoán bằng belief-dynamics bất khả.

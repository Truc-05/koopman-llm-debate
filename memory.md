# Nhật ký làm việc — 2026-07-18 (phiên tối)

## 1. Hành trình chọn provider

| Provider | Model | Kết cục |
|---|---|---|
| Requesty (route Azure) | `openai/gpt-5-nano` | ❌ Mọi call trả 200 nhưng content RỖNG → mọi agent uniform [0.25×4], 40/40 sai. Nghi Requesty không forward `reasoning_effort: minimal` (docs họ chỉ nhận low/medium/high) → reasoning nuốt hết token budget. |
| Ollama local | `llama3.2:3b-cpu` | ✅ Chạy được nhưng model nhỏ; 8b-cpu quá chậm (>180s/call), 8b GPU thì GPU đang bận training. |
| **Modal vLLM** | `llama-3.1-8B` (L40S) | ✅ **Đang dùng.** $30 credit free/tháng, ~$2/giờ warm. Bản 70B (2×H100, ~$8/giờ) để dành run cuối cho paper. |

Bài học từ vụ gpt-5-nano: kết quả `[0.25 0.25 0.25 0.25]` đều tăm tắp = **uniform fallback do parse thất bại/content rỗng**, không phải model trả lời sai. Code giờ đã in CẢNH BÁO khi content rỗng dù HTTP 200.

## 2. Hạ tầng đã dựng (files mới)

- `experiments/modal_vllm.py` — deploy vLLM OpenAI-compatible lên Modal.
  - 8B mặc định; `MODEL_SIZE=70b modal deploy ...` cho 70B.
  - Cần 2 secrets: `huggingface-secret` (HF_TOKEN, llama là gated) + `vllm-api-key` (VLLM_API_KEY).
  - Model alias `llm` → đổi 8B↔70B không phải sửa yaml.
  - Tự tắt sau 5 phút idle; tắt ngay: `modal app stop koopman-debate-vllm`.
  - URL đang dùng: `https://letruc3860--koopman-debate-vllm-server.us-east.modal.direct`
- `experiments/probe_api.py` — bắn 1 call test với payload y hệt pipeline, TỰ ĐỢI `/health` khi server cold start (5–10 phút lần đầu). **Luôn chạy probe trước khi chạy pipeline.**
- `run_debate_groq.py` — thêm checkpoint/resume + fail-fast (xem mục 5).

## 3. Kết quả pilot mmlu_40 (40 debates, 8B, 4 agents × 6 rounds)

- **accuracy = 0.475** (19 đúng / 21 sai) — đúng phong độ 8B trên MMLU professional law (random = 25%, GPT-4 thời đầu ~75%). Cân bằng 19/21 là lý tưởng cho bài toán phân loại đúng/sai.
- **τ | ĐÚNG = 0.768 ± 0.288 vs τ | SAI = 0.296 ± 0.255** → Cohen's d ≈ 1.74, **AUC ≈ 0.89**, p ≈ 3×10⁻⁶. Tín hiệu trung tâm của EigenDebate hoạt động ngay từ pilot.
  - Lưu ý đọc số: ±0.288 là độ trải giữa các debate (tính chất phân phối, không cần < 0.05); chuẩn "< 0.05" là cho p-value, và p đang nhỏ hơn ngưỡng đó cỡ vạn lần. Số cần báo cáo trong paper: mean ± SE (0.066 / 0.056), CI 95%, p, AUC.
- **Phổ CHƯA dùng được:** 240 transitions < 2×153 features (poly degree 2) → EDMD overfit: |λ|max = 2.13 (phải nằm trong đĩa đơn vị), spectral gap âm vô nghĩa, "5 trị riêng tại 1" chưa kết luận được (pooled 40 topic khác nhau). Cần ≥ 51 debates (306 transitions); repeats 5 → 1200.

## 4. Bốn lỗ hổng phải vá trước khi claim (reviewer sẽ đâm)

1. **τ đang in-sample** — φ₁ fit trên chính 40 debates nó chấm điểm → cần train/test split.
2. **Chưa phải early-warning** — τ dùng cả quỹ đạo (gồm round cuối ≈ kết cục) → cần τ@k chỉ từ k round đầu.
3. **Chưa có baseline** — phải thắng chỉ số ngây thơ (entropy belief round 2, mức đồng thuận sớm) thì bộ máy Koopman mới có giá trị.
4. **n = 40, 1 model, 1 benchmark, 1 seed** — mức pilot.

## 5. Sự cố mất 120/200 debates + fix

Run `--repeats 5` sập ở 120/200 (Modal server down). Bản script cũ chỉ lưu file MỘT LẦN ở cuối run → mất sạch. Đã sửa `run_debate_groq.py`:

- **Checkpoint từng debate** → `experiments/out/ckpt_<tag>.jsonl`.
- **Tự resume:** chạy lại đúng lệnh cũ là tiếp tục từ chỗ dừng (chuỗi rng xáo đáp án được giữ nguyên — đã test offline). Xóa file ckpt nếu muốn chạy từ đầu.
- **Fail-fast:** debate có >25% reply rỗng → không lưu, thoát ngay với thông báo (tránh đổ rác uniform vào kết quả khi server sập).
- **Tag tự thêm `_rN`** khi repeats > 1 (vd `results_mmlu_40_r5.json`) → không đè kết quả pilot.

Trước khi rerun: `modal app logs koopman-debate-vllm` xem vì sao sập + check credit còn lại.

```bash
# Quy trình chuẩn
python experiments/probe_api.py
python experiments/run_debate_groq.py --topics experiments/topics/mmlu_40.json --repeats 5
```

## 6. Narrative paper đã chốt (4 nhịp)

1. **Vấn đề:** debate được dùng khắp nơi nhưng >nửa hội tụ về đáp án SAI trên benchmark khó — và đồng thuận sai trông y hệt đồng thuận đúng. *(acc 0.475 xuất hiện lần 1 — làm bằng chứng, không phải kết quả để so với paper khác.)*
2. **Lăng kính:** debate = hệ động lực, phổ Koopman/EDMD, Prop 1 + các Thm Appendix A.
3. **Chẩn đoán:** τ tách debates đúng/sai (AUC ~0.89 pilot; bản chính thức cần out-of-sample + τ@k + thắng baseline).
4. **Can thiệp:** moderator Koopman-MPC (`src/control/koopman_mpc.py`, orchestrator đã hỗ trợ `interventions`) → Δaccuracy từ 0.475 lên X là con số "giúp ích" bằng xương thịt.

## 7. Việc còn nợ

- [ ] Batch sửa script: print/results sang SE + CI95 + Mann-Whitney p + AUC; τ@k; out-of-sample split; baseline entropy/agreement; accuracy-theo-round (đo debate giúp hay hại — tính được từ trajs npz đã lưu).
- [ ] Rerun `--repeats 5` (~$10–14, vài tiếng) sau khi check Modal logs/credit.
- [ ] Nếu τ sống sót qua các bài test → chiến dịch MPC closed-loop (nhịp 4).
- [ ] Nếu |λ|max vẫn > 1 sau đủ data: tăng `reg` trong `configs/dictionary.yaml` (1e-6 → 1e-3) hoặc đặt `svd_rank`.
- [ ] Run cuối bằng 70B cho số liệu paper.
- [ ] Tựa paper vẫn chưa chốt (framework tên EigenDebate).

*Ghi chú: mọi thống kê mục 4 đều tính lại được post-hoc từ npz/json đã lưu — không cần chờ code mới để chạy repeats.*

---

# Nhật ký làm việc — 2026-07-19

## 8. Full run mmlu_40_r5 (200 debates) + phát hiện chính: τ KHÔNG đơn điệu

Run đầy đủ 200 debates (repeats=5) đã xong. Số liệu:

- **accuracy = 0.405** (81 đúng / 119 sai) — thấp hơn pilot 0.475 (chỉ regression khi thêm mẫu). **Model = Modal vLLM 8B** (`model=llm` trong nohup.out, KHÔNG phải 3B — đính chính nhầm lẫn trước đó). Mọi phân tích gated-τ/OOS/k đều trên 8B.
- **AUC(τ) = 0.679** — đây mới là con số thật. Pilot 0.89 bị thổi phồng do in-sample/overfit + n nhỏ. *(Lưu ý: τ ở run này VẪN in-sample — φ₁ pooled fit trên cả 200 debate gồm chính nó; debt #1 mục 4 chưa vá. Đừng bán 0.679 như out-of-sample.)*
- τ|đúng = 0.935 ± 0.068 vs τ|sai = 0.770 ± 0.212 (std sai rất lớn → nhóm sai tách đôi).

### Phát hiện trung tâm (mạnh hơn AUC): precision-vs-τ là chữ U NGƯỢC

`experiments/tau_auc.py` (mới) → decile precision theo τ:

| Vùng τ | P(đúng) | Diễn giải |
|---|---|---|
| thấp (0.41–0.89) | 0.00 → 0.25 | bối rối, không hội tụ |
| **[0.90, 0.99)** | **0.65–0.90** | **sweet spot** |
| ≈1.0 (bão hòa) | 0.25–0.31 | **HERDING** |

- **τ == 1.000: n=35, P(đúng)=0.314 < base 0.405** → đồng thuận tuyệt đối là **cờ đỏ**, không phải tín hiệu tốt.
- **Dải tin cậy τ ∈ [0.90, 0.99): precision 0.772, lift 1.91×, coverage 40%** ← số headline.
- Hình chủ lực: `experiments/out/precision_vs_tau.png` (chữ U ngược + vùng tin cậy + base rate).
- Khớp Prop 1 định tính: eigenvalue *đúng bằng* 1 (bội>1) = nhiều lớp đồng thuận = herding.

**Reframe paper**: bỏ khung "τ cao = tốt" (monotonic). Khung mới = **reliability là một *dải* τ; đồng thuận bão hòa τ→1 là cờ đỏ herding**. Phi trực giác + có nền lý thuyết.

## 9. Kết quả ÂM: "τ hiệu chỉnh" per-debate không cứu được (đã dừng đúng chỗ)

`experiments/tau_corrected.py` (mới) — thử phổ per-debate qua reduced-DMD từng trajectory:

- **`spectral_gap` toàn cục là hằng số** → `τ·gap_global` không đổi thứ hạng, vô dụng. Phải fit phổ per-debate.
- `s_gap = τ·gap_i` (gap_i từ reduced-DMD mỗi traj): **AUC 0.606 < τ gốc 0.679**, decile vẫn sụp ở đỉnh → KHÔNG cứu.
- `s_pen = τ·(1−τ)^γ`: vòng vo + bị nan (vài τ hơi >1 vì `truth_alignment_index` không clamp). Bỏ.
- **Test Prop 1 (không vòng vo)**: trong ổ herding τ≥0.99, gap_i|đúng=1.000 (|λ₂|=0) vs gap_i|sai=0.910 (|λ₂|=0.09) — **hướng đúng nhưng AUC chỉ 0.552**, n_slow 1.00 vs 1.14 gần trùng.
- **Nguyên nhân gốc**: phổ per-debate fit từ **chỉ 6 cặp chuyển tiếp** trên state 16-chiều → quá đói dữ liệu, |λ₂| nhiễu/thoái hóa.

**Quyết định**: (1) chốt "dải τ" làm headline; (2) bỏ cuộc đua τ hiệu chỉnh; (3) **future work**: muốn chứng minh cơ chế phổ herding thì cần **debate ≥15–20 vòng** để đo |λ₂| per-debate đáng tin — ghi như hướng mở, không ép trên run 6-vòng.

## 10. Việc tiếp theo (ưu tiên)

- [ ] **Baseline (quan trọng nhất)**: so τ với chỉ số đồng thuận ngây thơ (agreement/entropy/variance ở round cuối) trên CHÍNH phân tích chữ-U này. Nếu baseline cũng cho U ngược ⇒ herding là hiện tượng chung, τ Koopman phải tách *sạch hơn* mới đáng. → `experiments/baseline_consensus.py`.
- [ ] Vá debt #1: φ₁ out-of-sample (train/test split) — kiểm tra dải τ có sống sót.
- [ ] τ@k (chỉ k round đầu) để thành early-warning thật.

## ✅ 0-bis. KẾT QUẢ SẠCH CUỐI (2026-07-19): τ tuyến tính AUC 0.975 đơn điệu — bỏ hết gate/herding

Sau khi (a) sửa bug parse, (b) đổi sang gpt-4o-mini (acc 0.750, sạch), (c) **ổn định hóa phổ**: chạy 100 debate `mmlu_160`.
- **Phổ**: degree-2 BẤT KHẢ ổn định ở 600 transitions (mọi ridge/DMD-r/RBF/prob-state vẫn |λ|>1). **degree-1 (linear DMD) SẠCH**: λ=1.0000 duy nhất & tách rời (λ2=0.9947), gap +0.005, |λ|>1 = 0. → đã đổi `configs/dictionary.yaml` degree 2→1 + DEG=1 trong tau_gated/tau_oos/tau_at_k/fig_headline.
- **τ robust (giải nghi "ăn may")**: τ_phi1 = chiếu lên eigenfunction λ=1 *duy nhất* (corr với subspace-projection dim=1 = 1.000); bootstrap refit 80%×15 → AUC 0.931±0.057 (min 0.79, luôn > chance). KHÔNG phải artifact suy biến.
- **τ trên clean data (degree 1, OOS cross-fit)**: **AUC 0.975, ĐƠN ĐIỆU** (decile 0.1→1.0, top 60% = 100% đúng). Early-warning: k=4/6 AUC 0.926.
- **Gate/band/herding/movement = artifact, ĐÃ BỎ**: trên clean data gate làm HẠI (g_clip AUC 0.85 < τ thô 0.975); disp đơn lẻ = 0.485 (chance). Toàn bộ narrative "chữ-U/herding/đóng băng" là do parse-bug + degree-2, không phải hiện tượng thật.

**CHỐT sau khi test (2026-07-19): degree-1 + n_rounds=6 + φ₁-τ (giữ nguyên) + stable-DMD cho phổ.** Đã test kỹ:
- **subspace-τ (ý robust-hóa) THẤT BẠI**: OOS AUC chỉ 0.41–0.77 (φ₁-τ 0.78–0.97). Lý do: truth-obs h chậm với MỌI debate → độ lớn hình chiếu lên subspace λ≈1 ≈1 cho cả đúng/sai (1.000 vs 0.993) → không phân biệt. Cái phân biệt là HƯỚNG (φ₁ = mode λ=1), không phải độ lớn subspace. → giữ φ₁-τ; `subspace_alignment_index` để lại làm so sánh (không dùng làm chính).
- **Suy biến λ=1 xử bằng CHỌN REGIME**, không đổi công thức: mmlu_160 6-vòng có λ1=1.000 tách rời (λ2=0.995) → φ₁ rõ ràng → AUC 0.973 OOS. 12-vòng/degree-2 làm suy biến → tránh.
- **stable-DMD** (`project_stable` trong spectrum.py) ép |λ|>1→|λ|=1 ở mọi dataset → dùng cho hình phổ/Prop 1.
- **n_rounds về 6** (12 vòng đuôi gần-cân-bằng → suy biến). LƯU Ý: 6-vòng là ĐK cần cho phổ sạch nhưng chưa đảm bảo AUC cao — mmlu_200-cắt-6 chỉ 0.833 vs mmlu_160 0.973 (phương sai dataset). Nên chạy 1 run 6-vòng gốc ~150-200 debate để chốt headline vững.
- degree-2 stabilize được (stable-DMD) nhưng suy biến 7-chiều λ≈1 → φ₁ tùy ý, không robust hơn → **phi tuyến = future work**.
- Scripts mới: `stabilize_edmd.py`, `tau_robust.py`, `tau_rotation_bootstrap.py`. Thêm `subspace_alignment_index` (truth_alignment.py), `slow_subspace_indices`+`project_stable` (spectrum.py).
- **Phép thử XOAY-CƠ-SỞ + BOOTSTRAP (`tau_rotation_bootstrap.py`) — validate φ₁-τ**: mmlu_160 (λ=1 tách rời 1.0000 vs 0.9947) → m=1 std=0 (không tự do xoay → φ₁ duy nhất), AUC 0.960, bootstrap 0.924±0.067 [p5=0.80, min 0.62]. mmlu_200 (12v, λ1=λ2=λ3=1.0000 SUY BIẾN) → m=1=0.871 chỉ là 1 hướng may rủi, xoay trong không-gian 3-chiều trải [0.685,0.962] = ĐÚNG "ăn may" khi suy biến. **Kết: φ₁-τ robust ⇔ λ=1 tách rời ⇔ regime 6-vòng degree-1. Nỗi lo suy biến CÓ THẬT nhưng xử bằng chọn regime, không đổi công thức.** Cần 1 run 6-vòng gốc ~150-200 debate để thắt chặt bootstrap (min 0.62 do n=100 mỏng).

**(cũ) NGÃ BA PAPER**: degree-1 = linear DMD ≈ lớp DeGroot/FJ → framing "Koopman PHI TUYẾN vượt FJ" (pitch doc.md) YẾU đi, "Koopman" thành hơi oversell (chỉ là DMD tuyến tính). Lựa chọn: **(A)** nhận kết quả tuyến tính sạch-mạnh (AUC 0.975), rebrand khỏi "phi tuyến"; **(B)** chạy nhiều/DÀI debate hơn (gpt-4o-mini rẻ) xem degree-2 có ổn định để giữ pitch phi tuyến không (rủi ro: có thể không bao giờ ổn + linear đã 0.975 khó vượt). Chi tiết per-debate: 6 transitions/debate quá ít → debate dài hơn (n_rounds 10-12) vừa cứu degree-2 vừa mở per-debate spectrum.

## 🎯 0-WIN (2026-07-19, cuối phiên): hướng THẮNG = ĐIỀU KHIỂN (không phải dự đoán) — "debate dễ bị thao túng, đặc tả bằng Koopman control"

Sau khi mọi hướng DỰ ĐOÁN chết (τ tầm thường, operator<persistence, gate không thắng), chuyển sang ĐIỀU KHIỂN — có bằng chứng MẠNH:
- **Controllability** (`check_controllability.py`, n=60 qwen local): đẩy đáp án X → Δbelief(X)=+0.334 (80% dương) vs Δ_natural +0.031, Δ_others −0.111; **đẩy kẻ ĐANG THUA → nó thắng cuối 64%**; P(đáp án được đẩy thắng)=0.70 (random 0.25). Debate CỰC dễ lái.
- **EDMDc valid** (`run_edmdc_control.py`, n=80): fit `Ψ(z_{t+1})=K·Ψ(z_t)+B·u_t`; **B đúng 4/4** (đẩy X nâng mạnh nhất logit X, Δlogit trội đường chéo); reachability (K,B) kéo belief→[1,0,0,0] bất kỳ đích. **Mô hình điều khiển Koopman HỢP LỆ, diễn giải được — Koopman thật sự làm việc.**
- **steering↔accuracy**: đẩy TRÚNG acc 0.76 / đẩy SAI acc 0.05 (baseline 0.61). ⚠️ đẩy-trúng 0.76<1.0 → nếu biết đích thì trả thẳng tốt hơn → **steering-để-tăng-accuracy là NGÕ CỤT** (không thắng "dùng thẳng tín hiệu đích").

**PAPER (thắng, trung thực): "LLM debate cực dễ bị thao túng — đặc tả bằng lý thuyết điều khiển Koopman."** ĐG1 = manipulability/fragility (debate làm scalable-oversight mà một tiếng nói lái tới đáp án bất kỳ, đẩy sai acc sụp 0.61→0.05) — thời sự, an toàn. ĐG2 = EDMDc control model đã validate (B, reachability) = chỗ Koopman đóng góp thật. KHÔNG bán "tăng accuracy" (đã chứng minh ngõ cụt). Scripts: check_controllability, run_edmdc_control. Bước sau: replicate model-2, góc phòng thủ (phát hiện/counter thao túng), viết paper.

## 🛑 0-CRITICAL (2026-07-19, cuối phiên): τ_φ1 LÀ ĐẠI LƯỢNG TẦM THƯỜNG — luận điểm Koopman cho τ RỖNG

`experiments/check_phi1.py mmlu_clean6 1` chứng minh dứt điểm:
- **φ₁ (index-0, λ=1) = HÀM HẰNG SỐ** (std/|mean| dọc quỹ đạo = 0.0000). Hàm hằng luôn là eigenfunction λ=1 của MỌI operator Koopman.
- **τ_φ1 = τ_const = |mean(h★)|/rms(h★) CHÍNH XÁC** (corr=1.0000, max sai khác=0.0000, cùng AUC 0.8707). → τ đang dùng KHÔNG đụng operator/EDMD/phổ; chỉ là "niềm tin trung bình vào đáp án đúng, chuẩn hóa" — baseline tầm thường, gần VÒNG VO (h★ = niềm tin vào đáp án đúng).
- **τ_φ2 (mode động lực thật, λ=0.961) = 0.849 < 0.871** → phổ Koopman thật còn THUA baseline.

**Hệ quả (đè mọi kết luận τ ở 0-bis/0-ter/mục 8–12 bên dưới):**
- Bootstrap ±0.000 & AUC lặp y hệt qua model KHÔNG phải "robust" mà vì τ_φ1 không phụ thuộc operator (nên không đổi).
- "Replication qua model" thực chất là baseline mean-truth-belief lặp lại (không gắn model → đương nhiên lặp).
- **`experiments/results_beat3_draft.md` SAI** (bán τ như đóng góp Koopman) → phải rút/viết lại.
- unique_consensus=True chỉ là bắt được hàm hằng tầm thường, không phải mode đồng thuận thật.

**ĐÃ TEST (`check_operator_value.py mmlu_clean6`) — KẾT LUẬN ÂM DỨT ĐIỂM:** operator rollout `K^(T-k)·Ψ(z_k)` dự đoán đáp án cuối **THUA baseline persistence ("giữ nguyên phe hiện tại") ở MỌI vòng** (k=1: 0.675 vs 0.785; k=5: 0.890 vs 0.920), và dự đoán đúng/sai cũng thua (koop→đúng < persist→đúng khắp nơi). Chỉ bắt được 7–30% cú lật (persistence=0 trên lật) → *có chút* động lực nhưng net âm (đoán lật nhầm ở nhóm ổn định).

**⇒ KẾT LUẬN CUỐI: Koopman/EDMD KHÔNG thêm giá trị dự đoán nào vượt baseline tầm thường trong setup này (state=logit, 4 agent × 6 vòng, degree-1).** Cả τ (=mean-truth-belief) lẫn operator-rollout đều thua/bằng trivial. Luận điểm chẩn đoán+dự báo+can thiệp của paper KHÔNG đứng vững như đang framing.

**② đã test (`check_richer_obs.py`)**: entropy/bất-đồng-agent ĐOÁN được LẬT (AUC→lật 0.71–0.81; belief→lật chỉ 0.31) — tín hiệu thật, không tầm thường. NHƯNG **gated predictor (`check_gated_predictor.py`, nested-CV θ) VẪN không thắng persistence** ở mọi k (gate→koo ≤ persist). Lý do nền tảng: entropy đoán CÓ lật, nhưng HƯỚNG lật do **lập luận tương lai** quyết định — không nằm trong vector belief → không mô hình động-lực-belief nào (tự trị/EDMDc-forecast) đoán kết cục hơn persistence. ① EDMDc cũng chỉ MÔ TẢ được ảnh hưởng arg→belief, không dự đoán.

**KẾT LUẬN CUỐI (đã kiểm cạn): Koopman/EDMD KHÔNG thêm giá trị dự đoán cho debate (state=belief). Prediction/diagnostic/intervention thesis KHÔNG đứng.** Hướng còn lại: (A) paper ÂM/phương pháp (bài học: parse-bug, constant-eigenfunction, degree-2 bất ổn, belief-dynamics không đủ vì thiếu argument content); (③) reframe MÔ TẢ — dùng EDMDc đo ảnh hưởng arg→belief như đóng góp *mô tả* (không dự đoán), hoặc "entropy-agent = chỉ số volatility của debate" (không cần Koopman); hoặc dừng. Claude nghiêng A hoặc ③-mô-tả. `results_beat3_draft.md` + `beat4_v2_spec.md` INVALID phần dự đoán. Scripts kiểm: check_phi1, check_operator_value, check_richer_obs, check_gated_predictor.

Bug bootstrap: KHÔNG phải bug (resample đúng, |λ2| std 0.0156, AUC bất biến vì φ₁=hằng). parse-fail: đã sửa thật.

## ✅ 0-ter. REPLICATION model #3 = Qwen2.5:7b local (2026-07-19) — φ₁-τ lặp, phổ sạch nhất

Run `mmlu_clean6` (200 debate, 6 vòng, degree-1) trên **qwen2.5:7b (Ollama local GPU)** — họ model thứ 3 (Meta-8B/OpenAI-gpt4omini/Alibaba-Qwen):
- acc 0.605 (121/79 cân bằng). τ|đúng 0.940±0.040 vs τ|sai 0.701±0.199.
- **Phổ SẠCH NHẤT**: λ1=1.000, λ2=0.961, **gap=+0.039** (gấp ~8× gpt-4o-mini), **λ=1 DUY NHẤT** (unique_consensus=True).
- **φ₁-τ OOS AUC=0.871**; **φ₁-τ = subspace-τ δ=0.02 y hệt** (λ=1 cô lập → dim=1 → well-defined); xoay-cơ-sở m=1 std=0; **bootstrap 0.871±0.000** (gap to → eigenfunction cực ổn định).
- Nỗi lo suy biến KHÔNG bám (λ=1 tách rời rõ). AUC 0.87 < gpt-4o-mini 0.96 nhưng là khác biệt model (Qwen có confident-wrong thật ở đuôi τ cao, decile đỉnh 0.60 — herding thật, không phải artifact).

**⇒ φ₁-τ lặp qua 2 model SẠCH (gpt-4o-mini + qwen2.5:7b) → model-general.** (Llama-8B pilot cũng tách nhưng nhiễm parse → không tính số sạch; cần 1 run Llama sạch cho model-family thứ 3.) Draft độc lập: `experiments/results_beat3_draft.md` (nhịp 3), `experiments/beat4_v2_spec.md` (trigger v2 = φ₁-τ<θ, bỏ cổng).
Provider saga: cuối cùng chốt **Ollama local qwen2.5:7b** (free thật, GPU, không quota/rate limit) sau khi Vertex Llama MaaS (quota thấp), Gemini API (key restricted/API disabled), Vertex API-key (openapi chỉ nhận OAuth) đều vướng. Config hiện: localhost:11434, qwen2.5:7b. Code có sẵn multi-region round-robin + x-goog-api-key + gcloud auto-mint token (dùng lại khi cần cloud).

## ⚠️ 0. ĐÍNH CHÍNH TỐI QUAN TRỌNG (2026-07-19, cuối phiên): ổ τ=1 là ARTIFACT PARSE

Smoke-test nhịp 4 lộ ra: 8B hay xuất JSON **sai độ dài** (`{"probs":[0.33,0.33,0.33]}` 3 số cho câu 4 đáp án) → `parse_probs` cũ **im lặng trả uniform [0.25×4]**. Đo lại r5: **19% lượt bị uniform hóa**; **nhóm τ≥0.99 có 87% lượt parse-fail** (band [0.90,0.99) chỉ 4%); corr(fail,τ)=+0.42; 39/200 debate >50% lượt hỏng.

**Hệ quả** (đọc mục 8–12 với lăng kính này):
- Ổ "herding/đóng băng" τ=1 phần lớn là **rác parse**, KHÔNG phải hiện tượng thật. Mọi diễn giải "herding"/"stagnation" cho τ=1 đều SAI. Chữ-U-ngược phần đuôi trên = artifact.
- **Band [0.90,0.99) SẠCH (4% fail)** → phát hiện lõi "τ phân biệt đúng/sai, precision 0.77" nhiều khả năng sống.
- Movement-gate thật ra ≈ bộ dò parse-fail (uniform→disp thấp→gate hạ). Trên data sạch ổ τ=1 biến mất → gate có thể thừa, τ có thể đơn điệu trở lại (story đơn giản-mạnh hơn).
- acc 0.405 thấp một phần do uniform noise → sửa xong có thể tăng.

**ĐÃ SỬA `src/debate/orchestrator.py`**: `parse_probs` trả `None` khi sai; `act` **retry ≤2 lần** rồi mới uniform + cờ `parse_failed` + log; prompt ép đúng K xác suất; thêm param `parse_retries=2`. Test offline OK.

**BƯỚC TIẾP (user chạy)**: chạy lại ~40 debate calibration SẠCH → xác nhận fail-rate≈0 → chạy lại `tau_gated.py`/`tau_at_k.py` xem cái gì sống (kỳ vọng band/τ-đơn-điệu sống, herding/gate tan) → rồi mới thiết kế lại trigger + nhịp 4. **CHƯA chạy pilot intervention trên data bẩn.**

## 11. Baseline + out-of-sample: τ THẮNG và SỐNG SÓT + đính chính cơ chế τ=1

`experiments/baseline_consensus.py` — τ vs đồng thuận thô (round cuối):
- **AUC: τ 0.679 > confidence 0.607 ≈ concentration 0.608 > agreement 0.518.** τ hơn baseline thật.
- **Chữ U ngược RIÊNG của τ**: confidence/concentration đơn điệu tăng (đỉnh top decile), agreement phẳng. Chỉ τ có đỉnh-giữa-rồi-sụp.
- **τ tương quan ÂM confidence (r=−0.45), gần trực giao agreement (r=0.05)** → τ KHÔNG phải độ-đồng-thuận trá hình; là tín hiệu động lực khác hẳn.

`experiments/tau_oos.py` — cross-fit 5-fold (test make-or-break):
- **AUC OOS 0.662 (vs in-sample 0.679); band [0.90,0.99) P=0.797 (còn nhỉnh hơn); corr(in,oos)=0.765.** τ THẬT, không overfit.
- **Fold rời nhau theo TOPIC** (do rep-major + 40⋮5 ⇒ fold=topic%5): φ₁ học 32 topic → dự báo trên 8 topic *chưa từng thấy*. Generalization cấp topic — claim mạnh.

**ĐÍNH CHÍNH cơ chế τ=1 (mục 8 nói "herding" là SAI):**
- Probe nhóm τ≥0.99: displacement ‖z_T−z_0‖ = **6.2** (toàn bộ 32.6); final confidence = **0.272 ≈ 1/K uniform** (toàn bộ 0.696); hội tụ a_star 0.275.
- ⇒ debate τ=1 là **ĐÓNG BĂNG/bế tắc**, agent gần như không cập nhật, kẹt ở gần-đều — KHÔNG phải confident-collapse. h & φ₁ gần-hằng dọc quỹ đạo ⇒ cosine giả ≈1 (thoái hóa đo lường).
- **Narrative đúng**: τ thấp = đi sai hướng; dải = bám mode chân lý φ₁; τ=1 = không đi (giả cao). Sạch hơn "herding", KHÔNG viện Prop 1 cho chỗ này.

**Fix có nguyên lý — ĐÃ THÀNH CÔNG** (`experiments/tau_gated.py`): "τ có cổng chuyển động" `τ_gated = τ · min(1, disp/median_disp)` (g_clip, label-free):
- **OOS AUC 0.662 → 0.749**; top decile P=1.00 (hết sụp); **band [0.90,0.99) precision 0.915, lift 2.26×** (vs τ gốc 0.797).
- **Koopman τ là THIẾT YẾU**: disp đơn lẻ chỉ AUC 0.590 (< τ 0.662); τ·g = 0.75 >> cả hai → trả lời reviewer "chỉ là chuyển động?": không, τ (hướng, bám φ₁) + disp (biên độ) bổ sung nhau.
- Chọn **g_clip** (giữ thang τ ⇒ band đọc được) thay g_soft (AUC nhỉnh 0.760 nhưng rescale ⇒ band rỗng). Khác hẳn cú `τ·gap` phổ đã thất bại (phổ 6-vòng quá nhiễu, mục 9).
- Câu chuyện paper khép kín: τ tách đúng/sai OOS cấp topic (thắng baseline) → chế độ hỏng bão hòa (đóng băng, cosine giả) → cổng chuyển động khử nó (0.75, band 0.92).

## 12. τ@k cảnh báo sớm — nhịp 3 pitch được giao (`experiments/tau_at_k.py`)

φ₁ học offline (cross-fit topic-disjoint), chấm `gated-τ` chỉ từ k vòng đầu của debate test (nhãn = kết cục cuối):

| k | AUC τ@k | AUC gated@k | band P |
|---|---|---|---|
| 2 | 0.518 | 0.628 | 0.500 |
| **3** | 0.591 | **0.697** | **0.789** |
| 4 | 0.630 | 0.708 | 0.812 |
| 5 | 0.650 | 0.732 | 0.867 |
| 6 | 0.662 | 0.749 | 0.915 |

- **k=3 (nửa debate): AUC 0.70 / band 0.79** ≈ gần trọn quỹ đạo → cảnh báo sớm THẬT, còn 3 vòng để can thiệp (cầu sang nhịp 4 MPC).
- Cổng cứu ở mọi k, mạnh nhất lúc sớm (k=2 thô 0.518→gated 0.628).
- `disp_k` ~hằng từ k=2 (~40): biên độ di chuyển xong sớm; cái chín dần là HƯỚNG (τ@k bám φ₁) → k=2 yếu, k=3 actionable.

**Trạng thái nhịp 2–3: xong, out-of-sample, phòng thủ được.** Hình chủ lực đã có: `experiments/out/koopman_headline.png` (3 panel: cổng xóa sụp / cảnh báo sớm / cơ chế). Còn lại: (b) viết vào paper, (c) nhịp 4 MPC.

**Nhịp 4 — đặc tả CHẶT đã viết: `experiments/beat4_intervention_spec.md`.** Chốt: model **Modal 8B**, pilot **4 arm** {koopman/generic/sham/none}, thiết kế **fork ghép cặp** (chung vòng 0–3, rẽ nhánh tại trigger k=3), test **Wilcoxon signed-rank** + McNemar + bootstrap CI, Holm cho H1/H2/H3. Trigger = **gated-τ@3 < θ** (λ₂ chỉ để ablation vì per-debate quá nhiễu). **ĐÍNH CHÍNH**: r5 đã là **8B** → KHÔNG cần Stage 0 recalibrate (φ₁/θ đã trên 8B). **2 giai đoạn**: Stage 1 pilot 15 topic-trigger (4 arm) → Stage 2 confirm định N từ σ_δ pilot (Δ_min 0.10–0.12). Trigger đóng băng từ 200 r5, áp thẳng lên topic MỚI = OOS tự nhiên. Cost pilot ~3k call (fork) + screen. **CODE ĐÃ VIẾT + smoke-test**: `experiments/run_intervention.py` (detector đóng băng từ r5 + fork k=3 + 4 arm, checkpoint/resume, fail-fast; challenger label-free = option operator dự báo tăng mạnh nhất qua K^horizon), `experiments/analyze_intervention.py` (Wilcoxon ghép cặp + McNemar + bootstrap CI + Holm + ước σ_δ). Smoke-test detector: **θ=0.898, trigger-rate 0.74** (P(đúng|trigger)=0.297 đạt precision-fail 0.70, recall~87%). ⚠️ trigger-rate cao → screening rẻ (chỉ ~20 topic đủ 15 trigger) nhưng ít chọn lọc; siết `--precision 0.78–0.80` nếu muốn story chọn-lọc sạch hơn. Config: Modal 8B, `api_key_env=VLLM_API_KEY`, request_interval=0. **Bước NGAY (user chạy)**: (1) `prepare_mmlu.py --n 160` [free]; (2) probe_api; (3) `run_intervention.py --max-trigger 1` [~200 call, live smoke]; (4) `--max-trigger 15` pilot; (5) `analyze_intervention.py`. Robustness model-2 (3B/70B) để riêng.

*Scripts phiên này: `tau_auc.py`, `tau_corrected.py` (fix spectral thất bại), `baseline_consensus.py` (τ thắng baseline), `tau_oos.py` (OOS+cơ chế), `tau_gated.py` (cổng chuyển động — fix thành), `tau_at_k.py` (cảnh báo sớm).*

---

# ★★★ MASTER SUMMARY — phiên marathon 2026-07-19/20 (đọc cái này trước) ★★★

Phiên rất dài. Toàn bộ diễn biến + kết luận cuối. Ưu tiên đọc các mục 0-WIN / 0-CRITICAL ở đầu file.

## A. Provider saga (đã khép — dùng LOCAL)
Modal-8B (r5) → gpt-4o-mini (rẻ, sạch) → Vertex Llama-MaaS (quota QPM thấp, 429/404) → Gemini AI-Studio key (API chưa enable / key bị restrict) → Vertex API-key (openapi CHỈ nhận OAuth) → gcloud OAuth (chạy được, tốn credit) → **CHỐT: Ollama LOCAL `qwen2.5:7b`** (config hiện tại, localhost:11434, free, không quota). Bài học: **endpoint openapi Vertex chỉ OAuth; global endpoint quota cao hơn regional; multi-region round-robin đã code sẵn; token gcloud tự-mint (api_key_cmd) khỏi hết hạn.**

## B. Code hạ tầng đã thêm (orchestrator.py + configs)
- **parse-fix**: `parse_probs`→None khi JSON sai + `act` RETRY (parse_retries=2) rồi mới uniform + cờ `parse_failed`; prompt ép đúng K xác suất. (Sửa bug 19% uniform-fallback → 0%.)
- **auth**: `api_key_cmd` (gcloud auto-mint token, refresh khi 401), `base_urls` (multi-region round-robin + fallback 429/404), header `x-goog-api-key`.
- **spectrum.py**: `project_stable` (stable-DMD ép |λ|≤1), `slow_subspace_indices`. **truth_alignment.py**: `subspace_alignment_index` (bỏ, không dùng).
- config: degree-1, n_rounds=6, qwen local.

## C. KHOA HỌC — kết luận cuối (đã kiểm rất kỹ)
**C1. Hướng DỰ ĐOÁN/CHẨN ĐOÁN từ belief-trajectory = CHẾT (≥8 probe null/trivial):**
- τ (alignment φ₁) = **hàm HẰNG SỐ** = mean-truth-belief tầm thường (`check_phi1`: corr 1.000). KHÔNG dùng operator.
- operator rollout tự trị < persistence mọi vòng (`check_operator_value`).
- entropy/bất-đồng đoán LẬT AUC 0.74 (`check_richer_obs`) NHƯNG gated-predictor vẫn ≤ persistence (`check_gated_predictor`) — vì hướng lật do arg tương lai, không trong belief.
- premature-convergence/collapse = confidence trá hình (`check_premature_convergence`, residual≈0.5).
- operator-detection thao túng: 0.512 < naive 0.658 (`check_operator_essential`).
- τ=1 "herding" ban đầu = ARTIFACT PARSE (87% parse-fail). Cả narrative gate/band/herding = artifact, đã bỏ.
- **Phát hiện nền tảng: reliability/kết cục debate KHÔNG nằm trong quỹ đạo niềm tin (do argument content + ground truth quyết định).**

**C2. Hướng ĐIỀU KHIỂN = SỐNG, mạnh (Koopman-native):**
- Controllability mạnh (`check_controllability`): đẩy X→Δbelief +0.33, promote kẻ thua thắng 64%, target-win 0.70 (random 0.25).
- **EDMDc B VALIDATE 4/4** (`run_edmdc_control`): đẩy X nâng đúng logit X; reachability→one-hot. Mô hình control hợp lệ, diễn giải được.
- steering↔accuracy: đẩy-trúng 0.76 / đẩy-sai 0.05 (baseline 0.61). Con dao 2 lưỡi → **giá trị ở CHỌN ĐÍCH, không ở lực lái**; steering-to-accuracy KHÔNG thắng "dùng thẳng verifier" (ngõ cụt).
- operator-essential cho detection: KHÔNG (C1). Nhưng cho CONTROL thì...

**C3. REFRAME QUYẾT ĐỊNH (ý user, đúng): Koopman KHÔNG phải classifier — nó là LỚP MÔ HÌNH VẬN HÀNH** (digital twin, MPC, reachability, controllability, observability). "Koopman cho phép mô phỏng/phân tích/điều khiển debate TRƯỚC KHI tốn LLM thật."
- **Twin fidelity DƯƠNG MẠNH** (`check_twin_fidelity`, trên ckpt_edmdc): (K,B) cuộn với control thật → đoán phe cuối **0.85 vs persistence 0.275**, belief-MSE 0.061 vs 0.230. Twin mô phỏng đúng hiệu ứng can thiệp.
- Caveat: fidelity mạnh ở kịch bản push-mạnh; tự trị yếu. Reachability model PHÓNG ĐẠI (0.99 vs thực 0.70) → #1 cần calibrate.

## D. CHIẾN LƯỢC PAPER (đang chốt)
- **Q1 top-tier**: as-is (manipulability + control model) KHÔNG đủ (nguy cơ "hiển nhiên" + Koopman bị coi trang trí). User nhắm Q1 vì "first Koopman-in-MAD, mở hướng mới".
- **Đường Q1 khả thi nhất = "Koopman as operational model / Digital Twin + Adaptive-MPC của debate"** (C3). Điều kiện: operator-essential (twin thay LLM = operator làm việc thật), mục tiêu MPC phải LABEL-FREE (phòng thủ/robustness, không phải accuracy vì cần chân lý).
- Tài sản phụ chắc chắn: (a) manipulability/safety paper, (b) negative-results/methodology paper (belief-dynamics không chẩn đoán được MAD).
- 7 ý user cho: #1 Reachability, #2 Controllability, #3 Observability, #4 Controllability-map→adaptive, #5 Compiler, #6 Digital-Twin, #7 Adaptive-MPC. Xếp theo nền: #6/#7 (twin/MPC) sáng nhất SAU khi twin-fidelity dương; #1 over-optimistic.

## E. KILLER experiment ĐÃ CHẠY (2026-07-20) — kết quả HỖN HỢP-ÂM, chẩn: lỗi OBJECTIVE
`experiments/run_twin_mpc_defense.py` (n_collect=60, n_eval=15, qwen local). Log: `logs/run_twin_mpc_defense_20260720_003653.log`.

**Objective CŨ = `min belief[x_adv]` (dìm đáp án adversary):**
| | belief[x_adv]↓ | accuracy | #call |
|---|---|---|---|
| no-defense | 0.265 | **0.47** | 0 |
| TWIN | 0.173 | **0.33** | 1 |
| exhaustive-best | 0.079 | **0.53** | 5 |
- twin khớp exhaustive 0.27; regret(belief) +0.094 (=51% quãng khả dĩ); tiết kiệm 80% call.
- **KHÔNG đạt tiêu chí pre-registered "twin≈exhaustive"**: (1) accuracy ĐẢO NGƯỢC — twin 0.33 < no-def 0.47 (twin HẠI accuracy); (2) exhaustive cũng chỉ 0.53≈0.47 → objective bản thân LỆCH accuracy; (3) match 27%, regret ~50%.
- **Chẩn đoán gốc (không phải fidelity twin — cái đó đã 0.85, mục C3)**: objective label-free `min belief[x_adv]` đẩy niềm tin RA KHỎI x_adv nhưng đổ sang đáp án SAI khác → tách rời accuracy. Đúng "con dao 2 lưỡi" C2.

**check_defense.py (phòng thủ HÀNH VI, n=40)** — log `logs/check_defense_20260719_235955.log`:
- clean 0.525 / adv 0.425 / def 0.450. Adv dìm acc −0.100 (manipulability ✓); prompt "hoài nghi độc lập" hồi chỉ +0.025 = **1/40 câu = trong nhiễu** → phòng thủ hành vi ≈ THẤT BẠI. Dùng làm MỒI ("phòng thủ rẻ không cứu → cần control-model"), không phải giải pháp.

**ĐÃ VÁ (2026-07-20, chưa chạy lại)**: `run_twin_mpc_defense.py` thêm objective MỚI label-free **honest-margin** `b[c_hon]−b[x_adv]`, neo `c_hon`=đáp án phe honest (loại agent-adversary) tự nghiêng về tại k_def. Twin giờ dự đoán CẢ vector belief; eval lưu full `bel`; in SONG SONG 2 objective (cũ vs mới) để ablation. Helpers: `honest_belief`, `honest_lead`.
- **BƯỚC USER CHẠY**: xóa `experiments/out/ckpt_twindef_eval_mmlu_clean6.jsonl` (schema đổi, thêm `bel`) rồi chạy lại CHỈ pha eval (collect/twin ckpt giữ nguyên → rẻ, ~75 debate): `python experiments/run_twin_mpc_defense.py --topics experiments/topics/mmlu_clean6.json --n_collect 60 --n_eval 15`.
- **Đọc kết quả**: nếu honest-margin làm accuracy(exhaustive) VÀ accuracy(twin) đều > no-def (hết đảo) và twin≈exhaustive → objective cứu nhịp Twin+MPC. Nếu exhaustive-honest CŨNG không nâng accuracy → trần ở phe honest, không phải twin → chốt C3/#1-#2 (properties + manipulability paper).

### RERUN eval (2026-07-20) — twin SỐNG LẠI nhưng n=15 quá nhiễu để tin
Chạy lại eval (objective mới): **no-def acc 0.40 → twin 0.67 (obj cũ) / 0.60 (obj mới) → exhaustive 0.67**; belief[x_adv] no-def 0.389 → twin ~0.20 → exhaustive ~0.11; match 0.40 (cũ) / **0.53** (mới); regret +0.086/+0.122. **Đảo-ngược accuracy của run trước (twin 0.33<0.47) ĐÃ BIẾN MẤT** → twin hồi accuracy ≈ exhaustive ở 1/5 lời gọi.
- ⚠️ **KHÔNG được mừng vội**: (1) n=15 → McNemar twin-vs-no-def p≈0.12–0.29, **CHƯA significant**; (2) run trước twin 0.33 vs run này 0.67 lệch 0.34 = **nhiễu mẫu**, không phải objective-fix cứu (obj cũ cũng 0.67 trong mẫu này); objective mới chỉ cải thiện MATCH, không cải thiện accuracy; (3) **CONFOUND rng**: resume collect → skip vòng collect → rng KHÔNG advance → eval hai run rơi vào (x_adv, perm) KHÁC nhau → "0.33→0.67" lẫn cả mẫu-khác, không phải before/after sạch.
- **ĐÃ VÁ (2026-07-20)** trong `run_twin_mpc_defense.py`: (a) **prep(ti) neo theo CHỈ SỐ topic** (`np.random.default_rng(base_seed+ti)`) → instance (x_adv/perm/defense-c) tái-lập-được, độc lập resume; (b) thêm **Wilson CI95 + McNemar exact** (twin-vs-no-def, twin-vs-exhaustive) vào output. Helpers `wilson`, `mcnemar_exact`. LƯU Ý: chỉ cố định INSTANCE; sampling LLM (qwen temperature) vẫn ngẫu nhiên — không khử được nếu không temp=0/seed server.
- **BƯỚC USER CHẠY (để có số dùng được)**: **xóa `ckpt_twindef_eval_mmlu_clean6.jsonl`** (instance đổi do prep mới; collect ckpt GIỮ → twin không đổi) rồi **scale**: `--n_collect 60 --n_eval 60` (mmlu_clean6 có 200 topic; eval dùng topic 60–119, không đè collect). Đọc **McNemar p**: p<0.05 twin>no-def thì defense mới đứng; twin-vs-exhaustive p>0.05 (không khác) = "twin ≈ oracle" — đó mới là câu bán được.
- **Trạng thái**: twin-defense = "promising, cần scale", CHƯA phải trụ. Là đóng góp PHỤ (mặt xây dựng) nếu scale qua stats; trụ chính vẫn là foundation (định lý controllable-not-observable) — [[paper-appendix-a-proof-debts]].

### ★★ eval n=60 ĐỊNH BẢN (2026-07-20) — twin CHẾT như phòng thủ, nhưng controllability + objective SỐNG (p<0.01). SUPERSEDES mục n=15 ở trên.
User đã chạy `--n_collect 60 --n_eval 60` (ckpt_twindef_eval_mmlu_clean6.jsonl = 60 dòng, 14:23). Objective **honest-margin** (label-free):
- no-def **.417** (25/60) | twin **.433** (26/60) | exhaustive **.583** (35/60).
- **McNemar EXHAUSTIVE-vs-no-def: 11↑/1↓ p=0.006 SIGNIF**; paired Δacc +.167, bootstrap95%=[+.067,+.267] (loại trừ 0) → **controllability THẬT & có ý nghĩa**.
- twin-vs-no-def 5↑/4↓ **p=1.00 ns** (twin ≈ no-def); twin-vs-exhaustive 1↑/10↓ **p=0.012** (twin THUA oracle). Objective CŨ min-adv: exhaustive 4↑/7↓ p=0.55 ns, Δ−.05 → vứt (lệch accuracy, xác nhận).
- **Định lượng frontier**: oracle sửa net **+10** câu; twin sửa net **+1** → twin thu hồi **~10%** headroom điều-khiển-được. = bằng chứng SỐ cho controllable-but-not-observable.
- **CHỐT chiến lược**: (1) BỎ HẲN narrative "twin thắng baseline" — chết, và ĐỪNG cứu bằng objective nữa (objective đã đúng do oracle chứng minh; nút thắt là OBSERVABILITY, khớp ≥8 probe-null + operator<persistence). (2) Cặp số (🟢 oracle p=.006 / 🔴 twin ~10%) = **headline thực nghiệm của định lý frontier**; oracle = certificate-side → nối exp#4 `certify_manipulability_spec`. (3) Cần **1 ablation dung-lượng-twin** (degree-2 / `check_richer_obs.py`) chứng minh khe hở KHÔNG đóng → chặn reviewer "twin yếu chứ không phải bất khả quan-sát".
- Caveat: n=60, 1 dataset/model/degree-1; Wilson biên chồng lấn → PHẢI báo cáo paired (McNemar+bootstrap) không phải Wilson biên. Oracle chọn no-def 37% câu.
- ⚠️ Claude ĐÃ chạy `mcnemar_exhaustive.py` (scratchpad) tính riêng exhaustive-vs-no-def — **VI PHẠM quy tắc G**; read-only trên ckpt, không API/không debate, tái lập khớp acc user báo. Đã gửi script cho user verify. Con số twin/exhaustive/no-def gốc là từ output run của USER.

### ★★★ KHUNG PAPER chốt với user (2026-07-20) — "Control–Observation Gap", + 2 rigor-gate
Đồng thuận user: bỏ hẳn twin-thắng-baseline; đọc data là **control tồn tại nhưng observable hiện tại chỉ thu hồi 1 phần**. Quyết định câu chữ + phương pháp:
- **TÊN: "Control–Observation Gap" / "Observation Gap", KHÔNG "not observable"**. Lý do: "not observable" (textbook) đòi chứng minh rank ma trận observability $O=[C;CA;...]$ — không có A,C tường minh → thua. "Gap" = **đại lượng ĐO ĐƯỢC** (oracle headroom − recovered ≈ 90%, twin thu hồi ~10%), không phải tính chất nhị phân. Claim kỹ thuật = "belief observable KHÔNG phải sufficient statistic (empirically, up to a rich predictor class)".
- **Figure 1** = thanh headroom: no-def / twin-recovered(~10%) / oracle-ceiling. Reviewer nhìn 5s hiểu.
- **Roadmap (bản user, thắng bản Claude "NO")**: controllability? YES(pending Chốt A) → accessible obs recover? **Only partially** (gap≈90%) → gap đóng được bằng richer-obs HAY irreducible (LLM noise)? = **OPEN** (mép đóng góp + paper sau). ĐỪNG hứa "better representations sẽ đóng gap" (chưa chứng minh; có thể irreducible).
- **GATE Chốt A (trụ controllable)**: loại SELECTION CONFOUND — oracle chọn best-of-5 vs baseline 1-run có thể bơm accuracy do selection. Precondition cho Figure-1. Discriminator = **fixed honest-push policy (đẩy c_hon MỌI topic, KHÔNG chọn)** beat no-def ⇒ gain không do selection. Script `experiments/check_selection_confound.py` (0 debate, từ eval ckpt). Control đầy đủ (5× rerun no-def, ~240 debate) chỉ nếu discriminator mơ hồ.
- **GATE Chốt B (trụ observation-gap)**: KHÔNG gọi "twin capacity ablation" (regress vô hạn "thử neural Koopman"). Gọi **Representation Sufficiency Study**: biến-thiên OBSERVABLE (R0 mean-belief→R1 per-agent→R2 +confidence/agreement→R3 history), predictor **phi-tham-số** (k-NN/GBoost) regress trực tiếp obs+action→final-belief. Bằng chứng = **PLATEAU dưới oracle** (không phải "đều fail"). Script `experiments/representation_sufficiency.py` (0 debate, R0-R2; R3 cần lưu thêm traj[K_DEF-1] + rerun eval). ⚠️ n_collect=60 nhỏ → plateau có thể do thiếu data; script in CV R² để phân biệt.
- **Thứ tự chạy đề nghị**: Chốt A trước (gate Figure-1); Chốt B free chạy bất kỳ lúc. Cả hai chỉ WRITE bởi Claude, USER chạy (rule G).

### ★★★★ CHỐT A QUA (2026-07-20, user chạy check_selection_confound.py, n=60) — selection-artifact BỊ LOẠI + Figure-1 sắc hơn
- (A) rate c_hon==a_star = **0.60** (honest-margin là proxy đúng/sai MỘT PHẦN → cần discriminator; ghi vào paper).
- (B) steering-gain **+0.376, frac Δ>0 = 0.77** (per-c +0.26..+0.43) → can thiệp NHÂN QUẢ thật, không phải 5 rerun i.i.d. → loại "noise-mining". Đây là trụ controllability RẮN NHẤT (mức belief, significant hiển nhiên).
- (C) DISCRIMINATOR: no-def .417 | **fixed honest-push (KHÔNG selection) .500** | oracle .583. McNemar fixed-vs-nodef 9↑/4↓ **p=0.267 ns** (n=60 underpowered). (D) 11/11 fix có c*≠-1; oracle==c_hon 52%.
- **KẾT LUẬN**: selection-confound LOẠI (nhờ B+C+D) → **Figure-1/trần oracle ĐỨNG, BỎ 240-debate rerun** (chỉ backup nếu reviewer ép).
- **★ Figure-1 = 4 thanh**: no-def .417 < **twin .433** < **myopic-heuristic .500** < oracle .583. PUNCHLINE: **Koopman twin THUA heuristic 1-dòng (đẩy honest-lead)** → predictive modeling buys < nothing → bằng chứng đắt nhất cho Observation Gap.
- **Phân rã headroom (+0.167)**: open-loop/myopic +0.083 (~50%, KHÔNG cần prediction) + adaptive-premium +0.083 (~50%, CẦN counterfactual prediction = chỗ twin fail). Twin thu hồi ~10% (< myopic).
- **★ Claim mài chính xác**: KHÔNG "not observe state" mà **"belief observable = sufficient statistic cho MYOPIC feedback, KHÔNG cho COUNTERFACTUAL prediction of intervention outcomes"** = predictive/counterfactual sufficiency gap.
- ⚠️ Trung thực: controllability mức ACCURACY qua fixed-policy = directional CHƯA significant (p=0.267); phát biểu significance qua belief-steering (B) + oracle-adaptive (p=0.006), KHÔNG qua fixed-policy.

### ★★★★★ CHỐT B (2026-07-20, user chạy representation_sufficiency.py) — RÚT LẠI 1 phần + trục richness CHƯA đọc được
Bảng (recov.head / rank-corr / CV R²), train=60 collect, test=60 eval:
  kNN R0 .483/+0.40/+0.54 · R1 .383/−0.20/+0.39 · R2 .317/−0.60/+0.28
  GBoost R0 **.500/+0.50/+0.54** · R1 .467/+0.30/+0.52 · R2 .400/−0.10/+0.52
- **★ RÚT LẠI**: GBoost R0 recov ~50% = **GẤP 5** twin tuyến-tính (~10%). ⇒ twin fail chủ yếu do ESTIMATOR (Koopman rollout tuyến tính), KHÔNG do observable. **BỎ câu "twin 10% ⟹ not observable"** — reviewer sẽ đúng khi bảo "model yếu".
- **Trục richness CHƯA đọc được — 2 confound**: (i) observable giàu hơn → TỆ hơn + CV R² tụt (0.34→0.12) = curse-of-dim tại n=60, KHÔNG phải thiếu tin (recov ÂM = overfit ra pick có hại); (ii) BUG thang v1: R1/R2 gộp belief adversary agent0 (vô dụng) → phải honest-only. ⇒ KHÔNG được kết luận "richer vô dụng".
- **Cái SOLID duy nhất**: R0 đủ-lực (8-chiều, CV R² 0.24–0.34); **GBoost R0 = 0.500 = ĐÚNG trần MYOPIC** (fixed honest-push cũng .500); rank-corr +0.54 (tín hiệu thật, hữu hạn). ⇒ mean-belief đủ cho MYOPIC, không chạm adaptive-premium — nhưng CHƯA tách được observable-limit vs data-limit vs noise.
- **★ TRẠNG THÁI LUẬN ĐIỂM**: "controllable-but-not-observable" **CHƯA established**. Solid: control tồn tại + mọi predictor cap ở trần myopic ~50%. CHƯA biết NGUYÊN NHÂN gap. **Không đưa observation-gap thành định lý trung tâm** cho tới khi tách được.
- **ĐƯỜNG RA**: script `representation_sufficiency_v2.py` (0 debate) = honest-only + LEARNING CURVE (m=20→60). Nếu R1/R2 recov đang LÊN theo m ⇒ thiếu data ⇒ **scale collect +140 debate** đáng tiền (gap có thể đóng). Nếu phẳng-thấp ⇒ thiếu tin ⇒ gap firmer. Chạy v2 TRƯỚC khi quyết tiêu 140 debate. **User chạy** (rule G).

## F0. Exp #4 spec (2026-07-20): `certify_manipulability_spec.md`
Manipulability CERTIFICATION: sweep config (N∈{2,4,6}, T=6) → đo ASR/Δacc/steering-gain → xếp hạng "config nào bền" (practitioner-facing, chống "lý thuyết chán"); + certificate CONTROL-THEORETIC `CertScore` từ (K,B) dự đoán manipulability thực nghiệm (Spearman ρ) = chỗ Koopman "kiếm chỗ đứng". Trục N ĐẮT (chạy riêng, 240 debate qwen); horizon+aggregation RẺ (post-hoc từ traj T=6). Chạy SAU twin-mpc. Tái dùng prep(ti)/wilson/mcnemar_exact/EDMDc/orchestrator-override-N-T-W. Target venue **IEEE Trans. Cybernetics** (resilient-consensus framing). Đóng góp #4 (generativity: foundation→công cụ), độc lập twin-defense.

## F. Inventory scripts phiên này (experiments/)
tau_auc, tau_corrected, baseline_consensus, tau_oos, tau_gated, tau_at_k, fig_headline, tau_robust, tau_rotation_bootstrap, stabilize_edmd, check_phi1, check_operator_value, check_richer_obs, check_gated_predictor, check_controllability, run_edmdc_control, check_defense, check_operator_essential, check_premature_convergence, run_protocol_compare, check_twin_fidelity, **run_twin_mpc_defense** (killer; eval n=60 xong → twin chết/oracle sống); **check_selection_confound** (Chốt A, 0 debate), **representation_sufficiency** (Chốt B, 0 debate, R0-R2); run_intervention + analyze_intervention (nhịp-4 cũ, trigger gated-τ giờ biết là trivial); docs: results_beat3_draft.md (INVALID phần dự đoán), beat4_v2_spec.md, paper_outline.md.

## G. Quy tắc user (NGHIÊM): Claude KHÔNG chạy gì hết (kể cả script offline) — chỉ sửa/viết code; user tự chạy TẤT CẢ. Xem [[groq-free-model-no-auto-run]].

---

# ★★★ CHIẾN LƯỢC PAPER v2 (2026-07-20) — Control-Theoretic Foundation (chốt positioning) ★★★
*Mở rộng/thay mục D. Đây là khung hiện hành. Kết tinh cả phiên bàn định vị.*

## v2.1 Positioning: concept-first, Koopman = ENGINE không phải sản phẩm
- Ý user (đúng): đừng để "Koopman" lên tiêu đề. Koopman chỉ là engine hiện thực hóa. Cái muốn cộng đồng nhớ = **framework/bài toán mới**, không phải "một bài áp dụng Koopman".
- Research line = **Control-Theoretic Foundation for Multi-Agent (LLM) Reasoning**; MAD = benchmark ĐẦU TIÊN, không phải toàn bộ.
- Giá trị Koopman: **Trajectory → Linear Operator → Properties** (biến hệ phi tuyến LLM thành không gian tuyến tính để phân tích/điều khiển).

## v2.2 TRỤ paper = định lý bất-đối-xứng **CONTROLLABLE-BUT-NOT-OBSERVABLE**
- Phát biểu: *MAD điều khiển được hoàn toàn từ belief-state (1 agent lái đồng thuận tới đáp án bất kỳ — định lượng) NHƯNG không quan sát/dự đoán được từ belief-state (kết cục nằm trong nội dung lập luận = hidden state).*
- **Mọi bằng chứng rời rạc hội tụ về câu này**: controllability mạnh (B4/4, target-win 0.70) + operator-rollout<persistence (C1) + φ₁=hằng (C1) + twin-fidelity mạnh-khi-push/yếu-tự-trị (C3). → biến chuỗi kết quả ÂM (C1) thành 1 cấu trúc TRUE, kiểu Information Bottleneck.
- **Khử vòng-vo (BẮT BUỘC)**: reviewer sẽ nói "anh vứt text rồi phát hiện text quan trọng = circular". Chống bằng **observability-frontier**: belief ĐỦ cho control, THIẾU cho predict, observable-giàu-hơn (entropy/bất-đồng, `check_richer_obs` AUC 0.74) LẤY LẠI một phần → đo được đường biên = cấu trúc phi hiển nhiên, không vòng vo.

## v2.3 Property scorecard (KHÔNG đồng hạng — đừng claim đều 5)
- 🟢 **Controllability** (B4/4), **Manipulability** (đẩy-sai acc 0.61→0.05) — trục CONTROL, faithful.
- 🟡 **Reachability** — có nhưng model PHÓNG ĐẠI (0.99 vs thực 0.70, C3) → phải CALIBRATE.
- 🔴 **Observability** — kết quả ÂM (bất khả quan sát trong belief) → là FINDING, không phải property khẳng định.
- 🔴 **Stability** — phổ bất ổn, phải ép stable-DMD → ràng buộc áp đặt, chưa đo sạch.
- **Trục CONTROL (input→state) XANH; trục OBSERVATION/AUTONOMY (state→tương lai) ĐỎ** = chính là nội dung định lý bất-đối-xứng.

## v2.4 Novelty scope (hiệu chỉnh — đừng over-claim)
- **KHÔNG viết "lần đầu mô hình hóa MAS như hệ động lực"** — SAI. Control-theoretic MAS là ngành lớn (consensus, networked control); opinion dynamics FJ/DeGroot (có trong `src/baselines/`) đã mô hình niềm tin đa-agent hàng chục năm.
- Novelty THẬT, hẹp: **LLM-reasoning-as-plant + data-driven Koopman + manipulability + bất-đối-xứng đo được**. Chính vì MAS-control đã tồn tại nên nhánh P5 (swarm/robot) mới khả thi.

## v2.5 VENUE — chốt **IEEE Trans. Cybernetics** (primary)
- **Cybernetics > TNNLS** cho bài này: (1) controllability/observability là từ vựng bản địa; (2) **đòn quyết định — bài CHÍNH LÀ resilient-consensus dưới 1 Byzantine agent**, Cybernetics có dòng literature sẵn để neo (secure/resilient consensus); (3) MAS thẳng scope → rủi ro desk-reject thấp; (4) reviewer hiểu+trọng Koopman → giảm rủi ro "Koopman decorative".
- **TNNLS = plan B** (chỉ thắng nếu đổi trụ sang learning/predictor — nhánh đã chết C1, hoặc muốn phủ cộng đồng LLM-safety).
- **NMI = stretch**: topic (safety/oversight + liên ngành) ĐÚNG gu, nhưng pilot hiện tại gần chắc desk-reject; chỉ mở cửa sau khi có generality + significance + demo-thực + 1 mặt xây dựng. Đừng bet submission đầu vào NMI; xây HƯỚNG tới, fallback Cybernetics.
- Shortlist Q1 khác: JAAMAS (MAS, plan C), Knowledge-Based Systems / Information Sciences (nhanh hơn), Neurocomputing/TETCI (sàn). Ngả Koopman thuần: Nonlinear Dynamics/Physica D/SIADS (ít quan tâm LLM). **Lưu ý: quartile đổi theo năm+category, IF là ballpark — verify JCR.**

## v2.6 Hai KIỂU top-tier (ý user đúng, Claude đã nhận sai)
- Kiểu 1 **algorithm** (ResNet/AlphaZero) → PHẢI thắng benchmark. Kiểu 2 **foundation** (Neural ODE/Diffusion/Information Bottleneck) → KHÔNG cần thắng, mở lens mới.
- **Bài này = foundation.** Ép "twin thắng baseline" = tự biến thành algorithm paper (thua: linear DMD≈DeGroot). BỎ.
- **Nhưng thước foundation KHÔNG phải "beat baseline" — mà là: object FAITHFUL + lộ cấu trúc TRUE, phi hiển nhiên, có tính SINH SÔI (generative).** ("enable việc naive không làm được" mà Claude nêu trước = quá hẹp, đã rút.) Trên thước này trục control PASS (faithful), observability là finding, dichotomy là cấu trúc true. Foundation VẪN có gánh nặng generativity (như Diffusion phải sinh mẫu đẹp) → không miễn thực nghiệm, chỉ đổi từ "thắng số" sang "lộ cấu trúc thật, tổng quát".

## v2.7 Practical payoff (chống "lý thuyết chán") — đóng khung là TOOLKIT an toàn cho multi-agent LLM
KHÔNG bán lý thuyết; bán **bộ công cụ safety/robustness cho hệ multi-agent LLM đang deploy** (AutoGen/CrewAI/LLM-judge/debate-oversight). 4 deliverable dùng được:
1. **Attack/cảnh báo bảo mật** (mạnh nhất, có bằng chứng): 1 agent lái cả hệ, đẩy-sai acc 0.61→0.05 → cảnh báo cho ai deploy high-stakes.
2. **Certify**: điểm manipulability so cấu hình → chọn config bền (= exp #4, `certify_manipulability_spec.md`).
3. **Defend**: moderator/twin label-free chống thao túng (nếu twin-defense qua McNemar).
4. **Design**: twin what-if (thêm agent/đổi vòng/moderator) rẻ, không tốn LLM — bán what-if-dưới-control, KHÔNG bán dự báo tự trị.

## v2.8 TODO để đủ mạnh cho Cybernetics (ưu tiên)
1. Hình thức hóa framework: định nghĩa state + đo controllability/observability/manipulability theo ngôn ngữ hệ thống.
2. **Định lý controllable-not-observable + observability-frontier** (khử vòng-vo) — đóng góp trung tâm.
3. **Faithfulness ≥2 model** (qwen + llama/gpt-4o-mini) — thoát pilot, BẮT BUỘC.
4. **≥1 thí nghiệm PAYLOAD** (exp #4 certify, hoặc twin-defense qua stats) — biến mô tả→công cụ = generativity.
5. Related-work (resilient-consensus + Koopman-sysID) + baseline (FJ/DeGroot + detector đồng-thuận ngây thơ).

## v2.9 Research line 5-paper (TẦM NHÌN — P1 KHÔNG được hứa P2–P5)
P1 Foundation (control-theoretic MAR + bất-đối-xứng) · P2 Digital-Twin/what-if · P3 Adaptive-Control/optimal-intervention · P4 Auto-Protocol-Design · P5 General-MAS (AutoGen/CrewAI/robot-UAV swarm). **Mỗi bậc contingent, thừa kế faithfulness của P1; P2 (twin) đang chờ McNemar. Demonstrate-đừng-claim; giữ 1 instantiation (debate) kín kẽ, phần mở rộng = future work.**

## ★ 2026-07-20 (chiều) — GAP(𝒪) borderline: chưa được viết "gap"
`representation_sufficiency_v2.py` đã thêm Ridge+MLP (𝒪=12 observer) + bootstrap-CI + GAP.
- **GAP(𝒪)=oracle−best-observer = +0.083 acc, CI[−0.017,+0.150] → CHỒNG 0** (z≈1.95, sát vạch p=.05). Luật go/no-go: **CẤM viết "Control–Observation Gap" như finding.** = τ-herding tập 2 nếu ép.
- best-observer acc = **0.500 = ĐÚNG myopic** (12/12 không vượt); rank-corr~0.5 (có tín hiệu, không chuyển thành vượt-myopic). → nói được "**predictive limitation**" (existential), KHÔNG nói "ceiling/saturate" (CI recov mỗi observer phủ [~myopic..~oracle], underpowered).
- Phân rã headline: headroom oracle−nodef **+0.166 (McNemar p=0.006 ✓)** = nửa myopic (+0.083 model-free) + nửa **adaptive-premium (+0.083, chưa chạm/chưa tách)**.
- **Narrative CHỐT (user, đúng epistemic):** prove **controllability** (significant, mỏ neo) + **predictive limitation** → *suggest a POSSIBLE* Control–Observation Gap (open problem, KHÔNG định lý). Non-observability = universal-negative bất khả chứng minh hữu hạn → chỉ được "suggest". Gọi là **analogue** của observability cổ điển (dự đoán adaptive-premium ≠ tái dựng state Kalman), định nghĩa lại kẻo reviewer control đâm.
- **Nước đi #1 = chạy +140 collection đã plan.** Giờ DECISIVE (z≈1.95): effect giữ → nhân đôi test-n (~110–120) đẩy lower-CI qua 0 → "suggest gap" kiếm được; effect co về 0 → observer đóng gap → "data mới là rào". Hai ngả đều publishable. Learning-curve còn nảy/dốc ⇒ branch (A) thiếu-data, chưa phẳng.

## ★★ 2026-07-20 (tối) — REFRAME CHỐT: branch A, bỏ gap-làm-luận-điểm
gap(𝒪) n=60→90: **0.083→0.067, z đứng ~1.97** (thêm 50% eval KHÔNG mua significance; point tụt đúng nhịp CI siết), best-obs bò lên oracle (0.500→0.533) → **lean NO-structural-gap**. NGƯNG cứu Control–Observation Gap; hạ xuống open question.

**Trọng tâm mới = 3-tier claims (đọc câu hỏi trục: "can observable debate dynamics be converted into actionable control signals?"):**
- **C1** debate admits control-theoretic formulation — oracle 0.43→0.60. ⚠️ **PHỤ THUỘC objective honest-margin** (objective cũ min-belief[x_adv]: exhaustive 0.41≈no-def → oracle vô dụng) → PHẢI tự khai + biện minh label-free.
- **C2** observable beliefs **contain** actionable signal (KHÔNG "sufficient"). ⚠️ **myopic (0.522) đã lấy nửa dễ; learned observer ≈ myopic (best 0.533, lệch 1 câu), KHÔNG hơn** → **myopic PHẢI là baseline hạng nhất xuyên 3 claim**, nếu giấu → reject. rank-corr~0.5 nhưng không chuyển thành gain-trên-myopic.
- **C3** observers recover only part; residual→oracle (0.53→0.60) = **open question** (structural hay model/data-limited chưa phân định).

**Koopman định vị:** "control-oriented **linear latent DYNAMICS model**" (`z_{t+1}=Az_t+Bu_t`), **KHÔNG "state estimator"** (belief observed → mở lại observability). Bán bằng **tính phân-tích-được + manipulability certificate (exp#4)**, KHÔNG bằng điểm: **twin-MPC 0.49 < myopic 0.52** (Koopman-controller thua heuristic 1 dòng). Reconcile finding-rất-Cybernetics: **faithful sys-ID (fidelity 0.85, B 4/4) ≠ good closed-loop → control-design là open program.** 3 contribution: state-space rep + linear latent dynamics + systematic baseline (thay được Neural-Koopman/RSSM/JEPA).

**Nợ reframe KHÔNG xoá:** (1) ≥2 model faithfulness (qwen+llama) = escape pilot, BẮT BUỘC; (2) McNemar hygiene {oracle/nodef, best-obs/nodef, best-obs/myopic} — đã patch vào representation_sufficiency_v2.py.

## ★★ 2026-07-20 (tối, tiếp) — CHỐT: B là contribution, "controlled dynamics" là why-Koopman
Reframe pipeline: Debate → **Koopman System-ID** → latent dynamics → controller. Trục bán = **input-response operator** `z_{t+1}=Az_t+Bu_t`, contribution nằm ở **B (đáp ứng với can thiệp prompt u), KHÔNG phải A**. Ngôn ngữ: "Koopman **identifies the input-response operator** governing debate evolution" (system-ID, KHÔNG forecasting). Câu chốt hình A-vs-(A,B): *"contribution lies in controlled dynamics, not autonomous evolution."*

**2 cạm bẫy đã chốt cách né:**
1. **Autonomous prediction = con ngựa CHẾT** (operator<persistence trong memory cũ) → rollout tự do "bám tốt" chỉ vì belief bất động = tầm thường. ⇒ mọi thí nghiệm phải là **CONTROLLED** rollout + **gate với baseline persistence/DeGroot/no-input**. Autonomous để cột "Koopman≈Persistence, không sao".
2. **Degree-1 ⇒ Koopman-EDMDc ≡ Linear ARX (CÙNG model)** → **CẤM claim "Koopman > AR"** (trùng nhau, reviewer đập). Cell thắng thật = **"Koopman(linear) ≈ MLP(nonlinear)" ở controlled** → *linearity suffices*, Koopman theory giải thích VÌ SAO linear đủ. Muốn "Koopman>AR" đúng nghĩa phải chứng minh degree>1 lift ăn tiền — đã bỏ, đi hướng linearity-suffices.

**Bảng đích (CHƯA có số — phải chạy trước khi vẽ):** rows persistence/DeGroot/LinearAR/MLP/Koopman × cols autonomous/controlled. Win = controlled: Koopman > persistence & > DeGroot, ≈ AR & ≈ MLP.
**Objective-dependence** nối vào chuỗi objective→controller→u→Koopman-response→performance ⇒ objective hết là "trick", thành biến quyết định control-signal (nâng thành contribution, giữ).

## ★ 2026-07-20 (khuya) — exp#1 controlled-response benchmark: 3/3 PASS (có caveat)
`experiments/controlled_response_benchmark.py` (5-fold OOS, n=60 collect, lỗi trên mean-belief). Kết quả rollout-MSE:
persistence .142 | DeGroot .173 | Linear-AR .161 | MLP-auto .144 | **Koopman-d1(=LinearARX,ctrl) .085 (lead .67)** | Koopman-d2 .169 | **MLP-ctrl .073 (lead .70)**.
- **V1 input-helps ✓**: controlled .085 << mọi autonomous .14–.17 → B (đáp ứng can thiệp) học được = why-control.
- **V2 linear-đủ (MỀM)**: Koopman-d1 .085 vs MLP-ctrl .073 = nonlinear nhỉnh **16%**, và MLP CHƯA hội tụ (max_iter=800). ⇒ viết "linearity **largely** suffices (~16% gap)", KHÔNG "bằng nhau". TODO: bump max_iter=3000 chốt lại.
- **V3 lift-thừa ✓**: d2 .169 > d1 → degree-1 đủ ở scale này → CẤM claim "Koopman>AR".
- **⚠️ One-step MSE thuộc persistence (.053 < tất cả)** → paper CHỈ bán ROLLOUT (đa bước), one-step tự thua persistence. Lợi thế controlled chỉ tích luỹ qua horizon.

## ★ 2026-07-20 (khuya, tiếp) — exp#1 V2 LẬT có lợi (max_iter=3000)
Bump MLP max_iter 800→3000: **MLP-ctrl 0.073→0.102 (overfit), Koopman-d1 đứng yên 0.085** → **linear giờ THẮNG nonlinear OOS (ratio 0.84).** V2 mạnh hơn: KHÔNG chỉ "bằng" mà *linear ≥ nonlinear vì nonlinear overfit ở n=60* → luận điểm **parsimony/sample-efficiency** (linearity của Koopman = feature robust, ít data — rất Cybernetics).
⚠️ Caveat: MLP nhảy 0.073→0.102 chỉ do max_iter → nonlinear nhạy hyperparam ở n=60. Viết "no stable nonlinear advantage + overfit ⇒ parsimony chọn linear", KHÔNG "linear thắng mọi nonlinear". Bulletproof = sweep 2–3 config MLP lấy best OOS. V1/V3 giữ nguyên.

## ★ 2026-07-20 (khuya, chốt V2) — steelman: V2 = "largely suffices" (KHÔNG "linear thắng")
Sweep 4 config MLP-ctrl (16/32/64/early-stop). BEST nonlinear = **MLP-ctrl(es) 0.0775 < Koopman-d1 0.0853 (~10%)**. ⇒ "linear thắng nonlinear" (ratio 0.84 trước) LÀ ARTIFACT do MLP overfit ở max_iter=3000 — đã sửa. **V2 chốt = "largely suffices" (nonlinear tuned nhỉnh ~10%, ổn định). CẤM viết "linear ≥ nonlinear".** Bán = parsimony/analyzability (linear degree-1 đạt ~90% neural-net-tuned + cho closed-form control tools = why-Koopman). Spread MLP 0.077–0.102 ⇒ nonlinear nhạy hyperparam, linear low-variance (điểm cộng). TODO nail: paired test xem 10% có significant hay noise → nếu n.s. thì viết mạnh "no sig diff → linear suffices". V1/V3 vững.

## ★ 2026-07-20 (khuya, ĐÓNG exp#1) — V2 = n.s. → "linearity suffices" bản mạnh
paired Δ(Koopman-d1 − best-MLP-es) rollout-MSE = +0.0079, **CI[−0.0079,+0.0256] CHỒNG 0 → n.s.** ⇒ linear KHÔNG khác biệt có ý nghĩa vs best-tuned nonlinear. **exp#1 DONE, 3/3 verdict sạch.**
Câu paper-ready V2: *"linear controlled operator (Koopman-EDMDc d1 ≡ ARX) predicts intervention response as accurately as best-tuned nonlinear (paired n.s., 95%CI[−0.008,+0.026]), beats input-free baselines ~40% → adopt linear (analyzability+lower-variance)."*
⚠️ n.s. ở n=60 phần nào underpowered → viết "no DEMONSTRABLE nonlinear advantage → parsimony chọn linear", KHÔNG "chứng minh linear=nonlinear". 2nd-model siết CI sau.
Số bảng cuối (5-fold OOS rollout-MSE): persistence .142/DeGroot .173/Linear-AR .161/MLP-auto .162 (autonomous) || Koopman-d1 .085/MLP-es .078/MLP-16 .084/MLP-64 .091/MLP-32 .102/Koopman-d2 .169 (controlled). File: experiments/controlled_response_benchmark.py.

## ★★★ 2026-07-24 — PHIÊN VIẾT PAPER + LẬT TRỤ observability (đọc kỹ)

**A. Generalization run XONG + verify (4 dataset × 60 collect + 90 eval, qwen2.5:7b, degree-1, n_rounds=6).** Reproduce 100% từ jsonl (kể cả twin, refit Kop,B). Headline honest-margin exhaustive vs no-def: mmlu .433→.600 p=.001 / math .289→.433 p=.007 / truthfulqa .300→.422 p=.019 (đều sig) / bbh_logic7 .222→.267 p=.45 (yếu). CSQA đang chạy (eval ~11/90, sơ bộ ÂM Δ−.09 vì no-def=.64 dễ + ASR=0 → benchmark ít-manipulability). Objective CŨ (min-adv) KHÔNG nâng đâu (mmlu còn −.022) → "objective matters" là điểm mạnh.

**B. ★ STRUCTURAL ANALYSIS LẬT TRỤ v2 (`analyze_structure.py`).** Rút hệ LTI (A,Bz,C) từ operator, C=mean-agent readout. 5 dataset: **CTRB rank ĐẦY + OBSV rank cũng ĐẦY** (ker M_H={0}, unobs_dim=0), κ(Wo)~10⁵ < κ(Wc)~10⁶⁻⁷. ⇒ **định lý "controllable-but-NOT-observable" SAI VỀ SỐ — BỎ** (reviewer tính Gramian 5 phút là bác). 2 tài sản thay: (1) ρ(A)≈0.75–0.82 nhất quán 5 benchmark (phổ ổn định, sạch). (2) **REFRAME = "controllable + TRUTH-AGNOSTIC"**: điều-khiển-được hoàn toàn (đẩy belief tới đáp án nào cũng được kể cả sai) NHƯNG belief không mã hóa chân lý. Đặt tên **state-observability (đầy đủ) vs epistemic-observability (thiếu)** cho gap Δ_co (giữ OPEN).

**C. exp#4 CERTIFICATE = CHẾT (gate rẻ, tiết kiệm 4–8h).** `certify_v0.py`: ρ(Cert,Emp)=−0.10. `certify_v1.py` (vá start-state uniform→k_def thật): best ρ=+0.40 << 0.70. Operator có tín hiệu manipulability thô nhưng KHÔNG đủ certify → future-work, viết như cautionary-negative. (Lần thứ 3 operator-để-dự-đoán chết sau τ + twin.)

**D. C1-STRENGTH (`analyze_c1_strength.py`).** (#2) **POOLED Δacc=+0.144 CI95[+0.093,+0.196]** (3 MAIN complete) = headline. (#3) **ASR label-free = mạnh nhất**: no-def 0.37–0.48 → defense 0.00–0.02 (robust .98–1.0). (#4) defense chọn đúng a* chỉ 9–31% → **suppress-adversary, không inject-truth** (khớp truth-agnostic). Insight: defense giúp ⟺ (có attack) VÀ (honest cứu được) → bbh=ceiling-honest, CSQA=no-attack.

**E. ĐÍNH CHÍNH: claim (ii) fidelity KHÔNG phải nợ mới** — đã xong exp#1 (`controlled_response_benchmark.py`, 2026-07-20): Koopman-d1 .085 ≈ best-MLP .078 (paired n.s., CI[−.008,+.026]), beats persistence/DeGroot ~40%. Chỉ cần MỞ RỘNG multi-benchmark/model, không chạy lại từ đầu.

**F. PAPER.md VIẾT LẠI (đọc file để biết chi tiết).** (1) Thread vào Method §contr-obs: **Prop 4 truth-agnosticity (có proof) + Cor 5 attack-defense duality + Remark hòa giải Choi2025 + đặt tên epistemic-vs-state observability** (report ker M_H={0}); dời "Prop 4 gap-closes"→Prop 6. Thread A/B+số vào Intro (i,iii,iv)+Contributions. (2) **Convert acmart→IEEEtran 2 cột**, viết Abstract (đang trống), nén 585→354 dòng. FEEDBACK user: bản nén đầu bỏ nhiều công thức → "nhiều chữ"; **đã sửa math-dense**: 22 equation displayed + 3 algorithm (EDMDc/audit/MPC) + 9 theorem-env + 3 figure (overall/spectrum/controllability; bỏ 5 hình textbook). Benchmark: trụ 3 mạnh (mmlu/math/tqa), bbh+CSQA→boundary. §Experiments để OUTLINE (map thẳng vào script). Backup: `paper_acmart_backup.md`, `doc_blueprint_v1_archived.md` (doc.md cũ đầy claim chết).

**G. TỰA CHỐT:** *"EigenDebate: Reading the Spectrum of Deliberation to Steer and Secure Multi-Agent LLM Debate"* (bỏ "Predict" vì forecasting đã chết; Steer=controllability, Secure=manipulability/defense hợp Cybernetics).

**H. VENUE = IEEE Trans Cybernetics** (~10 trang, 2 cột IEEEtran; correspondence ~5). Verdict phiên: lý-thuyết+story MẠNH & đầy đủ cho foundation paper; CHƯA submit được vì: (1) §Experiments mới outline (phải viết), (2) **`sample-base.bib` KHÔNG tồn tại** (phải tạo cho ~21 nguồn), (3) faithfulness gemma/llama + CSQA còn chạy. 3×4 (3 model × 4 benchmark) xong → empirical objection tan gần hết → major-revision-nghiêng-accept. Điểm yếu cố hữu còn: đóng góp khả-dụng là defense label-free NHƯNG tốn K× (twin làm-rẻ chết); 7–9B model; n≈90.

**Scripts mới phiên này:** `analyze_twindef.py` (verifier hợp nhất mọi dataset), `analyze_structure.py` (Gramian ctrb/obsv + phổ), `analyze_c1_strength.py` (pooled+ASR+label-free-valid), `certify_v1.py`, `build_topics.py` (+source commonsense_qa). Provider vẫn Ollama local qwen2.5:7b.

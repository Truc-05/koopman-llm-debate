# EDMDc-under-LLM-sampling-noise — Feasibility sketch (giá-vào-cửa TCyb)

*Mục tiêu: biết cái bound có (a) **derive được**, (b) **novel đủ cho TCyb**, (c) **không vacuous** —
TRƯỚC khi đổ 2–4 tuần chứng minh. Đây là "cược có kiểm soát" cho lựa chọn TCyb-gamble.*

---

## 0. Cái ta thực sự đang ước lượng (từ code, không phải mơ)

- State `z_t ∈ R^D`, `D = N·Kans` = **logits per-agent ghép lại** (`build_state`), belief `b_{a,t}=softmax(z_{a,t})`.
- Degree-1 dictionary ⇒ `Ψ(z)=[1,z]` ⇒ Koopman-restricted-to-state = **A (D×D)**, control **B (D×Kans)**.
- `fit_edmdc`: `Θ̂ = [Â B̂] = (Σ_t z_{t+1} φ_tᵀ)(Σ_t φ_t φ_tᵀ + λI)^{-1}`, `φ_t=[z_t;u_t]`, `λ=1e-6` (ridge).

**⇒ "EDMDc bound" = finite-sample error bound cho RIDGE LINEAR SYSTEM-ID từ trajectory snapshots.**
Đây là dữ kiện then chốt: **degree-1 ⇒ không có phần "Koopman lift" phi tuyến** — nó là DMDc thuần.

---

## 1. Model & error decomposition (phần này CHẮC chắn đúng)

Giả định model: `z_{t+1} = A z_t + B u_t + ξ_t`, `ξ_t = ξ_t^{stoch} + b_t`
- `ξ_t^{stoch}` = sampling noise của LLM (zero-mean có điều kiện, martingale-difference nếu model well-specified),
- `b_t = E[z_{t+1}|F_t] − (A z_t + B u_t)` = **misspecification bias** (dynamics thật phi tuyến).

Ridge estimate error (chuẩn, không cãi được):
```
Θ̂ − Θ = ( Σ_t ξ_t φ_tᵀ ) V_n^{-1}  −  λ Θ V_n^{-1},     V_n = Σ_t φ_t φ_tᵀ + λI
        = [ noise term ]           + [ regularization bias ]
```
Tách `ξ_t = ξ_t^{stoch} + b_t`:
```
‖Θ̂ − Θ‖₂ ≲ ‖ Σ ξ^{stoch} φᵀ ‖ / λ_min(V_n)   (A) variance, → 0
           + ‖ Σ b_t φᵀ ‖ / λ_min(V_n)         (B) MISSPEC BIAS FLOOR, KHÔNG → 0
           + √λ ‖Θ‖ / λ_min(V_n)               (C) reg, bé
```

## 2. Bound cho phần variance (A) — self-normalized martingale (đã có sẵn)

Nếu `ξ^{stoch}_t` là **σ-sub-Gaussian có điều kiện** trên filtration `F_t`, thì theo self-normalized
tail inequality (Abbasi-Yadkori–Pál–Szepesvári 2011, bản matrix), w.p. ≥ 1−δ:
```
‖ V_n^{-1/2} ( Σ_t φ_t ξ^{stoch}_tᵀ ) ‖  ≤  σ · sqrt( 2 log( det(V_n)^{1/2} / (det(λI)^{1/2} δ) ) )
```
⇒ phần variance của error, chuyển sang spectral:
```
‖Θ̂ − Θ‖₂^{(var)}  ≤  σ · sqrt( (D+Kans)·log( (1 + n·B_φ²/λ) / δ ) )  /  sqrt( λ_min(V_n) )
```
với `B_φ = max_t ‖φ_t‖`. **Dạng này standard.** Rate `~ σ·sqrt(D log n / λ_min(V_n))`.

## 3. Ba chỗ LLM-specific (đây là TOÀN BỘ novelty — nếu không có → reject "incremental")

Vì §2 là corollary của linear-sysID có sẵn, novelty **phải** đến từ 3 chỗ sau:

**(N1) σ từ simplex geometry (sạch, honest).** Belief sống trên simplex `Δ^{Kans-1}`. Nếu belief ước
lượng từ `m` mẫu vote → `ξ^{stoch}` là multinomial deviation bounded ⇒ **σ ≲ 1/√m** (Hoeffding trên
simplex), không cần giả định phân phối. Nếu chỉ 1 generation → σ tied to temperature τ (xem N2).
→ *Đóng góp:* bound σ **không** bằng giả định sub-Gaussian trừu tượng mà bằng **cấu trúc lấy mẫu thật**.

**(N2) σ(τ) — nối temperature vào bound (novel nhất NHƯNG shaky nhất).** Logits = logprobs của model
(deterministic given prompt); randomness vào qua **transcript được sample** ở nhiệt độ τ, đổi prompt vòng
sau. `σ ≲ L_gen · g(τ)`, `g` tăng theo τ (τ→0 ⇒ σ→0). **Rủi ro:** `L_gen` (Lipschitz của next-logits
theo transcript sample) khó bound chặt — reviewer có thể bác. *Giảm rủi ro:* chỉ claim monotone `σ(τ)↑`
+ verify thực nghiệm (chạy 2–3 temperature, đo σ residual), không claim hằng số tuyệt đối.

**(N3) bias `b_t` qua softmax curvature (nối vào manipulability — đắt giá nhất).** Belief=softmax(z);
linearization error của belief-update phi tuyến bị chặn bởi Jacobian softmax `J=diag(b)−bbᵀ`,
`‖J‖≤1/2`, **suy biến khi agent tự tin (gần đỉnh simplex)**. ⇒ **bias `b_t` lớn nhất đúng lúc agent
confident** — nối thẳng vào positional-bias / manipulability đã đo. *Đây là insight LLM-specific thật,
non-decorative, và testable.*

## 4. Verdict feasibility (HONEST)

| Câu hỏi | Trả lời | Độ tin |
|---|---|---|
| Derive được bound? | **Có** — skeleton §1–2 đóng chắc | ~cao |
| Novel một mình (§2)? | **Không** — corollary Simchowitz/Sarkar–Rakhlin | ~cao |
| Cứu novelty được? | Chỉ nếu N1+N2+N3 gánh + validate được N3 | trung bình |
| N3 validate sẽ đứng? | **Rủi ro** — là claim *predictive về operator*, đã chết 3× (τ/twin/certificate) | thấp–TB |

## 5. BA CỬA TỬ — mỗi cửa test RẺ trên data có sẵn (gate script) TRƯỚC khi cam kết

Bound có thể **đúng mà vẫn là đồ bỏ**. Ba cách nó chết, mỗi cách một gate offline (`gate_noise_bound.py`):

**Cửa 1 — Misspecification nuốt variance (bias floor (B) ≫ (A)).** Nếu dynamics thật phi tuyến mạnh,
term (B) không → 0, bound variance vô nghĩa. *Test:* residual `r_t=z_{t+1}−Âz_t−B̂u_t`; (i) correl
`r_t` với `z_t` (structured ⇒ bias); (ii) R²(deg-1) vs R²(deg-2) — nếu deg-2 giảm residual mạnh ⇒
linear misspecified ⇒ bias-floor thống trị ⇒ **bound decorative**.

**Cửa 2 — Quasi-static (A≈I, B≈0), bound đúng nhưng vacuous.** Fidelity đã chết + "belief quasi-static"
⇒ nghi `z_{t+1}≈z_t`. Nếu vậy A≈I fit residual bé (bound thỏa tầm thường) nhưng **không có dynamics để
nhận dạng**, và control effect B chìm dưới noise. *Test:* `‖A−I‖`, spectral content, và **control SNR** =
‖B u‖ điển hình / ‖ξ‖ điển hình. SNR≲1 ⇒ B bất-khả-nhận-dạng ⇒ bound vô hồn.

**Cửa 3 — Variance structure của bound có THẬT không (đồng thời validate N3).** Bound nói
`Var(Θ̂) theo hướng i ∝ σ²/λ_i(V_n)`. *Test:* bootstrap refit (A,B) trên B resample trajectory → đo
Var per eigendirection của `V_n` → `Spearman(log Var_i, −log λ_i)`. ≈+1 ⇒ cấu trúc bound thật (đèn
xanh N3); ≈0 ⇒ decorative (đèn đỏ, nguy cơ chết-lần-4).

## 6. Quy tắc quyết định (sau khi chạy gate)

- **3 cửa đều xanh** (bias nhỏ so noise, control-SNR>1, Spearman>~0.6) ⇒ **derive full** (§1–4),
  ~2–4 tuần, TCyb-plausible. Killer figure = **predicted-vs-actual per-direction error** (validate N3).
- **Bất kỳ cửa nào đỏ** ⇒ bound **decorative/vacuous** = dấu-hiệu-thứ-4 operator không có răng vận hành
  ⇒ **KHÔNG derive**; quay lại xét venue (TCyb-không-bound = reject ⇒ tụt TNNLS/TAI). Trung thực > cược mù.

## 7. Vị trí trong bài (nếu xanh)
- §Theory: Thm (finite-sample ID của debate-plant dưới LLM sampling): dạng §2 + N1–N3.
- §Experiment: validate N3 (predicted vs actual error theo hướng) + σ(τ) monotone (2–3 temperature).
- Nối related-work: linear-sysID finite-sample (Simchowitz, Sarkar–Rakhlin) + EDMD approx (Nüske,
  Zhang–Zuazua, Ziemann) — **định vị: ta specialize cho softmax-logit LLM dynamics + simplex noise + bias-curvature**.

---
*Gate trước, chứng minh sau. Data có sẵn: `ckpt_twindef_collect_<tag>.jsonl`. Script: `gate_noise_bound.py`.*

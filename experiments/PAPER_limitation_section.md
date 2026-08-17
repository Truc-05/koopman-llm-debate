# §Limitation — "Controllable empirically, but not cheaply modelable"
*(drop vào paper.md/doc.md; số từ gate_noise_bound.py + gate_noise_bound_v2.py, mmlu_clean6, n=60, 3 model llama3.1-8b / mistral-7b / gemma2-9b)*

Our identified Koopman operator (A, B) is a **descriptive** object, not a predictive or
model-based-control tool. We quantify this explicitly, because it delimits what the framework
can and cannot claim — and it explains why belief-forecasting, the digital-twin controller, and
an operator-based manipulability certificate all failed in our experiments.

**(L1) The linear model is the right complexity — the residual is irreducible sampling noise,
not correctable nonlinearity.** A degree-1 EDMDc (linear) model explains only
**17–42%** of out-of-sample belief-transition variance (6-fold CV, R²_OOS across models). Crucially,
enriching the dictionary does **not** help: a degree-2 EDMD **overfits catastrophically** at this
data scale (R²_OOS strongly negative; 153 features vs ≈360 transitions), so the unexplained variance
is dominated by *sampling noise*, not by recoverable nonlinear structure. The linear operator is
therefore an appropriate — but intrinsically noisy — surrogate.

**(L2) The control channel is below the noise floor.** The per-step control signal is smaller than
the sampling noise: control-SNR = ‖B u‖ / ‖ξ‖ ≈ **0.42–0.56 < 1** across models. Consequently the
control matrix B is identifiable only at the standard √n rate and remains loosely pinned at practical
sample sizes: bootstrap relative error of B̂ is **≈ 40–50%** at n = 60 and decreases like 1/√n,
implying ≈ 20× more trajectories to reach ≈ 10% relative error. A single one-hot answer push moves
the collective belief *less* than round-to-round sampling jitter.

**(L3) These are fundamental data/noise limits, not estimation artifacts.** The estimator's variance
matches least-squares theory *exactly*: per-eigendirection variance scales as 1/λ_i of the data Gramian
(Spearman **+0.99** between log Var_i and −log λ_i). The Gramian is strongly anisotropic
(cond ≈ 1.5–2×10⁴), the operator-theoretic signature of the positional bias we report in §[Structure]:
some directions are well-excited, others barely — but the estimation behaves as theory predicts
throughout. Hence a finite-sample estimation bound for (A, B) would be *correct but vacuous*: it would
certify precisely that the control channel it concerns is buried under noise.

**Consequence for the contribution.** Controllability in this paper is established **empirically and
end-to-end** — a single adversary shifts the collective answer, and a label-free oracle intervention
recovers accuracy (McNemar p = 0.006; §[Controllability]) — *without* relying on the operator as a
predictor. Steering is therefore **empirical/oracle-driven, not model-driven**. The value of the
Koopman operator is *analytic*: its structural invariants (full controllability rank, anisotropic
Gramian, positional bias) characterize **where** a debate is manipulable, while its noise-limited,
low-SNR dynamics preclude cheap model-based prediction, control, or certification. We regard closing
this "controllable-but-not-cheaply-modelable" gap — e.g. via lower-variance belief estimation or
richer intervention channels that raise control-SNR above 1 — as the central open problem.

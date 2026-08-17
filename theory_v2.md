# Theory core v2 — Koopman response-collapse (insert-ready, §III–V)

**Thesis.** *MAD's identifiable controlled dynamics collapse empirically to a low-order static response map.*

**Final claim (safe).** The first Koopman-operator treatment of multi-agent LLM debate, revealing that its
identifiable controlled dynamics collapse empirically to a low-order static response map.
> Wording: use **"estimate and characterize intervention responsiveness / manipulability"** — NOT "certify".
> "Certificate" only if a separate calibration/coverage test proves the bound holds (old reachability
> predicted realized reach weakly → no certificate claim).

---

## Setup
MAD = controlled stochastic (Markov) process. State \(z_t\in\mathbb R^{D}\) (per-agent belief-logits,
\(D=N K\)), intervention \(u_t\in\mathcal U\) (one-hot channel active for \(t\ge K_{\mathrm{DEF}}\)),
observable \(g:\mathbb R^{D}\!\to\!\mathbb R^{m}\), output \(y_t=g(z_t)\).

**Definition 1 (controlled Koopman + response).**
\[(\mathcal K_u g)(z)=\mathbb E[g(z_{t+1})\mid z_t=z,\,u_t=u],\qquad \Delta\mathcal K_u=\mathcal K_u-\mathcal K_0.\]

## Theorem 1 — Controlled stochastic Koopman formulation
For each \(u\), \(\mathcal K_u\) is a bounded linear operator on bounded observables
(\(\|\mathcal K_u g\|_\infty\le\|g\|_\infty\)), and \(\{\mathcal K_u\}_{u\in\mathcal U}\) is a family indexed
by the intervention. Hence the nonlinear, stochastic debate admits an **exact linear representation in
observable space**, with the control entering as operator selection.
*Proof.* Linearity of conditional expectation; boundedness by the Markov/contraction property. ∎

## Theorem 2 — Finite-horizon response-collapse
Suppose the lifted dynamics are affine, \(z_{t+1}=A z_t+B u_t\), with output \(y=Cz\). Then the \(H\)-round
**interventional response** (difference between the driven and undriven rollouts from a common state) is
\[
\Delta y_H \;=\; C\sum_{k=0}^{H-1} A^{\,H-1-k} B\,u_k
\;=\; \Theta^\star\,\mathbf u_{0:H-1},
\qquad \Theta^\star=\big[\,CA^{H-1}B\ \cdots\ CB\,\big],
\]
i.e. a **static linear map of the intervention sequence** \(\mathbf u_{0:H-1}\). Therefore the hypothesis
class of an unconstrained direct linear regression on \((z,\mathbf u)\) **contains the entire response
class**, and the Koopman/EDMDc factorization \(\{A,B,C\}\) offers **no representational advantage** for
predicting or selecting interventions in this regime; it can only match, not exceed, direct estimation.
*Empirical evidence.* Paired cluster-bootstrap TOST across **4 models** (n=1710): selection
\(\Delta_{\text{sel}}=+0.002\), 90% CI \([-0.005,+0.009]\subset\pm0.03\) → **equivalent** (McNemar \(p=0.68\));
effect-prediction shows **no consistent advantage** of either estimator. The dynamic factorization collapses
to the static response predicted by the theorem. ∎

## Proposition — Identifiability under randomized interventions
Under paired/randomized control with sufficient excitation and overlap (positivity), the response map
\(\Delta\mathcal K_u g(z)=\mathbb E[g\mid z,u]-\mathbb E[g\mid z,0]\) is identifiable and its paired estimator
is consistent, with finite-sample error \(O(\sqrt{d/n})\) under conditional exchangeability.
*(The twindef design supplies the required paired \(u\) vs. \(0\) at matched states.)*

## Empirical observation (stated as finding, not theorem)
- **One-step belief dynamics carry a predictable convergence drift** — a low-order autonomous operator
  improves one-step belief prediction over persistence (SNR \(\approx\) 4–12; skill \(+3\%\) to \(+13\%\),
  4 models, LOO-CV). So belief is *not* persistence-trivial at one step (DeGroot-like consensus).
- **Multi-step and semantic fidelity are unstable**, and across every regime **Koopman does not beat direct
  models**: multi-step forecast (koopman > persistence only 2/7), semantic-observable operator (n=24,
  skill \(-0.33\), CI wholly negative), manipulability from reachability (loses to persistence, n=1710).
- Steering is nonetheless effective: response-based intervention selection **recovers ~77% of oracle
  headroom** over no-defense (koopman .42 vs no-def .33 vs oracle .45; \(\Delta_{\text{oracle}}\) CI excludes
  0 → report as *fraction of headroom*, **not** "near-oracle").

---

## Positioning (must-cite + differentiate)
- **Wilson & Akrout 2026 (arXiv:2605.05134)** — Koopman/EDMD of a *single* LLM (hallucination). We differ:
  multi-agent debate + *controlled* Koopman + inter-agent belief dynamics.
- **Pokharel & Dantu 2026, "Hidden Anchors" (arXiv:2606.19494)** — LLM debate as closed-loop
  **DeGroot + Friedkin–Johnsen anchor**, no Koopman. Closest framing neighbor. **Our Theorem 2 explains why
  their hand-specified static model suffices**: the finite-horizon response is a static linear map, so
  dynamic factorization adds nothing. Cite head-on as corroboration, not competition.
- Also cite: Niemann–Klus–Schütte (Koopman generator for agent-based social dynamics, arXiv:2012.07718);
  "Koopman Performance Analysis of Nonlinear Consensus Networks" (arXiv:1807.04237).
- **Do NOT claim**: first Koopman-on-LLMs, first Koopman-opinion-dynamics, first control-account-of-debate.
  Claim only the **conjunction** (Koopman × multi-agent LLM debate) + the response-collapse finding.

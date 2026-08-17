# koopman-debate

**A Koopman operator theory of multi-agent LLM debate.**

We treat a multi-agent LLM debate (MAD) as a *controlled dynamical system*: the
per-agent belief distribution is the state, a moderator intervention is the
control input, and the round-to-round update is an unknown nonlinear stochastic
map. Using Extended Dynamic Mode Decomposition (EDMD / EDMDc) we fit a **linear
latent-dynamics model** of that map in observable space, and use it to *analyze*
and *steer* the debate.

The central finding is summarized by the project's thesis:

> **Controllable beyond prediction.** A debate's consensus is *reliably
> steerable* by an external intervention, even though its belief trajectory is
> *not cheaply predictable* from the state.

---

## What this project shows

- **Debate is controllable.** An oracle intervention that injects the truth-
  aligned channel reliably moves the group toward the correct answer. On the
  anchor benchmark cell it lifts accuracy from ≈0.42 to ≈0.58 (exact McNemar
  *p* = 0.006, paired Δ ≈ +0.17).
- **The effect replicates.** Across **4 open models** (`qwen2.5:7b`,
  `llama3.1:8b`, `mistral:7b`, `gemma2:9b`) × **4 benchmarks** (MMLU, GSM8K-style
  math, TruthfulQA, ARC), the pooled accuracy gain from the oracle intervention
  is Δ ≈ +0.13 with a stratified-bootstrap 95% CI that excludes 0.
- **Koopman gives an analyzable model.** Fitting `(A, B)` with EDMDc yields a
  linear controlled system `Ψ(z_{t+1}) = A·Ψ(z_t) + B·u_t` whose structure is
  interpretable: full controllability rank (16/16), a per-target reachability
  index reaching ≈1.0 (**manipulability**), spectral radius ρ(A) ≈ 0.62–0.80,
  and a measurable positional/answer-order bias in `B`.
- **Prediction is the hard side.** A myopic (current-belief) observer is already
  a strong predictor of the debate outcome; learned observers do not beat it.
  So the value of the Koopman model is *analyzability and manipulability*, not
  forecasting — control is the "green" axis, observation the "red" one.

Everything below is code for the results above. Exploratory directions that did
not survive validation (belief-trajectory early-warning, model-predictive
"twin" defense, operator-as-forecaster) have been retired and are not part of
the maintained pipeline.

---

## Repository layout

### Core library — `src/`

| Module | What it does |
|---|---|
| `src/debate/state.py`, `orchestrator.py` | Debate state (per-agent belief logits) and the multi-agent orchestrator |
| `src/debate/observables_truth.py` | Truth-aligned observables / channel used by the oracle intervention |
| `src/koopman/edmd.py` | EDMD and **EDMDc** (`fit_edmdc` → returns `A, B`) |
| `src/koopman/dictionary.py` | Observable dictionaries (polynomial lifting, degree-1 default) |
| `src/koopman/dmd_reduced.py`, `spectrum.py` | DMD-SVD reduction and spectral decomposition |
| `src/koopman/stochastic.py` | Stochastic (ensemble) Koopman estimate |
| `src/control/` | Koopman-MPC / LQR controllers on the fitted operator |
| `src/criteria/` | Consensus, truth-alignment, and early-warning metrics |
| `src/baselines/` | Opinion-dynamics baselines (DeGroot, Friedkin–Johnsen, hidden anchors) |
| `src/utils/metrics.py`, `viz.py` | Fidelity/AUC metrics and plotting |

### Experiments — `experiments/`

| Script | Role |
|---|---|
| `run_synthetic.py` | Offline end-to-end demo (nonlinear debate sim, fidelity vs baselines) — **no API needed** |
| `run_debate_groq.py` | Run real LLM debates, save trajectories/transcripts, fit EDMD |
| `run_edmdc_control.py` | Collect debates with a random intervention channel and fit `(A, B)` |
| `control_analysis.py` | Control-theoretic analysis of `(A, B)`: stability, controllability rank, reachability |
| `check_fidelity.py` | H-step rollout accuracy of the Koopman model vs persistence / DeGroot / MLP |
| `paper_numbers.py` | **One-command verifier** — recomputes every reported number from saved checkpoints |
| `prepare_mmlu.py` | Download the benchmark topics |

Saved run checkpoints live in `experiments/out/` (llama / mistral / gemma) and
`experiments/out_qwen/` (qwen).

---

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.10 (numpy, scipy, scikit-learn, matplotlib, pyyaml).

## Quickstart

```bash
# 1. Unit tests — math checks on linear systems with known answers (no API)
pytest

# 2. Offline demo — nonlinear debate sim, fidelity vs FJ/DeGroot (no API)
python experiments/run_synthetic.py

# 3. Reproduce the paper numbers from the saved checkpoints (no API, offline)
python experiments/paper_numbers.py
OUT=experiments/out_qwen python experiments/check_fidelity.py   # per-model view

# 4. Analyze the fitted controlled operator (A, B)
python experiments/control_analysis.py
```

### Running fresh debates (needs a model endpoint)

```bash
# Smoke test on 3 sample questions
python experiments/run_debate_groq.py

# Full benchmark
python experiments/prepare_mmlu.py --n 40
python experiments/run_debate_groq.py --topics experiments/topics/mmlu_40.json

# Collect controlled debates and fit the Koopman-with-control operator
python experiments/run_edmdc_control.py --topics experiments/topics/mmlu_clean6.json --n 80
```

## Models & providers

The runtime speaks an OpenAI-style chat endpoint; switch providers by editing
`configs/debate.yaml`. The default is a **local Ollama** model (free, no rate
limits):

```yaml
base_url: http://localhost:11434/v1/chat/completions
model: qwen2.5:7b   # also validated: llama3.1:8b, mistral:7b, gemma2:9b
```

Config: `n_agents: 4`, `n_rounds: 6`, degree-1 observable dictionary. Six rounds
give a clean spectrum (the λ=1 consensus mode stays separated); more rounds push
the tail toward equilibrium and degrade the fit.

> **Do not** use a safety-classifier model (e.g. `llama-prompt-guard`) as a
> debate agent — it cannot emit arguments + JSON probabilities, so every agent
> collapses to a uniform fallback and the debate has no dynamics.

## Math conventions

- **Column snapshots.** `Psi_X, Psi_Y ∈ R^{M×S}`, with `K = A·G⁻¹` so that
  `psi_{t+1} = K·psi_t`.
- **Eigenfunction = left eigenvector.** `phi_i(z) = W[i,:] @ Psi(z)` where
  `W = V⁻¹` from `spectral_decomposition(K)`. Using the right eigenvector is
  wrong — see the docstring in `src/koopman/spectrum.py` and
  `test_eigenfunction_property`.
- **Trajectories include `z₀`** (the uniform prior) so the first transition is
  available to downstream analysis.

## Status

Research code accompanying the paper *"Controllable Beyond Prediction: A Koopman
Response Theory of Multi-Agent Debate."* The controllability result (C1) and its
4-model replication are the load-bearing claims and are reproduced offline by
`experiments/paper_numbers.py`.

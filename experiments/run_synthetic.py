"""Offline end-to-end demo of the whole pipeline (NO API calls).

Simulates nonlinear debates in two regimes (truth-convergent vs herding), then:
  1. EDMD fit + spectrum per regime          (doc §4, §3)
  2. Multi-step fidelity vs FJ / DeGroot     (doc §6.2, §6.5 — make-or-break)
  3. Early warning from the first rounds     (doc §6.3, Thm 3) -> AUC

Run:  python experiments/run_synthetic.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.debate.state import softmax
from src.koopman.dictionary import PolynomialDictionary
from src.koopman.edmd import build_snapshots, fit_edmd
from src.koopman.dmd_reduced import fit_dmd_reduced, full_operator
from src.koopman.spectrum import spectral_decomposition, spectral_gap, eigenfunctions
from src.criteria.truth_alignment import truth_alignment_index
from src.criteria.early_warning import per_debate_mode_ratios, evaluate_auc
from src.debate.observables_truth import h_star_soft, along_trajectory
from src.utils.metrics import fidelity_curves_over_debates
from src.baselines.friedkin_johnsen import fit_fj
from src.baselines.degroot import fit_degroot

N_AGENTS, N_ANSWERS, T = 4, 3, 10
D = N_AGENTS * N_ANSWERS
A_STAR = 0
EARLY = 3           # rounds visible to the early-warning detector
N_PER_REGIME = 40
N_TRAIN = 30
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

rng = np.random.default_rng(0)


def simulate_debate(herding):
    """Nonlinear opinion dynamics: conformity pull through softmax (nonlinear),
    plus either evidence toward the true answer or an anchor on the first
    speaker's initial answer."""
    x = rng.normal(0.0, 1.0, size=(N_AGENTS, N_ANSWERS))
    anchor = np.zeros(N_ANSWERS)
    anchor[np.argmax(x[0])] = 2.0
    traj = [x.copy()]
    for _ in range(T):
        pbar = softmax(x).mean(axis=0)
        social = np.log(pbar + 1e-9)
        if herding:
            x = 0.55 * x + 0.35 * social + 0.10 * anchor
        else:
            evid = np.zeros(N_ANSWERS)
            evid[A_STAR] = 2.0
            x = 0.60 * x + 0.20 * social + 0.20 * evid
        x = x + 0.05 * rng.normal(size=x.shape)
        x = x - x.mean(axis=1, keepdims=True)
        traj.append(x.copy())
    return np.array(traj)                      # (T+1, N, K)


def flat(X):
    return X.reshape(X.shape[0], -1)           # (T+1, N*K)


def pooled_snapshots(debates):
    Z_t = np.concatenate([flat(X)[:-1] for X in debates], axis=0)
    Z_tp1 = np.concatenate([flat(X)[1:] for X in debates], axis=0)
    return Z_t, Z_tp1


def baseline_curve(step_fn, debates, max_h):
    """Per-horizon error of a baseline stepper x' = step_fn(x, x0)."""
    errs = [[] for _ in range(max_h)]
    for X in debates:
        Tn = X.shape[0]
        for t0 in range(Tn - 1):
            H = min(max_h, Tn - 1 - t0)
            x = X[t0].copy()
            for h in range(1, H + 1):
                x = step_fn(x, X[0])
                errs[h - 1].append(np.linalg.norm((x - X[t0 + h]).ravel()))
    return np.array([np.mean(e) for e in errs])


def main():
    os.makedirs(OUT, exist_ok=True)

    truth_debates = [simulate_debate(False) for _ in range(N_PER_REGIME)]
    herd_debates = [simulate_debate(True) for _ in range(N_PER_REGIME)]
    train = truth_debates[:N_TRAIN] + herd_debates[:N_TRAIN]
    test = truth_debates[N_TRAIN:] + herd_debates[N_TRAIN:]
    test_labels = np.array([0] * (N_PER_REGIME - N_TRAIN)
                           + [1] * (N_PER_REGIME - N_TRAIN))

    d = PolynomialDictionary(degree=2)

    # ---------- 1. Spectrum per regime ----------
    print("=" * 60)
    print("1. Koopman spectrum per regime (train)")
    for name, debates in [("truth", truth_debates[:N_TRAIN]),
                          ("herding", herd_debates[:N_TRAIN])]:
        Z_t, Z_tp1 = pooled_snapshots(debates)
        K, _, _ = fit_edmd(*build_snapshots(d, Z_t, Z_tp1), reg=1e-4)
        eigvals, V, W = spectral_decomposition(K)
        h_vals = along_trajectory(h_star_soft, Z_t, n_agents=N_AGENTS,
                                  n_answers=N_ANSWERS, a_star=A_STAR)
        Phi = eigenfunctions(d, W, Z_t)
        hc = h_vals - h_vals.mean()            # drop the trivial constant part
        tau1 = truth_alignment_index(h_vals, Phi[:, 0])
        tau2 = truth_alignment_index(hc, Phi[:, 1])
        print(f"  [{name:8s}] |lam| top-4 = {np.round(np.abs(eigvals[:4]), 4)}"
              f"  gap = {spectral_gap(eigvals):.4f}"
              f"  tau1 = {tau1:.3f}  tau2(centered) = {tau2:.3f}")

    # ---------- 2. Fidelity: Koopman vs FJ vs DeGroot on held-out ----------
    print("=" * 60)
    print("2. Multi-step fidelity on held-out debates (state RMSE per horizon)")
    Z_t, Z_tp1 = pooled_snapshots(train)
    test_flat = [flat(X) for X in test]

    # degree-1 Koopman (linear + constant; the fair 'linear' Koopman)
    d1 = PolynomialDictionary(degree=1)
    K1, _, _ = fit_edmd(*build_snapshots(d1, Z_t, Z_tp1), reg=1e-4)
    koop1 = fidelity_curves_over_debates(K1, d1, test_flat, 5, relift=True)

    # degree-2 via SVD-truncated DMD (doc §4: raw EDMD rollouts blow up on the
    # spurious |lambda|>1 noise modes; truncation is the standard fix)
    Psi_X, Psi_Y = build_snapshots(d, Z_t, Z_tp1)
    K_tilde, U_svd, _, _ = fit_dmd_reduced(Psi_X, Psi_Y, rank=30)
    K_all = full_operator(K_tilde, U_svd)
    koop2 = fidelity_curves_over_debates(K_all, d, test_flat, 5, relift=True)

    lam_fj, W_fj, _ = fit_fj(train)
    fj = baseline_curve(lambda x, x0: lam_fj * (W_fj @ x) + (1 - lam_fj) * x0,
                        test, 5)
    W_dg = fit_degroot(train)
    dg = baseline_curve(lambda x, x0: W_dg @ x, test, 5)

    print(f"  horizon           : {list(range(1, 6))}")
    print(f"  Koopman d=1       : {np.round(koop1, 3)}")
    print(f"  Koopman d=2 (SVD) : {np.round(koop2, 3)}")
    print(f"  FJ (fitted)       : {np.round(fj, 3)}   (lam={lam_fj:.2f})")
    print(f"  DeGroot           : {np.round(dg, 3)}")

    # ---------- 3. Early warning from first EARLY rounds ----------
    print("=" * 60)
    print(f"3. Early warning from rounds 0..{EARLY} (shared-K mode ratios)")
    Z_te, Z_tpe = pooled_snapshots([X[:EARLY + 1] for X in train])
    K_early, _, _ = fit_edmd(*build_snapshots(d, Z_te, Z_tpe), reg=1e-4)
    _, _, W_early = spectral_decomposition(K_early)
    scores = []
    for X in test:
        r = per_debate_mode_ratios(W_early, d, flat(X)[:EARLY + 1], n_modes=3)
        scores.append(r[1])                    # effective |lambda_2| of this debate
    scores = np.array(scores)
    auc, fpr, tpr, _ = evaluate_auc(scores, test_labels)
    print(f"  per-debate effective |lam2|: truth mean = "
          f"{scores[test_labels == 0].mean():.3f}, herding mean = "
          f"{scores[test_labels == 1].mean():.3f}")
    print(f"  AUC (herding detection) = {auc:.3f}")

    try:
        from src.utils.viz import plot_fidelity, plot_roc, plot_spectrum
        eigvals_all, _, _ = spectral_decomposition(K_all)
        plot_spectrum(eigvals_all, os.path.join(OUT, "spectrum.png"))
        plot_fidelity({"Koopman d=1": koop1, "Koopman d=2 (SVD)": koop2,
                       "FJ": fj, "DeGroot": dg},
                      os.path.join(OUT, "fidelity.png"))
        plot_roc(fpr, tpr, auc, os.path.join(OUT, "roc.png"))
        print(f"  plots -> {OUT}/")
    except Exception as e:
        print(f"  (plots skipped: {e})")


if __name__ == "__main__":
    main()

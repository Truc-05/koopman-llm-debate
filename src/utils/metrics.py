# src/utils/metrics.py
import numpy as np


def multistep_prediction_error(pred_traj, true_traj):
    """Per-step L2 error between (T, D) trajectories -> (T,) array."""
    pred = np.asarray(pred_traj)
    true = np.asarray(true_traj)
    return np.linalg.norm(pred - true, axis=-1)


def fidelity_curve(K, dictionary, Z, max_horizon, relift=True):
    """The make-or-break check (doc §6.2): does K^t predict held-out rounds?

    Z: (T, D) one held-out trajectory. From every start t0, roll out up to
    max_horizon and collect the state-space error per horizon. Returns (H,)
    mean error at horizons 1..H (NaN where no sample)."""
    from src.koopman.edmd import rollout_states
    Z = np.atleast_2d(Z)
    T = Z.shape[0]
    errs = [[] for _ in range(max_horizon)]
    for t0 in range(T - 1):
        H = min(max_horizon, T - 1 - t0)
        pred = rollout_states(K, dictionary, Z[t0], H, relift=relift)
        for h in range(1, H + 1):
            errs[h - 1].append(np.linalg.norm(np.real(pred[h]) - Z[t0 + h]))
    return np.array([np.mean(e) if e else np.nan for e in errs])


def fidelity_curves_over_debates(K, dictionary, Z_list, max_horizon, relift=True):
    """Mean fidelity curve across a list of held-out debates."""
    curves = np.stack([fidelity_curve(K, dictionary, Z, max_horizon, relift)
                       for Z in Z_list])
    return np.nanmean(curves, axis=0)


def dictionary_sensitivity(errors_by_dict):
    means = {k: float(np.mean(v)) for k, v in errors_by_dict.items()}
    stds = {k: float(np.std(v)) for k, v in errors_by_dict.items()}
    return means, stds


def herding_rate(labels):
    return float(np.mean(np.asarray(labels)))


def accuracy_from_predictions(preds, targets):
    return float(np.mean(np.asarray(preds) == np.asarray(targets)))


def token_cost_summary(costs):
    costs = np.asarray(costs)
    return {"mean": float(np.mean(costs)), "total": float(np.sum(costs)),
            "std": float(np.std(costs))}

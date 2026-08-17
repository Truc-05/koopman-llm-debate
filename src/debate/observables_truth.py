# src/debate/observables_truth.py
import numpy as np
from src.debate.state import mean_belief, split_state


def h_star_hard(z, n_agents, n_answers, a_star):
    """1 if the mean belief currently ranks the true answer first."""
    p_bar = mean_belief(z, n_agents, n_answers)
    return 1.0 if np.argmax(p_bar) == a_star else 0.0


def h_star_soft(z, n_agents, n_answers, a_star):
    """Mean probability mass on the true answer (smooth version)."""
    p_bar = mean_belief(z, n_agents, n_answers)
    return float(p_bar[a_star])


def agreement_index(z, n_agents, n_answers):
    """Fraction of agents voting for the current plurality answer."""
    _, p = split_state(z, n_agents, n_answers)
    votes = np.argmax(p, axis=1)
    counts = np.bincount(votes, minlength=n_answers)
    return counts.max() / n_agents


def order_alignment(z, n_agents, n_answers, first_speaker_answer):
    """Anchor observable: mean mass on the first speaker's initial answer."""
    p_bar = mean_belief(z, n_agents, n_answers)
    return float(p_bar[first_speaker_answer])


def along_trajectory(fn, Z, **kwargs):
    """Evaluate a scalar observable along a (T, D) trajectory -> (T,) array."""
    return np.array([fn(z, **kwargs) for z in np.atleast_2d(Z)])

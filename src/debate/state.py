# src/debate/state.py
import numpy as np

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

def build_state(logits_per_agent):
    return np.concatenate([x.reshape(-1) for x in logits_per_agent])

def split_state(z, n_agents, n_answers):
    x = z.reshape(n_agents, n_answers)
    p = softmax(x)
    return x, p

def mean_belief(z, n_agents, n_answers):
    _, p = split_state(z, n_agents, n_answers)
    return p.mean(axis=0)
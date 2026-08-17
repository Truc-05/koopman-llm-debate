# src/control/rl_policy.py
import numpy as np


class KoopmanWarmStartPolicy:
    def __init__(self, lqr_gain, psi_target):
        self.lqr_gain = lqr_gain
        self.psi_target = psi_target

    def act(self, psi_state, spectral_features=None):
        return -self.lqr_gain @ (psi_state - self.psi_target)


class QLearningRefiner:
    """Tabular refinement on top of the Koopman-LQR warm start.

    select_action returns (idx, action) so that idx feeds update() directly."""

    def __init__(self, action_space, lr=0.1, gamma=0.9, epsilon=0.1, seed=0):
        self.Q = {}
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def _key(self, state):
        return tuple(np.round(np.asarray(state), 2))

    def _ensure(self, key, warm_start_action=None):
        if key not in self.Q:
            self.Q[key] = np.zeros(len(self.action_space))
            if warm_start_action is not None:
                idx = int(np.argmin([
                    np.linalg.norm(np.asarray(a) - warm_start_action)
                    for a in self.action_space
                ]))
                self.Q[key][idx] = 1.0

    def select_action(self, state, warm_start_action=None):
        key = self._key(state)
        self._ensure(key, warm_start_action)
        if self.rng.random() < self.epsilon:
            idx = int(self.rng.integers(len(self.action_space)))
        else:
            idx = int(np.argmax(self.Q[key]))
        return idx, self.action_space[idx]

    def update(self, state, action_idx, reward, next_state):
        key = self._key(state)
        nkey = self._key(next_state)
        self._ensure(key)
        self._ensure(nkey)
        best_next = np.max(self.Q[nkey])
        self.Q[key][action_idx] += self.lr * (
            reward + self.gamma * best_next - self.Q[key][action_idx])

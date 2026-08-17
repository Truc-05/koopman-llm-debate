# src/koopman/dictionary.py
import numpy as np


class PolynomialDictionary:
    """[1, z, {z_i z_j}] up to `degree`. `state_slice` marks where z sits in Psi(z)
    so states can be decoded back from lifted vectors (set on first transform)."""

    def __init__(self, degree=2):
        self.degree = degree
        self.state_slice = None

    def transform(self, Z):
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        S, D = Z.shape
        self.state_slice = slice(1, 1 + D)
        feats = [np.ones((S, 1)), Z]
        if self.degree >= 2:
            for i in range(D):
                for j in range(i, D):
                    feats.append((Z[:, i] * Z[:, j]).reshape(-1, 1))
        return np.concatenate(feats, axis=1)


class RBFDictionary:
    """[1, z, exp(-gamma ||z - c_m||^2)]."""

    def __init__(self, centers, gamma=1.0):
        self.centers = np.atleast_2d(centers)
        self.gamma = gamma
        self.state_slice = slice(1, 1 + self.centers.shape[1])

    def transform(self, Z):
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        S = Z.shape[0]
        M = self.centers.shape[0]
        out = np.zeros((S, M + Z.shape[1] + 1))
        out[:, 0] = 1.0
        out[:, 1:1 + Z.shape[1]] = Z
        diff = Z[:, None, :] - self.centers[None, :, :]
        sqd = np.sum(diff ** 2, axis=2)
        out[:, 1 + Z.shape[1]:] = np.exp(-self.gamma * sqd)
        return out


def rbf_centers_from_data(Z, n_centers, seed=0):
    """Pick RBF centers by subsampling observed states (cheap k-means substitute)."""
    Z = np.atleast_2d(Z)
    rng = np.random.default_rng(seed)
    idx = rng.choice(Z.shape[0], size=min(n_centers, Z.shape[0]), replace=False)
    return Z[idx]


class LearnedDictionary:
    """Wrap an arbitrary encoder. Pass state_slice if the encoder embeds z linearly
    somewhere (needed for decoding / fidelity in state space)."""

    def __init__(self, encoder_fn, state_slice=None):
        self.encoder_fn = encoder_fn
        self.state_slice = state_slice

    def transform(self, Z):
        return self.encoder_fn(np.atleast_2d(Z))

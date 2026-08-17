# src/utils/viz.py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_spectrum(eigvals, path="spectrum.png", title="Koopman spectrum"):
    """Eigenvalues on the complex plane with the unit circle."""
    fig, ax = plt.subplots(figsize=(5, 5))
    th = np.linspace(0, 2 * np.pi, 256)
    ax.plot(np.cos(th), np.sin(th), "k--", lw=0.8)
    sc = ax.scatter(np.real(eigvals), np.imag(eigvals),
                    c=np.abs(eigvals), cmap="viridis", zorder=3)
    fig.colorbar(sc, ax=ax, label="|lambda|")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_fidelity(curves, path="fidelity.png", title="Multi-step prediction error"):
    """curves: dict name -> (H,) mean error per horizon."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, err in curves.items():
        ax.plot(np.arange(1, len(err) + 1), err, marker="o", label=name)
    ax.set_xlabel("horizon (rounds)")
    ax.set_ylabel("state RMSE")
    ax.set_title(title)
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc(fpr, tpr, auc, path="roc.png", title="Early-warning ROC"):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

import os
import numpy as np
import matplotlib.pyplot as plt

from src.experiments import run_many_runs
from src.optimizers import get_cmaes, get_de


# =========================================================
# DATA PROCESSING
# =========================================================

def to_matrix(curves):
    """
    Converts list of convergence curves into:
    shape = (n_runs, min_length)
    """

    min_len = min(len(c) for c in curves)

    mat = np.stack([
        np.asarray(c[:min_len], dtype=float).reshape(-1)
        for c in curves
    ])

    return mat


def mean_std(mat):
    """
    Returns 1D mean and std over runs
    """

    mat = np.asarray(mat, dtype=float)

    mean = np.mean(mat, axis=0).reshape(-1)
    std = np.std(mat, axis=0).reshape(-1)

    return mean, std


# =========================================================
# PLOTTING
# =========================================================

def plot_mean_std(ax, mean, std, label):
    """
    Safe plotting with full shape enforcement
    """

    mean = np.asarray(mean, dtype=float).reshape(-1)
    std = np.asarray(std, dtype=float).reshape(-1)

    x = np.arange(mean.shape[0])

    ax.plot(x, mean, label=label)
    ax.fill_between(
        x,
        mean - std,
        mean + std,
        alpha=0.2
    )


# =========================================================
# MAIN EXPERIMENT PLOT
# =========================================================

def plot_cmaes_vs_de_convergence(dims=range(2, 26, 2), n_runs=10):

    fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharey=True)
    axes = axes.flatten()

    for i, n in enumerate(dims):

        # -------------------------
        # CMA-ES
        # -------------------------
        cma_curves = run_many_runs(n, get_cmaes, n_runs=n_runs)
        cma_mat = to_matrix(cma_curves)
        cma_mean, cma_std = mean_std(cma_mat)

        # -------------------------
        # DE
        # -------------------------
        de_curves = run_many_runs(n, get_de, n_runs=n_runs)
        de_mat = to_matrix(de_curves)
        de_mean, de_std = mean_std(de_mat)

        # -------------------------
        # PLOT
        # -------------------------
        ax = axes[i]

        plot_mean_std(ax, cma_mean, cma_std, "CMA-ES")
        plot_mean_std(ax, de_mean, de_std, "DE")

        ax.set_title(f"Dimension = {n}")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Best Objective")
        ax.legend()

    # turn off extra axes
    for j in range(len(dims), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    # -------------------------
    # SAVE FIGURE
    # -------------------------
    os.makedirs("results/figures", exist_ok=True)

    plt.savefig(
        "results/figures/convergence_grid.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
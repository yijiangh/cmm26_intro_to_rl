#!/usr/bin/env python3
"""Generate slide-ready visuals for policy-gradient variance reduction.

The figures are intentionally schematic rather than tied to one exact run.
They are designed to illustrate:
1. Why subtracting a baseline recenters trajectory weights.
2. Why a baseline can reduce gradient-estimate variance without changing the mean.
3. How constant, time-dependent, and state-dependent baselines differ.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "generated_figures" / "variance_reduction"


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "axes.facecolor": "#fbfbf8",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def ensure_outdir() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str) -> Path:
    path = OUTDIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def make_weight_centering_figure() -> Path:
    """Trajectory returns vs centered advantages."""
    rng = np.random.default_rng(4)
    n = 14
    returns = np.array([2, 3, 4, 5, 5, 6, 7, 7, 8, 8, 9, 12, 15, 23], dtype=float)
    returns += rng.normal(0.0, 0.35, size=n)
    baseline = returns.mean()
    advantages = returns - baseline
    x = np.arange(1, n + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)

    left = axes[0]
    left.scatter(x, returns, s=85, color="#d95f02", edgecolors="black", linewidths=0.6, zorder=3)
    left.axhline(baseline, color="#666666", linestyle="--", linewidth=1.5, label=f"batch mean baseline = {baseline:.1f}")
    for xi, yi in zip(x, returns):
        left.plot([xi, xi], [baseline, yi], color="#bbbbbb", linewidth=1.0, zorder=1)
    left.set_title("Raw REINFORCE weights")
    left.set_xlabel("trajectory in batch")
    left.set_ylabel("trajectory return  $R(\\tau)$")
    left.text(
        0.02,
        0.98,
        "All weights are positive.\nOne high-return outlier dominates.",
        transform=left.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc"},
    )
    left.legend(frameon=False, loc="upper left")

    right = axes[1]
    colors = np.where(advantages >= 0, "#1b9e77", "#7570b3")
    right.scatter(x, advantages, s=85, c=colors, edgecolors="black", linewidths=0.6, zorder=3)
    right.axhline(0.0, color="#333333", linewidth=1.5)
    for xi, yi in zip(x, advantages):
        right.plot([xi, xi], [0.0, yi], color="#bbbbbb", linewidth=1.0, zorder=1)
    right.set_title("After subtracting a baseline")
    right.set_xlabel("trajectory in batch")
    right.set_ylabel("centered weight  $R(\\tau)-b$")
    right.text(
        0.02,
        0.98,
        "Weights are recentered around zero.\nGood trajectories stay positive;\nbad ones become negative.",
        transform=right.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc"},
    )

    fig.suptitle("Variance reduction intuition: recenter the trajectory weights", fontsize=18, fontweight="bold")
    fig.tight_layout()
    return save(fig, "variance_weight_centering.png")


def draw_mean_arrow(ax: plt.Axes, origin: np.ndarray, target: np.ndarray, color: str) -> None:
    arrow = FancyArrowPatch(
        posA=origin,
        posB=target,
        arrowstyle="-|>",
        mutation_scale=20,
        linewidth=2.6,
        color=color,
        zorder=4,
    )
    ax.add_patch(arrow)


def make_gradient_variance_figure() -> Path:
    """Toy 2D gradient samples with same mean and different covariance."""
    rng = np.random.default_rng(7)
    mean = np.array([1.25, 0.85])
    raw = rng.multivariate_normal(mean, [[1.2, 0.55], [0.55, 0.9]], size=70)
    reduced = rng.multivariate_normal(mean, [[0.28, 0.08], [0.08, 0.2]], size=70)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    for ax, pts, title, color in [
        (axes[0], raw, "Raw estimator", "#d95f02"),
        (axes[1], reduced, "With baseline", "#1b9e77"),
    ]:
        ax.scatter(pts[:, 0], pts[:, 1], s=26, alpha=0.65, color=color, edgecolors="none")
        ax.add_patch(Circle(mean, radius=0.08, color="black", zorder=5))
        draw_mean_arrow(ax, np.zeros(2), mean, "black")
        ax.text(
            mean[0] + 0.12,
            mean[1] + 0.10,
            "mean gradient\n$\\mathbb{E}[\\hat g]$",
            ha="left",
            va="bottom",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc"},
        )
        ax.text(
            0.05,
            0.07,
            "Dots = sampled gradient estimates",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            color="#444444",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#dddddd"},
        )
        ax.set_title(title)
        ax.set_xlabel("gradient component 1")
        ax.grid(alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
        if ax is axes[0]:
            ax.set_ylabel("gradient component 2")

    axes[0].text(
        0.03,
        0.97,
        "Same expected update direction,\nbut noisy samples spread widely.",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc"},
    )
    axes[1].text(
        0.03,
        0.97,
        "Mean stays the same.\nSample cloud tightens: lower variance.",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc"},
    )

    for ax in axes:
        ax.set_xlim(-2.2, 3.2)
        ax.set_ylim(-1.6, 2.8)

    fig.suptitle("What the baseline really reduces: variance of the gradient estimate", fontsize=18, fontweight="bold")
    fig.tight_layout()
    return save(fig, "variance_gradient_cloud.png")


def make_baseline_types_figure() -> Path:
    """Constant vs time-dependent vs state-dependent baselines."""
    t = np.arange(0, 14)
    expected_return_curve = 9.5 - 0.55 * t + 0.4 * np.sin(t / 2.1)
    constant = np.full_like(t, expected_return_curve.mean(), dtype=float)

    fig = plt.figure(figsize=(13, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    ax0.plot(t, expected_return_curve, color="#1f78b4", linewidth=3, label="typical $\\mathbb{E}[G_t]$")
    ax0.plot(t, constant, color="#e31a1c", linestyle="--", linewidth=2.5, label="constant baseline")
    ax0.set_title("1. Constant baseline")
    ax0.set_xlabel("time step  $t$")
    ax0.set_ylabel("return scale")
    ax0.legend(frameon=False, loc="upper right")
    ax0.grid(alpha=0.25)
    ax0.text(0.03, 0.05, "One scalar for the whole batch / episode.", transform=ax0.transAxes)

    ax1.plot(t, expected_return_curve, color="#1f78b4", linewidth=3, label="typical $\\mathbb{E}[G_t]$")
    ax1.fill_between(t, expected_return_curve - 0.35, expected_return_curve + 0.35, color="#a6cee3", alpha=0.35)
    ax1.set_title("2. Time-dependent baseline")
    ax1.set_xlabel("time step  $t$")
    ax1.set_ylabel("return scale")
    ax1.grid(alpha=0.25)
    ax1.text(0.03, 0.05, "$b_t$ changes with time.\nUseful when early and late returns differ.", transform=ax1.transAxes)

    grid = np.array(
        [
            [7.5, 8.2, 9.0, 9.8],
            [6.6, 7.3, 8.1, 8.8],
            [5.4, 6.1, 6.9, 7.7],
            [4.0, 4.8, 5.6, 6.4],
        ]
    )
    im = ax2.imshow(grid, origin="lower", cmap="viridis")
    ax2.set_title("3. State-dependent baseline")
    ax2.set_xlabel("state feature 1")
    ax2.set_ylabel("state feature 2")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax2.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("$V(s)$ estimate")
    ax2.text(
        0.04,
        -0.16,
        "Best baseline in practice:\npredict expected return from the state.",
        transform=ax2.transAxes,
    )

    fig.suptitle("Increasingly informative baselines", fontsize=18, fontweight="bold")
    fig.tight_layout()
    return save(fig, "baseline_types.png")


def main() -> None:
    setup_matplotlib()
    ensure_outdir()
    outputs = [
        make_weight_centering_figure(),
        make_gradient_variance_figure(),
        make_baseline_types_figure(),
    ]
    print("Generated figures:")
    for path in outputs:
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

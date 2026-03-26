#!/usr/bin/env python3
"""Generate slide-ready visuals for likelihood-ratio policy gradients."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "generated_figures" / "likelihood_ratio"


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


def make_probability_shift_figure() -> Path:
    labels = [r"$\tau_1$", r"$\tau_2$", r"$\tau_3$", r"$\tau_4$", r"$\tau_5$"]
    returns = np.array([-2.0, 0.5, 5.0, -1.5, 3.0])
    before = np.array([0.18, 0.22, 0.20, 0.24, 0.16])
    after = np.array([0.09, 0.17, 0.35, 0.12, 0.27])
    x = np.arange(len(labels))
    colors = ["#c0392b" if r < 0 else "#1f78b4" if r < 2 else "#1b9e77" for r in returns]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=True)

    for ax, probs, title in [
        (axes[0], before, "Before policy update"),
        (axes[1], after, "After policy update"),
    ]:
        ax.bar(x, probs, color=colors, edgecolor="black", linewidth=0.8, width=0.68)
        ax.set_xticks(x, labels)
        ax.set_ylim(0.0, 0.42)
        ax.set_xlabel("trajectory identity")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for xi, yi, ret in zip(x, probs, returns):
            ax.text(xi, yi + 0.012, f"R={ret:+.1f}", ha="center", va="bottom", fontsize=10)

    axes[0].set_ylabel(r"trajectory probability  $P(\tau;\theta)$")
    axes[0].text(
        0.02,
        0.96,
        "These are fixed path types in the environment.\nThe update does not edit their geometry.",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc"},
    )
    axes[1].text(
        0.02,
        0.96,
        "Probability mass shifts toward good sampled paths.\nBad sampled paths become less likely next time.",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc"},
    )

    fig.suptitle("Likelihood-ratio update reweights trajectories, it does not redraw them", fontsize=18, fontweight="bold")
    fig.tight_layout()
    return save(fig, "trajectory_probability_shift.png")


def draw_box(ax: plt.Axes, xy: tuple[float, float], w: float, h: float, text: str, *, fc: str, ec: str, text_color: str = "black") -> None:
    rect = Rectangle(xy, w, h, facecolor=fc, edgecolor=ec, linewidth=2.0)
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", color=text_color, fontsize=13)


def make_factorization_figure() -> Path:
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(0.4, 5.2, r"$P(\tau;\theta)$", fontsize=26)
    ax.text(2.5, 5.2, r"$=$", fontsize=26)

    draw_box(ax, (3.1, 4.45), 2.0, 1.15, r"$\rho(s_0)$", fc="#ececec", ec="#999999")
    ax.text(5.4, 5.0, r"$\times$", fontsize=24)
    draw_box(ax, (6.0, 4.45), 2.4, 1.15, r"$\pi_\theta(a_0|s_0)$", fc="#fde0c5", ec="#d95f02")
    ax.text(8.7, 5.0, r"$\times$", fontsize=24)
    draw_box(ax, (9.2, 4.45), 2.7, 1.15, r"$P(s_1|s_0,a_0)$", fc="#ececec", ec="#999999")
    ax.text(12.25, 5.0, r"$\times \cdots \times$", fontsize=20)
    draw_box(ax, (14.25, 4.45), 2.4, 1.15, r"$\pi_\theta(a_t|s_t)$", fc="#fde0c5", ec="#d95f02")

    ax.text(1.1, 3.35, r"$\nabla_\theta \log P(\tau;\theta)$", fontsize=24)
    ax.text(6.0, 3.35, r"$=$", fontsize=24)
    draw_box(ax, (6.7, 2.7), 3.35, 1.15, r"$\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)$", fc="#fde0c5", ec="#d95f02")
    ax.text(10.45, 3.25, r"$+$", fontsize=24)
    draw_box(ax, (11.15, 2.7), 4.2, 1.15, r"$\nabla_\theta \log P(s_{t+1}|s_t,a_t)$", fc="#ececec", ec="#999999")
    ax.text(15.65, 3.25, r"$= 0$", fontsize=24)

    ax.text(
        0.5,
        1.55,
        "Only the policy terms depend on $\\theta$.\nThe environment transition model is fixed, so the gradient cannot 'change the path physics'.",
        fontsize=15,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#cccccc"},
    )
    ax.text(6.0, 0.45, "orange = can be changed by the policy", fontsize=13, color="#d95f02")
    ax.text(11.0, 0.45, "gray = fixed by the environment", fontsize=13, color="#666666")

    fig.suptitle("Why the likelihood-ratio gradient only changes path probabilities", fontsize=18, fontweight="bold")
    fig.tight_layout()
    return save(fig, "trajectory_factorization.png")


def make_future_rollout_figure() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), sharex=True, sharey=True)

    x = np.linspace(0, 10, 300)
    good_y = 0.55 * x + 0.65 * np.sin(0.7 * x)
    bad_y = -0.4 * x + 0.55 * np.sin(0.8 * x) + 2.2

    for ax, title, alpha_good, alpha_bad in [
        (axes[0], "Current rollouts under $\\pi_\\theta$", 0.55, 0.55),
        (axes[1], "Future rollouts under $\\pi_{\\theta+\\Delta\\theta}$", 0.95, 0.20),
    ]:
        ax.fill_between(x, good_y - 0.45, good_y + 0.45, color="#ccebc5", alpha=0.65 * alpha_good)
        ax.fill_between(x, bad_y - 0.45, bad_y + 0.45, color="#f4cccc", alpha=0.65 * alpha_bad)
        ax.plot(x, good_y, color="#1b9e77", linewidth=4, alpha=alpha_good)
        ax.plot(x, bad_y, color="#c0392b", linewidth=4, alpha=alpha_bad)
        ax.scatter([0], [0], s=80, color="black", zorder=5)
        ax.text(0.12, 0.15, "start", fontsize=11)
        ax.text(9.0, good_y[-1] + 0.25, "high return", color="#1b9e77", fontsize=12, fontweight="bold")
        ax.text(8.1, bad_y[-1] - 0.55, "low return", color="#c0392b", fontsize=12, fontweight="bold")
        ax.set_title(title)
        ax.set_xlabel("state feature 1")
        ax.grid(alpha=0.18)

    axes[0].set_ylabel("state feature 2")
    axes[0].text(
        0.03,
        0.97,
        "Both path families are possible.\nWe sampled one good and one bad trajectory.",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc"},
    )
    axes[1].text(
        0.03,
        0.97,
        "The update makes good paths more likely next time.\nThe actual sampled path from the last rollout is unchanged.",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc"},
    )
    arrow = FancyArrowPatch((0.48, 0.52), (0.54, 0.52), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=18, linewidth=2.2, color="#444444")
    fig.patches.append(arrow)

    fig.suptitle("Policy gradient changes what future rollouts are likely to look like", fontsize=18, fontweight="bold")
    fig.tight_layout()
    return save(fig, "future_rollout_reweighting.png")


def main() -> None:
    setup_matplotlib()
    ensure_outdir()
    outputs = [
        make_probability_shift_figure(),
        make_factorization_figure(),
        make_future_rollout_figure(),
    ]
    print("Generated figures:")
    for path in outputs:
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

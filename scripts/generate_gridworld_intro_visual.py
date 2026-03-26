#!/usr/bin/env python3
"""Generate a slide-style gridworld overview visual for the notebook."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "generated_figures" / "gridworld"
OUTPATH = OUTDIR / "gridworld_goal_policy_intro.png"


def draw_main_grid(ax):
    ax.set_xlim(-0.4, 4.1)
    ax.set_ylim(-0.2, 3.4)
    ax.set_aspect("equal")
    ax.axis("off")

    base_fc = "#7d766a"
    wall_fc = "#c8c2bc"
    pit_fc = "#7b5c4b"

    for x in range(4):
        for y in range(3):
            fc = base_fc
            if (x, y) == (1, 1):
                fc = wall_fc
            if (x, y) == (3, 1):
                fc = pit_fc
            ax.add_patch(
                Rectangle((x, y), 1, 1, facecolor=fc, edgecolor="#5c554c", linewidth=1.6)
            )

    for x in range(1, 5):
        ax.text(x - 0.5, -0.08, f"{x}", ha="center", va="top", fontsize=12, fontweight="bold")
    for y in range(1, 4):
        ax.text(-0.08, y - 0.5, f"{y}", ha="right", va="center", fontsize=12, fontweight="bold")

    ax.text(0.15, 0.28, "START", fontsize=18, color="#2a221d", fontstyle="italic")

    ax.text(1.5, 1.5, " ", ha="center", va="center")
    ax.text(3.52, 1.52, "-1", color="#ffb347", fontsize=30, fontweight="bold", ha="center", va="center")
    ax.text(3.52, 2.55, "+1", color="#cfe0ff", fontsize=22, fontweight="bold", ha="center", va="center")

    # Wall sketch lines.
    ax.plot([1.05, 1.18, 1.1, 1.04], [1.8, 1.9, 2.02, 1.95], color="#867e78", lw=1.4)
    ax.plot([1.92, 2.03, 2.0, 1.95], [1.88, 1.98, 1.78, 1.7], color="#867e78", lw=1.4)
    ax.plot([1.15, 1.88], [1.03, 1.05], color="#867e78", lw=1.4)

    # Pit burst.
    burst = [
        (3.08, 1.18), (3.22, 1.34), (3.08, 1.55), (3.26, 1.72), (3.10, 1.95),
        (3.34, 2.00), (3.46, 2.20), (3.63, 2.01), (3.88, 2.06), (3.80, 1.82),
        (3.98, 1.70), (3.82, 1.55), (3.97, 1.34), (3.73, 1.30), (3.65, 1.08),
        (3.46, 1.22), (3.24, 1.10),
    ]
    ax.add_patch(Polygon(burst, closed=True, facecolor="#ff9d3c", edgecolor="#e46a21", linewidth=2.2))

    # Diamond goal.
    diamond = [(3.52, 2.95), (3.74, 2.72), (3.52, 2.46), (3.30, 2.72)]
    ax.add_patch(Polygon(diamond, closed=True, facecolor="#2f77ff", edgecolor="#2855b8", linewidth=2.0))

    # Agent body.
    ax.add_patch(plt.Circle((2.52, 0.55), 0.18, facecolor="#c9d2d8", edgecolor="#59646f", lw=2.0))
    ax.add_patch(plt.Circle((2.52, 0.55), 0.05, facecolor="#c53030", edgecolor="#7a1717", lw=1.0))
    ax.add_patch(Polygon([(2.33, 0.72), (2.18, 0.86), (2.22, 0.58)], closed=True,
                         facecolor="#8f9aa4", edgecolor="#59646f", lw=1.6))
    ax.add_patch(Polygon([(2.71, 0.72), (2.86, 0.86), (2.82, 0.58)], closed=True,
                         facecolor="#8f9aa4", edgecolor="#59646f", lw=1.6))

    green = "#73f542"
    for start, end in [
        ((2.52, 0.82), (2.52, 1.48)),
        ((2.28, 0.55), (1.86, 0.55)),
        ((2.76, 0.55), (3.18, 0.55)),
    ]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="simple", mutation_scale=28,
                                     facecolor=green, edgecolor="#4a8b2c", lw=1.2))


def draw_goal_box(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.2)
    ax.axis("off")
    ax.add_patch(Rectangle((0.25, 0.2), 9.5, 1.7, facecolor="white", edgecolor="black", linewidth=2.0))
    ax.text(0.8, 1.05, "Goal:", fontsize=26, fontweight="bold", va="center")
    eq = r"$\max_{\pi}\,\mathbb{E}\!\left[\sum_{t=0}^{H}\gamma^t\,R(S_t,A_t,S_{t+1}) \mid \pi \right]$"
    ax.text(5.7, 1.02, eq, fontsize=28, ha="center", va="center")


def draw_policy_sheet(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    ax.text(0.55, 2.35, r"$\pi$:", fontsize=38, fontweight="bold", va="center")

    paper = Polygon(
        [(3.0, 0.7), (8.4, 0.7), (8.0, 3.2), (3.2, 3.2)],
        closed=True,
        facecolor="#efe5c8",
        edgecolor="#b6aa86",
        linewidth=2.0,
    )
    ax.add_patch(paper)

    x0, y0, w, h = 3.45, 1.0, 4.1, 1.85
    for i in range(5):
        ax.plot([x0 + i * w / 4, x0 + i * w / 4], [y0, y0 + h], color="#8a836b", lw=1.5)
    for j in range(4):
        ax.plot([x0, x0 + w], [y0 + j * h / 3, y0 + j * h / 3], color="#8a836b", lw=1.5)

    # Wall hatch.
    ax.add_patch(Rectangle((x0 + w / 4, y0 + h / 3), w / 4, h / 3,
                           facecolor="#948c77", edgecolor="#5d584a", linewidth=1.4, hatch="////"))

    arrow_color = "#55c43d"
    def cell_center(ix, iy):
        return (x0 + (ix + 0.5) * w / 4, y0 + (iy + 0.5) * h / 3)

    directions = {
        (0, 0): (0.35, 0.0), (1, 0): (0.35, 0.0), (2, 0): (0.0, 0.35), (3, 0): (-0.35, 0.0),
        (0, 1): (0.0, 0.35), (2, 1): (0.0, 0.35),
        (0, 2): (0.35, 0.0), (1, 2): (0.35, 0.0), (2, 2): (0.35, 0.0),
    }
    for (ix, iy), (dx, dy) in directions.items():
        cx, cy = cell_center(ix, iy)
        ax.add_patch(FancyArrowPatch((cx - dx * 0.35, cy - dy * 0.35), (cx + dx * 0.35, cy + dy * 0.35),
                                     arrowstyle="simple", mutation_scale=16,
                                     facecolor=arrow_color, edgecolor="#4a8b2c", lw=1.0))

    # Goal / pit icons.
    dx, dy = cell_center(3, 2)
    ax.add_patch(Polygon([(dx, dy + 0.34), (dx + 0.22, dy + 0.06), (dx, dy - 0.22), (dx - 0.22, dy + 0.06)],
                         closed=True, facecolor="#2f77ff", edgecolor="#2855b8", linewidth=1.5))
    ax.text(dx, dy + 0.05, "+", color="#dce8ff", fontsize=15, fontweight="bold", ha="center", va="center")

    px, py = cell_center(3, 1)
    burst = [
        (px - 0.22, py - 0.11), (px - 0.12, py + 0.02), (px - 0.23, py + 0.14),
        (px - 0.03, py + 0.16), (px + 0.05, py + 0.31), (px + 0.14, py + 0.15),
        (px + 0.30, py + 0.18), (px + 0.20, py + 0.02), (px + 0.31, py - 0.12),
        (px + 0.10, py - 0.10), (px + 0.02, py - 0.26), (px - 0.07, py - 0.11),
    ]
    ax.add_patch(Polygon(burst, closed=True, facecolor="#ff9d3c", edgecolor="#e46a21", linewidth=1.6))
    ax.text(px, py + 0.02, "☠", color="#ffefcc", fontsize=17, ha="center", va="center")

    # Hands.
    left_hand = Polygon([(2.6, 0.8), (3.0, 0.8), (3.0, 1.5), (2.65, 1.7), (2.45, 1.3)],
                        closed=True, facecolor="#59646f", edgecolor="#2f3942", linewidth=1.6)
    right_hand = Polygon([(8.45, 0.85), (8.8, 1.05), (8.65, 1.7), (8.2, 1.45), (8.18, 0.9)],
                         closed=True, facecolor="#59646f", edgecolor="#2f3942", linewidth=1.6)
    ax.add_patch(left_hand)
    ax.add_patch(right_hand)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 10), facecolor="white")
    gs = fig.add_gridspec(3, 1, height_ratios=[4.9, 1.8, 3.0], hspace=0.15)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    draw_main_grid(ax1)
    draw_goal_box(ax2)
    draw_policy_sheet(ax3)

    fig.savefig(OUTPATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(OUTPATH.relative_to(ROOT))


if __name__ == "__main__":
    main()

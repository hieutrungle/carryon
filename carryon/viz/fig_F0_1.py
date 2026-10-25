"""F0.1 — System architecture: the acting loop, then logging, learning and reporting.

Each box names the package modules that implement it. Reads no runs.
Usage: python carryon/viz/fig_F0_1.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from carryon.viz.common import FIGURES_DIR, FONT_SIZES, apply_style, save_figure

FIG_ID = "F0.1"
BASE, SECONDARY, SMALL = FONT_SIZES
INK = "#333333"

# The context manager is the object of study, so it alone gets the saturated colour.
FOCAL = ("#fbe3c9", "#c8641a")
ACTING = ("#dfe9f5", "#3f6fa6")
LEARNING = ("#ececec", "#6b6b6b")
BOX_W, BOX_H = 2.8, 1.8
TOP, BOTTOM = 4.7, 1.3  # row centres
COLUMNS = (1.6, 5.8, 10.0, 14.2)


@dataclass(frozen=True, slots=True, kw_only=True)
class Box:
    x: float  # centre
    y: float
    title: str
    detail: str
    colors: tuple[str, str]

    def left(self) -> tuple[float, float]:
        return (self.x - BOX_W / 2, self.y)

    def right(self) -> tuple[float, float]:
        return (self.x + BOX_W / 2, self.y)

    def top(self, dx: float = 0.0) -> tuple[float, float]:
        return (self.x + dx, self.y + BOX_H / 2)

    def bottom(self, dx: float = 0.0) -> tuple[float, float]:
        return (self.x + dx, self.y - BOX_H / 2)


QUESTIONS = Box(
    x=COLUMNS[0], y=TOP, title="Questions", detail="multi-hop QA\ntrain / dev / test", colors=ACTING
)
ENV = Box(
    x=COLUMNS[1],
    y=TOP,
    title="Environment",
    detail="BM25 search, open,\nanswer, reward\n(envs)",
    colors=ACTING,
)
CONTEXT = Box(
    x=COLUMNS[2],
    y=TOP,
    title="Context manager",
    detail="active-token budget\nappend · truncate\nsummarize · fold\n(context)",
    colors=FOCAL,
)
POLICY = Box(
    x=COLUMNS[3],
    y=TOP,
    title="Policy model",
    detail="Qwen3.5-4B + LoRA\nserved by vLLM\n(agent, rollout)",
    colors=ACTING,
)
REPORT = Box(
    x=COLUMNS[1],
    y=BOTTOM,
    title="Figures and report",
    detail="one script per figure ID\nliving report\n(eval, viz)",
    colors=LEARNING,
)
LOG = Box(
    x=COLUMNS[2],
    y=BOTTOM,
    title="Trajectory log",
    detail="runs/<run_id>/\ntrajectories, run card\n(rollout, run)",
    colors=LEARNING,
)
TRAINER = Box(
    x=COLUMNS[3],
    y=BOTTOM,
    title="Trainer",
    detail="SFT: Unsloth\nGRPO: TRL\n(train)",
    colors=LEARNING,
)


def draw_box(ax: Axes, box: Box) -> None:
    fill, edge = box.colors
    ax.add_patch(
        FancyBboxPatch(
            (box.x - BOX_W / 2, box.y - BOX_H / 2),
            BOX_W,
            BOX_H,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.2,
        )
    )
    ax.text(
        box.x,
        box.y + 0.58,
        box.title,
        ha="center",
        va="center",
        fontsize=BASE,
        fontweight="bold",
        color="#222222",
    )
    ax.text(
        box.x,
        box.y + 0.3,
        box.detail,
        ha="center",
        va="top",
        fontsize=SECONDARY,
        color=INK,
        linespacing=1.35,
    )


def arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
    label_xy: tuple[float, float],
    ha: str = "center",
    curve: float = 0.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.1,
            color="#444444",
            connectionstyle=f"arc3,rad={curve}",
            shrinkA=2,
            shrinkB=2,
        )
    )
    ax.text(*label_xy, label, ha=ha, va="bottom", fontsize=SMALL, color="#444444", linespacing=1.2)


def between(a: Box, b: Box) -> tuple[float, float]:
    """Label position just above a horizontal arrow from a to b."""
    return ((a.x + b.x) / 2, a.y + 0.14)


def draw(ax: Axes) -> None:
    for box in (QUESTIONS, ENV, CONTEXT, POLICY, REPORT, LOG, TRAINER):
        draw_box(ax, box)

    # Acting loop: observation -> context manager -> policy; the action returns to the env.
    arrow(ax, QUESTIONS.right(), ENV.left(), "question", between(QUESTIONS, ENV))
    arrow(ax, ENV.right(), CONTEXT.left(), "observation", between(ENV, CONTEXT))
    arrow(ax, CONTEXT.right(), POLICY.left(), "active\ncontext", between(CONTEXT, POLICY))
    arrow(
        ax,
        POLICY.top(-0.4),
        ENV.top(0.4),
        "action: tool call or answer",
        label_xy=(CONTEXT.x, 6.0),
        curve=0.24,
    )

    # Learning and reporting.
    arrow(
        ax,
        CONTEXT.bottom(),
        LOG.top(),
        "every episode,\nlogged once",
        label_xy=(CONTEXT.x - 0.15, 2.75),
        ha="right",
    )
    arrow(ax, LOG.right(), TRAINER.left(), "token-exact\nsequences", between(LOG, TRAINER))
    arrow(
        ax,
        TRAINER.top(),
        POLICY.bottom(),
        "LoRA\nupdate",
        label_xy=(TRAINER.x + 0.15, 2.75),
        ha="left",
    )
    arrow(ax, LOG.left(), REPORT.right(), "rebuilt\nfrom runs/", between(LOG, REPORT))

    ax.text(0.2, 6.85, "Acting loop", fontsize=BASE, fontweight="bold", color=ACTING[1])
    ax.text(
        0.2, 2.75, "Learning and reporting", fontsize=BASE, fontweight="bold", color=LEARNING[1]
    )
    ax.set_xlim(0.1, 15.7)
    ax.set_ylim(0.3, 7.5)
    ax.set_aspect("equal")
    ax.axis("off")


def main(out_dir: Path = FIGURES_DIR) -> list[Path]:
    apply_style()
    fig, ax = plt.subplots(figsize=(10.8, 4.9))
    draw(ax)
    return save_figure(fig, FIG_ID, run_ids=[], out_dir=out_dir)


if __name__ == "__main__":
    for path in main():
        print(path)

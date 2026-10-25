"""Shared helpers for figure scripts: one style, one way to save a figure and its inputs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figure scripts run headless (CI, Slurm, make figures)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from pydantic import TypeAdapter  # noqa: E402

FIGURES_DIR = Path("figures")

# Three font sizes, mapped to role: titles and labels, secondary text, small annotations.
FONT_SIZES = (9.0, 8.0, 7.0)

_RUN_IDS = TypeAdapter(list[str])


def apply_style() -> None:
    base, secondary, small = FONT_SIZES
    plt.rcParams.update(
        {
            "font.size": base,
            "axes.titlesize": base,
            "axes.labelsize": base,
            "legend.fontsize": secondary,
            "xtick.labelsize": small,
            "ytick.labelsize": small,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "svg.fonttype": "none",  # keep text as text in SVG
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def save_figure(
    fig: Figure, fig_id: str, run_ids: Sequence[str], out_dir: Path = FIGURES_DIR
) -> list[Path]:
    """Write <ID>.png, <ID>.svg and <ID>.inputs.json (the run ids read), then close the figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{fig_id}.png", out_dir / f"{fig_id}.svg"]
    for path in paths:
        fig.savefig(path)
    plt.close(fig)
    inputs = out_dir / f"{fig_id}.inputs.json"
    inputs.write_bytes(_RUN_IDS.dump_json(list(run_ids)))
    return [*paths, inputs]

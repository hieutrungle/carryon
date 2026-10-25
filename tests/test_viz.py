"""Figure scripts and the report build: F0.1 renders, and the report embeds what exists."""

import json
import shutil
from pathlib import Path

from carryon.viz import fig_F0_1
from carryon.viz.report import CHAPTERS, build_report

REPO = Path(__file__).resolve().parents[1]


def test_fig_F0_1_writes_png_svg_and_inputs(tmp_path: Path) -> None:
    paths = fig_F0_1.main(out_dir=tmp_path)
    assert sorted(p.name for p in paths) == ["F0.1.inputs.json", "F0.1.png", "F0.1.svg"]
    assert all(p.stat().st_size > 0 for p in paths)
    assert json.loads((tmp_path / "F0.1.inputs.json").read_text()) == []  # a diagram reads no runs


def test_report_embeds_existing_figures_and_marks_missing_ones(tmp_path: Path) -> None:
    report = tmp_path / "report"
    shutil.copytree(REPO / "report", report, ignore=shutil.ignore_patterns("build"))
    fig_F0_1.main(out_dir=tmp_path / "figures")  # chapters link ../figures/<ID>.png

    html = build_report(report).read_text()

    assert html.count("data:image/png;base64,") == 1  # only F0.1 exists
    assert 'alt="F0.1"' in html
    assert "Figure F2.1 not generated yet" in html
    for chapter in CHAPTERS:  # every chapter heading made it in
        title = (report / chapter).read_text().splitlines()[0].removeprefix("# ")
        assert title in html

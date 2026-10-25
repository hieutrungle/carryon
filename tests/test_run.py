"""Run directories and run cards, on a fake run with two logged episodes."""

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from test_trajectory import make_episode

from carryon.eval.run_card import main as run_card_main
from carryon.rollout.trajectory import append_episode, read_episodes
from carryon.run import RunDir, create_run, finish_run, load_run_info

START = datetime(2026, 10, 25, 15, 30, 12, tzinfo=UTC)


@pytest.fixture
def config(tmp_path: Path) -> Path:
    path = tmp_path / "grpo_append_toy_4k.yaml"
    path.write_text("model: toy\nbudget_tokens: 4096\nlearning_rate: 1.0e-5\n")
    return path


@pytest.fixture
def run(tmp_path: Path, config: Path) -> RunDir:
    return create_run(config, kind="grpo", seed=3, runs_root=tmp_path / "runs", now=START)


def test_create_run_layout(run: RunDir, config: Path) -> None:
    assert run.run_id == "20261025-153012-grpo_append_toy_4k-s3"
    assert run.config.read_text() == config.read_text()
    info = load_run_info(run)
    assert info.status == "running"
    assert info.seed == 3
    assert info.started_at == START
    assert len(info.config_sha256) == 64
    assert "pydantic" in info.packages


def test_create_run_never_reuses_a_directory(run: RunDir, config: Path, tmp_path: Path) -> None:
    with pytest.raises(FileExistsError):
        create_run(config, kind="grpo", seed=3, runs_root=tmp_path / "runs", now=START)


def test_finish_run_gzips_trajectories_and_writes_card(run: RunDir) -> None:
    episodes = [make_episode(), make_episode(episode_id="ep-1")]
    for episode in episodes:
        append_episode(run.trajectories, episode)

    info = finish_run(run, {"reward": 0.75, "exact_match": 0.5}, now=START + timedelta(minutes=90))

    assert not run.trajectories.exists()
    assert run.trajectories_file().name == "trajectories.jsonl.gz"
    assert list(read_episodes(run.trajectories_file())) == episodes
    assert info.status == "finished"
    assert load_run_info(run) == info
    card = run.card.read_text()
    assert run.run_id in card
    assert "| reward | 0.75 |" in card
    assert "90.0 min" in card
    assert "learning_rate: 1.0e-5" in card


def test_card_cli_links_figures_that_use_the_run(run: RunDir, tmp_path: Path) -> None:
    figures = tmp_path / "figures"
    figures.mkdir()
    (figures / "F4.1.inputs.json").write_text(json.dumps([run.run_id, "other-run"]))
    (figures / "F2.1.inputs.json").write_text(json.dumps(["other-run"]))

    run_card_main([str(run.path), "--figures-dir", str(figures)])

    card = run.card.read_text()
    assert "[F4.1](../../figures/F4.1.png)" in card
    assert "F2.1" not in card


def test_card_warns_on_uncommitted_changes(run: RunDir) -> None:
    info = load_run_info(run).model_copy(update={"git_dirty": True})
    run.info.write_text(info.model_dump_json())
    run_card_main([str(run.path)])
    assert "**Warning:** uncommitted changes" in run.card.read_text()


def test_module_entry_point(run: RunDir) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "carryon.eval.run_card", str(run.path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == str(run.card)

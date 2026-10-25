"""CPU-side pieces of the M0 training stack: configs, the toy task, and CLI overrides."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml
from pydantic import ValidationError

from carryon.config import GrpoExperiment, load_config
from carryon.train.toy_task import exact_answer_reward, make_prompts

REPO = Path(__file__).resolve().parents[1]
GRPO_CONFIGS = sorted((REPO / "configs").rglob("grpo_*.yaml"))


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # pydantic resolves the script's annotations through sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("path", GRPO_CONFIGS, ids=[p.name for p in GRPO_CONFIGS])
def test_grpo_configs_validate(path: Path) -> None:
    assert isinstance(load_config(path), GrpoExperiment)


def test_config_rejects_unknown_keys_and_partial_groups() -> None:
    data = yaml.safe_load((REPO / "configs/m0/grpo_toy_qwen35-4b.yaml").read_text())
    with pytest.raises(ValidationError, match="learning_rte"):
        GrpoExperiment.model_validate({**data, "grpo": {**data["grpo"], "learning_rte": 1e-5}})
    with pytest.raises(ValidationError, match="multiple of num_generations"):
        GrpoExperiment.model_validate(
            {**data, "grpo": {**data["grpo"], "per_device_train_batch_size": 6}}
        )


def test_toy_prompts_are_deterministic_and_correct() -> None:
    rows = make_prompts(n=50, max_operand=99, seed=7)
    assert rows == make_prompts(n=50, max_operand=99, seed=7)
    for row in rows:
        a, b = row["prompt"].removesuffix(" = ").split(" + ")
        assert int(a) + int(b) == int(row["answer"])


def test_exact_answer_reward() -> None:
    completions = ["42", " 42 because", "41", "", [{"role": "assistant", "content": "42."}]]
    assert exact_answer_reward(completions, answer=["42"] * 5) == [1.0, 1.0, 0.0, 0.0, 1.0]


def test_cli_overrides_reach_the_resolved_config() -> None:
    train_grpo = load_script("train_grpo")
    args = train_grpo.TrainArgs(
        config=REPO / "configs/m0/grpo_toy_qwen35-4b.yaml", seed=2, vllm_mode="server"
    )
    cfg = train_grpo.resolve(args)
    assert (cfg.seed, cfg.vllm.mode) == (2, "server")
    assert cfg.grpo == load_config(args.config).grpo  # nothing else changes


def test_final_metrics_keeps_last_numeric_values() -> None:
    train_grpo = load_script("train_grpo")
    history = [{"step": 1, "reward": 0.1}, {"step": 2, "reward": 0.4, "note": "x", "done": True}]
    assert train_grpo.final_metrics(history) == {"step": 2.0, "reward": 0.4}

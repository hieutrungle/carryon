"""GRPO training (M0: toy arithmetic task) with TRL + vLLM + LoRA, logged to runs/<run_id>/.

Usage (inside the container, from the repo root):
    python scripts/train_grpo.py --config configs/m0/grpo_toy_qwen35-4b.yaml
        [--seed 1] [--vllm-mode server] [--run-dir runs/<run_id>]

--run-dir resumes an existing run from its latest checkpoint; without it a new run is created.
TRL and PEFT are imported here, not in the carryon package, so CI runs without the GPU stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_settings import CliApp

from carryon.config import GrpoExperiment, VllmMode, load_config
from carryon.run import RunDir, create_run, finish_run
from carryon.train.toy_task import exact_answer_reward, make_prompts


class TrainArgs(BaseModel):
    """Train with GRPO on the experiment in --config."""

    config: Path
    seed: int | None = None  # overrides the config (seed arrays)
    vllm_mode: VllmMode | None = None  # overrides the config (job type)
    run_dir: Path | None = None  # resume this run instead of creating one
    runs_root: Path = Path("runs")

    def cli_cmd(self) -> None:
        train(self)


def resolve(args: TrainArgs) -> GrpoExperiment:
    cfg = load_config(args.config)
    updates: dict[str, Any] = {}
    if args.seed is not None:
        updates["seed"] = args.seed
    if args.vllm_mode is not None:
        updates["vllm"] = cfg.vllm.model_copy(update={"mode": args.vllm_mode})
    return cfg.model_copy(update=updates)


def train(args: TrainArgs) -> None:
    cfg = resolve(args)
    if args.run_dir is not None:
        run = RunDir(path=args.run_dir)
    else:
        run = create_run(args.config, kind="grpo", seed=cfg.seed, runs_root=args.runs_root)
    # Overrides are not in the copied YAML, so the config that actually ran is saved too.
    (run.path / "config.resolved.json").write_text(cfg.model_dump_json(indent=2) + "\n")
    checkpoints = run.path / "checkpoints"
    resume = any(checkpoints.glob("checkpoint-*"))
    print(f"run {run.run_id}: {'resuming' if resume else 'starting'}, vllm {cfg.vllm.mode}")

    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    task = cfg.task
    dataset = Dataset.from_list(make_prompts(task.n_prompts, task.max_operand, task.seed))
    g, v = cfg.grpo, cfg.vllm
    grpo_args = GRPOConfig(
        output_dir=str(checkpoints),
        seed=cfg.seed,
        learning_rate=g.learning_rate,
        max_steps=g.max_steps,
        per_device_train_batch_size=g.per_device_train_batch_size,
        gradient_accumulation_steps=g.gradient_accumulation_steps,
        num_generations=g.num_generations,
        max_completion_length=g.max_completion_length,
        temperature=g.temperature,
        beta=g.beta,
        loss_type=g.loss_type,
        logging_steps=g.logging_steps,
        bf16=True,
        model_init_kwargs={"dtype": "bfloat16"},
        use_vllm=True,
        vllm_mode=v.mode,
        vllm_gpu_memory_utilization=v.gpu_memory_utilization,
        vllm_max_model_length=v.max_model_len,
        vllm_importance_sampling_correction=v.importance_sampling_correction,
        vllm_server_base_url=f"http://127.0.0.1:{v.server_port}" if v.mode == "server" else None,
        save_steps=cfg.checkpoint.save_steps,
        save_total_limit=cfg.checkpoint.save_total_limit,
        report_to="none",  # runs/ is the source of truth; metrics.parquet is written below
    )
    lora = cfg.lora
    trainer = GRPOTrainer(
        model=cfg.model,
        reward_funcs=exact_answer_reward,
        args=grpo_args,
        train_dataset=dataset,
        peft_config=LoraConfig(
            r=lora.r,
            lora_alpha=lora.alpha,
            lora_dropout=lora.dropout,
            target_modules=lora.target_modules,
            task_type="CAUSAL_LM",
        ),
    )
    trainer.train(resume_from_checkpoint=True if resume else None)
    trainer.save_model(str(run.path / "adapter"))

    history = trainer.state.log_history
    write_metrics(run, history)
    finish_run(run, final_metrics(history))
    print(f"run {run.run_id}: finished at step {trainer.state.global_step}")


def write_metrics(run: RunDir, history: list[dict[str, Any]]) -> None:
    import pandas as pd

    rows = [row for row in history if "step" in row]
    pd.DataFrame(rows).to_parquet(run.metrics, index=False)


def final_metrics(history: list[dict[str, Any]]) -> dict[str, float]:
    """Last logged value of every numeric metric (reward, loss, step, train_runtime, ...)."""
    summary: dict[str, float] = {}
    for row in history:
        for key, value in row.items():
            if isinstance(value, int | float) and not isinstance(value, bool):
                summary[key] = float(value)
    return summary


if __name__ == "__main__":
    CliApp.run(TrainArgs)

"""Experiment configs: one YAML per run, validated here before anything touches a GPU.

Only what the M0 GRPO scripts need exists so far; M1+ add their own sections when they need them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

VllmMode = Literal["colocate", "server"]


class _Config(BaseModel):
    # A misspelled key in a YAML file must fail, not silently fall back to a default.
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToyArithmeticTask(_Config):
    """M0 stand-in task: "a + b = " prompts, reward 1 for the right sum (scripts only, no env)."""

    kind: Literal["toy_arithmetic"] = "toy_arithmetic"
    n_prompts: PositiveInt = 2048
    max_operand: PositiveInt = 99
    seed: int = 0  # dataset seed, independent of the training seed


class LoraSettings(_Config):
    r: PositiveInt = 16
    alpha: PositiveInt = 32
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    target_modules: str | list[str] = "all-linear"


class GrpoSettings(_Config):
    learning_rate: PositiveFloat
    max_steps: PositiveInt
    per_device_train_batch_size: PositiveInt
    gradient_accumulation_steps: PositiveInt = 1
    num_generations: PositiveInt
    max_completion_length: PositiveInt
    temperature: PositiveFloat = 1.0
    beta: NonNegativeFloat = 0.0  # KL weight; TRL's default is 0
    loss_type: str = "dapo"
    logging_steps: PositiveInt = 1

    @model_validator(mode="after")
    def _check_group_size(self) -> GrpoSettings:
        # TRL needs whole groups per optimizer step (single process; multi-GPU training is M4+).
        batch = self.per_device_train_batch_size * self.gradient_accumulation_steps
        if batch % self.num_generations:
            raise ValueError(f"batch {batch} is not a multiple of num_generations")
        return self


class VllmSettings(_Config):
    mode: VllmMode = "colocate"
    gpu_memory_utilization: float = Field(default=0.3, gt=0.0, le=1.0)  # colocate share
    server_gpu_memory_utilization: float = Field(default=0.9, gt=0.0, le=1.0)
    server_port: PositiveInt = 8000
    max_model_len: PositiveInt = 1024
    importance_sampling_correction: bool = True


class CheckpointSettings(_Config):
    save_steps: PositiveInt
    save_total_limit: PositiveInt = 2


class GrpoExperiment(_Config):
    kind: Literal["grpo"] = "grpo"
    model: str  # Hugging Face id; resolved from the offline cache on FASTER
    seed: int = 0
    task: ToyArithmeticTask
    lora: LoraSettings = LoraSettings()
    grpo: GrpoSettings
    vllm: VllmSettings = VllmSettings()
    checkpoint: CheckpointSettings


def load_config(path: Path) -> GrpoExperiment:
    return GrpoExperiment.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

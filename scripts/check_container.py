"""Container check (M0.9) and offline model/dataset load (M0.10), run inside the image with --nv.

Usage:
    python scripts/check_container.py                       # versions, GPU, unsloth import
    python scripts/check_container.py --model Qwen/Qwen3.5-4B
    python scripts/check_container.py --dataset ../carryon-data/datasets/flashrag/musique/dev.jsonl

Load one model per process: vLLM does not free GPU memory reliably within a process.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import CliApp

PACKAGES = ("torch", "vllm", "trl", "peft", "transformers", "unsloth", "flashinfer")


class CheckArgs(BaseModel):
    """Check the container's GPU stack, optionally load one model and read one dataset."""

    model: str | None = None
    dataset: Path | None = None
    max_model_len: int = 1024
    gpu_memory_utilization: float = 0.6

    def cli_cmd(self) -> None:
        check_stack()
        if self.model:
            check_model(self.model, self.max_model_len, self.gpu_memory_utilization)
        if self.dataset:
            check_dataset(self.dataset)


def check_stack() -> None:
    from importlib import metadata

    import torch
    import unsloth  # noqa: F401  # needs a GPU; must be imported before transformers/trl

    for name in PACKAGES:
        dist = "flashinfer-python" if name == "flashinfer" else name
        print(f"{name:13s} {metadata.version(dist)}")
    print(f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')} HF_HOME={os.environ.get('HF_HOME')}")
    assert torch.cuda.is_available(), "no CUDA device visible (run with --nv)"
    name, cap = torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    ok = torch.isfinite(x @ x).all().item()
    print(
        f"gpu          {name} sm_{cap[0]}{cap[1]}, cuda {torch.version.cuda}, bf16 matmul ok={ok}"
    )


def check_model(model: str, max_model_len: int, gpu_memory_utilization: float) -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},  # text only
    )
    out = llm.generate(["12 + 30 = "], SamplingParams(temperature=0.0, max_tokens=8))
    print(f"model        {model} -> {out[0].outputs[0].text!r}")


def check_dataset(path: Path) -> None:
    with path.open(encoding="utf-8") as f:
        first = json.loads(f.readline())
    print(f"dataset      {path}: keys {sorted(first)}")


if __name__ == "__main__":
    CliApp.run(CheckArgs)

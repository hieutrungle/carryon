"""Start TRL's vLLM server for the experiment in --config (2-GPU server mode, GPU 0).

Usage: python scripts/serve_vllm.py --config configs/m0/grpo_toy_qwen35-4b.yaml
The trainer on GPU 1 connects to http://127.0.0.1:<vllm.server_port> and syncs LoRA weights.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import CliApp

from carryon.config import load_config


class ServeArgs(BaseModel):
    """Serve the config's model with `trl vllm-serve`."""

    config: Path

    def cli_cmd(self) -> None:
        cfg = load_config(self.config)
        command = [
            "trl",
            "vllm-serve",
            "--model", cfg.model,
            "--port", str(cfg.vllm.server_port),
            "--gpu-memory-utilization", str(cfg.vllm.server_gpu_memory_utilization),
            "--max-model-len", str(cfg.vllm.max_model_len),
            "--dtype", "bfloat16",
        ]  # fmt: skip
        print(" ".join(command), flush=True)
        os.execvp(command[0], command)  # replace this process so Slurm signals reach vLLM


if __name__ == "__main__":
    CliApp.run(ServeArgs)

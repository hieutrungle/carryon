# 002 — Pinned GPU stack, smoke test and default job type

- Status: draft. Part 1 (pinned stack) is filled in at M0.8; part 2 (smoke-test table and
  default job type) is filled in at M0.13.
- Date: 2026-10-25
- Milestone: M0

## Part 1 — Pinned GPU stack (M0.8)

### Context
The stack must run Qwen3.5-4B (hybrid Gated DeltaNet) and Gemma 4 E4B in vLLM, TRL GRPO with
vLLM importance-sampling correction, and Unsloth LoRA SFT, on the RTX 5070 Ti (Blackwell, sm_120)
and on FASTER A100 (sm_80) and A40 (sm_86). Drivers: local 595.99.02, FASTER 580.126.20.

### Constraints found (PyPI metadata, 2026-10-25)
- vLLM 0.32.0 (latest) pins `torch==2.13.0` and needs `transformers>=5.16.1,<5.20.0`.
- Unsloth and unsloth_zoo 2026.10.8 (latest) need `transformers<=5.17.0`, `trl<=1.15.0`,
  `datasets<5`.
- So TRL is held at 1.15.0 (1.16.0 is out) and transformers at 5.17.0 (5.20.2 is out).

### Choice
The `train` extra pins these exactly; `uv.lock` (245 packages, Linux x86-64 only) pins the rest.

| Package | Version |
| --- | --- |
| torch | 2.13.0 |
| vllm | 0.32.0 |
| trl | 1.15.0 |
| unsloth | 2026.10.8 |
| unsloth-zoo | 2026.10.8 |
| transformers | 5.17.0 |
| peft | 0.21.2 |
| accelerate | 1.15.0 |
| datasets | 4.8.5 |
| tokenizers | 0.23.3 |
| xformers | 0.0.35 |
| bitsandbytes | 0.50.2 |
| flashinfer-python | 0.7.0.post1 |
| triton | 3.7.1 |
| bm25s | 0.3.15 |
| nvidia-cuda-runtime | 13.0.96 |

The wheels are built for **CUDA 13.0**, so the container base moves from CUDA 12.8 to
`nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04` (FlashInfer JIT-compiles with the image's nvcc).
CUDA 13.0 needs driver ≥ 580, which both machines meet.

The environment library (OpenEnv or verifiers) is not pinned here: choosing it is an M1 task.

### Checked without a GPU
- vLLM 0.32.0 ships `qwen3_5.py` and `gemma4.py` model files; transformers 5.17.0 ships the
  `qwen3_5` and `gemma4` model packages; Unsloth 2026.10.8 references both families, including a
  Gated DeltaNet kernel (`_gdn_op.py`).
- TRL 1.15.0 `GRPOConfig` has `vllm_mode`, `vllm_importance_sampling_correction`,
  `vllm_importance_sampling_cap` and `mask_truncated_completions`.
- Real tokenizers (Qwen3.5-4B, Gemma 4 E4B): re-encoding sampled text changes the token ids
  (`tests/test_real_tokenizers.py`), so trajectories must store sampled ids.

### Not checked yet
Import and GPU execution on each GPU type (M0.9 container test) and multi-turn GRPO steps (M0.13).
If any fails, step back one vLLM minor version first, then re-solve the lock.

## Part 2 — Smoke test and default job type (M0.13)

20 multi-turn GRPO steps per cell; peak memory and episodes per hour.

| Model | GPU | Mode | Peak memory | Episodes / h | Notes |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-4B | A100 | 1-GPU colocate | | | |
| Qwen3.5-4B | A100 | 2-GPU server | | | |
| Qwen3.5-4B | A40 | 1-GPU colocate | | | |
| Qwen3.5-4B | A40 | 2-GPU server | | | |
| Gemma 4 E4B | A100 | 1-GPU colocate | | | |
| Gemma 4 E4B | A100 | 2-GPU server | | | |
| Gemma 4 E4B | A40 | 1-GPU colocate | | | |
| Gemma 4 E4B | A40 | 2-GPU server | | | |

Default job type per model: to be decided from this table.

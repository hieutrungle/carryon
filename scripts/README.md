# scripts/

Thin entry points; logic lives in `carryon/`. Planned (added in the milestone that needs them):

| Script | Milestone | Purpose |
| --- | --- | --- |
| `build_index.py` | M1 | Build the BM25 index over the Wikipedia subset |
| `run_eval.py` | M1–M2 | Run an agent + context strategy on a split, log trajectories |
| `train_sft.py` | M3 | Unsloth LoRA warm-start |
| `train_grpo.py` | M0 smoke, M4+ | TRL GRPO with vLLM (colocate or server mode) |
| `serve_vllm.py` | M0 smoke, M4+ | vLLM server for 2-GPU server mode |

# configs/

One YAML file per experiment; scripts take `--config <file>` and never hard-code hyperparameters.

Naming: `configs/<milestone>/<what>_<model>_<budget>.yaml`, e.g. `configs/m4/grpo_append_qwen35-4b_8k.yaml`.
Every file here is parsed by `tests/test_smoke.py`, so a broken YAML fails CI.

#!/bin/bash
# UNTESTED until M0.11. 1-GPU colocate: vLLM and the trainer share one GPU.
# Usage: sbatch [--gres=gpu:a40:1] slurm/colocate_1gpu.sbatch <config.yaml>
#SBATCH --job-name=carryon-colocate
#SBATCH --account=142788516569
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
CONFIG="${1:?usage: sbatch slurm/colocate_1gpu.sbatch <config.yaml>}"
source slurm/common.sh

# Resume: RUN_DIR=runs/<run_id> sbatch ... continues that run from its latest checkpoint.
run_in_container python scripts/train_grpo.py --config "$CONFIG" --vllm-mode colocate \
    ${RUN_DIR:+--run-dir "$RUN_DIR"}

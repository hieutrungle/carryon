#!/bin/bash
# UNTESTED until M0.11. 1-GPU colocate job array: one seed per array task.
# Usage: sbatch --array=0-2 [--gres=gpu:a40:1] slurm/seed_array.sbatch <config.yaml>
#SBATCH --job-name=carryon-seeds
#SBATCH --account=142788516569
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x-%j-%a.out

set -euo pipefail
CONFIG="${1:?usage: sbatch --array=0-2 slurm/seed_array.sbatch <config.yaml>}"
SEED="${SLURM_ARRAY_TASK_ID:?submit with --array}"
source slurm/common.sh

run_in_container python scripts/train_grpo.py --config "$CONFIG" --seed "$SEED" \
    --vllm-mode colocate --resume auto

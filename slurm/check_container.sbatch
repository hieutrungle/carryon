#!/bin/bash
# M0.9/M0.10: GPU stack check plus offline load of both models and one dataset.
# Usage: sbatch [--gres=gpu:a40:1] slurm/check_container.sbatch
#SBATCH --job-name=carryon-check
#SBATCH --account=142788516569
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
source slurm/common.sh

run_in_container python scripts/check_container.py \
    --dataset "$CARRYON_DATA/datasets/flashrag/musique/dev.jsonl"
for model in Qwen/Qwen3.5-4B google/gemma-4-E4B-it; do
    run_in_container python scripts/check_container.py --model "$model"
done
echo "check_container: all passed"

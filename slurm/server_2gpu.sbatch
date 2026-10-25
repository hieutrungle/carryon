#!/bin/bash
# UNTESTED until M0.11. 2-GPU server mode: GPU 0 serves vLLM, GPU 1 trains.
# Usage: sbatch [--gres=gpu:a40:2] slurm/server_2gpu.sbatch <config.yaml>
#SBATCH --job-name=carryon-server
#SBATCH --account=142788516569
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
CONFIG="${1:?usage: sbatch slurm/server_2gpu.sbatch <config.yaml>}"
source slurm/common.sh

CUDA_VISIBLE_DEVICES=0 run_in_container python scripts/serve_vllm.py --config "$CONFIG" --port 8000 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

# Wait up to 15 minutes for the server before starting the trainer.
for _ in $(seq 180); do
    curl -sf http://127.0.0.1:8000/health/ >/dev/null && break
    sleep 5
done
curl -sf http://127.0.0.1:8000/health/ >/dev/null || { echo "vLLM server never became healthy" >&2; exit 1; }

CUDA_VISIBLE_DEVICES=1 run_in_container python scripts/train_grpo.py \
    --config "$CONFIG" --vllm-mode server --vllm-port 8000 --resume auto

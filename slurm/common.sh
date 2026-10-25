# Sourced by every sbatch template. Paths default to the FASTER layout in docs/sync.md.
# shellcheck shell=bash

CARRYON_DATA="${CARRYON_DATA:-$SCRATCH/research/carryon-data}"
CARRYON_SIF="${CARRYON_SIF:-$CARRYON_DATA/sif/carryon.sif}"

# Compute nodes have no internet: everything comes from the pre-filled cache.
export HF_HOME="$CARRYON_DATA/hf"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline

# Run cards record the git hash, so the hash must describe the code that actually ran.
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "refusing to run: uncommitted changes in $(pwd)" >&2
    exit 1
fi
echo "carryon git $(git rev-parse HEAD) on $(hostname), job ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# FASTER compute nodes ship SingularityCE (/usr/bin/singularity, no module, not on login nodes);
# the local machine uses Apptainer. Both run the same .sif with the same flags.
CONTAINER_CMD="$(command -v apptainer || command -v singularity)" || {
    echo "no apptainer or singularity on PATH (container commands only work on compute nodes)" >&2
    exit 1
}
run_in_container() {
    "$CONTAINER_CMD" exec --nv \
        --bind "$CARRYON_DATA" --bind "$PWD" --pwd "$PWD" \
        --env PYTHONPATH="$PWD" \
        "$CARRYON_SIF" "$@"
}

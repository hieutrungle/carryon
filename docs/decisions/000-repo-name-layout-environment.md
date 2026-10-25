# 000 — Repo name, layout and environment

- Status: accepted
- Date: 2026-10-25
- Milestone: M0

## Context
The project plan names the repository `carry-on` and the package `carryon`; the GitHub repository was
created as `hieutrungle/carryon`. Code has to run identically on the local RTX 5070 Ti and on the TAMU
FASTER cluster (A100/A40), with one pinned environment. Facts checked on 2026-10-25
([M0 roadmap](../roadmap/M0.md), "Verified environment facts"):

- FASTER compute nodes have SingularityCE 4.4.1 at `/usr/bin/singularity`; there is no `apptainer`
  command or module, and no container runtime on login nodes.
- GPU drivers: local 595.99.02, FASTER A40 and A100 580.126.20 (all ≥ 580, enough for the
  CUDA 13.0 image of decision 002).
- FASTER scratch allows 2M files and about 680k are free; home is 10 GB / 10k files.
- An image built locally with Apptainer runs under SingularityCE with `--nv` on a FASTER A40.

## Options
1. **Name:** `carryon` everywhere, or `carry-on` folders with a `carryon` package.
2. **Pinning:** uv with `uv.lock`, or conda with `conda-lock`.
3. **Runtime environment on FASTER:** one container image, or a uv virtualenv on scratch.

## Choice
1. `carryon` for the GitHub repo, both clones (`/home/hieule/research/carryon`,
   `/scratch/user/hieult/research/carryon`) and the package.
2. uv: `pyproject.toml` with ranges, `uv.lock` with exact pins; CI runs `uv sync --locked`.
3. One image from `container/carryon.def`, built locally with Apptainer from `uv.lock`, copied to
   `carryon-data/sif/` on FASTER and run there with `singularity exec --nv`. `slurm/common.sh` picks
   whichever runtime is on PATH.

Code moves only through GitHub; models, datasets, indexes, images and runs live outside git in a
sibling `carryon-data/` folder on each machine ([sync guide](../sync.md)).

## Why
One name removes a whole class of path mistakes. uv gives one lock file that CI, the container and a
local venv all install from; vLLM, TRL and Unsloth are distributed as wheels, so conda adds solve time
without adding packages. The container wins on FASTER because a torch + vLLM + TRL + Unsloth venv is
tens of thousands of files against about 680k free, while a `.sif` is one file. It also guarantees that
final checkpoints are scored with the same software they were trained with.

Revisit if a full image fails under SingularityCE in M0.9 (fallback: a uv venv on scratch from the same
lock), or if HPRC changes the container runtime on FASTER.

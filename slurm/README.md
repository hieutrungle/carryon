# slurm/

UNTESTED placeholders: they become real in M0 (docs/roadmap/M0.md, items M0.9 and M0.11).
Submit from the repo root on FASTER so `logs/slurm/` and relative config paths resolve.

| Template | Use | Example |
| --- | --- | --- |
| `colocate_1gpu.sbatch` | 1 GPU, vLLM and trainer share it | `sbatch slurm/colocate_1gpu.sbatch configs/m0/smoke.yaml` |
| `server_2gpu.sbatch` | 2 GPUs: GPU 0 runs the vLLM server, GPU 1 trains | `sbatch slurm/server_2gpu.sbatch configs/m4/main.yaml` |
| `seed_array.sbatch` | 1-GPU colocate job array, one seed per task | `sbatch --array=0-2 slurm/seed_array.sbatch configs/m4/main.yaml` |

`#SBATCH` lines cannot read variables, so the defaults (account 142788516569, partition `gpu`, A100) are
written in each file. Override them at submit time, e.g. `sbatch --gres=gpu:a40:1 ...`.
`common.sh` sets offline Hugging Face/W&B mode, refuses to run from a dirty git tree, and defines `run_in_container`
(Apptainer if present, else SingularityCE, which is what FASTER compute nodes have).

FASTER's submit filter rejects any job without `--nodes`, `--ntasks` or `--ntasks-per-node`; the templates set
`--nodes=1 --ntasks=1`. For an interactive shell on a GPU node:

```bash
srun --partition=gpu --gres=gpu:a40:1 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G \
     --time=00:30:00 --account=142788516569 --pty bash -i
```

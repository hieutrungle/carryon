# container/

One Apptainer image (`carryon.def`) for FASTER and the local 5070 Ti, built from the same `uv.lock`.
The `.sif` is never committed; it lives at `$SCRATCH/research/carryon-data/sif/carryon.sif` on FASTER and
`~/research/carryon-data/sif/carryon.sif` locally (see docs/sync.md). Rebuild only between milestones.

Runtimes: Apptainer locally (builds need `sudo` or `--fakeroot`); SingularityCE 4.4.1 on FASTER, available only on
compute nodes at `/usr/bin/singularity` (no module, nothing on login nodes). The commands are interchangeable
(`singularity exec --nv carryon.sif ...`), and `slurm/common.sh` picks whichever is on PATH. Build locally, then
rsync the `.sif` to FASTER.

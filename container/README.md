# container/

One Apptainer image (`carryon.def`) for FASTER and the local 5070 Ti, built from the same `uv.lock`.
The `.sif` is never committed; it lives at `$SCRATCH/research/carryon-data/sif/carryon.sif` on FASTER and
`~/research/carryon-data/sif/carryon.sif` locally (see docs/sync.md). Rebuild only between milestones.

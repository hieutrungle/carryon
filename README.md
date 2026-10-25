# Carry-On

An LLM agent that learns, with RL, what to keep in its own context under a fixed active-token budget.

> Status: M0 (foundation). Nothing to run yet beyond `make check`. Headline figure (F7.2) goes here at v1.0.

## Quickstart (development)

```bash
git clone https://github.com/hieutrungle/carryon.git && cd carryon
make install      # uv sync --locked
make check        # ruff, mypy, pytest on CPU
```

## Where to look

- [ROADMAP.md](ROADMAP.md): phases, milestones, gates and releases
- [docs/roadmap/](docs/roadmap/): one detailed roadmap per milestone (`M0.md` first)
- [docs/sync.md](docs/sync.md): how the local machine, GitHub and the FASTER cluster stay in sync
- [docs/decisions/](docs/decisions/): one page per design decision
- [report/](report/): living report, chapters 0, A, B, C

## Layout

```
carryon/      Python package: envs, context, agent, rollout, train, eval, viz
configs/      one YAML per experiment
scripts/      entry points (build_index, run_eval, train_sft, train_grpo)
slurm/        sbatch templates: 1-GPU colocate, 2-GPU server, seed arrays
container/    Apptainer definition shared by FASTER and the local 5070 Ti
tests/        masking, token-exact trajectories, rewards, context managers
runs/         logged runs (gitignored; released as archives)
figures/      generated PNG/SVG, never hand-edited (gitignored, rebuilt by `make figures`)
report/       living report
notes/        lab notebook, one file per session
docs/         roadmaps, decisions, sync guide
```

The repository, folders and package are all named `carryon` (the plan's `carry-on` was renamed to match the GitHub repo).

## License

MIT. See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).

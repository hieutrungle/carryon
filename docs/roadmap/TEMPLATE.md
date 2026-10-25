# Mx · <title> — detailed roadmap

- Phase: 0 | A | B | C · Budget: <h> h · Status: not started | in progress | done
- Written: YYYY-MM-DD, after M(x-1) closed · Plan section: <link or heading>

## Goal
One sentence, copied from the plan.

## Entry conditions
What must already be true (previous exit checklist, gate result, pinned versions, data on FASTER).

## Work items
Ordered. Each item is small enough for one or two sessions and has its own done-criterion.

| # | Item | Where | Est. h | Done when |
| --- | --- | --- | --- | --- |
| Mx.1 | ... | Local / FASTER / both | | |

## Runs to queue
Configs and job types (`slurm/*.sbatch`), expected GPU-hours, which can run overnight.

## Figures
| ID | Script | Input runs |
| --- | --- | --- |

## Decisions expected
Decision notes this milestone should produce (`docs/decisions/NNN-*.md`).

## Risks and early signs
From the plan's risk table, plus anything the previous milestone surfaced.

## Exit checklist
- [ ] Plan's "done when" criterion met
- [ ] `make check` green locally and in CI
- [ ] Figures regenerated with `make figures`; report chapter updated
- [ ] Run cards exist for every run cited
- [ ] ROADMAP.md status box ticked; next `Mx.md` written from this template
- [ ] Release tag / gate decision, if this milestone has one

# Carry-On Roadmap

The project-level roadmap. Source of truth for scope and decisions is the project plan
(Oct 24, 2026); this file tracks progress against it. Any deviation from the plan is agreed first and
recorded in [docs/decisions/](docs/decisions/).

**How the roadmaps fit together**

- This file: all 14 milestones, gates and releases, one status box each.
- `docs/roadmap/Mx.md`: the detailed roadmap for one milestone, written from
  [TEMPLATE.md](docs/roadmap/TEMPLATE.md) when the previous milestone closes, so it reflects what
  that milestone actually found. Only [M0.md](docs/roadmap/M0.md) exists now.
- Closing a milestone: tick its exit checklist, regenerate figures (`make figures`), update its report
  chapter, tick the box below, write the next `Mx.md`.

## Status

| Done | ID | Phase | Milestone | Hours | Where | Gate / release | Detailed roadmap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [ ] | M0 | 0 | Repo, cluster setup and logging | 25 | Local + FASTER |  | [docs/roadmap/M0.md](docs/roadmap/M0.md) |
| [ ] | M1 | 0 | Environment and agent loop | 25 | Local |  | not yet |
| [ ] | M2 | 0 | Task set and prompted baselines | 20 | Local | Gate 1 · v0.1 · blog 1 | not yet |
| [ ] | M3 | A | Supervised warm-start | 25 | Teacher local; SFT on FASTER (1 GPU); eval local |  | not yet |
| [ ] | M4 | A | Multi-turn GRPO baseline | 45 | FASTER (2-GPU server; 1-GPU arrays); eval local | Gate 2 | not yet |
| [ ] | M5 | A | RL with summarization | 30 | FASTER; eval local |  | not yet |
| [ ] | M6 | A | RL with folding | 45 | FASTER; eval local |  | not yet |
| [ ] | M7 | A | Ablations, final matrix and chapter A | 40 | FASTER (training); local (all final scoring) | v0.5 | not yet |
| [ ] | M8 | B | Systems cost study | 30 | Local 5070 Ti + one A100 comparison | can start during M5–M7 | not yet |
| [ ] | M9 | B | A + B release | 25 | Local | Gate 3 · v1.0 · blog 2 | not yet |
| [ ] | M10 | C | Credit diagnostics | 25 | FASTER logs; analysis local |  | not yet |
| [ ] | M11 | C | Credit-assignment method | 45 | FASTER; eval local |  | not yet |
| [ ] | M12 | C | Generalization check | 35 | Harbor task local/Apptainer; 9B on FASTER (2× A100) |  | not yet |
| [ ] | M13 | C | Final report and v2.0 | 25 | Local | v2.0 · blog 3 | not yet |

Total about 440 h. Standard pace 15 h/week ≈ 30 weeks (v1.0 ≈ week 21); full-time 14–16 weeks; Lean MVP
(M0–M9, RL for fold only, lighter M7/M8) ≈ 260 h. Most of M8 can be coded while Phase A runs train overnight.

## Decision gates

| Gate | When | Pass condition | If it fails |
| --- | --- | --- | --- |
| Gate 1 | End of M2 | Strategies differ by a clear margin at 4k; the unlimited reference beats `append` at 4k | Make the task harder (more hops, longer passages, smaller budget) before any training |
| Gate 2 | End of M4 | RL beats SFT on dev for `append` at 8k | Debug masking, reward hacking and learning rate; do not add folding to a broken pipeline |
| Gate 3 | End of M9 | A learned strategy beat its prompted version in M7 | Phase C becomes a diagnostic study of why learning failed |

## Releases

| Tag | Milestone | Contents |
| --- | --- | --- |
| v0.1 | M2 | Environment + context-budget wrapper on the HF Hub, prompted baselines, chapter 0, blog post 1 |
| v0.5 | M7 | Phase A results, chapter A |
| v1.0 | M9 | A + B package: workshop draft, LoRA adapters, model/data cards, blog post 2 |
| v2.0 | M13 | Full report, weights, data cards, reproduction guide, blog post 3 |

Each tag gets release notes listing new figures and results.

## Where work runs

| Work | Where |
| --- | --- |
| Environment building, debugging, smoke tests | Local 5070 Ti |
| Teacher trajectories (27B, 4-bit, ~13 GB) | Local |
| SFT warm-start (M3) | FASTER, 1 GPU |
| GRPO training, seeds, ablations (M4–M7, M11) | FASTER: 2-GPU server mode for main runs, 1-GPU colocate arrays for extra seeds |
| 9B scale check (M12) | FASTER, 2 GPUs (A100 preferred) |
| Checkpoint evaluation, BrowseComp-Plus, figures, viewer | Local (one machine for all final scores) |
| Phase B serving measurements | Local, plus one A100 comparison |

Prefer A100 over A40 for RL (memory bandwidth); bf16 everywhere. Code moves through GitHub;
data, models and runs move as described in [docs/sync.md](docs/sync.md).

## Phase 0 — Foundation (about 70 h)

First public result: context strategies differ at a 4k budget. No training.

### M0 · Repo, cluster setup and logging (25 h)

- **Where:** Local + FASTER
- **Figures:** F0.1
- **Done when:** `make report` builds chapter 0 with F0.1, CI is green, and decision 002 records the smoke-test table and the default job type.
- **Roadmap:** [M0](docs/roadmap/M0.md)

### M1 · Environment and agent loop (25 h)

- **Where:** Local
- **Figures:** F1.1, F1.2
- **Done when:** 100 questions run end to end and every one opens in the viewer.
- **Roadmap:** M1 (roadmap written when M0 closes)

### M2 · Task set and prompted baselines (20 h)

- **Where:** Local
- **Figures:** F2.1 ★, F2.2, F2.3
- **Done when:** Gate 1 passes; environment + wrapper on the HF Hub; chapter 0 and blog post 1 written; tag v0.1.
- **Roadmap:** M2 (roadmap written when M1 closes)

## Phase A — Learned context management (about 185 h)

Trained strategies vs. rule-based ones at fixed budgets, with seeds and ablations.

### M3 · Supervised warm-start (25 h)

- **Where:** Teacher local; SFT on FASTER (1 GPU); eval local
- **Figures:** F3.1, F3.2, F3.3
- **Done when:** The SFT model emits valid fold and summarize actions in nearly every episode.
- **Roadmap:** M3 (roadmap written when M2 closes)

### M4 · Multi-turn GRPO baseline (45 h)

- **Where:** FASTER (2-GPU server; 1-GPU arrays); eval local
- **Figures:** F4.1–F4.4
- **Done when:** Gate 2 passes: RL beats SFT on dev for `append` at 8k.
- **Roadmap:** M4 (roadmap written when M3 closes)

### M5 · RL with summarization (30 h)

- **Where:** FASTER; eval local
- **Figures:** F5.1–F5.3
- **Done when:** RL summarization beats prompted summarization on dev at 4k across three seeds.
- **Roadmap:** M5 (roadmap written when M4 closes)

### M6 · RL with folding (45 h)

- **Where:** FASTER; eval local
- **Figures:** F6.1 ★, F6.2–F6.4
- **Done when:** Folding training is stable for three seeds, whatever its final rank.
- **Roadmap:** M6 (roadmap written when M5 closes)

### M7 · Ablations, final matrix and chapter A (40 h)

- **Where:** FASTER (training); local (all final scoring)
- **Figures:** F7.1, F7.2 ★, F7.3 ★, F7.4–F7.6
- **Done when:** Chapter A reads end to end with every figure regenerated from `runs/`; tag v0.5.
- **Roadmap:** M7 (roadmap written when M6 closes)

## Phase B — Systems cost and v1.0 (about 55 h)

What each strategy costs to serve and to train; first complete public package.

### M8 · Systems cost study (30 h)

- **Where:** Local 5070 Ti + one A100 comparison
- **Figures:** F8.1, F8.2, F8.3 ★, F8.4, F8.5
- **Done when:** Every number in chapter B comes from one rerunnable command.
- **Roadmap:** M8 (roadmap written when M7 closes)

### M9 · A + B release (25 h)

- **Where:** Local
- **Figures:** (uses F2.1, F6.1, F7.2, F7.3, F8.3)
- **Done when:** Workshop draft, README quickstart on a 16 GB GPU, adapters + model/data cards, blog post 2; tag v1.0; Gate 3 decided.
- **Roadmap:** M9 (roadmap written when M8 closes)

## Phase C — Credit assignment (about 130 h)

Can a better credit signal for context actions beat outcome-only GRPO?

### M10 · Credit diagnostics (25 h)

- **Where:** FASTER logs; analysis local
- **Figures:** F10.1–F10.3
- **Done when:** One concrete, testable hypothesis about the credit signal is written as a decision note.
- **Roadmap:** M10 (roadmap written when M9 closes)

### M11 · Credit-assignment method (45 h)

- **Where:** FASTER; eval local
- **Figures:** F11.1 ★, F11.2, F11.3
- **Done when:** The chosen method vs. outcome-only GRPO and the M6 process reward, three seeds each at 4k, with a reward-hacking check.
- **Roadmap:** M11 (roadmap written when M10 closes)

### M12 · Generalization check (35 h)

- **Where:** Harbor task local/Apptainer; 9B on FASTER (2× A100)
- **Figures:** F12.1–F12.3
- **Done when:** Second task wrapped by the unchanged context-budget wrapper; Qwen3.5-9B scale check; figure scripts unchanged.
- **Roadmap:** M12 (roadmap written when M11 closes)

### M13 · Final report and v2.0 (25 h)

- **Where:** Local
- **Figures:** (figure catalog appendix)
- **Done when:** Full report (0, A, B, C), venue chosen, blog post 3, v2.0 with weights, data cards and reproduction guide; retrospective.
- **Roadmap:** M13 (roadmap written when M12 closes)

## Main risks (from the plan)

| Risk | Early sign | Fallback |
| --- | --- | --- |
| Small model never learns to fold | Fold actions vanish in early RL | More and better warm-start data; process reward for valid folds |
| Segmented trajectories break existing tools | Loss or advantage bugs on split episodes | Own rollout and loss code, guarded by the token-exact unit test |
| Cluster queue or time limits stall runs | Jobs wait for days or are killed mid-run | Frequent checkpoints with tested resume; short 1-GPU job arrays; smoke-test locally first |
| vLLM does not handle the hybrid model well | Qwen3.5 rollouts fail, drift or run slowly | Pin the vLLM version that passed the M0 smoke test; promote Gemma 4 E4B to main model if needed |
| Experiment matrix explodes | More than 40 planned runs | RL only for summarize and fold; rule-based strategies are evaluation only; Gemma replicates the headline only |
| Environment libraries change under you | An upgrade breaks rollouts or rewards | Pin versions in the container; upgrade only between milestones, unit tests as the gate |
| A community environment has a buggy verifier | Reward rises while real accuracy does not | Oracle, no-op and unsolved-state checks before training; drop failing tasks |
| Someone publishes the same idea | A close arXiv paper appears | Reproduce it as a baseline; systems and credit chapters still differ |
| Motivation dips mid-project | Notebook entries stop | Ship the next milestone's show-off item, however small; post progress |

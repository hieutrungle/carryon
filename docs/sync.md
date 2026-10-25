# Sync: local ↔ GitHub ↔ FASTER

Code moves only through GitHub. Data, models, images and run outputs never go into git; they live in a
`carryon-data/` folder next to the repo on each machine and move with rsync or the Hugging Face Hub.

## Layout on each machine

| | Local (RTX 5070 Ti) | FASTER |
| --- | --- | --- |
| Repo (git) | `/home/hieule/research/carryon` | `/scratch/user/hieult/research/carryon` |
| Data (not git) | `/home/hieule/research/carryon-data/` | `/scratch/user/hieult/research/carryon-data/` |
| `carryon-data/hf/` | HF cache (`HF_HOME`) | HF cache, pre-filled; compute nodes run offline |
| `carryon-data/datasets/`, `indexes/` | datasets, BM25 index | same |
| `carryon-data/sif/` | `carryon.sif` | `carryon.sif` (same image) |
| `carryon/runs/` (gitignored) | evaluation runs + runs synced back | training runs |

FASTER scratch is not backed up; anything you need to keep comes back to local or goes to the HF Hub.

## What goes where

| Thing | Git | rsync | HF Hub (private) |
| --- | --- | --- | --- |
| Code, configs, tests, docs, notes, report text, `uv.lock` | ✓ | | |
| Run logs: `trajectories.jsonl`, metrics Parquet, `card.md`, W&B offline dirs | | ✓ FASTER → local | optional archive |
| Final LoRA adapters | | ✓ FASTER → local | ✓ (released at v1.0) |
| Intermediate checkpoints | | stay on FASTER | |
| Base models, datasets, BM25 index | | | downloaded on each side from the Hub |
| `carryon.sif` | | ✓ local → FASTER (if built locally) | |
| Figures | rebuilt by `make figures` from `runs/` on local | | |

## 1. One-time: local repo → GitHub

The repo already exists locally with one commit and `origin` set. From your own terminal:

```bash
cd /home/hieule/research/carryon
git log --oneline            # one commit: "M0: repo skeleton, roadmaps and sync guide"
git push -u origin main
```

If HTTPS asks for a password, GitHub needs a personal access token (or `gh auth login`). If you already use
SSH keys with GitHub locally, switch the remote instead:

```bash
git remote set-url origin git@github.com:hieutrungle/carryon.git
git push -u origin main
```

Then open the Actions tab and check that the `ci` workflow is green.

## 2. One-time: FASTER clone

In your own FASTER session (login node; git is light work):

```bash
mkdir -p /scratch/user/hieult/research && cd /scratch/user/hieult/research
```

Pick one way to authenticate to GitHub from FASTER.

**Option A: SSH key (recommended if outbound SSH works)**

```bash
ssh-keygen -t ed25519 -f ~/.ssh/github_faster -C "faster-carryon"
cat ~/.ssh/github_faster.pub        # add at GitHub → Settings → SSH and GPG keys
                                    # (or repo → Settings → Deploy keys, tick "Allow write access", for repo-only access)
cat >> ~/.ssh/config <<'EOF'
Host github.com
    IdentityFile ~/.ssh/github_faster
    IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config
ssh -T git@github.com               # expect: "Hi hieutrungle! You've successfully authenticated..."
```

If port 22 to github.com is blocked, use GitHub's SSH-over-443 endpoint by adding `HostName ssh.github.com`
and `Port 443` to that `Host github.com` block, then rerun `ssh -T git@github.com`.

```bash
git clone git@github.com:hieutrungle/carryon.git
```

**Option B: HTTPS with a fine-grained token** (scope: only `hieutrungle/carryon`, Contents read/write)

```bash
git config --global credential.helper 'cache --timeout=28800'   # keeps the token in memory for 8 h, not on disk
git clone https://github.com/hieutrungle/carryon.git           # username: hieutrungle, password: the token
```

Then, either way:

```bash
cd /scratch/user/hieult/research/carryon
git config user.name  "hieule"
git config user.email "<the email you commit with>"
mkdir -p ../carryon-data/{hf,datasets,indexes,sif}
git log --oneline -1                 # same hash as local
```

## 3. Daily loop

Edit code locally by default; FASTER pulls and runs.

```bash
# local
make check                           # never push a red tree
git add -A && git commit -m "M0.3: trajectory schema"
git push

# FASTER
cd /scratch/user/hieult/research/carryon
git pull --ff-only
sbatch slurm/colocate_1gpu.sbatch configs/<exp>.yaml
```

Rules:

- `git pull --ff-only` on FASTER. If it refuses, something was committed on FASTER: push that first, then pull locally.
- If you must fix something on FASTER (a path, a Slurm flag), commit it there on a branch and push it:
  `git switch -c faster/<topic> && git commit -am "..." && git push -u origin faster/<topic>`, then merge locally.
- `slurm/common.sh` refuses to start a job from a tree with uncommitted changes, so every run card's git hash is the code that ran.
- Branches: `main` is always green; work branches are `m<k>/<topic>` (e.g. `m0/trajectory-schema`); releases are tags (`v0.1`, `v0.5`, `v1.0`, `v2.0`).
- Pin changes (`uv.lock`, `container/carryon.def`) happen only between milestones; after one, rebuild the `.sif` and copy it to FASTER before the next job.

## 4. Results back to local

Run these from your **local** terminal, through your existing FASTER login session (`faster-claude` in `~/.ssh/config`).
Make sure the session is alive first (`ssh -O check faster-claude`, or log in again with password + Duo).

```bash
# one run: logs, card, W&B offline data and the final adapter; skip intermediate checkpoints
RUN=<run_id>
rsync -avh --progress \
    --exclude 'checkpoint-*/' \
    faster-claude:/scratch/user/hieult/research/carryon/runs/$RUN/ \
    /home/hieule/research/carryon/runs/$RUN/

# all runs (logs only, small)
rsync -avh --prune-empty-dirs --include '*/' --include '*.jsonl' --include '*.parquet' --include 'card.md' --include 'config.yaml' \
    --exclude '*' \
    faster-claude:/scratch/user/hieult/research/carryon/runs/ /home/hieule/research/carryon/runs/

# W&B: compute nodes are offline; sync from local after rsync
uvx wandb sync /home/hieule/research/carryon/runs/$RUN/wandb/offline-run-*
```

Alternative for adapters (useful once they are many or large): push them from the FASTER login node to a private
HF repo with the `hf` CLI from `huggingface_hub`, e.g. `hf upload hieutrungle/carryon-adapters runs/$RUN/adapter $RUN --private`,
and pull locally with `hf download hieutrungle/carryon-adapters --include "$RUN/*"`.

## 5. Images and big inputs to FASTER

```bash
# container built locally → FASTER (from your local terminal)
rsync -avh --progress /home/hieule/research/carryon-data/sif/carryon.sif \
    faster-claude:/scratch/user/hieult/research/carryon-data/sif/
```

Models and datasets are downloaded directly on the FASTER login node into `carryon-data/hf` (downloads are light
work; check `showquota` first, HF caches create many files). Compute nodes then run with `HF_HUB_OFFLINE=1`.

## 6. FASTER rules (from the cluster setup)

- No heavy computation on the login node: git, rsync, downloads and `sbatch` only. Everything else goes through Slurm.
- Account `142788516569`, partition `gpu`, `--gres=gpu:a100:N` preferred, `a40` acceptable (both 128 SU per GPU-hour).
- Submit from the repo root so `logs/slurm/` and relative config paths resolve.

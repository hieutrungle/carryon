# 001 — Why this task and budget

- Status: accepted (quantitative check pending: F1.1 in M1, Gate 1 at the end of M2)
- Date: 2026-10-25
- Milestone: M0

## Context
Research question A asks when an RL-trained context strategy (summarize, fold) beats hand-written rules
(append, truncate) at a fixed active-context budget. That needs a task where:

- the context really overflows the budget, so what the agent keeps decides the outcome;
- the reward is verifiable and cheap, so RL can run thousands of episodes;
- the environment runs locally and offline, because FASTER compute nodes have no internet;
- difficulty can be turned up or down (Gate 1's fallback is to make the task harder);
- a 4B model can learn it with LoRA on 1–2 GPUs.

## Options
1. **Multi-hop QA over a local Wikipedia BM25 index** (MuSiQue, HotpotQA or 2WikiMultiHopQA), with
   search returning full passages.
2. **Open-web browsing** (e.g. BrowseComp): realistic, but too hard for a 4B model to train on, and it
   needs internet during rollouts.
3. **Terminal or coding tasks** (Harbor format): long tool outputs, but sandboxes per episode make RL
   rollouts expensive and verifiers can be buggy.
4. **Single long-document QA:** overflows easily, but has no search decisions, so it tests reading more
   than context management by an agent.

## Choice
Option 1 as the training and main evaluation task. Search returns full passages, so a 3–4 hop question
accumulates context quickly. Budgets: **4k active tokens** as the main setting and **8k** as the second,
plus an unlimited-context reference. The budget is the experimental knob, not a hardware limit.

Option 2 enters as evaluation only, through a BrowseComp-Plus subset: a fixed corpus with its own local
BM25 retriever and human-verified supporting documents (M7). Option 3 enters in Phase C (M12) as the
generalization check, wrapped by the same context-budget wrapper. The dataset and the frozen test split
are chosen in M2.

## Why
Multi-hop QA meets all five requirements: exact match and F1 give a cheap verifiable reward, BM25 runs
offline, and the number of hops and passage length tune difficulty. Its supporting-paragraph labels also
give a retention probe for free (M7) and a candidate training signal for Phase C. A 4k budget is small
enough that full-passage searches overflow it on 3–4 hop questions, so rules like append must fail and
learned strategies have room to help; 8k shows how the gap changes when the pressure drops.

The overflow claim is not measured yet. F1.1 (tokens per search, M1) and F2.1 (context growth against
the budget line, M2) test it. Revisit at Gate 1: if strategies do not differ clearly at 4k, or the
unlimited reference does not beat append, make the task harder (more hops, longer passages, smaller
budget) before any training.

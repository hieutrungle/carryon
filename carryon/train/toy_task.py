"""M0 toy task for testing the training stack: "a + b = " prompts with an exact-answer reward.

Plain-text prompts (no chat template) keep the model out of its thinking mode, so completions
stay short. Replaced by the multi-hop environment from M1 on.
"""

from __future__ import annotations

import random
import re
from typing import Any

_FIRST_INTEGER = re.compile(r"-?\d+")


def make_prompts(n: int, max_operand: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a, b = rng.randint(0, max_operand), rng.randint(0, max_operand)
        rows.append({"prompt": f"{a} + {b} = ", "answer": str(a + b)})
    return rows


def completion_text(completion: Any) -> str:
    """TRL passes plain strings for text prompts and message lists for chat prompts."""
    if isinstance(completion, str):
        return completion
    return "".join(str(message.get("content", "")) for message in completion)


def exact_answer_reward(completions: list[Any], answer: list[str], **_: Any) -> list[float]:
    """1.0 when the first integer in the completion is the sum, else 0.0 (TRL reward signature)."""
    rewards = []
    for completion, gold in zip(completions, answer, strict=True):
        match = _FIRST_INTEGER.search(completion_text(completion))
        rewards.append(1.0 if match and match.group(0) == gold else 0.0)
    return rewards

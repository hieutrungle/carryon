"""Phase B instrumentation: what the rollout loop measures per model call, and per-turn summaries.

The loop records three numbers on every ``Turn``:

- ``active_tokens``: tokens in the active context at the end of the turn (``Segment.n_tokens``
  of the spans so far);
- ``prefix_reused_tokens``: ``common_prefix_length(previous_call, current_input)``, where
  ``previous_call`` is the previous model call's input plus its generated tokens. This is the
  part a prefix cache (vLLM automatic prefix caching) can reuse: everything under ``append``,
  only the shared prompt after a summarize or truncate;
- ``wall_time_s``: ``time.perf_counter()`` around the model call and the tool call.

``turn_token_counts`` turns a logged episode into per-turn token counts for F1.1 and chapter B.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from carryon.rollout.trajectory import Episode


def common_prefix_length(previous: Sequence[int], current: Sequence[int]) -> int:
    """Number of leading tokens of ``current`` that equal those of ``previous``."""
    for i, (a, b) in enumerate(zip(previous, current, strict=False)):
        if a != b:
            return i
    return min(len(previous), len(current))


@dataclass(frozen=True, slots=True, kw_only=True)
class TurnTokens:
    turn: int
    generated: int  # tokens the policy sampled in this turn
    tool: int  # tool-output tokens added in this turn
    added: int  # all tokens added in this turn, including chat-template tokens
    active: int
    prefix_reused: int | None
    wall_time_s: float | None


def turn_token_counts(episode: Episode) -> list[TurnTokens]:
    """Per-turn token counts; a span counts once, in the turn and segment where it was added."""
    generated = [0] * len(episode.turns)
    tool = [0] * len(episode.turns)
    added = [0] * len(episode.turns)
    for turn in episode.turns:
        # Spans copied into a new segment (the prompt, a summary) keep their original turn;
        # counting only the turn's own segment avoids counting them twice.
        for span in episode.segments[turn.segment].spans:
            if span.turn != turn.index:
                continue
            n = len(span.token_ids)
            added[turn.index] += n
            generated[turn.index] += n if span.sampled else 0
            tool[turn.index] += n if span.role == "tool" else 0
    return [
        TurnTokens(
            turn=t.index,
            generated=generated[t.index],
            tool=tool[t.index],
            added=added[t.index],
            active=t.active_tokens,
            prefix_reused=t.prefix_reused_tokens,
            wall_time_s=t.wall_time_s,
        )
        for t in episode.turns
    ]

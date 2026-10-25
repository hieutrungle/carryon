"""Trajectory format: one episode per JSONL line, written once, read by the trainer and the figures.

Layout
------
An episode holds *segments*. A segment is one contiguous token sequence exactly as the model
saw it (one KV-cache lifetime). A new segment starts whenever a context operation rewrites the
active context (truncate, summarize, fold return), because the next model input is no longer an
extension of the previous one. Each segment is a list of *spans*: runs of tokens with one role.

Token exactness
---------------
Spans store the token ids the sampler produced, never text to re-tokenize: decoding and
re-encoding is not the identity (tests/test_trajectory.py shows a case). The trainer builds its
inputs only from ``Segment.token_ids()``, ``Segment.loss_mask()`` and
``Segment.sampled_logprobs()``. ``Span.text`` is for humans and the viewer.

Turns and context operations are the agent-level view used by the trajectory viewer and the
behaviour figures. ``active_tokens``, ``prefix_reused_tokens`` and ``wall_time_s`` feed the
Phase B cost study.
"""

from __future__ import annotations

import gzip
import itertools
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Literal, TextIO

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    ValidationError,
    model_validator,
)

SCHEMA_VERSION: Literal[1] = 1  # bump on any breaking change; readers reject other versions

Role = Literal["system", "user", "assistant", "tool"]
Strategy = Literal["append", "truncate", "summarize", "fold"]
ContextOpKind = Literal["truncate", "summarize", "fold_branch", "fold_return"]
Termination = Literal["answered", "max_turns", "budget_exceeded", "invalid_action", "error"]

# A log-probability is finite and <= 0; this catches probabilities or -inf leaking in.
LogProb = Annotated[float, Field(le=0.0, allow_inf_nan=False)]


class _Record(BaseModel):
    # extra="forbid" turns silent schema drift (a renamed or misspelled field) into a load error.
    model_config = ConfigDict(extra="forbid", frozen=True)


class Span(_Record):
    """A run of tokens with one role inside one segment."""

    role: Role
    token_ids: list[NonNegativeInt] = Field(min_length=1)
    sampled: bool  # True only for tokens the policy generated; exactly these tokens get loss.
    logprobs: list[LogProb] | None = None  # sampler logprobs, one per token, iff sampled
    turn: NonNegativeInt | None = None  # None for the initial prompt, before turn 0
    text: str | None = None  # decoded text for the viewer; never re-tokenized

    @model_validator(mode="after")
    def _check_sampled(self) -> Span:
        if self.sampled:
            if self.role != "assistant":
                raise ValueError("only assistant spans can be sampled")
            if self.logprobs is None or len(self.logprobs) != len(self.token_ids):
                raise ValueError("a sampled span needs exactly one logprob per token")
        elif self.logprobs is not None:
            raise ValueError("only sampled spans carry logprobs")
        return self


class Segment(_Record):
    """One contiguous model-visible token sequence."""

    index: NonNegativeInt
    spans: list[Span] = Field(min_length=1)

    @property
    def n_tokens(self) -> int:
        return sum(len(span.token_ids) for span in self.spans)

    def token_ids(self) -> list[int]:
        return [t for span in self.spans for t in span.token_ids]

    def loss_mask(self) -> list[int]:
        return [int(span.sampled) for span in self.spans for _ in span.token_ids]

    def sampled_logprobs(self) -> list[float | None]:
        """Sampler logprob per token; None where the token was not sampled."""
        out: list[float | None] = []
        for span in self.spans:
            out.extend(span.logprobs if span.logprobs is not None else [None] * len(span.token_ids))
        return out


class ToolCall(_Record):
    name: str
    arguments: dict[str, JsonValue]


class Turn(_Record):
    """One agent step: the model generates, then the environment answers."""

    index: NonNegativeInt
    segment: NonNegativeInt  # segment the model was generating in
    tool_call: ToolCall | None  # None when the output could not be parsed as an action
    active_tokens: NonNegativeInt  # active context size at the end of the turn, before any op
    prefix_reused_tokens: NonNegativeInt | None = None  # leading tokens shared with the last call
    wall_time_s: NonNegativeFloat | None = None


class ContextOp(_Record):
    """A context-manager action applied after a turn."""

    kind: ContextOpKind
    turn: NonNegativeInt  # the turn after which the op ran
    tokens_before: NonNegativeInt
    tokens_after: NonNegativeInt
    new_segment: NonNegativeInt | None  # segment that starts here; None only for fold_branch

    @model_validator(mode="after")
    def _check_segment(self) -> ContextOp:
        if self.new_segment is None and self.kind != "fold_branch":
            raise ValueError(f"a {self.kind} op rewrites the context and must start a new segment")
        return self


class Outcome(_Record):
    termination: Termination
    answer: str | None
    gold_answers: list[str] = Field(min_length=1)
    reward: float
    exact_match: bool
    f1: float = Field(ge=0.0, le=1.0)
    format_ok: bool


class Episode(_Record):
    """One episode; the unit written to and read from trajectories.jsonl."""

    schema_version: Literal[1] = SCHEMA_VERSION
    run_id: str
    episode_id: str
    question_id: str
    strategy: Strategy
    budget_tokens: PositiveInt | None  # None = unlimited reference
    model: str
    adapter: str | None = None  # LoRA checkpoint, None for the base model
    seed: int
    segments: list[Segment] = Field(min_length=1)
    turns: list[Turn]
    context_ops: list[ContextOp] = Field(default_factory=list)
    outcome: Outcome

    @model_validator(mode="after")
    def _check_references(self) -> Episode:
        n_segments, n_turns = len(self.segments), len(self.turns)
        if [s.index for s in self.segments] != list(range(n_segments)):
            raise ValueError("segment indices must be 0..n-1 in order")
        if [t.index for t in self.turns] != list(range(n_turns)):
            raise ValueError("turn indices must be 0..n-1 in order")
        if any(t.segment >= n_segments for t in self.turns):
            raise ValueError("a turn refers to a missing segment")
        if any(b.segment < a.segment for a, b in itertools.pairwise(self.turns)):
            raise ValueError("turn segments must not decrease")
        span_turns = (s.turn for seg in self.segments for s in seg.spans if s.turn is not None)
        if any(t >= n_turns for t in span_turns):
            raise ValueError("a span refers to a missing turn")
        if [op.turn for op in self.context_ops] != sorted(op.turn for op in self.context_ops):
            raise ValueError("context ops must be in turn order")
        for op in self.context_ops:
            if op.turn >= n_turns:
                raise ValueError("a context op refers to a missing turn")
            if op.new_segment is not None and op.new_segment >= n_segments:
                raise ValueError("a context op refers to a missing segment")
        return self


def append_episode(path: Path, episode: Episode) -> None:
    """Append one episode as a JSONL line; the rollout loop calls this once per episode."""
    with _open_text(path, "a") as f:
        f.write(episode.model_dump_json() + "\n")


def read_episodes(path: Path) -> Iterator[Episode]:
    """Yield episodes from a .jsonl or .jsonl.gz file; a bad line fails with its line number."""
    with _open_text(path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield Episode.model_validate_json(line)
            except ValidationError as err:
                raise ValueError(f"{path}:{line_no}: not a valid episode") from err


def _open_text(path: Path, mode: Literal["r", "a"]) -> TextIO:
    # Finished runs are gzipped (carryon.run.finish_run); the suffix alone decides.
    if path.suffix == ".gz":
        if mode == "r":
            return gzip.open(path, "rt", encoding="utf-8")
        return gzip.open(path, "at", encoding="utf-8")
    return path.open(mode, encoding="utf-8")

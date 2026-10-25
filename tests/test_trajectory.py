"""Trajectory schema: JSONL round trip, token exactness, and the invariants that guard training."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from carryon.rollout.trajectory import (
    ContextOp,
    Episode,
    Outcome,
    Segment,
    Span,
    ToolCall,
    Turn,
    append_episode,
    read_episodes,
)

# Toy tokenizer: the canonical split of "hello" is one token, but a sampler may emit "hell" + "o".
# Real BPE tokenizers behave the same way; a real-tokenizer test comes with the GPU stack (M0.8).
VOCAB = {"Q": 0, ":": 1, " ": 2, "h": 3, "e": 4, "l": 5, "o": 6}
VOCAB |= {"he": 7, "ll": 8, "hell": 9, "hello": 10}
ID_TO_TOKEN = {i: t for t, i in VOCAB.items()}
LONGEST = max(map(len, VOCAB))


def toy_encode(text: str) -> list[int]:
    """Greedy longest match: one canonical split per string, like a BPE tokenizer."""
    ids, i = [], 0
    while i < len(text):
        for n in range(min(LONGEST, len(text) - i), 0, -1):
            if text[i : i + n] in VOCAB:
                ids.append(VOCAB[text[i : i + n]])
                i += n
                break
        else:
            raise ValueError(f"cannot encode {text[i]!r}")
    return ids


def toy_decode(ids: list[int]) -> str:
    return "".join(ID_TO_TOKEN[i] for i in ids)


def prompt_span(text: str = "Q: ") -> Span:
    return Span(role="user", token_ids=toy_encode(text), sampled=False, text=text)


def sampled_span(ids: list[int], turn: int) -> Span:
    return Span(
        role="assistant",
        token_ids=ids,
        sampled=True,
        logprobs=[-0.5] * len(ids),
        turn=turn,
        text=toy_decode(ids),
    )


def make_episode(**overrides: Any) -> Episode:
    """Two segments: turn 0 searches, a summarize op compacts, turn 1 answers."""
    search = [VOCAB["he"], VOCAB["l"]]
    segment_0 = Segment(
        index=0,
        spans=[
            prompt_span(),
            sampled_span(search, turn=0),
            Span(role="tool", token_ids=toy_encode("hello hello"), sampled=False, turn=0),
            sampled_span([VOCAB["hello"]], turn=0),  # the summary, an RL-trained action
        ],
    )
    segment_1 = Segment(
        index=1,
        spans=[prompt_span(), prompt_span("hello"), sampled_span([VOCAB["hell"], VOCAB["o"]], 1)],
    )
    fields: dict[str, Any] = {
        "run_id": "test-run",
        "episode_id": "ep-0",
        "question_id": "q-0",
        "strategy": "summarize",
        "budget_tokens": 4096,
        "model": "toy",
        "seed": 0,
        "segments": [segment_0, segment_1],
        "turns": [
            Turn(
                index=0,
                segment=0,
                tool_call=ToolCall(name="search", arguments={"query": "hel"}),
                active_tokens=segment_0.n_tokens,
                prefix_reused_tokens=0,
                wall_time_s=0.25,
            ),
            Turn(
                index=1,
                segment=1,
                tool_call=ToolCall(name="answer", arguments={"text": "hello"}),
                active_tokens=segment_1.n_tokens,
            ),
        ],
        "context_ops": [
            ContextOp(
                kind="summarize",
                turn=0,
                tokens_before=segment_0.n_tokens,
                tokens_after=5,
                new_segment=1,
            )
        ],
        "outcome": Outcome(
            termination="answered",
            answer="hello",
            gold_answers=["hello"],
            reward=1.0,
            exact_match=True,
            f1=1.0,
            format_ok=True,
        ),
    }
    fields.update(overrides)
    return Episode(**fields)


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "trajectories.jsonl"
    episodes = [make_episode(), make_episode(episode_id="ep-1", seed=1)]
    for episode in episodes:
        append_episode(path, episode)
    assert path.read_text().count("\n") == 2
    assert list(read_episodes(path)) == episodes


def test_sampled_ids_survive_where_retokenization_would_not(tmp_path: Path) -> None:
    sampled = [VOCAB["hell"], VOCAB["o"]]
    # The hazard: decoding and re-encoding the sampled text yields different ids.
    assert toy_encode(toy_decode(sampled)) != sampled

    path = tmp_path / "trajectories.jsonl"
    append_episode(path, make_episode())
    (episode,) = read_episodes(path)
    segment = episode.segments[1]
    assert segment.token_ids()[-2:] == sampled
    assert segment.loss_mask()[-2:] == [1, 1]


def test_training_views_align() -> None:
    for segment in make_episode().segments:
        ids, mask, logprobs = segment.token_ids(), segment.loss_mask(), segment.sampled_logprobs()
        assert len(ids) == len(mask) == len(logprobs) == segment.n_tokens
        assert all((m == 1) == (lp is not None) for m, lp in zip(mask, logprobs, strict=True))


def test_loss_only_on_sampled_tokens() -> None:
    segment_0 = make_episode().segments[0]
    expected = [int(s.sampled) for s in segment_0.spans for _ in s.token_ids]
    assert segment_0.loss_mask() == expected
    assert sum(segment_0.loss_mask()) == 3  # search (2 tokens) + summary (1 token)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"role": "assistant", "token_ids": [1, 2], "sampled": True, "logprobs": [-0.1]},
        {"role": "assistant", "token_ids": [1], "sampled": True},
        {"role": "tool", "token_ids": [1], "sampled": False, "logprobs": [-0.1]},
        {"role": "user", "token_ids": [1], "sampled": True, "logprobs": [-0.1]},
        {"role": "assistant", "token_ids": [1], "sampled": True, "logprobs": [0.3]},
        {"role": "assistant", "token_ids": [1], "sampled": True, "logprobs": [float("-inf")]},
        {"role": "user", "token_ids": [], "sampled": False},
        {"role": "user", "token_ids": [-1], "sampled": False},
        {"role": "user", "token_ids": [1], "sampled": False, "mask": [0]},
    ],
    ids=[
        "logprob-count",
        "missing-logprobs",
        "logprobs-on-unsampled",
        "sampled-non-assistant",
        "positive-logprob",
        "infinite-logprob",
        "empty-span",
        "negative-token-id",
        "unknown-field",
    ],
)
def test_invalid_span(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Span(**kwargs)


def test_rewriting_op_needs_new_segment() -> None:
    with pytest.raises(ValidationError):
        ContextOp(kind="truncate", turn=0, tokens_before=10, tokens_after=5, new_segment=None)
    ContextOp(kind="fold_branch", turn=0, tokens_before=10, tokens_after=10, new_segment=None)


def test_invalid_references() -> None:
    episode = make_episode()
    bad_turn = episode.turns[1].model_copy(update={"segment": 5})
    with pytest.raises(ValidationError, match="missing segment"):
        make_episode(turns=[episode.turns[0], bad_turn])
    with pytest.raises(ValidationError, match="missing turn"):
        make_episode(turns=episode.turns[:1])
    with pytest.raises(ValidationError, match="must not decrease"):
        swapped = [
            episode.turns[1].model_copy(update={"index": 0}),
            episode.turns[0].model_copy(update={"index": 1}),
        ]
        make_episode(turns=swapped)


def test_other_schema_version_rejected() -> None:
    data = make_episode().model_dump()
    with pytest.raises(ValidationError):
        Episode.model_validate({**data, "schema_version": 2})


def test_bad_line_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "trajectories.jsonl"
    append_episode(path, make_episode())
    with path.open("a") as f:
        f.write('{"schema_version": 1}\n')
    with pytest.raises(ValueError, match=r"trajectories\.jsonl:2"):
        list(read_episodes(path))

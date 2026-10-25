"""Phase B instrumentation on a dummy episode built by a fake rollout loop.

The fake loop does what the real one will: before each model call it measures prefix reuse against
the previous call, it times the turn, and after a summarize it starts a new segment that repeats
only the prompt and the summary.
"""

import time

from carryon.rollout.instrumentation import common_prefix_length, turn_token_counts
from carryon.rollout.trajectory import (
    ContextOp,
    Episode,
    Outcome,
    Segment,
    Span,
    ToolCall,
    Turn,
)

PROMPT = Span(role="user", token_ids=list(range(100, 110)), sampled=False)  # 10 tokens


def generate(turn: int, n: int) -> Span:
    return Span(
        role="assistant",
        token_ids=[200 + turn] * n,
        sampled=True,
        logprobs=[-0.1] * n,
        turn=turn,
    )


def tool_output(turn: int, n: int) -> Span:
    return Span(role="tool", token_ids=[300 + turn] * n, sampled=False, turn=turn)


def ids(spans: list[Span]) -> list[int]:
    return [t for s in spans for t in s.token_ids]


def run_dummy_episode() -> Episode:
    """Turn 0 and 1 search under append; a summarize op compacts; turn 2 answers."""
    segments: list[list[Span]] = [[PROMPT]]
    turns: list[Turn] = []
    ops: list[ContextOp] = []
    previous_call: list[int] = []
    plan = [(5, 40), (6, 50), (4, 0)]  # (generated, tool) tokens per turn
    for index, (n_gen, n_tool) in enumerate(plan):
        start = time.perf_counter()
        current = segments[-1]
        model_input = ids(current)
        reused = common_prefix_length(previous_call, model_input)
        output = generate(index, n_gen)
        previous_call = model_input + output.token_ids
        current.append(output)
        if n_tool:
            current.append(tool_output(index, n_tool))
        turns.append(
            Turn(
                index=index,
                segment=len(segments) - 1,
                tool_call=ToolCall(name="search" if n_tool else "answer", arguments={}),
                active_tokens=len(ids(current)),
                prefix_reused_tokens=reused,
                wall_time_s=time.perf_counter() - start,
            )
        )
        if index == 1:  # the context manager compacts after turn 1
            summary = generate(index, 3)
            before = len(ids(current)) + len(summary.token_ids)
            segments[-1].append(summary)
            segments.append([PROMPT, summary])
            ops.append(
                ContextOp(
                    kind="summarize",
                    turn=index,
                    tokens_before=before,
                    tokens_after=len(ids(segments[-1])),
                    new_segment=len(segments) - 1,
                )
            )
    return Episode(
        run_id="dummy",
        episode_id="ep-0",
        question_id="q-0",
        strategy="summarize",
        budget_tokens=4096,
        model="toy",
        seed=0,
        segments=[Segment(index=i, spans=s) for i, s in enumerate(segments)],
        turns=turns,
        context_ops=ops,
        outcome=Outcome(
            termination="answered",
            answer="a",
            gold_answers=["a"],
            reward=1.0,
            exact_match=True,
            f1=1.0,
            format_ok=True,
        ),
    )


def test_common_prefix_length() -> None:
    assert common_prefix_length([], [1, 2]) == 0
    assert common_prefix_length([1, 2, 3], [1, 2, 3, 4]) == 3
    assert common_prefix_length([1, 2, 3, 4], [1, 2]) == 2
    assert common_prefix_length([1, 9, 3], [1, 2, 3]) == 1


def test_dummy_episode_fields_are_populated() -> None:
    episode = run_dummy_episode()
    assert all(t.prefix_reused_tokens is not None for t in episode.turns)
    assert all(t.wall_time_s is not None and t.wall_time_s >= 0 for t in episode.turns)
    assert [t.active_tokens for t in episode.turns] == [
        10 + 5 + 40,
        10 + 5 + 40 + 6 + 50,
        10 + 3 + 4,
    ]


def test_prefix_reuse_follows_the_strategy() -> None:
    reused = [t.prefix_reused_tokens for t in run_dummy_episode().turns]
    # Turn 0: nothing cached. Turn 1 (append): the whole previous call is a prefix (10 + 5).
    # Turn 2 (after summarize): only the 10 prompt tokens survive; the summary is new text there.
    assert reused == [0, 15, 10]


def test_turn_token_counts() -> None:
    counts = turn_token_counts(run_dummy_episode())
    assert [(c.generated, c.tool) for c in counts] == [(5, 40), (6 + 3, 50), (4, 0)]
    assert [c.added for c in counts] == [45, 59, 4]

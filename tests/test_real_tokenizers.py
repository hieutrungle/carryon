"""Token exactness with the real tokenizers of both model families (needs the Hugging Face Hub).

A sampler can emit a non-canonical split ("hel" + "lo"); decoding and re-encoding returns the
canonical one. Storing sampled ids (carryon.rollout.trajectory) is what keeps training exact.
"""

from pathlib import Path

import pytest
from test_trajectory import make_episode
from tokenizers import Tokenizer

from carryon.rollout.trajectory import Segment, Span, append_episode, read_episodes

MODELS = ["Qwen/Qwen3.5-4B", "google/gemma-4-E4B-it"]


def load(model: str) -> Tokenizer:
    try:
        return Tokenizer.from_pretrained(model)
    except Exception as err:  # offline machine: skip locally; CI has network
        pytest.skip(f"cannot download the {model} tokenizer: {err}")


@pytest.mark.parametrize("model", MODELS)
def test_sampled_ids_survive_where_retokenization_would_not(model: str, tmp_path: Path) -> None:
    tok = load(model)
    # Two tokens a model could sample in a row, whose joined text has a different encoding.
    hel = tok.encode("hel", add_special_tokens=False).ids
    lo = tok.encode("lo", add_special_tokens=False).ids
    sampled = hel + lo
    assert tok.decode(sampled) == "hello"
    assert tok.encode("hello", add_special_tokens=False).ids != sampled

    base = make_episode()
    answer = Span(
        role="assistant", token_ids=sampled, sampled=True, logprobs=[-0.2] * len(sampled), turn=1
    )
    segment = Segment(index=1, spans=[*base.segments[1].spans, answer])
    path = tmp_path / "trajectories.jsonl"
    append_episode(path, make_episode(segments=[base.segments[0], segment]))

    (episode,) = read_episodes(path)
    assert episode.segments[1].token_ids()[-len(sampled) :] == sampled
    assert episode.segments[1].loss_mask()[-len(sampled) :] == [1] * len(sampled)

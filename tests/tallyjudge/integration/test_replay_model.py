import asyncio
import json

import pytest
from pydantic import BaseModel

from tallyjudge import (
    Dataset,
    Evaluator,
    ExecutionConfig,
    create_cache_backend,
    create_language_model,
    create_metric,
)
from tallyjudge.sdk import ModelError, PromptInput
from tallyjudge.sdk.dataset import canonical_sha256

FAITH = "metric.rag.faithfulness"
REPLAY = "model.replay.jsonl"
ROW = {
    "user_input": "图书馆什么时候开放？",
    "response": "工作日上午九点开放，周末也开放。",
    "retrieved_contexts": ["图书馆公告：工作日上午九点开放。"],
}


def _score(model):
    metric = create_metric(FAITH, llm=model)
    return Evaluator([metric]).arun(Dataset.from_records([ROW]))


@pytest.mark.asyncio
async def test_recorded_replies_reproduce_the_score_without_requests(client, tmp_path):
    recording = tmp_path / "judge.jsonl"
    client.chat_reply(json.dumps({"statements": ["工作日九点开放", "周末开放"]}))
    client.chat_reply(
        json.dumps(
            {
                "statements": [
                    {"statement": "工作日九点开放", "reason": "公告写明", "verdict": 1},
                    {"statement": "周末开放", "reason": "公告未提", "verdict": 0},
                ]
            }
        )
    )
    live = create_language_model(
        "model.openai_compatible.chat_completions", client=client, model="judge"
    )
    recorded = await _score(create_language_model(REPLAY, path=recording, llm=live))
    assert recorded.rows[0].metrics[FAITH].value == 0.5
    assert len(client.calls) == 2

    replayed = await _score(create_language_model(REPLAY, path=recording))
    score = replayed.rows[0].metrics[FAITH]
    assert score.value == 0.5 and len(client.calls) == 2
    assert score.usage.untracked_sources == [REPLAY]


@pytest.mark.asyncio
async def test_prompt_without_a_recording_is_a_model_error(tmp_path):
    result = await _score(create_language_model(REPLAY, path=tmp_path / "empty.jsonl"))
    score = result.rows[0].metrics[FAITH]
    assert (score.status, score.error.category) == ("error", "model_error")


async def _score_votes(judge, cases, *, votes=1, cache=None):
    metric = create_metric(
        "metric.custom.numeric",
        llm=judge,
        criteria="quality",
        levels=[0, 1],
        strictness=votes,
    )
    data = Dataset.from_records(
        [{"case_id": case, "response": "same"} for case in cases]
    )
    run = await Evaluator(
        [metric], config=ExecutionConfig(max_concurrency=1), cache=cache
    ).arun(data)
    return {row.case_id: row.metrics[metric.spec.id] for row in run.rows}


@pytest.mark.asyncio
async def test_replay_preserves_votes_and_their_cases_after_reordering(
    client, tmp_path
):
    path = tmp_path / "votes.jsonl"
    for vote in [0, 1, 1, 1, 0, 0]:
        client.chat_reply(json.dumps({"value": vote, "reason": str(vote)}))
    llm = create_language_model(
        "model.openai_compatible.chat_completions", client=client, model="judge"
    )
    cache = create_cache_backend(
        "cache.local.sqlite", path=tmp_path / "scores.db", namespace="repeat"
    )
    original = await _score_votes(
        create_language_model(REPLAY, path=path, llm=llm),
        ["a", "b"],
        votes=3,
        cache=cache,
    )
    replay = await _score_votes(
        create_language_model(REPLAY, path=path), ["b", "a"], votes=3, cache=cache
    )
    assert {case: s.value for case, s in original.items()} == {"a": 1.0, "b": 0.0}
    assert {case: s.value for case, s in replay.items()} == {"a": 1.0, "b": 0.0}
    assert len(client.calls) == 6


@pytest.mark.asyncio
async def test_different_recordings_with_same_filename_do_not_share_scores(
    client, tmp_path
):
    cache = create_cache_backend(
        "cache.local.sqlite", path=tmp_path / "scores.db", namespace="test"
    )
    llm = create_language_model(
        "model.openai_compatible.chat_completions", client=client, model="judge"
    )
    for vote in [0, 1]:
        path = tmp_path / str(vote) / "votes.jsonl"
        client.chat_reply(json.dumps({"value": vote, "reason": str(vote)}))
        await _score_votes(create_language_model(REPLAY, path=path, llm=llm), ["a"])
        for expected_hit in [False, True]:
            scores = await _score_votes(
                create_language_model(REPLAY, path=path), ["a"], cache=cache
            )
            assert (scores["a"].value, scores["a"].cache_hit) == (vote, expected_hit)


class Answer(BaseModel):
    text: str


@pytest.mark.asyncio
async def test_concurrent_calls_replay_in_invocation_order(client, tmp_path):
    path = tmp_path / "overlap.jsonl"
    llm = create_language_model(
        "model.openai_compatible.chat_completions", client=client, model="judge"
    )
    recorder = create_language_model(REPLAY, path=path, llm=llm)
    prompt = PromptInput.from_text("same prompt")
    entered, release = asyncio.Event(), asyncio.Event()
    client.chat_reply('{"text":"first"}')
    first_reply = client.responses.popleft()

    async def delayed():
        entered.set()
        await release.wait()
        return first_reply

    client.responses.append(delayed)
    client.chat_reply('{"text":"second"}')
    first = asyncio.create_task(recorder.agenerate(prompt, Answer))
    await entered.wait()
    assert (await recorder.agenerate(prompt, Answer)).data.text == "second"
    release.set()
    assert (await first).data.text == "first"
    replay = create_language_model(REPLAY, path=path)
    assert (await replay.agenerate(prompt, Answer)).data.text == "first"
    assert (await replay.agenerate(prompt, Answer)).data.text == "second"


@pytest.mark.asyncio
async def test_legacy_entry_can_be_played_once(tmp_path):
    prompt = PromptInput.from_text("legacy")
    key = canonical_sha256(
        {"messages": [["user", "legacy"]], "schema": Answer.model_json_schema()}
    )
    path = tmp_path / "legacy.jsonl"
    path.write_text(
        json.dumps({"key": key, "reply": {"text": "old answer"}}), encoding="utf-8"
    )
    replay = create_language_model(REPLAY, path=path)
    assert (await replay.agenerate(prompt, Answer)).data.text == "old answer"
    with pytest.raises(ModelError, match="No recorded reply"):
        await replay.agenerate(prompt, Answer)

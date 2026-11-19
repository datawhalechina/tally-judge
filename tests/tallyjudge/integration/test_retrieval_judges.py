"""Public model behavior of the two ranked context judges."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from tallyjudge import Dataset, Evaluator, create_language_model, create_metric
from tallyjudge.core.errors import InputValidationError


@pytest.mark.asyncio
async def test_recall_requires_a_reference_and_maps_malformed_classifications(client):
    judge = make_judge(client)
    metric = create_metric("metric.rag.context_recall", llm=judge)
    data = Dataset.from_records(
        [{"user_input": "q", "retrieved_contexts": ["doc"], "reference": ""}]
    )
    invalid = (await Evaluator([metric]).arun(data)).rows[0].metrics[metric.spec.id]
    assert invalid.error.field == "reference" and not client.calls
    client.chat_reply('{"classifications":[{"reason":"missing attribution"}]}')
    data = Dataset.from_records(
        [{"user_input": "q", "retrieved_contexts": ["doc"], "reference": "answer"}]
    )
    failed = (await Evaluator([metric]).arun(data)).rows[0].metrics[metric.spec.id]
    assert (failed.status, failed.error.category, failed.error.code) == (
        "error",
        "model_output_error",
        "schema_validation",
    )
    assert failed.usage.generation.requests == 1


def make_judge(client, retries=0):
    return create_language_model(
        "model.openai_compatible.chat_completions",
        client=client,
        model="offline-judge",
        structured_retries=retries,
    )


@pytest.mark.parametrize("without_reference", [True, False])
@pytest.mark.asyncio
async def test_precision_rejects_empty_answer_before_any_call(
    client, without_reference
):
    suffix = "without_reference" if without_reference else "with_reference"
    metric = create_metric(
        "metric.rag.context_precision_" + suffix, llm=make_judge(client)
    )
    with pytest.raises(InputValidationError):
        await metric.ascore("question", "", ["evidence"])
    assert client.calls == []


@pytest.mark.asyncio
async def test_failed_later_passage_keeps_usage_from_earlier_attempts(client):
    metric = create_metric(
        "metric.rag.context_precision_with_reference", llm=make_judge(client, retries=1)
    )
    client.chat_reply("not JSON")
    client.chat_reply(
        json.dumps({"verdict": 1, "reason": "match"}),
        tokens=SimpleNamespace(prompt_tokens=4, completion_tokens=2),
    )
    client.responses.append(RuntimeError("transport failed"))
    result = await Evaluator([metric]).arun(
        Dataset.from_records(
            [
                {
                    "user_input": "q",
                    "reference": "r",
                    "retrieved_contexts": ["a", "b", "c"],
                }
            ]
        )
    )
    score = result.rows[0].metrics[metric.spec.id]
    assert (score.status, score.error.category) == ("error", "model_error")
    assert len(client.calls) == score.usage.generation.requests == 3
    assert score.usage.generation.known_tokens == {
        "input_tokens": 4,
        "output_tokens": 2,
    }
    assert score.usage.generation.requests_without_usage == 2


@pytest.mark.asyncio
async def test_direct_precision_cancellation_stops_following_passages(client):
    metric = create_metric(
        "metric.rag.context_precision_without_reference", llm=make_judge(client)
    )
    started, release, returned = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def slow():
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            await release.wait()
        returned.set()
        return client.responses.popleft()

    client.responses.append(slow)
    client.chat_reply('{"verdict":1,"reason":"late"}')
    task = asyncio.create_task(metric.ascore("q", "a", ["a", "b"]))
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
    await asyncio.wait_for(returned.wait(), 2)
    assert len(client.calls) == 1

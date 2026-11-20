"""Public request, cancellation and cache contracts for the six diagnostic judges."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from tallyjudge import (
    Dataset,
    Evaluator,
    MetricBinding,
    RunCollector,
    create_cache_backend,
    create_language_model,
    create_metric,
)
from tallyjudge.core.errors import PluginError
from tallyjudge.registry import registry

ROW = {
    "case_id": "example",
    "user_input": "question",
    "response": "answer",
    "reference": "gold",
    "retrieved_contexts": ["first", "second"],
}
FIRST_REPLIES = [
    ("context_entity_recall", {"entities": ["x"]}),
    ("factual_correctness", {"claims": ["x"]}),
    ("noise_sensitivity", {"statements": ["x"]}),
    ("answer_accuracy", {"rating": 4}),
    ("context_relevance", {"rating": 2}),
    ("response_groundedness", {"rating": 2}),
]


def model(client, name="diagnostic"):
    return create_language_model(
        "model.openai_compatible.chat_completions",
        client=client,
        model=name,
        temperature=0.0,
        structured_retries=0,
    )


@pytest.mark.parametrize(
    ("metric", "settings", "replies", "code"),
    [
        ("context_entity_recall", {}, [{"entities": []}], "no_reference_entities"),
        ("factual_correctness", {}, [{"claims": []}], "no_response_statements"),
        (
            "factual_correctness",
            {},
            [{"claims": ["x"]}, {"statements": []}],
            "no_verdicts",
        ),
        (
            "factual_correctness",
            {"mode": "precision"},
            [
                {"claims": ["x"]},
                {"statements": [{"statement": "x", "reason": "judge", "verdict": -1}]},
            ],
            "invalid_verdict",
        ),
        (
            "noise_sensitivity",
            {},
            [
                {"statements": ["gold"]},
                {"statements": ["x"]},
                {
                    "statements": [
                        {"statement": "gold", "reason": "judge", "verdict": 2}
                    ]
                },
            ],
            "invalid_verdict",
        ),
    ],
)
@pytest.mark.asyncio
async def test_terminal_judgment_stops_before_an_unrelated_transport_failure(
    client, metric, settings, replies, code
):
    plugin = create_metric("metric.rag." + metric, llm=model(client), **settings)
    for reply in replies:
        client.chat_reply(json.dumps(reply))
    client.responses.append(RuntimeError("an unnecessary request was made"))
    run = await Evaluator([plugin]).arun(Dataset.from_records([ROW]))
    score = run.rows[0].metrics[plugin.spec.id]
    category = "model_output_error" if code == "invalid_verdict" else "undefined_score"
    assert (score.error.category, score.error.code) == (category, code)
    assert score.usage.generation.requests == len(client.calls) == len(replies)


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "first"), FIRST_REPLIES)
async def test_failed_second_call_keeps_observed_usage(client, name, first):
    metric = create_metric("metric.rag." + name, llm=model(client))
    client.chat_reply(
        json.dumps(first), tokens=SimpleNamespace(prompt_tokens=7, completion_tokens=2)
    )
    client.responses.append(RuntimeError("offline transport failure"))
    run = await Evaluator([metric]).arun(Dataset.from_records([ROW]))
    score = run.rows[0].metrics[metric.spec.id]
    assert (score.status, score.error.category, score.error.exception_type) == (
        "error",
        "model_error",
        "RuntimeError",
    )
    usage = score.usage.generation
    assert len(client.calls) == usage.requests == 2
    assert usage.known_tokens == {"input_tokens": 7, "output_tokens": 2}
    assert usage.requests_without_usage == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "first"), FIRST_REPLIES)
async def test_direct_cancel_stops_following_requests_even_if_transport_swallows_it(
    client, name, first
):
    metric = create_metric("metric.rag." + name, llm=model(client))
    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def delayed():
        entered.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            await release.wait()
        finished.set()
        return client.responses.popleft()

    client.responses.append(delayed)
    client.chat_reply(json.dumps(first))
    task = asyncio.create_task(
        metric.ascore(**{key: ROW[key] for key in metric.spec.inputs})
    )
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
    await asyncio.wait_for(finished.wait(), 2)
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_cancelled_second_rating_preserves_the_first_judgment_usage(client):
    metric = create_metric("metric.rag.answer_accuracy", llm=model(client))
    client.chat_reply(
        '{"rating": 4}', tokens=SimpleNamespace(prompt_tokens=7, completion_tokens=2)
    )
    entered = asyncio.Event()

    async def blocked():
        entered.set()
        await asyncio.Future()

    client.responses.append(blocked)
    collector = RunCollector()
    run = asyncio.create_task(
        Evaluator([metric]).arun(Dataset.from_records([ROW]), collector=collector)
    )
    await asyncio.wait_for(entered.wait(), 2)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    partial = collector.snapshot()
    score = partial.rows[0].metrics[metric.spec.id]
    assert (partial.run_status, score.status) == ("cancelled", "cancelled")
    usage = score.usage.generation
    assert usage.requests == len(client.calls) == 2
    assert usage.known_tokens == {"input_tokens": 7, "output_tokens": 2}


@pytest.mark.asyncio
async def test_noise_mode_and_model_partition_cache_and_bindings(client, tmp_path):
    cache = create_cache_backend(
        "cache.local.sqlite", namespace="noise", path=tmp_path / "cache.db"
    )
    data = Dataset.from_records([ROW])

    def bind(mode, deployment):
        metric = create_metric(
            "metric.rag.noise_sensitivity", llm=model(client, deployment), mode=mode
        )
        return MetricBinding(metric, key=mode)

    def queue_responses():
        client.chat_reply('{"statements":["gold"]}')
        client.chat_reply('{"statements":["A","B","C","D"]}')
        for flags in ([1], [1, 1, 1, 0], [0], [0, 0, 1, 1], [1, 0, 0, 0]):
            client.chat_reply(
                json.dumps(
                    {
                        "statements": [
                            {"statement": str(i), "reason": "fixture", "verdict": v}
                            for i, v in enumerate(flags)
                        ]
                    }
                )
            )

    for deployment, mode, value in (
        ("judge", "relevant", 0.5),
        ("judge", "irrelevant", 0.25),
        ("other", "irrelevant", 0.25),
    ):
        queue_responses()
        cold = await Evaluator([bind(mode, deployment)], cache=cache).arun(data)
        score = cold.rows[0].metrics[mode]
        assert (score.status, score.value, score.cache_hit) == ("success", value, False)
        assert cold.usage.generation.requests == 7
        snapshot = cold.metrics[mode]["config_snapshot"]
        assert (
            snapshot["llm"]["config"]["model"] == deployment
            and snapshot["mode"] == mode
        )

    paired = [bind("relevant", "judge"), bind("irrelevant", "judge")]
    warm = await Evaluator(paired, cache=cache).arun(data)
    assert warm.usage is None and len(client.calls) == 21
    for key, value in (("relevant", 0.5), ("irrelevant", 0.25)):
        score = warm.rows[0].metrics[key]
        assert score.cache_hit and score.value == value
        assert warm.metrics[key]["direction"] == "lower_is_better"


@pytest.mark.parametrize(
    ("metric", "settings"),
    [
        ("factual_correctness", {"mode": "accuracy"}),
        ("factual_correctness", {"beta": float("nan")}),
        ("factual_correctness", {"atomicity": "medium"}),
        ("factual_correctness", {"coverage": "partial"}),
        ("noise_sensitivity", {"mode": "all"}),
    ],
)
def test_invalid_options_are_rejected_before_scoring(client, metric, settings):
    plugin_id = "metric.rag." + metric
    with pytest.raises(PluginError, match=plugin_id):
        registry.describe(plugin_id, **settings)
    with pytest.raises(PluginError, match=plugin_id):
        create_metric(plugin_id, llm=model(client), **settings)
    assert not client.calls

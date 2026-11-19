import asyncio
import sys
from types import SimpleNamespace as NS

import pytest
from pydantic import BaseModel

from tallyjudge import (
    Dataset,
    Evaluator,
    ExecutionConfig,
    MetricBinding,
    RunCollector,
    create_embedding_model,
    create_language_model,
    create_metric,
)
from tallyjudge.core.errors import (
    EvaluationAborted,
    InputValidationError,
    ModelOutputError,
    PluginError,
)
from tallyjudge.core.prompts import PromptInput
from tallyjudge.core.usage import UsageSummary
from tallyjudge.interfaces.models import GenerationResult
from tallyjudge.registry import PluginRegistry, PluginSpec, registry

CHAT = "model.openai_compatible.chat_completions"
EMBED = "embedding.openai_compatible.embeddings"
FAITH = "metric.rag.faithfulness"
RELEVANCE = "metric.rag.answer_relevance"
ROW = {
    "user_input": "question",
    "response": "answer",
    "retrieved_contexts": ["context"],
}


class Response(BaseModel):
    answer: str


def judge(client, **config):
    return create_language_model(CHAT, model="judge", client=client, **config)


def embedder(client, **config):
    return create_embedding_model(EMBED, model="embedding", client=client, **config)


def test_factories_and_lazy_descriptions(client):
    assert set(registry.describe(FAITH).inputs) == set(ROW)
    with pytest.raises(PluginError, match="Unexpected plugin kind"):
        create_metric(CHAT, client=client, model="x")
    with pytest.raises(PluginError, match="llm"):
        create_metric(FAITH)
    with pytest.raises(PluginError, match="embeddings"):
        create_metric(RELEVANCE, llm=judge(client), embeddings=object())
    with pytest.raises(PluginError, match="configuration"):
        judge(client, top_p=0.5, temperature=0.01)
    with pytest.raises(PluginError, match="configuration"):
        judge(client, api_key="private")
    with pytest.raises(PluginError, match="configuration"):
        create_metric(
            RELEVANCE, llm=judge(client), embeddings=embedder(client), strictness=True
        )
    with pytest.raises(PluginError) as error:
        judge(object())
    assert isinstance(error.value.__cause__, AttributeError)
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["json_object", "text"])
async def test_structured_retry_and_known_zero(client, mode):
    client.chat_reply("broken", tokens=NS(prompt_tokens=0, completion_tokens=3))
    client.chat_reply('```json\n{"answer":"ok"}\n```')
    model = judge(
        client,
        structured_output=mode,
        max_output_tokens=9,
        token_limit_param="max_completion_tokens",
    )
    out = await model.agenerate(PromptInput.from_text("test"), Response)
    assert (out.data.answer, out.model) == ("ok", "served-chat")
    assert out.usage.requests == 2 and out.usage.requests_without_usage == 1
    assert out.usage.known_counts == {"input_tokens": 1, "output_tokens": 1}
    assert out.usage.known_tokens == {"input_tokens": 0, "output_tokens": 3}
    assert not out.usage.complete
    body = client.calls[1][1]
    assert body["max_completion_tokens"] == 9 and "max_tokens" not in body
    assert body.get("response_format") == (
        {"type": "json_object"} if mode == "json_object" else None
    )
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assert "temperature" not in body and "top_p" not in body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "stop", "refusal", "code"),
    [
        ("", "stop", None, "empty_content"),
        ("{}", "length", None, "truncated"),
        (None, "stop", "refused", "refusal"),
        ("{}", "tool_calls", None, "unexpected_response"),
        ('prefix {"answer":"a"}', "stop", None, "invalid_json"),
        ("{}", "stop", None, "schema_validation"),
    ],
)
async def test_output_errors_have_stable_codes(client, text, stop, refusal, code):
    client.chat_reply(text, finish=stop, refusal=refusal)
    with pytest.raises(ModelOutputError) as info:
        await judge(client, structured_retries=0).agenerate(
            PromptInput.from_text("x"), Response
        )
    assert info.value.code == code
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_partial_embedding_usage_and_fail_fast_snapshot(client):
    client.chat_reply('{"question":"a","noncommittal":0}', tokens=NS(prompt_tokens=4))
    client.chat_reply('{"question":"b","noncommittal":1}')
    client.embedding_reply([(0, [1.0, 0.0])], tokens=NS(prompt_tokens=3))
    client.embedding_reply([(0, [1.0, 0.0])], tokens=NS(prompt_tokens=2))
    client.responses.append(OSError("private-endpoint"))
    m = create_metric(
        RELEVANCE,
        llm=judge(client),
        embeddings=embedder(client, batch_size=1),
        strictness=2,
    )
    with pytest.raises(EvaluationAborted) as e:
        await Evaluator(
            [m], config=ExecutionConfig(failure_policy="raise", max_concurrency=1)
        ).arun(Dataset.from_records([ROW, ROW]))
    r = e.value.partial_result
    score = r.rows[0].metrics[RELEVANCE]
    assert (r.run_status, score.error.category, score.error.exception_type) == (
        "aborted",
        "model_error",
        "OSError",
    )
    assert r.rows[1].metrics[RELEVANCE].status == "not_started"
    assert score.usage.generation.requests == 2 and score.usage.embedding.requests == 3
    assert score.usage.embedding.known_tokens == {
        "input_tokens": 5,
        "output_tokens": None,
    }
    assert score.usage.embedding.requests_without_usage == 1
    assert r.summary[RELEVANCE]["usage"]["embedding"]["requests"] == 3
    assert "private-endpoint" not in r.to_json()


@pytest.mark.asyncio
async def test_external_cancel_seals_inflight_and_stops_late_request(client):
    entered, release, returned = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def slow():
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        returned.set()
        return NS(
            choices=[
                NS(finish_reason="stop", message=NS(content='{"statements":["late"]}'))
            ],
            model="m",
            usage=NS(prompt_tokens=99),
        )

    client.responses.append(slow)
    m = create_metric(FAITH, llm=judge(client))
    collector = RunCollector()
    task = asyncio.create_task(
        Evaluator([m], config=ExecutionConfig(cleanup_timeout=0.02)).arun(
            Dataset.from_records([ROW]), collector=collector
        )
    )
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    snap = collector.snapshot()
    score = snap.rows[0].metrics[FAITH]
    assert (snap.run_status, score.status) == ("cancelled", "cancelled")
    assert (
        score.usage.generation.requests
        == score.usage.generation.in_flight_requests
        == 1
    )
    assert score.usage.generation.known_tokens["input_tokens"] is None
    before = snap.to_json()
    release.set()
    await asyncio.wait_for(returned.wait(), 1)
    await asyncio.sleep(0)
    assert collector.snapshot().to_json() == before
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_custom_untracked_source_and_validation(client):
    class Custom:
        plugin_id = "model.test.custom"
        usage_tracking = "untracked"

        def config_snapshot(self):
            return {}

        async def agenerate(self, prompt, response_model):
            return GenerationResult(response_model(statements=[]), UsageSummary())

    m = create_metric(FAITH, llm=Custom())
    rows = [ROW, dict(ROW, retrieved_contexts=[]), dict(ROW, response=1)]
    r = await Evaluator([m]).arun(Dataset.from_records(rows))
    scores = [row.metrics[FAITH] for row in r.rows]
    assert scores[0].status == "undefined" and r.run_status == "completed"
    assert scores[0].usage.untracked_sources == ["model.test.custom"]
    assert (
        scores[0].usage.generation.requests == 0
        and not scores[0].usage.generation.complete
    )
    assert [(s.error.field, s.error.expected, s.usage) for s in scores[1:]] == [
        ("retrieved_contexts", "non-empty list[str]", None),
        ("response", "non-empty str", None),
    ]
    assert r.summary[FAITH]["mean"] is None
    assert r.summary[FAITH]["status_counts"]["undefined"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "items",
    [
        [(0, [1]), (0, [2])],
        [(1, [1]), (0, [])],
        [(0, [True]), (1, [2])],
        [(1, [float("inf")]), (0, [0])],
        [(0, [0, 1]), (1, [1])],
        [(2, [1]), (0, [1])],
    ],
)
async def test_embedding_rejects_invalid_shapes(client, items):
    client.embedding_reply(items)
    with pytest.raises(ModelOutputError, match="invalid_embedding"):
        await embedder(client).aembed_texts(["a", "b"])


@pytest.mark.asyncio
async def test_embedding_restores_order_and_checks_batches(client):
    model = embedder(client, batch_size=2)
    assert (await model.aembed_texts([])).usage.requests == 0
    client.embedding_reply([(1, [0.0, 1.0]), (0, [1.0, 0.0])])
    assert (await model.aembed_texts(["a", "b"])).vectors == [[1.0, 0.0], [0.0, 1.0]]
    client.embedding_reply([(1, [1.0]), (0, [1.0])])
    client.embedding_reply([(0, [1.0, 1.0])])
    with pytest.raises(ModelOutputError, match="invalid_embedding"):
        await model.aembed_texts(["a", "b", "c"])


@pytest.mark.asyncio
async def test_cosine_handles_extreme_finite_components(client):
    client.chat_reply('{"question":"Q","noncommittal":0}')
    for _ in range(2):
        client.embedding_reply([(0, [sys.float_info.max] * 2)])
    m = create_metric(
        RELEVANCE, llm=judge(client), embeddings=embedder(client), strictness=1
    )
    r = await Evaluator([m]).arun(Dataset.from_records([ROW]))
    assert r.rows[0].metrics[RELEVANCE].value == pytest.approx(1.0, abs=1e-12, rel=0)


@pytest.mark.asyncio
async def test_named_scorer_and_evaluator_validate_once_before_requests(
    client, monkeypatch
):
    from tallyjudge.interfaces import metrics
    from tallyjudge.plugins.metrics.rag.answer_relevance import AnswerRelevanceMetric
    from tallyjudge.plugins.metrics.rag.faithfulness import FaithfulnessMetric

    checked = []
    validate = metrics.validate_inputs

    def track(spec, inputs):
        checked.append(spec.id)
        return validate(spec, inputs)

    monkeypatch.setattr(metrics, "validate_inputs", track)
    language, embeddings = judge(client), embedder(client)
    faith, relevance = (
        FaithfulnessMetric(language),
        AnswerRelevanceMetric(language, embeddings),
    )
    with pytest.raises(InputValidationError) as error:
        await faith.ascore("q", "a", "a single string")
    assert error.value.field == "retrieved_contexts"
    with pytest.raises(InputValidationError):
        await relevance.ascore("", "a")
    with pytest.raises(ValueError, match="strictness"):
        AnswerRelevanceMetric(language, embeddings, strictness=0)
    result = await Evaluator([faith, relevance]).arun(
        Dataset.from_records([dict(ROW, response="")])
    )
    assert all(
        s.error.field == "response" and s.usage is None
        for s in result.rows[0].metrics.values()
    )
    assert checked == [FAITH, RELEVANCE, FAITH, RELEVANCE]
    assert not client.calls


def test_misconfigured_lazy_plugin_schema_names_plugin():
    spec = PluginSpec(FAITH, "metric", 1, "unused:factory", config_type="builtins:str")
    with pytest.raises(PluginError, match=FAITH):
        PluginRegistry([spec]).create(FAITH)


@pytest.mark.asyncio
async def test_exhausted_retry_keeps_known_usage_and_safe_reason(client, caplog):
    client.chat_reply(
        "private malformed response", tokens=NS(prompt_tokens=-7, completion_tokens=0)
    )
    client.chat_reply("{}", tokens=NS(prompt_tokens=4))
    data = await Evaluator([create_metric(FAITH, llm=judge(client))]).arun(
        Dataset.from_records([ROW])
    )
    score = data.rows[0].metrics[FAITH]
    assert (score.status, score.error.category, score.error.code) == (
        "error",
        "model_output_error",
        "schema_validation",
    )
    assert score.usage.generation.requests == 2 and score.usage.generation.complete
    assert score.usage.generation.known_counts == {
        "input_tokens": 1,
        "output_tokens": 1,
    }
    assert score.usage.generation.known_tokens == {
        "input_tokens": 4,
        "output_tokens": 0,
    }
    assert "Invalid token count for prompt_tokens" in caplog.text
    assert "private malformed response" not in data.to_json()


@pytest.mark.asyncio
async def test_concurrent_rows_and_bindings_keep_their_own_usage(client):
    async def generate(*, messages, model, **options):
        await asyncio.sleep(0)
        units = 1 if '"answer": "small"' in messages[0]["content"] else 2
        tokens = units * (10 if model == "second" else 1)
        return NS(
            choices=[NS(finish_reason="stop", message=NS(content='{"statements":[]}'))],
            usage=NS(prompt_tokens=tokens),
        )

    client.chat.completions.create = generate
    bindings = [
        MetricBinding(
            create_metric(
                FAITH, llm=create_language_model(CHAT, client=client, model=name)
            ),
            key=name,
        )
        for name in ("first", "second")
    ]
    data = Dataset.from_records(
        [dict(ROW, response="small"), dict(ROW, response="large")]
    )
    result = await Evaluator(bindings, config=ExecutionConfig(max_concurrency=4)).arun(
        data
    )
    assert [
        [s.usage.generation.known_tokens["input_tokens"] for s in row.metrics.values()]
        for row in result.rows
    ] == [[1, 10], [2, 20]]
    assert [
        result.summary[key]["usage"]["generation"]["known_tokens"]["input_tokens"]
        for key in ("first", "second")
    ] == [3, 30]


@pytest.mark.asyncio
async def test_cancel_between_responses_preserves_known_tokens_and_queued_row(client):
    started = asyncio.Event()

    async def pending():
        started.set()
        await asyncio.Future()

    client.chat_reply(
        '{"statements":["one"]}', tokens=NS(prompt_tokens=2, completion_tokens=1)
    )
    client.responses.append(pending)
    metric = create_metric(FAITH, llm=judge(client))
    collector = RunCollector()
    task = asyncio.create_task(
        Evaluator([metric], config=ExecutionConfig(max_concurrency=1)).arun(
            Dataset.from_records([ROW, ROW]), collector=collector
        )
    )
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    first, queued = (row.metrics[FAITH] for row in collector.snapshot().rows)
    counted = first.usage.generation
    assert (
        first.status,
        counted.requests,
        counted.in_flight_requests,
        counted.requests_without_usage,
    ) == ("cancelled", 2, 1, 1)
    assert counted.known_tokens == {"input_tokens": 2, "output_tokens": 1}
    assert (queued.status, queued.usage) == ("not_started", None)

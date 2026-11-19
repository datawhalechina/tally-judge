import asyncio
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from tallyjudge import (
    Dataset,
    Evaluator,
    ExecutionConfig,
    RunCollector,
    create_cache_backend,
    create_embedding_model,
    create_event_sink,
    create_language_model,
    create_metric,
    create_storage_backend,
    experiment,
    numeric_metric,
)
from tallyjudge.core.prompts import PromptInput

CHAT = "model.openai_compatible.chat_completions"
EMBED = "embedding.openai_compatible.embeddings"
RELEVANCE = "metric.rag.answer_relevance"


class Text(BaseModel):
    answer: str


class Recorder:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


@pytest.mark.asyncio
async def test_logical_calls_attempts_and_total_usage(client, tmp_path):
    tokens = SimpleNamespace(prompt_tokens=4, completion_tokens=None)
    client.chat_reply("private malformed answer", tokens=tokens)
    client.chat_reply('{"answer":"private answer"}', tokens=tokens)
    client.embedding_reply([(0, [1.0])], tokens=SimpleNamespace(prompt_tokens=2))
    client.responses.append(OSError("private response"))
    judge = create_language_model(CHAT, client=client, model="test")
    embedder = create_embedding_model(EMBED, client=client, model="embed", batch_size=1)
    calls = []

    @experiment
    async def generate(row):
        output = await judge.agenerate(PromptInput.from_text(row["prompt"]), Text)
        calls.append(output.call_id)
        return {"answer": output.data.answer}

    @numeric_metric("metric.test.vector", revision="1")
    async def score(answer: str):
        await embedder.aembed_texts([answer, answer])
        return 1.0

    events = create_event_sink("event.local.jsonl", root=tmp_path)
    store = create_storage_backend("storage.local.jsonl", root=tmp_path)
    data = Dataset.from_records([{"case_id": "a", "prompt": "secret prompt"}])
    result = await generate.arun(
        data, metrics=[score()], event_sink=events, backend=store
    )
    assert result.events_complete is True
    assert result.row_for("a").task.usage.generation.requests == 2
    assert result.row_for("a").metrics[score.spec.id].status == "error"
    assert result.usage.generation.known_tokens == {
        "input_tokens": 8,
        "output_tokens": None,
    }
    assert (
        result.usage.embedding.requests == 2
        and result.usage.embedding.known_counts["input_tokens"] == 1
    )
    assert result.usage == store.load_run(result.run_id).usage
    content = (tmp_path / f"events-{result.run_id}.jsonl").read_text(encoding="utf-8")
    events = [json.loads(line) for line in content.splitlines()]
    assert events[0]["kind"] == "run_started" and events[-1]["kind"] == "run_finished"
    assert [e["sequence"] for e in events] == list(range(1, len(events) + 1))
    assert all(e["run_id"] == result.run_id for e in events)
    assert all(s not in content for s in ("private", "secret prompt"))
    started = [e for e in events if e["kind"] == "request_started"]
    finished = [e for e in events if e["kind"] == "request_finished"]
    assert len({e["attempt_id"] for e in started}) == 4
    assert [e["attempt_id"] for e in started] == [e["attempt_id"] for e in finished]
    assert [e["request_kind"] for e in started] == ["generation"] * 2 + [
        "embedding"
    ] * 2
    assert [e["metric_key"] for e in started] == [
        None,
        None,
        score.spec.id,
        score.spec.id,
    ]
    assert started[0]["call_id"] == started[1]["call_id"] == calls[0] and calls[0]
    assert started[2]["call_id"] == started[3]["call_id"] != calls[0]
    assert finished[-1]["status"] == "error"


@pytest.mark.asyncio
async def test_cold_sampling_cache_hits_and_deployment_invalidation(client, tmp_path):
    data = Dataset.from_records(
        [{"case_id": "sample", "user_input": "question", "response": "answer"}]
    )
    path = tmp_path / "scores.sqlite"
    sink = Recorder()

    def evaluator(model, namespace):
        metric = create_metric(
            RELEVANCE,
            llm=create_language_model(CHAT, model=model, client=client),
            embeddings=create_embedding_model(EMBED, model="vectors", client=client),
            strictness=2,
        )
        return Evaluator(
            [metric],
            event_sink=sink,
            cache=create_cache_backend(
                "cache.local.sqlite", namespace=namespace, path=path
            ),
        )

    def replies():
        for q in ("one", "two"):
            client.chat_reply(
                json.dumps({"question": q, "noncommittal": 0}),
                tokens=SimpleNamespace(prompt_tokens=3),
            )
        client.embedding_reply([(0, [1.0])])
        client.embedding_reply([(1, [1.0]), (0, [1.0])])

    replies()
    cold = await evaluator("judge", "v1").arun(data)
    assert not cold.rows[0].metrics[RELEVANCE].cache_hit
    assert cold.usage.generation.requests == 2 and cold.usage.embedding.requests == 2
    warm = await evaluator("judge", "v1").arun(data)
    assert len(client.calls) == 4 and warm.rows[0].metrics[RELEVANCE].cache_hit
    assert (
        warm.rows[0].metrics[RELEVANCE].value
        == cold.rows[0].metrics[RELEVANCE].value
        == 1.0
    )
    assert warm.usage is None and warm.summary[RELEVANCE]["usage"] is None
    assert [e.kind for e in sink.events if e.run_id == warm.run_id] == [
        "run_started",
        "step_started",
        "cache_hit",
        "step_finished",
        "run_finished",
    ]
    for model, ns in [("new-judge", "v1"), ("judge", "v2")]:
        replies()
        report = await evaluator(model, ns).arun(data)
        assert not report.rows[0].metrics[RELEVANCE].cache_hit
    assert len(client.calls) == 12


@pytest.mark.asyncio
async def test_cancelled_row_saves_only_observed_usage_and_emits_no_late_events(client):
    started = asyncio.Event()

    async def wait():
        started.set()
        await asyncio.Future()

    client.responses.append(wait)
    embed = create_embedding_model(EMBED, model="vec", client=client)
    sink = Recorder()

    @experiment
    async def run(row):
        await embed.aembed_texts(["private"])
        return {}

    collector = RunCollector()
    store = create_storage_backend("storage.local.memory")
    task = asyncio.create_task(
        run.arun(
            Dataset.from_records([{}, {}]),
            config=ExecutionConfig(max_concurrency=1),
            collector=collector,
            backend=store,
            event_sink=sink,
        )
    )
    await asyncio.wait_for(started.wait(), 1)
    task.cancel("keep me")
    with pytest.raises(asyncio.CancelledError) as exc:
        await task
    assert exc.value.args == ("keep me",)
    result = store.load_run(collector.snapshot().run_id)
    assert result.run_status == "cancelled" and result.events_complete
    assert result.usage.embedding.in_flight_requests == 1
    assert [e.kind for e in sink.events].count("request_started") == 1
    assert sink.events[-1].kind == "run_finished"
    assert "private" not in str([asdict(e) for e in sink.events])


@pytest.mark.asyncio
async def test_timed_out_score_never_populates_cache_and_inputs_are_part_of_key(
    tmp_path,
):
    release = asyncio.Event()
    calls = []

    @numeric_metric("metric.test.delay", revision="1")
    async def score(value: str):
        calls.append(value)
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return 1.0

    cache = create_cache_backend(
        "cache.local.sqlite", path=tmp_path / "cache.sqlite", namespace="test"
    )
    runner = Evaluator(
        [score()],
        config=ExecutionConfig(timeout=0.03, cleanup_timeout=0.02),
        cache=cache,
    )

    def data(value):
        return Dataset.from_records([{"case_id": "same", "value": value}])

    try:
        first = await runner.arun(data("old"))
        assert (
            first.run_status == "stopped"
            and first.rows[0].metrics[score.spec.id].status == "timeout"
        )
    finally:
        release.set()
        await asyncio.sleep(0)
    for value, hit in [("old", False), ("old", True), ("new", False)]:
        result = await runner.arun(data(value))
        assert result.rows[0].metrics[score.spec.id].cache_hit is hit
    assert calls == ["old", "old", "new"]


def test_cache_leaves_non_json_validated_inputs_scorable(tmp_path):
    from pathlib import Path
    from typing import Annotated

    from pydantic import AfterValidator

    from tallyjudge import MetricResult
    from tallyjudge.interfaces.metrics import MetricBase, MetricSpec

    def as_path(name: str) -> Path:
        return Path(name)

    class PathMetric(MetricBase):
        spec = MetricSpec(
            "metric.test.path", "1", {"name": Annotated[str, AfterValidator(as_path)]}
        )

        async def ascore(self, name: Path) -> MetricResult:
            return MetricResult(float(name.suffix == ".txt"))

    cache = create_cache_backend(
        "cache.local.sqlite", path=tmp_path / "scores", namespace="paths"
    )
    result = Evaluator([PathMetric()], cache=cache).run(
        Dataset.from_records([{"name": "a.txt"}])
    )
    score = result.rows[0].metrics[PathMetric.spec.id]
    assert score.value == 1.0 and not score.cache_hit


@pytest.mark.parametrize("failure", ["read", "write"])
def test_optional_cache_io_failure_preserves_score(failure, caplog):
    class UnavailableCache:
        namespace = "test"

        def get(self, key):
            if failure == "read":
                raise OSError("cache unavailable")

        def put(self, key, value):
            raise OSError("cache unavailable")

    result = Evaluator(
        [create_metric("metric.text.exact_match")], cache=UnavailableCache()
    ).run(Dataset.from_records([{"response": "a", "reference": "a"}]))
    assert result.rows[0].metrics["metric.text.exact_match"].value == 1.0
    assert "cache" in caplog.text


def test_failed_observer_reports_incomplete_events_and_keeps_scores():
    class Broken:
        def emit(self, event):
            raise OSError("full")

    result = Evaluator(
        [create_metric("metric.text.exact_match")], event_sink=Broken()
    ).run(Dataset.from_records([{"response": "x", "reference": "x"}]))
    assert result.events_complete is False
    assert (
        result.run_status == "completed"
        and result.rows[0].metrics["metric.text.exact_match"].value == 1.0
    )

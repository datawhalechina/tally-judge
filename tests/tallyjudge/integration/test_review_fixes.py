"""Regressions for the phase 3 commit review, each reproduced from a public entry."""

import asyncio
import json
import sqlite3
import threading
import time

import pytest

from tallyjudge import (
    Dataset,
    EvaluationAborted,
    Evaluator,
    ExecutionConfig,
    MetricBinding,
    RunCollector,
    RunResult,
    categorical_metric,
    compare_runs,
    create_cache_backend,
    create_event_sink,
    create_language_model,
    create_metric,
    create_storage_backend,
    experiment,
    numeric_metric,
    ranking_metric,
)
from tallyjudge.core.errors import ClientLoopError

EXACT = "metric.text.exact_match"


def one_row(case_id="a", **fields):
    return Dataset.from_records([{"case_id": case_id, **fields}])


@pytest.mark.asyncio
async def test_locked_cache_does_not_block_the_event_loop_or_outer_timeout(tmp_path):
    path = tmp_path / "cache.sqlite"
    cache = create_cache_backend("cache.local.sqlite", path=path, namespace="ns")
    locked = threading.Event()
    release = threading.Event()

    def hold_write_lock():
        connection = sqlite3.connect(path)
        connection.execute("BEGIN EXCLUSIVE")
        locked.set()
        release.wait(5)
        connection.rollback()
        connection.close()

    holder = threading.Thread(target=hold_write_lock)
    holder.start()
    locked.wait(5)
    timer_started = time.monotonic()
    timer = asyncio.create_task(asyncio.sleep(0.02))
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.05):
                await Evaluator([create_metric(EXACT)], cache=cache).arun(
                    one_row(response="x", reference="x")
                )
        elapsed = time.monotonic() - started
        await timer
        timer_delay = time.monotonic() - timer_started
    finally:
        release.set()
        holder.join()
    # The lock was held far longer; both deadlines must fire close to schedule.
    assert elapsed < 0.2, elapsed
    assert timer_delay < 0.2, timer_delay


def test_cache_reuses_scores_across_binding_keys_and_case_ids(tmp_path):
    cache = create_cache_backend(
        "cache.local.sqlite", path=tmp_path / "c.sqlite", namespace="ns"
    )
    metric = create_metric(EXACT)
    Evaluator([metric], cache=cache).run(one_row("x", response="a", reference="a"))
    renamed = Evaluator([MetricBinding(metric, key="renamed")], cache=cache).run(
        one_row("x", response="a", reference="a")
    )
    other_id = Evaluator([metric], cache=cache).run(
        one_row("y", response="a", reference="a")
    )
    changed = Evaluator([metric], cache=cache).run(
        one_row("x", response="b", reference="a")
    )
    assert renamed.rows[0].metrics["renamed"].cache_hit
    assert other_id.rows[0].metrics[EXACT].cache_hit
    assert not changed.rows[0].metrics[EXACT].cache_hit


def test_function_defaults_apply_in_batch_and_key_the_cache(tmp_path):
    @numeric_metric("metric.test.scaled", revision="1")
    def scaled(response: str, scale: float = 2.0) -> float:
        return len(response) * scale

    metric = scaled()
    assert metric.score(response="cat").value == 6.0
    cache = create_cache_backend(
        "cache.local.sqlite", path=tmp_path / "f.sqlite", namespace="ns"
    )
    runner = Evaluator([metric], cache=cache)
    omitted = runner.run(one_row(response="cat"))
    explicit = runner.run(one_row(response="cat", scale=2.0))
    override = runner.run(one_row(response="cat", scale=3.0))
    scores = [r.rows[0].metrics[metric.spec.id] for r in (omitted, explicit, override)]
    assert [s.value for s in scores] == [6.0, 6.0, 9.0]
    assert [s.cache_hit for s in scores] == [False, True, False]
    assert scores[0].inputs_sha256 == scores[1].inputs_sha256 != scores[2].inputs_sha256


def test_swapping_the_function_under_one_id_changes_its_snapshot():
    @numeric_metric("metric.test.same", revision="1")
    def first(response: str) -> float:
        return 1.0

    @numeric_metric("metric.test.same", revision="1")
    def second(response: str) -> float:
        return 0.0

    assert first().config_snapshot()["function"].endswith("first")
    assert first().config_snapshot() != second().config_snapshot()


def test_ordinary_comparison_checks_scored_defaults():
    def scaled(default):
        @numeric_metric("metric.test.scaled", revision="1")
        def score(response: str, scale: float = default) -> float:
            return len(response) * scale

        return score()

    data = one_row(response="abc")
    first, second = (Evaluator([scaled(value)]).run(data) for value in (1.0, 2.0))
    key = "metric.test.scaled"
    assert (
        first.metrics[key] == second.metrics[key] and first.manifest == second.manifest
    )
    difference = compare_runs(first, second).rows[0].metrics[key]
    assert difference.changes == ("metric_input_changed",) and difference.delta is None


def test_compare_distinguishes_json_types_in_settings():
    left = Evaluator([create_metric(EXACT)]).run(one_row(response="a", reference="a"))
    right = RunResult.from_dict(json.loads(left.to_json()))
    left.metrics[EXACT]["config_snapshot"] = {"mode": True}
    right.metrics[EXACT]["config_snapshot"] = {"mode": 1}
    score = compare_runs(left, right).rows[0].metrics[EXACT]
    assert score.changes == ("config_snapshot",)
    assert score.delta is None


@pytest.mark.parametrize("stores_function_name", [False, True])
def test_old_function_archive_still_enforces_its_declared_range(stores_function_name):
    @numeric_metric("metric.test.bounded", revision="1", value_range=(0, 1))
    def bounded(response: str) -> float:
        return 1.0

    key = bounded.spec.id
    result = Evaluator([bounded()]).run(one_row(response="a"))
    old = result.to_dict()
    del old["metrics"][key]["value_range"]
    if not stores_function_name:
        del old["metrics"][key]["config_snapshot"]["function"]
    restored = RunResult.from_dict(old)
    assert (
        "value_range" not in compare_runs(restored, result).rows[0].metrics[key].changes
    )
    old["rows"][0]["metrics"][key]["value"] = 5.0
    with pytest.raises(ValueError, match="Invalid stored score"):
        RunResult.from_dict(old)


def test_comparison_distinguishes_bounded_and_unbounded_scores():
    from tallyjudge import MetricResult
    from tallyjudge.interfaces.metrics import MetricBase, MetricSpec

    class Score(MetricBase):
        def __init__(self, bounds, levels=()):
            self.spec = MetricSpec(
                "metric.test.scale",
                "1",
                {"response": str},
                value_range=bounds,
                levels=levels,
            )

        async def ascore(self, response: str) -> MetricResult:
            return MetricResult(1.0)

    left, right = [
        Evaluator([Score(bounds)]).run(one_row(response="a"))
        for bounds in [(None, None), (0.0, 1.0)]
    ]
    difference = compare_runs(left, right).rows[0].metrics["metric.test.scale"]
    assert difference.changes == ("value_range",) and difference.delta is None

    discrete = Evaluator([Score((0, 1), levels=(0,))]).run(one_row(response="a"))
    assert discrete.rows[0].metrics["metric.test.scale"].status == "invalid_output"
    changed_scale = compare_runs(right, discrete).rows[0].metrics["metric.test.scale"]
    assert changed_scale.changes == ("levels",) and changed_scale.delta is None


@pytest.mark.parametrize("keep", [True, False])
def test_redacted_experiment_runs_remain_comparable(keep):
    @experiment
    def echo(row):
        return {"response": row["response"]}

    @experiment
    def shout(row):
        return {"response": row["response"].upper()}

    data = one_row(response="a", reference="a")
    store = create_storage_backend("storage.local.memory")
    plain = Evaluator([create_metric(EXACT)]).run(data)
    runs = [plain] + [
        fn.run(data, metrics=[create_metric(EXACT)]) for fn in (echo, shout)
    ]
    for result in runs:
        result.save(backend=store, include_inputs=keep)
    plain, echoed, shouted = (store.load_run(r.run_id) for r in runs)
    # A plain evaluation against an experiment that passes the answer through.
    same = compare_runs(plain, echoed).rows[0].metrics[EXACT]
    changed = compare_runs(plain, shouted).rows[0].metrics[EXACT]
    assert (same.changes, same.delta) == ((), 0.0)
    assert changed.changes == ("experiment_input_changed",) and changed.delta is None


def test_redacted_archives_without_digests_report_unverified_inputs():
    @experiment
    def echo(row):
        return {"response": row["response"]}

    result = echo.run(
        one_row(response="a", reference="a"), metrics=[create_metric(EXACT)]
    )
    data = result.to_dict(include_inputs=False)
    for row in data["rows"]:
        row["metrics"][EXACT]["inputs_sha256"] = None
    old = RunResult.from_dict(data)
    score = compare_runs(old, old).rows[0].metrics[EXACT]
    assert score.changes == ("experiment_input_unverified",)
    assert score.delta is None


@categorical_metric("metric.test.label", revision="1", labels=["ok", "bad"])
def label(response: str) -> str:
    return "ok"


@ranking_metric("metric.test.rank", revision="1", length=2)
def rank(response: str) -> list[str]:
    return ["a", "b"]


@pytest.mark.parametrize("factory,bad", [(label, "unknown"), (rank, [])])
def test_archives_apply_the_saved_output_contract(tmp_path, factory, bad):
    metric = factory()
    result = Evaluator([metric]).run(one_row(response="x"))
    store = create_storage_backend("storage.local.jsonl", root=tmp_path)
    result.save(backend=store)
    assert store.load_run(result.run_id).summary[metric.spec.id]["coverage"] == 1.0
    target = next(tmp_path.glob("run-*.jsonl"))
    header, line = target.read_text(encoding="utf-8").splitlines()
    record = json.loads(line)
    record["row"]["metrics"][metric.spec.id]["value"] = bad
    target.write_text(header + "\n" + json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid stored score"):
        store.load_run(result.run_id)


class _Completions:
    async def create(self, **kwargs):
        class Message:
            content = '{"statements": []}'
            refusal = None
            tool_calls = None

        class Choice:
            message = Message()
            finish_reason = "stop"

        class Response:
            choices = [Choice()]
            model = "judge"
            usage = None

        return Response()


class _Client:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _Completions()})()
        self.embeddings = _Completions()

    def with_options(self, **kwargs):
        return self


def test_client_bound_to_another_loop_aborts_the_run_once(caplog):
    judge = create_language_model(
        "model.openai_compatible.chat_completions",
        client=_Client(),
        model="judge",
        temperature=0.0,
    )
    metric = create_metric("metric.rag.faithfulness", llm=judge)
    rows = Dataset.from_records(
        [{"user_input": "q", "response": "r", "retrieved_contexts": ["c"]}] * 5
    )
    first = Evaluator([metric]).run(rows)
    assert first.run_status == "completed"
    collector = RunCollector()
    with pytest.raises(ClientLoopError, match="another event loop"):
        Evaluator([metric]).run(rows, collector=collector)
    partial = collector.snapshot()
    assert (partial.run_status, partial.stop_reason) == ("aborted", "client_loop")
    assert "Metric execution failed" not in caplog.text


def test_event_sink_keeps_one_handle_per_run_and_closes_it(tmp_path, monkeypatch):
    from pathlib import Path

    opened = []
    original = Path.open

    def track(self, *args, **kwargs):
        opened.append(self.name)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", track)
    sink = create_event_sink("event.local.jsonl", root=tmp_path)
    data = Dataset.from_records([{"response": "a", "reference": "a"}] * 50)
    result = Evaluator([create_metric(EXACT)], event_sink=sink).run(data)
    assert result.events_complete is True
    assert opened == [f"events-{result.run_id}.jsonl"]
    assert sink._streams == {}
    lines = (tmp_path / opened[0]).read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["kind"] == "run_finished"
    assert len(lines) == 2 + 2 * 50


def test_event_sink_closes_healthy_writer_when_another_file_fails(tmp_path):
    from tallyjudge.core.events import RunEvent
    from tallyjudge.plugins.tracing.jsonl import JsonlEventSink

    sink = JsonlEventSink(root=tmp_path)
    (tmp_path / "events-broken.jsonl").mkdir()
    sink.emit(RunEvent("good", 1, "run_started"))
    sink.emit(RunEvent("broken", 1, "run_started"))
    healthy_writer = sink._streams["good"].done
    try:
        with pytest.raises(OSError):
            sink.close()
        assert healthy_writer.done()
        assert (
            json.loads((tmp_path / "events-good.jsonl").read_text())["run_id"] == "good"
        )
    finally:
        sink.close()


def test_cross_loop_model_from_experiment_aborts_once():
    from pydantic import BaseModel

    from tallyjudge.core.prompts import PromptInput

    class Verdict(BaseModel):
        statements: list[str]

    judge = create_language_model(
        "model.openai_compatible.chat_completions", client=_Client(), model="judge"
    )

    @experiment
    async def answer(row):
        await judge.agenerate(PromptInput.from_text("question"), Verdict)
        return {"response": "ok"}

    rows = one_row(response="x")
    assert answer.run(rows).run_status == "completed"
    collector = RunCollector()
    with pytest.raises(ClientLoopError):
        answer.run(rows, collector=collector)
    assert (collector.snapshot().run_status, collector.snapshot().stop_reason) == (
        "aborted",
        "client_loop",
    )


@pytest.mark.asyncio
async def test_cancellation_shares_one_grace_with_cache_and_partial_storage():
    writing = threading.Event()
    release = threading.Event()

    class SlowCache:
        namespace = "test"

        def get(self, key):
            return None

        def put(self, key, result):
            writing.set()
            release.wait(3)

    class SlowStorage:
        def save_run(self, result, *, include_inputs=True):
            release.wait(3)

    @numeric_metric("metric.test.wait", revision="1")
    async def wait(response: str):
        if response == "second":
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await asyncio.to_thread(release.wait, 3)
        return 1.0

    dataset = Dataset.from_records([{"response": "first"}, {"response": "second"}])
    collector = RunCollector()
    task = asyncio.create_task(
        Evaluator(
            [wait()],
            config=ExecutionConfig(cleanup_timeout=0.08),
            cache=SlowCache(),
            backend=SlowStorage(),
        ).arun(dataset, collector=collector)
    )
    try:
        assert await asyncio.to_thread(writing.wait, 2)
        before = time.monotonic()
        task.cancel("original")
        with pytest.raises(asyncio.CancelledError, match="original"):
            await task
        assert time.monotonic() - before < 0.4
        assert (
            collector.snapshot().run_status == "cancelled"
            and collector.snapshot().cleanup_warnings
        )
    finally:
        release.set()


@pytest.mark.asyncio
async def test_slow_event_file_yields_and_write_failure_closes_handle(
    tmp_path, monkeypatch
):
    from pathlib import Path

    started, release, closed = (threading.Event() for _ in range(3))

    class BrokenStream:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            closed.set()

        def write(self, line):
            started.set()
            release.wait(3)
            raise OSError("disk full")

    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: BrokenStream())
    sink = create_event_sink("event.local.jsonl", root=tmp_path)
    task = asyncio.create_task(
        Evaluator([create_metric(EXACT)], event_sink=sink).arun(
            one_row(response="x", reference="x")
        )
    )
    try:
        assert await asyncio.to_thread(started.wait, 2)
        assert not closed.is_set()
    finally:
        release.set()
    result = await task
    assert closed.is_set() and result.events_complete is False
    assert result.rows[0].metrics[EXACT].value == 1.0


def test_aborted_run_is_still_saved_by_the_evaluator():
    @numeric_metric("metric.test.fail", revision="1")
    def fail(response: str) -> float:
        raise ValueError("boom")

    store = create_storage_backend("storage.local.memory")
    runner = Evaluator(
        [fail()], config=ExecutionConfig(failure_policy="raise"), backend=store
    )
    with pytest.raises(EvaluationAborted) as caught:
        runner.run(one_row(response="x"))
    saved = store.load_run(caught.value.partial_result.run_id)
    assert saved.run_status == "aborted"

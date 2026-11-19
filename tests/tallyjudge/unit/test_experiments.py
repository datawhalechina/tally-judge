import asyncio

import pytest

from tallyjudge import (
    Dataset,
    EvaluationAborted,
    Evaluator,
    ExecutionConfig,
    MetricBinding,
    MetricResult,
    RunCollector,
    RunSaveError,
    categorical_metric,
    compare_runs,
    create_metric,
    create_storage_backend,
    experiment,
    numeric_metric,
    ranking_metric,
)
from tallyjudge.registry import PluginRegistry, PluginSpec

EXACT = "metric.text.exact_match"


@pytest.mark.asyncio
async def test_experiment_steps_share_concurrency_and_keep_inputs():
    active = peak = 0
    order = []

    async def wait(kind, sample):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        order.append((kind, sample))
        try:
            await asyncio.sleep(0.01)
        finally:
            active -= 1

    @experiment
    async def generate(row):
        await wait("experiment", row["case_id"])
        if row["case_id"] == "skip":
            return None
        if row["case_id"] == "error":
            raise LookupError("sensitive content")
        if row["case_id"] == "timeout":
            await asyncio.Event().wait()
        return {"response": row["case_id"]}

    @numeric_metric("metric.test.numeric", revision="1", value_range=(0, 1))
    async def score(response: str):
        await wait("score", response)
        return 1.0

    data = Dataset.from_records(
        [
            {"case_id": case, "response": "original"}
            for case in ("a", "b", "skip", "error", "timeout")
        ]
    )
    store = create_storage_backend("storage.local.memory")
    run = await generate.arun(
        data,
        metrics=[score()],
        config=ExecutionConfig(max_concurrency=2, timeout=0.2),
        backend=store,
    )
    report = store.load_run(run.run_id)
    assert peak == 2 and active == 0
    assert [row.task.status for row in report.rows] == [
        "success",
        "success",
        "skipped",
        "error",
        "timeout",
    ]
    assert [row.metrics[score.spec.id].status for row in report.rows] == [
        "success",
        "success",
        "not_started",
        "not_started",
        "not_started",
    ]
    assert {s for step, s in order if step == "score"} == {"a", "b"}
    assert report.input_for("a")["response"] == "original"
    assert report.row_for("a").task.output == {"response": "a"}
    assert report.row_for("error").task.error.exception_type == "LookupError"
    assert "sensitive content" not in report.to_json()
    assert run.run_status == "completed"


def test_sync_experiment_and_mixed_function_metrics_round_trip():
    @experiment
    def solve(row):
        return {"answer": row["reference"]}

    @categorical_metric("metric.test.label", revision="2", labels=["yes", "no"])
    def label(response: str):
        return MetricResult("yes", "matched")

    @ranking_metric("metric.test.ranking", revision="1", length=2)
    async def ordered(response: str):
        return [response, {"source": 1}]

    registry = PluginRegistry([PluginSpec(label.spec.id, "metric", 1, label)])
    assert registry.describe(label.spec.id) == label.spec
    assert label() is not label()
    metrics = [
        MetricBinding(metric, input_map={"response": "answer"})
        for metric in (create_metric(EXACT), registry.create(label.spec.id), ordered())
    ]
    data = Dataset.from_records([{"case_id": "one", "reference": "x"}])
    store = create_storage_backend("storage.local.memory")
    result = solve.run(data, metrics=metrics, backend=store, include_inputs=False)
    loaded = store.load_run(result.run_id)
    assert loaded.inputs is None and loaded.usage is None
    assert loaded.row_for("one").task.output == {"answer": "x"}
    assert loaded.summary[EXACT]["mean"] == 1.0
    assert loaded.summary[label.spec.id]["label_counts"] == {"yes": 1, "no": 0}
    for factory in (label, ordered):
        assert loaded.summary[factory.spec.id]["mean"] is None
        assert loaded.summary[factory.spec.id]["coverage"] == 1.0
    assert loaded.rows[0].metrics[ordered.spec.id].value == ["x", {"source": 1}]
    assert loaded.rows[0].metrics[label.spec.id].reason == "matched"
    assert compare_runs(result, loaded).rows[0].metrics[ordered.spec.id].delta is None

    @experiment
    def other(row):
        return {"answer": "new"}

    diff = compare_runs(result, other.run(data, metrics=metrics)).rows[0].metrics[EXACT]
    assert diff.changes == ("experiment_input_changed",) and diff.delta is None


@pytest.mark.parametrize(
    "factory,value",
    [
        (numeric_metric("numeric", revision="1", value_range=(0, 1)), 2.0),
        (categorical_metric("category", revision="1", labels=["yes"]), "unknown"),
        (ranking_metric("ranking", revision="1", length=2), ["one"]),
        (ranking_metric("ranking", revision="1", length=1), [object()]),
    ],
)
def test_invalid_function_output_keeps_rows(factory, value):
    @factory
    def metric():
        return value

    result = Evaluator([metric()]).run(Dataset.from_records([{}]))
    assert result.rows[0].metrics[metric.spec.id].status == "invalid_output"
    assert result.summary[metric.spec.id]["coverage"] == 0.0


class BrokenStore:
    def save_run(self, result, *, include_inputs):
        raise OSError("disk full")


def test_autosave_failure_carries_result_but_preserves_abort():
    @experiment
    def work(row):
        if row.get("fail"):
            raise ValueError("private")
        return row

    with pytest.raises(RunSaveError) as saved:
        work.run(Dataset.from_records([{}]), backend=BrokenStore())
    assert isinstance(saved.value.__cause__, OSError)
    assert saved.value.result.rows[0].task.status == "success"
    with pytest.raises(EvaluationAborted) as abort:
        work.run(
            Dataset.from_records([{"fail": True}, {}]),
            config=ExecutionConfig(max_concurrency=1, failure_policy="raise"),
            backend=BrokenStore(),
        )
    assert [row.task.status for row in abort.value.partial_result.rows] == [
        "error",
        "not_started",
    ]


@pytest.mark.asyncio
async def test_cancellation_seals_experiment_and_preserves_original_exception():
    started, release = asyncio.Event(), asyncio.Event()

    @experiment
    async def wait(row):
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return {"answer": "late"}

    collector = RunCollector()
    task = asyncio.create_task(
        wait.arun(
            Dataset.from_records([{}, {}]),
            metrics=[create_metric(EXACT)],
            collector=collector,
            config=ExecutionConfig(max_concurrency=1, cleanup_timeout=0.02),
            backend=BrokenStore(),
        )
    )
    await started.wait()
    assert collector.snapshot().rows[0].task.status == "running"
    task.cancel("original")
    try:
        with pytest.raises(asyncio.CancelledError) as exc:
            await asyncio.wait_for(task, 1)
        assert exc.value.args == ("original",)
        result = collector.snapshot()
        assert result.run_status == "cancelled"
        assert [row.task.status for row in result.rows] == ["cancelled", "not_started"]
        assert all(row.metrics[EXACT].status == "not_started" for row in result.rows)
        assert result.cleanup_warnings[0].metric_key is None
    finally:
        release.set()
        await asyncio.sleep(0)
    assert collector.snapshot().rows[0].task.output is None

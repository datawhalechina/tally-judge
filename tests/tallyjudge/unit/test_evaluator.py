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
    create_metric,
)
from tallyjudge.interfaces.metrics import MetricBase, MetricSpec


class ControlledMetric(MetricBase):
    spec = MetricSpec("metric.test.controlled", "1", {"response": str})

    def __init__(self):
        self.active = 0
        self.peak = 0
        self.started = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.cleaned = []

    async def ascore(self, response):
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.started.append(response)
        self.entered.set()
        try:
            if response == "error":
                raise ValueError("SECRET raw provider response")
            if response == "timeout_error":
                raise TimeoutError("plugin timeout")
            if response == "self_cancel":
                raise asyncio.CancelledError
            if response == "wait":
                await self.release.wait()
            if response == "slow":
                await asyncio.sleep(0.02)
            if response == "nan":
                return MetricResult(float("nan"))
            if response == "inf":
                return MetricResult(float("inf"))
            if response == "bad_type":
                return "not a result"
            if response == "bad_reason":
                return MetricResult(1.0, 42)
            return MetricResult(1.0)
        finally:
            self.active -= 1
            self.cleaned.append(response)


def data(*responses):
    return Dataset.from_records(
        [
            {"case_id": index, "response": response, "reference": response}
            for index, response in enumerate(responses)
        ]
    )


def statuses(result, key="metric.test.controlled"):
    return [row.metrics[key].status for row in result.rows]


@pytest.mark.asyncio
async def test_mixed_order_and_independent_bindings():
    metric = ControlledMetric()
    report = await Evaluator([metric, create_metric("metric.text.exact_match")]).arun(
        data("slow", "error", "ok")
    )
    assert statuses(report) == ["success", "error", "success"]
    assert [row.case_id for row in report.rows] == ["0", "1", "2"]
    assert statuses(report, "metric.text.exact_match") == ["success"] * 3
    assert "SECRET" not in report.to_json()
    assert metric.active == 0


def test_binding_keys_mapping_and_field_errors():
    metric = create_metric("metric.text.exact_match")
    mapping = {"response": "other"}
    alternate = MetricBinding(metric, key="alternate", input_map=mapping)
    mapping["response"] = "reference"
    report = Evaluator([metric, alternate]).run(
        Dataset.from_records(
            [
                {"response": "a", "reference": "a", "other": "b"},
                {"response": 2, "reference": "a", "other": "a"},
                {"response": "a", "reference": "a"},
            ]
        )
    )
    assert report.rows[0].metrics[metric.spec.id].value == 1.0
    assert report.rows[0].metrics["alternate"].value == 0.0
    assert report.rows[1].metrics[metric.spec.id].error.category == "validation_error"
    assert report.rows[1].metrics["alternate"].status == "success"
    assert report.rows[2].metrics["alternate"].error.category == "validation_error"
    with pytest.raises(ValueError, match="unique"):
        Evaluator([metric, metric])
    with pytest.raises(ValueError, match="unknown"):
        MetricBinding(metric, input_map={"unknown": "response"})


@pytest.mark.asyncio
async def test_tasks_are_bounded_not_just_work(monkeypatch):
    metric = ControlledMetric()
    original_create_task = asyncio.create_task
    created = []

    def track(coroutine, *args, **kwargs):
        task = original_create_task(coroutine, *args, **kwargs)
        created.append(task)
        return task

    monkeypatch.setattr(asyncio, "create_task", track)
    run = original_create_task(
        Evaluator([metric], config=ExecutionConfig(max_concurrency=3)).arun(
            data(*(["wait"] * 100))
        )
    )
    await metric.entered.wait()
    assert len(created) == 3
    assert metric.peak == 3
    metric.release.set()
    report = await run
    assert len(report.rows) == 100
    assert metric.peak <= 3
    assert all(task.done() for task in created)


@pytest.mark.asyncio
async def test_external_cancel_and_snapshot_detachment():
    metric = ControlledMetric()
    collector = RunCollector()
    before = asyncio.all_tasks()
    task = asyncio.create_task(
        Evaluator([metric], config=ExecutionConfig(max_concurrency=2)).arun(
            data("wait", "wait", "wait"), collector=collector
        )
    )
    await metric.entered.wait()
    task.cancel("caller cancellation")
    with pytest.raises(asyncio.CancelledError) as error:
        await task
    assert error.value.args == ("caller cancellation",)
    assert statuses(collector.snapshot()) == ["cancelled", "cancelled", "not_started"]
    assert metric.active == 0
    snapshot = collector.snapshot()
    snapshot.rows.clear()
    assert len(collector.snapshot().rows) == 3
    assert not (asyncio.all_tasks() - before)


@pytest.mark.asyncio
async def test_asyncio_timeout_and_taskgroup_keep_native_semantics():
    metric = ControlledMetric()
    collector = RunCollector()
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.02):
            await Evaluator([metric]).arun(data("wait"), collector=collector)
    assert statuses(collector.snapshot()) == ["cancelled"]
    assert asyncio.current_task().cancelling() == 0
    assert metric.active == 0

    async def fail():
        await metric.entered.wait()
        raise ValueError("peer failed")

    metric.entered.clear()
    with pytest.raises(ExceptionGroup):
        async with asyncio.TaskGroup() as group:
            group.create_task(
                Evaluator([metric]).arun(data("wait"), collector=collector)
            )
            group.create_task(fail())
    assert statuses(collector.snapshot()) == ["cancelled"]
    assert metric.active == 0


@pytest.mark.asyncio
async def test_fail_fast_preserves_completed_and_cancels_remaining():
    metric = ControlledMetric()
    with pytest.raises(EvaluationAborted) as caught:
        await Evaluator(
            [metric], config=ExecutionConfig(max_concurrency=3, failure_policy="raise")
        ).arun(data("ok", "error", "wait", "never"))
    assert statuses(caught.value.partial_result) == [
        "success",
        "error",
        "cancelled",
        "not_started",
    ]
    assert "never" not in metric.started
    assert metric.active == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["record", "raise"])
async def test_invalid_outputs_do_not_abort(policy):
    report = await Evaluator(
        [ControlledMetric()],
        config=ExecutionConfig(max_concurrency=1, failure_policy=policy),
    ).arun(data("nan", "inf", "bad_type", "bad_reason", "ok"))
    assert statuses(report) == ["invalid_output"] * 4 + ["success"]
    assert report.summary["metric.test.controlled"]["mean"] == 1.0
    assert "NaN" not in report.to_json()


@pytest.mark.asyncio
async def test_timeout_record_and_raise():
    config = ExecutionConfig(max_concurrency=1, timeout=0.01)
    metric = ControlledMetric()
    report = await Evaluator([metric], config=config).arun(
        data("wait", "ok", "timeout_error", "self_cancel")
    )
    assert statuses(report) == ["timeout", "success", "error", "cancelled"]
    with pytest.raises(EvaluationAborted) as caught:
        await Evaluator(
            [metric],
            config=ExecutionConfig(
                max_concurrency=1, timeout=0.01, failure_policy="raise"
            ),
        ).arun(data("wait", "never"))
    assert statuses(caught.value.partial_result) == ["timeout", "not_started"]
    assert metric.active == 0


@pytest.mark.asyncio
async def test_sync_api_rejects_running_loop_and_collector_reuse():
    collector = RunCollector()
    metric = ControlledMetric()
    evaluator = Evaluator([metric])
    with pytest.raises(RuntimeError, match="await arun"):
        evaluator.run(data("ok"))
    with pytest.raises(RuntimeError, match="No run"):
        collector.snapshot()
    task = asyncio.create_task(evaluator.arun(data("wait"), collector=collector))
    await metric.entered.wait()
    with pytest.raises(ValueError, match="already collecting"):
        await evaluator.arun(data("ok"), collector=collector)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert statuses(await evaluator.arun(data("ok"), collector=collector)) == [
        "success"
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_concurrency": 0},
        {"max_concurrency": True},
        {"max_concurrency": 1.2},
        {"timeout": 0},
        {"timeout": -1},
        {"timeout": True},
        {"timeout": float("inf")},
        {"timeout": float("nan")},
        {"failure_policy": "ignore"},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        ExecutionConfig(**kwargs)


def test_invalid_snapshot_fails_before_run():
    class BadSnapshot(ControlledMetric):
        def config_snapshot(self):
            return {"client": object()}

    metric = BadSnapshot()
    collector = RunCollector()
    with pytest.raises(ValueError):
        Evaluator([metric]).run(data("ok"), collector=collector)
    assert metric.started == []
    with pytest.raises(RuntimeError):
        collector.snapshot()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_fast", [False, True])
async def test_repeated_cancel_does_not_interrupt_plugin_cleanup(fail_fast):
    class CleanupMetric(ControlledMetric):
        def __init__(self):
            super().__init__()
            self.cleaning = asyncio.Event()
            self.allow_cleanup = asyncio.Event()
            self.cleanup_complete = False

        async def ascore(self, response):
            if response == "error":
                raise ValueError("fail fast")
            self.entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cleaning.set()
                await self.allow_cleanup.wait()
                self.cleanup_complete = True

    metric = CleanupMetric()
    collector = RunCollector()
    evaluator = Evaluator(
        [metric],
        config=ExecutionConfig(
            max_concurrency=2, failure_policy="raise" if fail_fast else "record"
        ),
    )
    task = asyncio.create_task(
        evaluator.arun(
            data(*(["error", "wait"] if fail_fast else ["wait"])), collector=collector
        )
    )
    await metric.entered.wait()
    if not fail_fast:
        task.cancel("original")
    await metric.cleaning.wait()
    task.cancel("during cleanup")
    await asyncio.sleep(0)
    metric.allow_cleanup.set()
    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    assert caught.value.args == (("during cleanup",) if fail_fast else ("original",))
    assert metric.cleanup_complete
    assert statuses(collector.snapshot())[-1] == "cancelled"


@pytest.mark.asyncio
async def test_out_of_range_and_integer_outputs_are_invalid():
    class InvalidMetric(ControlledMetric):
        spec = MetricSpec(
            "metric.test.controlled", "1", {"response": str}, value_range=(0.0, 1.0)
        )

        async def ascore(self, response):
            return MetricResult(2.0 if response == "range" else 1)

    report = await Evaluator([InvalidMetric()]).arun(data("range", "integer"))
    assert statuses(report) == ["invalid_output", "invalid_output"]

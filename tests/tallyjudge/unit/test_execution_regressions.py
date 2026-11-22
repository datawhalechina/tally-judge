import asyncio
import gc
import json
import logging
import subprocess
import sys
import weakref

import pytest

from tallyjudge import (
    Dataset,
    EvaluationAborted,
    Evaluator,
    ExecutionConfig,
    MetricResult,
    RunCollector,
    create_metric,
)
from tallyjudge.interfaces.metrics import MetricBase, MetricSpec


class UncooperativeMetric(MetricBase):
    spec = MetricSpec("metric.test.uncooperative", "1", {"response": str})

    def __init__(self):
        self.started = asyncio.Event()
        self.swallowed = asyncio.Event()
        self.release = asyncio.Event()
        self.workers = []
        self.late_error = False

    async def ascore(self, response):
        self.workers.append(asyncio.current_task())
        self.started.set()
        if response == "error":
            raise ZeroDivisionError("secret context")
        while not self.release.is_set():
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.swallowed.set()
        if self.late_error:
            raise ZeroDivisionError("late secret")
        return MetricResult(0.5)

    async def finish(self):
        self.release.set()
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        await asyncio.sleep(0)  # run the detached-task done callbacks


def records(*responses):
    return Dataset.from_records([{"response": value} for value in responses])


async def assert_returns(task):
    # A regression must fail the test rather than hang the test runner.
    done, _ = await asyncio.wait({task}, timeout=1.0)
    assert task in done, "Evaluation did not return within its cleanup grace period"


@pytest.mark.asyncio
async def test_cancel_without_storage_defers_snapshot_until_requested(monkeypatch):
    started = asyncio.Event()

    class WaitingMetric(MetricBase):
        spec = MetricSpec("metric.test.wait", "1", {})

        async def ascore(self):
            started.set()
            await asyncio.Future()

    collector = RunCollector()
    snapshot = collector.snapshot
    snapshots = []

    def counted():
        snapshots.append(None)
        return snapshot()

    monkeypatch.setattr(collector, "snapshot", counted)
    evaluation = asyncio.create_task(
        Evaluator([WaitingMetric()]).arun(records("one", "two"), collector=collector)
    )
    await started.wait()
    evaluation.cancel("stop")
    with pytest.raises(asyncio.CancelledError, match="stop"):
        await evaluation
    assert snapshots == []
    assert collector.snapshot().run_status == "cancelled"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "external_cancel",
        "outer_timeout",
        "task_timeout",
        "task_timeout_raise",
        "fail_fast",
    ],
)
async def test_uncooperative_metric_cannot_block_evaluator(mode):
    metric = UncooperativeMetric()
    collector = RunCollector()
    config = ExecutionConfig(
        max_concurrency=2 if mode == "fail_fast" else 1,
        timeout=0.02 if mode.startswith("task_timeout") else None,
        cleanup_timeout=0.03,
        failure_policy="raise"
        if mode in ("task_timeout_raise", "fail_fast")
        else "record",
    )
    evaluator = Evaluator([metric], config=config)
    dataset = (
        records("error", "wait", "never")
        if mode == "fail_fast"
        else records("wait", "never")
    )

    async def evaluate():
        if mode == "outer_timeout":
            async with asyncio.timeout(0.02):
                return await evaluator.arun(dataset, collector=collector)
        return await evaluator.arun(dataset, collector=collector)

    task = asyncio.create_task(evaluate())
    try:
        await metric.started.wait()
        if mode == "external_cancel":
            task.cancel("original cancellation")
        await assert_returns(task)
        if mode == "external_cancel":
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            assert caught.value.args == ("original cancellation",)
        elif mode == "outer_timeout":
            with pytest.raises(TimeoutError):
                await task
        elif mode in ("task_timeout_raise", "fail_fast"):
            with pytest.raises(EvaluationAborted) as caught:
                await task
            assert caught.value.partial_result.cleanup_warnings
        else:
            assert (await task).cleanup_warnings
        report = collector.snapshot()
        expected_run = {
            "external_cancel": ("cancelled", "external_cancel"),
            "outer_timeout": ("cancelled", "external_cancel"),
            "task_timeout": ("stopped", "cleanup_timeout"),
            "task_timeout_raise": ("aborted", "failure_policy"),
            "fail_fast": ("aborted", "failure_policy"),
        }
        assert (report.run_status, report.stop_reason) == expected_run[mode]
        if mode in ("task_timeout_raise", "fail_fast"):
            assert caught.value.partial_result.run_status == "aborted"
        statuses = [row.metrics[metric.spec.id].status for row in report.rows]
        if mode.startswith("task_timeout"):
            assert statuses == ["timeout", "not_started"]
        elif mode == "fail_fast":
            assert statuses == ["error", "cancelled", "not_started"]
        else:
            assert statuses == ["cancelled", "not_started"]
        assert len(metric.workers) == config.max_concurrency
        warning = report.cleanup_warnings[0]
        assert warning.category == "cleanup_timeout"
        assert warning.metric_key == metric.spec.id
        assert (
            json.loads(report.to_json())["cleanup_warnings"][0]["case_id"]
            == warning.case_id
        )
        frozen = report.to_json()
        await metric.finish()
        assert (
            collector.snapshot().to_json() == frozen
        )  # late success cannot overwrite timeout/cancelled
    finally:
        await metric.finish()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_repeated_cancellation_has_a_fixed_deadline():
    metric = UncooperativeMetric()
    collector = RunCollector()
    task = asyncio.create_task(
        Evaluator([metric], config=ExecutionConfig(cleanup_timeout=0.03)).arun(
            records("wait"), collector=collector
        )
    )
    try:
        await metric.started.wait()
        task.cancel("first")
        await metric.swallowed.wait()
        for _ in range(30):
            if task.done():
                break
            task.cancel("again")
            # Windows loop clocks may have a 15.6 ms resolution; shorter
            # sleeps can all finish before the clock advances.
            await asyncio.sleep(0.02)
        assert task.done(), "Repeated cancellation extended the cleanup deadline"
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert caught.value.args == ("first",)
        assert len(collector.snapshot().cleanup_warnings) == 1
    finally:
        await metric.finish()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_late_exception_is_logged_without_mutating_result(caplog):
    metric = UncooperativeMetric()
    task = asyncio.create_task(
        Evaluator(
            [metric], config=ExecutionConfig(timeout=0.01, cleanup_timeout=0.01)
        ).arun(records("wait"))
    )
    try:
        await assert_returns(task)
        report = await task
        snapshot = report.to_json()
        metric.late_error = True
        with caplog.at_level(logging.ERROR, logger="tallyjudge.core.execution"):
            await metric.finish()
        assert report.to_json() == snapshot
        assert "late secret" not in snapshot
        assert any(
            record.exc_info and record.exc_info[0] is ZeroDivisionError
            for record in caplog.records
        )
    finally:
        await metric.finish()
        await asyncio.gather(task, return_exceptions=True)


def test_sync_entry_exits_even_if_metric_never_cooperates():
    # The subprocess watchdog also detects regressions in event-loop shutdown.
    code = """
import asyncio
from tallyjudge import Dataset, Evaluator, ExecutionConfig, MetricResult
from tallyjudge.interfaces.metrics import MetricBase, MetricSpec
class Stubborn(MetricBase):
    spec = MetricSpec('metric.test.stubborn', '1', {'response': str})
    async def ascore(self, response):
        while True:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                pass
# Leave time for the child process to enter the scorer before testing its refusal to cancel.
report = Evaluator([Stubborn()], config=ExecutionConfig(timeout=0.2, cleanup_timeout=0.02)).run(Dataset.from_records([{'response': 'x'}]))
assert report.rows[0].metrics['metric.test.stubborn'].status == 'timeout'
assert len(report.cleanup_warnings) == 1
print('returned')
"""
    process = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=5
    )
    assert process.returncode == 0, process.stderr
    assert "returned" in process.stdout


@pytest.mark.asyncio
async def test_error_type_and_local_traceback(caplog):
    class BrokenMetric(MetricBase):
        spec = MetricSpec("metric.test.broken", "1", {"response": str})

        async def ascore(self, response):
            raise ZeroDivisionError("secret original input")

    with caplog.at_level(logging.ERROR, logger="tallyjudge.core.execution"):
        result = await Evaluator([BrokenMetric()]).arun(records("a"))
    error = result.rows[0].metrics["metric.test.broken"].error
    assert error.category == "metric_error"
    assert error.exception_type == "ZeroDivisionError"
    assert "secret original input" not in result.to_json()
    assert '"message"' not in result.to_json()
    record = next(item for item in caplog.records if item.exc_info)
    assert record.exc_info[0] is ZeroDivisionError
    assert "metric.test.broken" in record.getMessage()
    assert "secret original input" in caplog.text


@pytest.mark.parametrize("value", [0, -1, True, None, float("nan"), float("inf")])
def test_cleanup_timeout_validation(value):
    with pytest.raises(ValueError, match="cleanup_timeout"):
        ExecutionConfig(cleanup_timeout=value)


@pytest.mark.asyncio
async def test_record_run_stopped_after_second_row_is_explicit():
    class MixedMetric(UncooperativeMetric):
        async def ascore(self, response):
            if response == "ok":
                return MetricResult(1.0)
            return await super().ascore(response)

    metric = MixedMetric()
    task = asyncio.create_task(
        Evaluator(
            [metric],
            config=ExecutionConfig(
                max_concurrency=1, timeout=0.02, cleanup_timeout=0.03
            ),
        ).arun(records("ok", "wait", "never", "never", "never"))
    )
    try:
        await assert_returns(task)
        result = await task
        assert (result.run_status, result.stop_reason) == ("stopped", "cleanup_timeout")
        assert [row.metrics[metric.spec.id].status for row in result.rows] == [
            "success",
            "timeout",
            "not_started",
            "not_started",
            "not_started",
        ]
        assert result.summary[metric.spec.id]["coverage"] == 0.2
    finally:
        await metric.finish()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_staggered_deadlines_fire_while_another_metric_is_cleaning():
    cleaning, release, later_cancelled = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )

    class Staggered(MetricBase):
        spec = MetricSpec("metric.test.staggered", "1", {"response": str})

        async def ascore(self, response):
            if response == "free":
                await asyncio.sleep(0.05)
                return MetricResult(1.0)
            try:
                await asyncio.Event().wait()
            finally:
                if response == "first":
                    cleaning.set()
                    await release.wait()
                else:
                    later_cancelled.set()

    metric = Staggered()
    run = asyncio.create_task(
        Evaluator(
            [metric],
            config=ExecutionConfig(max_concurrency=2, timeout=0.15, cleanup_timeout=2),
        ).arun(records("first", "free", "later"))
    )
    try:
        await asyncio.wait_for(cleaning.wait(), 2)
        # Keep the first worker in cleanup until the later worker sees its own timeout.
        await asyncio.wait_for(later_cancelled.wait(), 0.75)
    finally:
        release.set()
        report = await asyncio.wait_for(run, 3)
    assert [row.metrics[metric.spec.id].status for row in report.rows] == [
        "timeout",
        "success",
        "timeout",
    ]
    assert report.run_status == "completed"
    assert not report.cleanup_warnings


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [1, 2])
async def test_internal_failure_seals_every_started_row(monkeypatch, concurrency):
    collector = RunCollector()
    original = OSError("diagnostic output unavailable")

    def broken_log(*args, **kwargs):
        raise original

    monkeypatch.setattr(
        logging.getLogger("tallyjudge.core.execution"), "warning", broken_log
    )
    metric = create_metric("metric.text.exact_match")
    dataset = Dataset.from_records(
        [{"response": 7, "reference": "ok"} for _ in range(concurrency)]
        + [{"response": "ok", "reference": "ok"}]
    )
    with pytest.raises(OSError) as caught:
        await Evaluator(
            [metric], config=ExecutionConfig(max_concurrency=concurrency)
        ).arun(dataset, collector=collector)
    assert caught.value is original
    report = collector.snapshot()
    assert (report.run_status, report.stop_reason) == ("aborted", "internal_error")
    *failed_rows, unstarted = [row.metrics[metric.spec.id] for row in report.rows]
    for failed in failed_rows:
        assert failed.status == "error"
        assert (failed.error.category, failed.error.exception_type) == (
            "internal_error",
            "OSError",
        )
    assert unstarted.status == "not_started"


def test_closed_caller_loop_releases_abandoned_run():
    class Paused(MetricBase):
        spec = MetricSpec("metric.test.paused", "1", {"response": str})

        async def ascore(self, response):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                await asyncio.sleep(10)
            return MetricResult(1.0)

    def run_and_close():
        loop = asyncio.new_event_loop()
        loop.set_exception_handler(lambda *_: None)
        dataset = records("wait", "never")
        try:
            report = loop.run_until_complete(
                Evaluator(
                    [Paused()],
                    config=ExecutionConfig(
                        max_concurrency=1, timeout=0.02, cleanup_timeout=0.03
                    ),
                ).arun(dataset)
            )
            assert (report.run_status, len(report.cleanup_warnings)) == ("stopped", 1)
            dataset_ref = weakref.ref(dataset)
            del dataset
            gc.collect()
            assert dataset_ref() is None  # The abandoned worker retains only its row.
            return weakref.ref(loop)
        finally:
            loop.close()

    loop_ref = run_and_close()
    gc.collect()
    assert loop_ref() is None

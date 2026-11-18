"""Bounded scheduling, independent deadlines and bounded cancellation cleanup."""

import asyncio
import json
import logging
import math
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4

from tallyjudge.core.config import ExecutionConfig
from tallyjudge.core.dataset import Dataset, canonical_json
from tallyjudge.core.errors import EvaluationAborted, InputValidationError
from tallyjudge.core.results import (
    CleanupWarning,
    ErrorInfo,
    MetricResult,
    RowResult,
    RunCollector,
    RunResult,
    RunStatus,
    ScoreResult,
)
from tallyjudge.interfaces.metrics import MetricBinding

logger = logging.getLogger(__name__)


def _detached_tasks(loop: asyncio.AbstractEventLoop) -> set[asyncio.Task[Any]]:
    # A process-wide set would retain closed loops through Task.get_loop().
    return vars(loop).setdefault("_tallyjudge_detached_tasks", set())


def _consume(task: asyncio.Task[Any]) -> None:
    if not task.cancelled():
        error = task.exception()
        if error is not None:
            logger.error(
                "Background task failed after cleanup",
                exc_info=(type(error), error, error.__traceback__),
            )


def _detach(task: asyncio.Task[Any]) -> None:
    detached = _detached_tasks(task.get_loop())
    detached.add(task)
    task.add_done_callback(detached.discard)
    task.add_done_callback(_consume)


def run_sync[T](
    factory: Callable[[], Coroutine[Any, Any, T]],
    async_name: str,
    *,
    cleanup_timeout: float = 5.0,
) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            f"An event loop is already running; use await {async_name}()"
        )
    # asyncio.run/Runner.close waits indefinitely for cancellation-resistant tasks.
    # Own this loop explicitly, in the calling thread, and bound shutdown as well.
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(factory())
    finally:
        detached = _detached_tasks(loop)
        pending = asyncio.all_tasks(loop) - detached
        for task in pending:
            task.cancel()
        if pending:
            done, unfinished = loop.run_until_complete(
                asyncio.wait(pending, timeout=cleanup_timeout)
            )
            for task in done:
                _consume(task)
            if unfinished:
                logger.warning(
                    "Synchronous loop shutdown left %d unfinished tasks",
                    len(unfinished),
                )
        finalizers = [
            loop.create_task(loop.shutdown_asyncgens()),
            loop.create_task(loop.shutdown_default_executor()),
        ]
        done, unfinished = loop.run_until_complete(
            asyncio.wait(finalizers, timeout=cleanup_timeout)
        )
        for task in done:
            _consume(task)
        for task in unfinished:
            task.cancel()
        # Closing a loop discards pending coroutine work; it cannot kill SDK threads.
        loop.close()
        detached.clear()


def validate_output(output: Any, binding: MetricBinding) -> ScoreResult:
    if not isinstance(output, MetricResult) or not isinstance(output.value, float):
        return ScoreResult("invalid_output", error=ErrorInfo("invalid_output"))
    value = float(output.value)
    low, high = binding.metric.spec.value_range
    if (
        not math.isfinite(value)
        or (low is not None and value < low)
        or (high is not None and value > high)
    ):
        return ScoreResult("invalid_output", error=ErrorInfo("invalid_output"))
    if output.reason is not None and not isinstance(output.reason, str):
        return ScoreResult("invalid_output", error=ErrorInfo("invalid_output"))
    return ScoreResult("success", value, output.reason)


@dataclass
class _Job:
    row: RowResult
    binding: MetricBinding
    deadline: float | None
    finished_at: float | None = None

    def record(self, outcome: ScoreResult) -> None:
        self.row.metrics[self.binding.key] = outcome

    def mark_cancelled(self) -> None:
        if self.row.metrics[self.binding.key].status == "running":
            self.record(ScoreResult("cancelled"))

    async def score(self, record: dict[str, Any]) -> ScoreResult:
        clock = asyncio.get_running_loop().time
        self.record(ScoreResult("running"))
        try:
            output = await self.binding.ascore_row(record)
            return validate_output(output, self.binding)
        except InputValidationError as exc:
            logger.warning(
                "Metric input validation failed: plugin=%s binding=%s input_index=%d",
                self.binding.metric.spec.id,
                self.binding.key,
                self.row.input_index,
            )
            logger.debug("Metric input validation details", exc_info=True)
            return ScoreResult(
                "error",
                error=ErrorInfo(
                    "validation_error", field=exc.field, expected=exc.expected
                ),
            )
        except Exception as exc:
            logger.exception(
                "Metric execution failed: plugin=%s binding=%s input_index=%d",
                self.binding.metric.spec.id,
                self.binding.key,
                self.row.input_index,
            )
            return ScoreResult(
                "error", error=ErrorInfo("metric_error", type(exc).__name__)
            )
        finally:
            self.finished_at = clock()

    def finish(self, task: asyncio.Task[ScoreResult]) -> None:
        if task.cancelled():
            self.mark_cancelled()
            return
        try:
            outcome = task.result()
        except BaseException as exc:
            # A failing worker (including its diagnostics) still owes its row a terminal state.
            self.record(
                ScoreResult(
                    "error", error=ErrorInfo("internal_error", type(exc).__name__)
                )
            )
            raise
        if (
            self.deadline is not None
            and self.finished_at is not None
            and self.finished_at > self.deadline
        ):
            outcome = ScoreResult("timeout", error=ErrorInfo("timeout"))
        self.record(outcome)


class TaskRunner:
    async def arun(
        self,
        dataset: Dataset,
        bindings: Sequence[MetricBinding],
        config: ExecutionConfig,
        collector: RunCollector,
    ) -> RunResult:
        metadata = {}
        for binding in bindings:
            snapshot = binding.metric.config_snapshot()
            if not isinstance(snapshot, dict):
                raise ValueError("Metric config_snapshot must return a JSON object")
            metadata[binding.key] = {
                "id": binding.metric.spec.id,
                "revision": binding.metric.spec.revision,
                "input_map": dict(binding.input_map),
                "config_snapshot": json.loads(canonical_json(snapshot)),
            }
        manifest = dataset.manifest
        result = RunResult(
            str(uuid4()),
            manifest,
            metadata,
            asdict(config),
            [
                RowResult(
                    item["case_id"],
                    item["input_index"],
                    {key: ScoreResult("not_started") for key in metadata},
                )
                for item in manifest
            ],
            dataset.records,
        )
        collector._start(result)
        loop = asyncio.get_running_loop()
        active: dict[asyncio.Task[ScoreResult], _Job] = {}
        cleaning: dict[asyncio.Task[ScoreResult], tuple[_Job, float]] = {}
        jobs = ((row, binding) for row in result.rows for binding in bindings)

        def fill() -> None:
            while len(active) < config.max_concurrency:
                item = next(jobs, None)
                if item is None:
                    break
                row, binding = item
                deadline = (
                    None if config.timeout is None else loop.time() + config.timeout
                )
                job = _Job(row, binding, deadline)
                task = asyncio.create_task(job.score(dataset.record(row.input_index)))
                active[task] = job

        def cancel(task: asyncio.Task[ScoreResult], until: float) -> None:
            # Moving the task seals its row. Late completion only drains its exception.
            cleaning[task] = (active.pop(task), until)
            task.cancel()

        stopping: tuple[RunStatus, str] | None = None

        def stop(status: RunStatus, reason: str) -> None:
            nonlocal stopping
            stopping = (status, reason)
            until = min(
                (end for _, end in cleaning.values()),
                default=loop.time() + config.cleanup_timeout,
            )
            for task, (job, _) in cleaning.items():
                cleaning[task] = (job, until)
            for task, job in tuple(active.items()):
                if not task.done():
                    job.mark_cancelled()
                    cancel(task, until)

        error: BaseException | None = None
        try:
            while True:
                try:
                    if stopping is None and not cleaning:
                        fill()
                    tasks = active.keys() | cleaning.keys()
                    if not tasks:
                        break
                    deadlines = [
                        job.deadline
                        for job in active.values()
                        if job.deadline is not None
                    ]
                    deadlines.extend(end for _, end in cleaning.values())
                    delay = min(deadlines) - loop.time() if deadlines else None
                    if delay is not None and delay <= 0:
                        # A repeated external cancellation cannot postpone an overdue deadline.
                        done = {task for task in tasks if task.done()}
                    else:
                        done, _ = await asyncio.wait(
                            tasks, timeout=delay, return_when=asyncio.FIRST_COMPLETED
                        )

                    failed = False
                    for task in done:
                        if task in cleaning:
                            del cleaning[task]
                            _consume(task)
                        else:
                            job = active.pop(task)
                            job.finish(task)
                            failed |= job.row.metrics[job.binding.key].status in (
                                "error",
                                "timeout",
                            )

                    now = loop.time()
                    for task, job in tuple(active.items()):
                        if job.deadline is not None and job.deadline <= now:
                            job.record(
                                ScoreResult("timeout", error=ErrorInfo("timeout"))
                            )
                            cancel(task, now + config.cleanup_timeout)
                            failed = True

                    if failed and config.failure_policy == "raise" and stopping is None:
                        stop("aborted", "failure_policy")
                    if any(end <= now for _, end in cleaning.values()):
                        if stopping is None:
                            stop("stopped", "cleanup_timeout")
                        while cleaning:
                            task, (job, _) = cleaning.popitem()
                            _detach(task)
                            result.cleanup_warnings.append(
                                CleanupWarning(
                                    job.row.case_id,
                                    job.row.input_index,
                                    job.binding.key,
                                )
                            )
                            logger.warning(
                                "Metric did not finish within cleanup_timeout: plugin=%s binding=%s input_index=%d",
                                job.binding.metric.spec.id,
                                job.binding.key,
                                job.row.input_index,
                            )
                except asyncio.CancelledError as exc:
                    if not isinstance(error, asyncio.CancelledError):
                        error = exc
                        stop("cancelled", "external_cancel")
                except BaseException as exc:
                    if error is None:
                        error = exc
                        stop("aborted", "internal_error")

            result.run_status, result.stop_reason = stopping or ("completed", None)
            if error is not None:
                raise error
            if result.run_status == "aborted":
                raise EvaluationAborted(collector.snapshot())
            return collector.snapshot()
        finally:
            collector._finish()

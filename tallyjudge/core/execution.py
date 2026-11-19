"""Bounded scheduling, independent deadlines and bounded cancellation cleanup."""

import asyncio
import copy
import logging
from collections import deque
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import asdict
from typing import Any
from uuid import uuid4

from tallyjudge.core.config import ExecutionConfig
from tallyjudge.core.dataset import Dataset
from tallyjudge.core.errors import EvaluationAborted, RunSaveError
from tallyjudge.core.events import RunEvents
from tallyjudge.core.jobs import ExperimentJob, Job, MetricJob, RowFunction
from tallyjudge.core.results import (
    CleanupWarning,
    RowResult,
    RowTaskResult,
    RunCollector,
    RunResult,
    RunStatus,
    ScoreResult,
)
from tallyjudge.interfaces.cache import ScoreCache
from tallyjudge.interfaces.events import EventSink
from tallyjudge.interfaces.metrics import MetricBinding
from tallyjudge.interfaces.storage import StorageBackend

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


class TaskRunner:
    async def arun(
        self,
        dataset: Dataset,
        bindings: Sequence[MetricBinding],
        config: ExecutionConfig,
        collector: RunCollector,
        *,
        row_function: RowFunction | None = None,
        backend: StorageBackend | None = None,
        include_inputs: bool = True,
        event_sink: EventSink | None = None,
        cache: ScoreCache | None = None,
    ) -> RunResult:
        metadata = {binding.key: binding.snapshot() for binding in bindings}
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
                    RowTaskResult("not_started") if row_function is not None else None,
                )
                for item in manifest
            ],
            dataset.records,
        )
        collector._start(result)
        loop = asyncio.get_running_loop()
        events = RunEvents(result.run_id, event_sink)
        events.emit("run_started", status="running")
        active: dict[asyncio.Task[Any], Job[Any]] = {}
        cleaning: dict[asyncio.Task[Any], tuple[Job[Any], float]] = {}
        # Prefer ready metrics so completed experiments do not accumulate a second work queue.
        steps = (row_function,) if row_function is not None else bindings
        initial = ((row, step) for row in result.rows for step in steps)
        ready: deque[tuple[RowResult, MetricBinding]] = deque()

        def fill() -> None:
            while len(active) < config.max_concurrency:
                item = ready.popleft() if ready else next(initial, None)
                if item is None:
                    break
                row, step = item
                deadline = (
                    None if config.timeout is None else loop.time() + config.timeout
                )
                record = dataset.record(row.input_index)
                if isinstance(step, MetricBinding):
                    if row.task is not None and row.task.output is not None:
                        record.update(copy.deepcopy(row.task.output))
                    job = MetricJob(
                        row, deadline, events, step, metadata[step.key], cache
                    )
                else:
                    job = ExperimentJob(row, deadline, events, step)
                with job.usage.activate():
                    task = asyncio.create_task(job.run(record))
                active[task] = job

        def cancel(task: asyncio.Task[Any], until: float) -> None:
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
                    job.cancel()
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
                            status = job.finish(task)
                            if isinstance(job, ExperimentJob) and status == "success":
                                ready.extend((job.row, binding) for binding in bindings)
                            failed |= status in ("error", "timeout")

                    now = loop.time()
                    for task, job in tuple(active.items()):
                        if job.deadline is not None and job.deadline <= now:
                            job.timeout()
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
                                    job.key,
                                )
                            )
                            logger.warning(
                                "Task did not finish within cleanup_timeout: binding=%s input_index=%d",
                                job.key,
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
            events.emit("run_finished", status=result.run_status)
            result.events_complete = events.complete
            if backend is not None:
                try:
                    result.save(backend=backend, include_inputs=include_inputs)
                except Exception as exc:
                    if error is None and result.run_status != "aborted":
                        raise RunSaveError(collector.snapshot()) from exc
                    logger.exception("Saving partial run %s failed", result.run_id)
            if error is not None:
                raise error
            if result.run_status == "aborted":
                raise EvaluationAborted(collector.snapshot())
            return collector.snapshot()
        finally:
            collector._finish()

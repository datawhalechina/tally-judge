import asyncio
import copy
import logging
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

from tallyjudge.engine.config import ExecutionConfig
from tallyjudge.engine.jobs import ExperimentJob, Job, MetricJob, RowFunction
from tallyjudge.sdk._tasks import consume_task, detach_task
from tallyjudge.sdk.cache import ScoreCache
from tallyjudge.sdk.dataset import Dataset
from tallyjudge.sdk.errors import ClientLoopError, EvaluationAborted
from tallyjudge.sdk.events import EventSink, RunEvents
from tallyjudge.sdk.metrics import MetricBinding
from tallyjudge.sdk.results import (
    CleanupWarning,
    RowResult,
    RowTaskResult,
    RunCollector,
    RunResult,
    RunStatus,
    ScoreResult,
)

logger = logging.getLogger(__package__)

type Step = MetricBinding | RowFunction


def _log_cache_write(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and task.exception() is not None:
        logger.warning("Could not cache a score", exc_info=task.exception())


@dataclass
class _Run:
    result: RunResult
    records: list[dict[str, Any]]
    bindings: Sequence[MetricBinding]
    config: ExecutionConfig
    collector: RunCollector
    events: RunEvents
    row_function: RowFunction | None
    cache: ScoreCache | None
    loop: asyncio.AbstractEventLoop
    active: dict[asyncio.Task[Any], Job[Any]] = field(default_factory=dict)
    cleaning: dict[asyncio.Task[Any], tuple[Job[Any], float]] = field(
        default_factory=dict
    )
    cache_writes: set[asyncio.Task[None]] = field(default_factory=set)
    ready: deque[tuple[RowResult, Step]] = field(default_factory=deque)
    stopping: tuple[RunStatus, str] | None = None
    queued: Iterator[tuple[RowResult, Step]] = field(init=False)

    def __post_init__(self) -> None:
        steps: Sequence[Step] = (
            (self.row_function,) if self.row_function is not None else self.bindings
        )
        self.queued = ((row, step) for row in self.result.rows for step in steps)

    def fill(self) -> None:
        while len(self.active) < self.config.max_concurrency:
            item = self.ready.popleft() if self.ready else next(self.queued, None)
            if item is None:
                break
            self._start(*item)

    def _start(self, row: RowResult, step: Step) -> None:
        timeout = self.config.timeout
        deadline = None if timeout is None else self.loop.time() + timeout
        record = self.records[row.input_index]
        if isinstance(step, MetricBinding):
            if row.task is not None and row.task.output is not None:
                record = record | row.task.output
            job: Job[Any] = MetricJob(
                row,
                deadline,
                self.events,
                step,
                self.result.metrics[step.key],
                self.cache,
            )
        else:
            record = copy.deepcopy(record)
            job = ExperimentJob(row, deadline, self.events, step)
        with job.usage.activate():
            task = asyncio.create_task(job.run(record))
        self.active[task] = job

    async def wait(self) -> set[asyncio.Task[Any]]:
        tasks = self.active.keys() | self.cleaning.keys()
        deadlines = [
            job.deadline for job in self.active.values() if job.deadline is not None
        ]
        deadlines.extend(end for _, end in self.cleaning.values())
        delay = min(deadlines) - self.loop.time() if deadlines else None
        if delay is not None and delay <= 0:
            return {task for task in tasks if task.done()}
        done, _ = await asyncio.wait(
            tasks, timeout=delay, return_when=asyncio.FIRST_COMPLETED
        )
        return done

    def settle(self, done: set[asyncio.Task[Any]]) -> bool:
        failed = False
        for task in done:
            if task in self.cleaning:
                del self.cleaning[task]
                consume_task(task)
                continue
            job = self.active.pop(task)
            status = job.finish(task)
            if isinstance(job, MetricJob):
                self._write_cache(job)
            elif status == "success":
                self.ready.extend((job.row, binding) for binding in self.bindings)
            failed |= status in ("error", "timeout")
        return failed

    def expire(self, now: float) -> bool:
        expired = False
        for task, job in tuple(self.active.items()):
            if job.deadline is not None and job.deadline <= now:
                job.timeout()
                self._seal(task, now + self.config.cleanup_timeout)
                expired = True
        return expired

    def stop(self, status: RunStatus, reason: str) -> None:
        self.stopping = (status, reason)
        until = min(
            (end for _, end in self.cleaning.values()),
            default=self.loop.time() + self.config.cleanup_timeout,
        )
        self.collector.cleanup_deadline = until
        for task, (job, _) in self.cleaning.items():
            self.cleaning[task] = (job, until)
        for task, job in tuple(self.active.items()):
            if not task.done():
                job.cancel()
                self._seal(task, until)

    def abandon(self, now: float) -> None:
        if not any(end <= now for _, end in self.cleaning.values()):
            return
        if self.stopping is None:
            self.stop("stopped", "cleanup_timeout")
        while self.cleaning:
            task, (job, _) = self.cleaning.popitem()
            detach_task(task)
            self.result.cleanup_warnings.append(
                CleanupWarning(job.row.case_id, job.row.input_index, job.key)
            )
            logger.warning(
                "Task did not finish within cleanup_timeout: binding=%s input_index=%d",
                job.key,
                job.row.input_index,
            )

    async def drain_cache_writes(self) -> None:
        if not self.cache_writes:
            return
        deadline = self.collector.cleanup_deadline
        timeout = (
            self.config.cleanup_timeout
            if deadline is None
            else max(0.0, deadline - self.loop.time())
        )
        _, unfinished = await asyncio.wait(self.cache_writes, timeout=timeout)
        if unfinished:
            logger.warning("%d score cache writes still pending", len(unfinished))

    def _seal(self, task: asyncio.Task[Any], until: float) -> None:
        self.cleaning[task] = (self.active.pop(task), until)
        task.cancel()

    def _write_cache(self, job: MetricJob) -> None:
        entry = job.cache_entry()
        if entry is None:
            return
        store, key, score = entry
        task = asyncio.create_task(asyncio.to_thread(store.put, key, score))
        self.cache_writes.add(task)
        task.add_done_callback(self.cache_writes.discard)
        task.add_done_callback(_log_cache_write)


async def run_tasks(
    dataset: Dataset,
    bindings: Sequence[MetricBinding],
    config: ExecutionConfig,
    collector: RunCollector,
    *,
    row_function: RowFunction | None = None,
    experiment_metadata: dict[str, Any] | None = None,
    event_sink: EventSink | None = None,
    cache: ScoreCache | None = None,
) -> RunResult:
    metadata = {binding.key: binding.snapshot() for binding in bindings}
    records = dataset.records
    manifest = dataset.manifest_of(records)
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
        records,
        metadata=experiment_metadata,
    )
    collector.begin(result)
    loop = asyncio.get_running_loop()
    events = RunEvents(result.run_id, event_sink)
    events.emit("run_started", status="running")
    run = _Run(
        result,
        records,
        bindings,
        config,
        collector,
        events,
        row_function,
        cache,
        loop,
    )
    error: BaseException | None = None
    try:
        while True:
            try:
                if run.stopping is None and not run.cleaning:
                    run.fill()
                if not run.active and not run.cleaning:
                    break
                failed = run.settle(await run.wait())
                now = loop.time()
                failed |= run.expire(now)
                if failed and config.failure_policy == "raise" and not run.stopping:
                    run.stop("aborted", "failure_policy")
                run.abandon(now)
            except asyncio.CancelledError as exc:
                if not isinstance(error, asyncio.CancelledError):
                    error = exc
                    run.stop("cancelled", "external_cancel")
            except BaseException as exc:
                if error is None:
                    error = exc
                    reason = (
                        "client_loop"
                        if isinstance(exc, ClientLoopError)
                        else "internal_error"
                    )
                    run.stop("aborted", reason)

        try:
            await run.drain_cache_writes()
        except asyncio.CancelledError as exc:
            if error is None:
                error = exc
                run.stop("cancelled", "external_cancel")
        result.run_status, result.stop_reason = run.stopping or ("completed", None)
        events.emit("run_finished", status=result.run_status)
        try:
            await events.flush(collector.cleanup_deadline)
        except asyncio.CancelledError as exc:
            if error is None:
                error = exc
                if collector.cleanup_deadline is None:
                    collector.cleanup_deadline = loop.time() + config.cleanup_timeout
        result.events_complete = events.complete
        if error is not None:
            raise error
        if result.run_status == "aborted":
            raise EvaluationAborted(collector.snapshot())
        return collector.snapshot()
    finally:
        collector.end()

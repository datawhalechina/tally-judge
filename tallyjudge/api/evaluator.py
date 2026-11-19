"""Explicit evaluation entry points without model or field inference."""

import asyncio
import logging
from collections.abc import Sequence

from tallyjudge.core.config import ExecutionConfig
from tallyjudge.core.dataset import Dataset
from tallyjudge.core.errors import EvaluationAborted, RunSaveError
from tallyjudge.core.execution import TaskRunner, run_sync
from tallyjudge.core.jobs import RowFunction
from tallyjudge.core.results import RunCollector, RunResult
from tallyjudge.interfaces.cache import ScoreCache
from tallyjudge.interfaces.events import EventSink
from tallyjudge.interfaces.metrics import Metric, MetricBinding
from tallyjudge.interfaces.storage import StorageBackend

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate datasets with a fixed set of metric bindings and execution settings.

    With ``backend``, every terminal run is saved automatically from a worker thread.
    A failed save of a completed or stopped run raises ``RunSaveError`` carrying the
    result; while an abort, error or cancellation propagates, a failed save is only
    logged so the original exception reaches the caller.
    """

    def __init__(
        self,
        metrics: Sequence[Metric | MetricBinding],
        *,
        config: ExecutionConfig | None = None,
        backend: StorageBackend | None = None,
        include_inputs: bool = True,
        event_sink: EventSink | None = None,
        cache: ScoreCache | None = None,
    ) -> None:
        self.bindings = tuple(
            item if isinstance(item, MetricBinding) else MetricBinding(item)
            for item in metrics
        )
        keys = [binding.key for binding in self.bindings]
        if len(set(keys)) != len(keys):
            raise ValueError("Metric binding keys must be unique within a run")
        self.config = config or ExecutionConfig()
        self.backend = backend
        self.include_inputs = include_inputs
        self.event_sink = event_sink
        self.cache = cache

    async def arun(
        self, dataset: Dataset, *, collector: RunCollector | None = None
    ) -> RunResult:
        """Score a dataset in the caller's loop.

        On external cancellation, the caller's collector retains the partial result.
        Cleanup may delay propagation by up to the configured grace period.
        """
        return await self._arun(dataset, collector, None)

    async def _arun(
        self,
        dataset: Dataset,
        collector: RunCollector | None,
        row_function: RowFunction | None,
    ) -> RunResult:
        collector = collector if collector is not None else RunCollector()
        previous = collector._result
        try:
            result = await TaskRunner().arun(
                dataset,
                self.bindings,
                self.config,
                collector,
                row_function=row_function,
                event_sink=self.event_sink,
                cache=self.cache,
            )
        except EvaluationAborted as exc:
            await self._save_partial(exc.partial_result)
            raise
        except BaseException:
            # Save only a run this call started; a reused collector may hold an older one.
            if collector._result is not previous:
                await self._save_partial(collector.snapshot())
            raise
        if self.backend is not None:
            try:
                await asyncio.to_thread(
                    result.save,
                    backend=self.backend,
                    include_inputs=self.include_inputs,
                )
            except Exception as exc:
                raise RunSaveError(result) from exc
        return result

    async def _save_partial(self, result: RunResult) -> None:
        """Best-effort save while another exception propagates; failures are logged."""
        if self.backend is None or result.run_status == "running":
            return
        try:
            await asyncio.to_thread(
                result.save, backend=self.backend, include_inputs=self.include_inputs
            )
        except BaseException:
            # Never replace the original failure or cancellation with a storage error.
            logger.exception("Saving partial run %s failed", result.run_id)

    def run(
        self, dataset: Dataset, *, collector: RunCollector | None = None
    ) -> RunResult:
        """Score a dataset synchronously. Use arun inside an active event loop."""
        return run_sync(
            lambda: self.arun(dataset, collector=collector),
            "arun",
            cleanup_timeout=self.config.cleanup_timeout,
        )

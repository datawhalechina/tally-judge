import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from tallyjudge.engine.config import ExecutionConfig
from tallyjudge.engine.jobs import RowFunction
from tallyjudge.engine.scheduler import run_tasks
from tallyjudge.sdk.cache import ScoreCache
from tallyjudge.sdk.dataset import Dataset
from tallyjudge.sdk.errors import EvaluationAborted, RunSaveError
from tallyjudge.sdk.events import EventSink
from tallyjudge.sdk.metrics import Metric, MetricBinding
from tallyjudge.sdk.results import RunCollector, RunResult
from tallyjudge.sdk.storage import StorageBackend
from tallyjudge.sdk.sync import run_sync

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
        experiment_metadata: dict[str, Any] | None = None,
    ) -> RunResult:
        from tallyjudge.api.controls import dataset_bound

        collector = collector if collector is not None else RunCollector()
        previous = collector.run_id
        try:
            result = await run_tasks(
                dataset,
                self.bindings,
                self.config,
                collector,
                row_function=row_function,
                experiment_metadata=experiment_metadata,
                event_sink=self.event_sink,
                cache=self.cache,
                controls=dataset_bound,
            )
        except EvaluationAborted as exc:
            await self._save_partial(exc.partial_result, collector.cleanup_deadline)
            raise
        except BaseException:
            if self.backend is not None and collector.run_id != previous:
                await self._save_partial(
                    collector.snapshot(), collector.cleanup_deadline
                )
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

    async def _save_partial(self, result: RunResult, deadline: float | None) -> None:
        if self.backend is None or result.run_status == "running":
            return
        try:
            async with asyncio.timeout_at(deadline):
                await asyncio.to_thread(
                    result.save,
                    backend=self.backend,
                    include_inputs=self.include_inputs,
                )
        except TimeoutError:
            logger.warning(
                "Saving partial run %s exceeded cleanup_timeout", result.run_id
            )
        except BaseException:
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

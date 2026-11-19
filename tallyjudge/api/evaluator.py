"""Explicit evaluation entry points without model or field inference."""

from collections.abc import Sequence

from tallyjudge.core.config import ExecutionConfig
from tallyjudge.core.dataset import Dataset
from tallyjudge.core.execution import TaskRunner, run_sync
from tallyjudge.core.jobs import RowFunction
from tallyjudge.core.results import RunCollector, RunResult
from tallyjudge.interfaces.cache import ScoreCache
from tallyjudge.interfaces.events import EventSink
from tallyjudge.interfaces.metrics import Metric, MetricBinding
from tallyjudge.interfaces.storage import StorageBackend


class Evaluator:
    """Evaluate datasets with a fixed set of metric bindings and execution settings."""

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
        return await TaskRunner().arun(
            dataset,
            self.bindings,
            self.config,
            collector if collector is not None else RunCollector(),
            row_function=row_function,
            backend=self.backend,
            include_inputs=self.include_inputs,
            event_sink=self.event_sink,
            cache=self.cache,
        )

    def run(
        self, dataset: Dataset, *, collector: RunCollector | None = None
    ) -> RunResult:
        """Score a dataset synchronously. Use arun inside an active event loop."""
        return run_sync(
            lambda: self.arun(dataset, collector=collector),
            "arun",
            cleanup_timeout=self.config.cleanup_timeout,
        )

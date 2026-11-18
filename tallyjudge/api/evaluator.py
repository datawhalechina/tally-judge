"""Explicit evaluation entry points without model or field inference."""

from collections.abc import Sequence

from tallyjudge.core.config import ExecutionConfig
from tallyjudge.core.dataset import Dataset
from tallyjudge.core.execution import TaskRunner, run_sync
from tallyjudge.core.results import RunCollector, RunResult
from tallyjudge.interfaces.metrics import Metric, MetricBinding


class Evaluator:
    """Evaluate datasets with a fixed set of metric bindings and execution settings."""

    def __init__(
        self,
        metrics: Sequence[Metric | MetricBinding],
        *,
        config: ExecutionConfig | None = None,
    ) -> None:
        self.bindings = tuple(
            item if isinstance(item, MetricBinding) else MetricBinding(item)
            for item in metrics
        )
        keys = [binding.key for binding in self.bindings]
        if len(set(keys)) != len(keys):
            raise ValueError("Metric binding keys must be unique within a run")
        self.config = config or ExecutionConfig()

    async def arun(
        self, dataset: Dataset, *, collector: RunCollector | None = None
    ) -> RunResult:
        """Score a dataset in the caller's loop.

        On external cancellation, the caller's collector retains the partial result.
        Cleanup may delay propagation by up to the configured grace period.
        """
        return await TaskRunner().arun(
            dataset,
            self.bindings,
            self.config,
            collector if collector is not None else RunCollector(),
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

"""Run a per-sample function followed by metric bindings in the evaluator's scheduler."""

from collections.abc import Sequence
from typing import TypedDict, Unpack

from tallyjudge.api.evaluator import Evaluator
from tallyjudge.core.config import ExecutionConfig
from tallyjudge.core.dataset import Dataset
from tallyjudge.core.execution import run_sync
from tallyjudge.core.jobs import RowFunction
from tallyjudge.core.results import RunCollector, RunResult
from tallyjudge.interfaces.cache import ScoreCache
from tallyjudge.interfaces.events import EventSink
from tallyjudge.interfaces.metrics import Metric, MetricBinding
from tallyjudge.interfaces.storage import StorageBackend


class _RunOptions(TypedDict, total=False):
    config: ExecutionConfig
    backend: StorageBackend
    include_inputs: bool
    event_sink: EventSink
    cache: ScoreCache


class ExperimentWrapper:
    def __init__(self, function: RowFunction) -> None:
        self.function = function

    async def arun(
        self,
        dataset: Dataset,
        *,
        metrics: Sequence[Metric | MetricBinding] = (),
        collector: RunCollector | None = None,
        **options: Unpack[_RunOptions],
    ) -> RunResult:
        """Run the function once per row, then score its output overlaid on a copy of the input."""
        return await Evaluator(metrics, **options)._arun(
            dataset, collector, self.function
        )

    def run(
        self,
        dataset: Dataset,
        *,
        metrics: Sequence[Metric | MetricBinding] = (),
        collector: RunCollector | None = None,
        **options: Unpack[_RunOptions],
    ) -> RunResult:
        """Run the same experiment synchronously, outside an active event loop."""
        evaluator = Evaluator(metrics, **options)
        return run_sync(
            lambda: evaluator._arun(dataset, collector, self.function),
            "arun",
            cleanup_timeout=evaluator.config.cleanup_timeout,
        )


def experiment(function: RowFunction) -> ExperimentWrapper:
    """Decorate a short sync or async row function. Return a JSON object to score, or None to skip."""
    return ExperimentWrapper(function)

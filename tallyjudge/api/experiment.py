"""Run a per-sample function followed by metric bindings in the evaluator's scheduler."""

import json
from collections.abc import Callable, Sequence
from typing import Any, TypedDict, Unpack, overload

from tallyjudge.api.evaluator import Evaluator
from tallyjudge.core.config import ExecutionConfig
from tallyjudge.core.dataset import Dataset, canonical_json
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
    """Run a decorated row function per sample, then score its outputs.

    Metrics and evaluator options (config, backend, cache, event sink) are given per
    call, so one experiment can be scored with different settings.
    """

    def __init__(
        self, function: RowFunction, metadata: dict[str, Any] | None = None
    ) -> None:
        self.function = function
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("Experiment metadata must be a JSON object")
        self._metadata = canonical_json(metadata)

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
            dataset, collector, self.function, json.loads(self._metadata)
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
            lambda: evaluator._arun(
                dataset, collector, self.function, json.loads(self._metadata)
            ),
            "arun",
            cleanup_timeout=evaluator.config.cleanup_timeout,
        )


@overload
def experiment(
    function: RowFunction, *, metadata: dict[str, Any] | None = None
) -> ExperimentWrapper: ...


@overload
def experiment(
    function: None = None, *, metadata: dict[str, Any] | None = None
) -> Callable[[RowFunction], ExperimentWrapper]: ...


def experiment(
    function: RowFunction | None = None, *, metadata: dict[str, Any] | None = None
) -> ExperimentWrapper | Callable[[RowFunction], ExperimentWrapper]:
    """Decorate a row function and persist its JSON prompt/model metadata in each run."""
    if function is None:
        return lambda target: ExperimentWrapper(target, metadata)
    return ExperimentWrapper(function, metadata)

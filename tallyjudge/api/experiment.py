import json
from collections.abc import Callable, Sequence
from typing import Any, TypedDict, Unpack, overload

from tallyjudge.api.evaluator import Evaluator
from tallyjudge.engine.config import ExecutionConfig
from tallyjudge.engine.jobs import RowFunction
from tallyjudge.sdk.cache import ScoreCache
from tallyjudge.sdk.dataset import Dataset, canonical_json
from tallyjudge.sdk.events import EventSink
from tallyjudge.sdk.metrics import Metric, MetricBinding
from tallyjudge.sdk.results import RunCollector, RunResult
from tallyjudge.sdk.storage import StorageBackend
from tallyjudge.sdk.sync import run_sync


class _RunOptions(TypedDict, total=False):
    config: ExecutionConfig
    backend: StorageBackend
    include_inputs: bool
    event_sink: EventSink
    cache: ScoreCache


class ExperimentWrapper:
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
) -> ExperimentWrapper:
    """Decorate a row function and persist its JSON prompt/model metadata in each run."""
    ...


@overload
def experiment(
    function: None = None, *, metadata: dict[str, Any] | None = None
) -> Callable[[RowFunction], ExperimentWrapper]: ...


def experiment(
    function: RowFunction | None = None, *, metadata: dict[str, Any] | None = None
) -> ExperimentWrapper | Callable[[RowFunction], ExperimentWrapper]:
    if function is None:
        return lambda target: ExperimentWrapper(target, metadata)
    return ExperimentWrapper(function, metadata)

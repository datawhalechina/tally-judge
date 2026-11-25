from typing import TYPE_CHECKING, Any

from tallyjudge.api.comparison import compare_experiments, compare_runs
from tallyjudge.api.evaluator import Evaluator
from tallyjudge.api.experiment import experiment
from tallyjudge.api.factories import (
    create_analyzer,
    create_cache_backend,
    create_embedding_model,
    create_event_sink,
    create_importer,
    create_language_model,
    create_metric,
    create_optimizer,
    create_runner,
    create_storage_backend,
    create_synthesizer,
)
from tallyjudge.api.metrics import categorical_metric, numeric_metric, ranking_metric
from tallyjudge.api.suite import Suite
from tallyjudge.engine.config import ExecutionConfig
from tallyjudge.sdk.dataset import Dataset
from tallyjudge.sdk.errors import EvaluationAborted, RunSaveError
from tallyjudge.sdk.metrics import MetricBinding
from tallyjudge.sdk.results import MetricResult, RunCollector, RunResult

if TYPE_CHECKING:
    from tallyjudge.plugins.synthesis.contract import (
        SynthesisCollector,
        SynthesisResult,
    )


def __getattr__(name: str) -> Any:
    if name in ("SynthesisCollector", "SynthesisResult"):
        from tallyjudge.plugins.synthesis import contract

        return getattr(contract, name)
    raise AttributeError(name)


__all__ = [
    "Dataset",
    "compare_runs",
    "compare_experiments",
    "experiment",
    "numeric_metric",
    "categorical_metric",
    "ranking_metric",
    "create_storage_backend",
    "create_synthesizer",
    "create_optimizer",
    "Suite",
    "SynthesisCollector",
    "SynthesisResult",
    "create_cache_backend",
    "create_analyzer",
    "create_event_sink",
    "create_importer",
    "create_runner",
    "Evaluator",
    "MetricBinding",
    "RunCollector",
    "create_metric",
    "create_language_model",
    "create_embedding_model",
    "ExecutionConfig",
    "EvaluationAborted",
    "MetricResult",
    "RunResult",
    "RunSaveError",
]

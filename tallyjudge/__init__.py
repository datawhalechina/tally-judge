"""TallyJudge: explicit, framework-independent evaluation."""

from tallyjudge.api.comparison import compare_experiments, compare_runs
from tallyjudge.api.evaluator import Evaluator
from tallyjudge.api.experiment import experiment
from tallyjudge.api.factories import (
    create_cache_backend,
    create_embedding_model,
    create_event_sink,
    create_language_model,
    create_metric,
    create_optimizer,
    create_storage_backend,
    create_synthesizer,
)
from tallyjudge.api.metrics import categorical_metric, numeric_metric, ranking_metric
from tallyjudge.core.config import ExecutionConfig
from tallyjudge.core.dataset import Dataset
from tallyjudge.core.errors import EvaluationAborted, RunSaveError
from tallyjudge.core.results import MetricResult, RunCollector, RunResult
from tallyjudge.interfaces.metrics import MetricBinding
from tallyjudge.interfaces.synthesis import SynthesisCollector, SynthesisResult

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
    "SynthesisCollector",
    "SynthesisResult",
    "create_cache_backend",
    "create_event_sink",
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

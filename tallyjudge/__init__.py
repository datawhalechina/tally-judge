"""TallyJudge: explicit, framework-independent evaluation."""

from tallyjudge.api.evaluator import Evaluator
from tallyjudge.api.factories import (
    create_embedding_model,
    create_language_model,
    create_metric,
)
from tallyjudge.core.config import ExecutionConfig
from tallyjudge.core.dataset import Dataset
from tallyjudge.core.errors import EvaluationAborted
from tallyjudge.core.results import MetricResult, RunCollector, RunResult
from tallyjudge.interfaces.metrics import MetricBinding

__all__ = [
    "Dataset",
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
]

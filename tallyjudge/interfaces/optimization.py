"""Contracts for bounded optimization of application instructions."""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from tallyjudge.core.comparison import ExperimentComparison
from tallyjudge.core.dataset import Dataset
from tallyjudge.interfaces.metrics import Metric, MetricBinding


@dataclass(frozen=True)
class OptimizationResult:
    """Training selection and an independent paired holdout report."""

    instruction: str
    config_snapshot: dict[str, Any]
    training_scores: dict[str, float | None]
    requests: int
    budget_exhausted: bool
    holdout: ExperimentComparison


@runtime_checkable
class PromptOptimizer(Protocol):
    """Improve an instruction on labeled training rows with a disjoint holdout."""

    async def arun(
        self,
        instruction: str,
        *,
        train: Dataset,
        holdout: Dataset,
        metric: Metric | MetricBinding,
    ) -> OptimizationResult: ...
    def run(
        self,
        instruction: str,
        *,
        train: Dataset,
        holdout: Dataset,
        metric: Metric | MetricBinding,
    ) -> OptimizationResult: ...

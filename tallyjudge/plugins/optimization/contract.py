from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from tallyjudge.sdk.comparison import ExperimentComparison
from tallyjudge.sdk.dataset import Dataset
from tallyjudge.sdk.metrics import Metric, MetricBinding
from tallyjudge.sdk.plugin import KindSpec
from tallyjudge.sdk.sync import run_sync


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


OPTIMIZATION = KindSpec("optimization", PromptOptimizer)


class PromptOptimizerBase(ABC):
    """Implement arun; the synchronous run is provided."""

    @abstractmethod
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
    ) -> OptimizationResult:
        return run_sync(
            lambda: self.arun(instruction, train=train, holdout=holdout, metric=metric),
            "arun",
        )

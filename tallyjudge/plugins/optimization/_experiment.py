"""Shared scoring, candidate selection and holdout accounting for prompt optimization."""

import hashlib
from abc import ABC, abstractmethod
from typing import Any

from pydantic import Field

from tallyjudge.core.comparison import compare_experiments
from tallyjudge.core.config import ExecutionConfig
from tallyjudge.core.dataset import Dataset
from tallyjudge.core.execution import TaskRunner, run_sync
from tallyjudge.core.prompts import Message, PromptInput
from tallyjudge.core.results import RunCollector, RunResult
from tallyjudge.core.usage import RequestBudget, RequestLimitReached, mark_usage_source
from tallyjudge.interfaces.metrics import Metric, MetricBinding
from tallyjudge.interfaces.models import LanguageModel
from tallyjudge.interfaces.optimization import OptimizationResult
from tallyjudge.registry.specs import EmptyConfig


class OptimizationConfig(EmptyConfig):
    """Bound the population (including the baseline), all actual requests and selection randomness."""

    max_candidates: int = Field(default=6, ge=1)
    max_requests: int = Field(default=100, ge=1)
    seed: int = 0


class Answer(EmptyConfig):
    response: str


class Training:
    """Keep evaluated candidates and the best fully scored instruction on the training set."""

    def __init__(
        self,
        instruction: str,
        model: LanguageModel,
        metric: MetricBinding,
        train: Dataset,
        budget: RequestBudget,
        settings: OptimizationConfig,
    ):
        self.original = self.best = instruction
        self.model = model
        self.metric = metric
        self.train = train
        self.budget = budget
        self.settings = settings
        self.runs: dict[str, RunResult] = {}
        self.scores: dict[str, float | None] = {}

    async def evaluate(self, instruction: str, data: Dataset) -> RunResult:
        """Run the business instruction and objective with separately tracked model usage."""
        metadata = {
            "prompt_version": hashlib.sha256(instruction.encode()).hexdigest(),
            "model": self.model.plugin_id,
            "model_config": self.model.config_snapshot(),
        }

        async def answer(row):
            mark_usage_source(
                "generation", self.model.plugin_id, self.model.usage_tracking
            )
            try:
                result = await self.model.agenerate(
                    PromptInput(
                        (
                            Message(
                                "system",
                                instruction
                                + '\nReturn a JSON object with the string field "response".',
                            ),
                            Message("user", row["user_input"]),
                        )
                    ),
                    Answer,
                )
            except RequestLimitReached:
                return None
            return {"response": result.data.response}

        return await TaskRunner().arun(
            data,
            (self.metric,),
            ExecutionConfig(max_concurrency=1),
            RunCollector(),
            row_function=answer,
            experiment_metadata=metadata,
        )

    async def consider(self, instruction: str) -> None:
        if (
            instruction in self.scores
            or len(self.scores) >= self.settings.max_candidates
        ):
            return
        run = await self.evaluate(instruction, self.train)
        summary = run.summary[self.metric.key]
        value = summary["mean"] if summary["coverage"] == 1.0 else None
        self.runs[instruction], self.scores[instruction] = run, value
        current = self.scores.get(self.best)
        if value is not None and (
            current is None or self.utility(value) > self.utility(current)
        ):
            self.best = instruction
        if self.budget.exhausted:
            raise RequestLimitReached

    def utility(self, value: float) -> float:
        """Orient a raw objective score so higher utility is always preferred."""
        return (
            -value if self.metric.metric.spec.direction == "lower_is_better" else value
        )


class OptimizerBase(ABC):
    """Select using training labels only, then rerun the original and winner on held-out IDs."""

    plugin_id: str

    def __init__(self, *, llm: LanguageModel, **settings: Any):
        if llm.usage_tracking != "per_request":
            raise ValueError("Optimization requires a per_request model")
        self.model, self.settings = llm, OptimizationConfig(**settings)

    @abstractmethod
    async def search(self, training: Training) -> None:
        """Generate and evaluate candidate instructions using the training partition."""

    def run(
        self,
        instruction: str,
        *,
        train: Dataset,
        holdout: Dataset,
        metric: Metric | MetricBinding,
    ) -> OptimizationResult:
        """Run bounded optimization outside an existing event loop."""
        return run_sync(
            lambda: self.arun(instruction, train=train, holdout=holdout, metric=metric),
            "arun",
        )

    async def arun(
        self,
        instruction: str,
        *,
        train: Dataset,
        holdout: Dataset,
        metric: Metric | MetricBinding,
    ) -> OptimizationResult:
        """Fit on labeled training IDs and compare on untouched holdout IDs."""
        binding = metric if isinstance(metric, MetricBinding) else MetricBinding(metric)
        if binding.metric.spec.output_type is not float:
            raise ValueError("Optimization needs a numeric metric")
        partitions = []
        for dataset in (train, holdout):
            if not len(dataset) or dataset.id_source != "provided":
                raise ValueError(
                    "Each partition needs labeled records with explicit case IDs"
                )
            for row in dataset.records:
                if not isinstance(row.get("reference"), str) or not isinstance(
                    row.get("user_input"), str
                ):
                    raise ValueError(
                        "Each optimization row needs string user_input and reference"
                    )
            partitions.append({row["case_id"] for row in dataset.manifest})
        if partitions[0] & partitions[1]:
            raise ValueError("Training and holdout case IDs overlap")
        budget = RequestBudget(self.settings.max_requests)
        t = Training(instruction, self.model, binding, train, budget, self.settings)
        config = {
            **self.settings.model_dump(),
            "id": self.plugin_id,
            "model": self.model.config_snapshot(),
            "model_id": self.model.plugin_id,
            "objective": binding.snapshot(),
            "direction": binding.metric.spec.direction,
        }
        try:
            with budget.activate():
                # Leave at least one generation per holdout arm and row when the cap permits it.
                budget.limit = max(0, self.settings.max_requests - 2 * len(holdout))
                try:
                    await t.consider(instruction)
                    if self.settings.max_candidates > 1:
                        await self.search(t)
                except RequestLimitReached:
                    pass
                budget.limit = self.settings.max_requests
                left = await t.evaluate(instruction, holdout)
                right = await t.evaluate(t.best, holdout)
        finally:
            budget.closed = True
        return OptimizationResult(
            t.best,
            config,
            t.scores,
            budget.requests,
            budget.exhausted,
            compare_experiments(left, right),
        )

"""AnswerCorrectness from Ragas collections/answer_correctness (ec00339, Apache-2.0).

AnswerCorrectnessMetric retains the judge prompts and signed cosine. Undefined F-beta
denominators and zero-norm embeddings have explicit reasons instead of fallback scores.
"""

import math
from typing import Any

from tallyjudge.plugins.metrics._cosine import embedding_cosine
from tallyjudge.plugins.metrics._judge import ModelJudge
from tallyjudge.plugins.metrics.rag._correctness_prompt import prompt
from tallyjudge.sdk.config import EmptyConfig
from tallyjudge.sdk.errors import UndefinedScore
from tallyjudge.sdk.gate import gated
from tallyjudge.sdk.metrics import MetricSpec
from tallyjudge.sdk.models import EmbeddingModel, LanguageModel, model_snapshot
from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.results import MetricResult

type Weights = tuple[float, float] | list[float]


class AnswerCorrectnessConfig(EmptyConfig):
    model_config = {"allow_inf_nan": False}

    weights: Weights = (0.75, 0.25)
    beta: float | int = 1.0

    def model_post_init(self, context: Any) -> None:
        if len(self.weights) != 2 or min(self.weights) < 0 or max(self.weights) == 0:
            raise ValueError("Supply two nonnegative weights with a positive sum")
        if not isinstance(self.beta, float):
            raise ValueError("beta must be a float")


def _weight_shares(weights: Weights) -> tuple[float, float]:
    scale = max(weights)
    factual, semantic = (w / scale for w in weights)
    total = factual + semantic
    return factual / total, semantic / total


def _f_beta(tp: int, fp: int, fn: int, beta: float) -> float:
    if tp == fp == fn == 0:
        raise UndefinedScore("no_classified_statements")
    if beta == 0.0 and tp + fp == 0:
        raise UndefinedScore("no_predicted_statements")
    if tp == 0:
        return 0.0
    norm = math.hypot(1.0, beta)
    return tp / (tp + (1.0 / norm) ** 2 * fp + (beta / norm) ** 2 * fn)


class AnswerCorrectnessMetric(ModelJudge):
    plugin = Plugin(
        "metric",
        dependencies={"llm": "language_model"},
        optional_dependencies={"embeddings": "embedding_model"},
    )
    plugin_id = "metric.rag.answer_correctness"

    Config = AnswerCorrectnessConfig

    @staticmethod
    def spec_from_config(
        *, weights: Weights = (0.75, 0.25), beta: float = 1.0
    ) -> MetricSpec:
        _, semantic = _weight_shares(weights)
        return MetricSpec(
            AnswerCorrectnessMetric.plugin_id,
            "1",
            {"user_input": str, "response": str, "reference": str},
            value_range=(-semantic, 1.0),
        )

    def __init__(
        self,
        llm: LanguageModel,
        embeddings: EmbeddingModel | None = None,
        weights: Weights = (0.75, 0.25),
        beta: float = 1.0,
    ):
        config = AnswerCorrectnessConfig(weights=weights, beta=beta)
        if config.weights[1] > 0 and embeddings is None:
            raise ValueError(
                "embeddings is required when the similarity weight is positive"
            )
        super().__init__(llm)
        self.embeddings = (
            gated(embeddings) if embeddings and config.weights[1] > 0 else None
        )
        self.weights = (config.weights[0], config.weights[1])
        self.beta = config.beta
        self.spec = self.spec_from_config(weights=self.weights)
        self._shares = _weight_shares(self.weights)

    def config_snapshot(self) -> dict[str, Any]:
        snapshot = {
            **super().config_snapshot(),
            "weights": list(self.weights),
            "beta": self.beta,
        }
        if self.embeddings is not None:
            snapshot["embeddings"] = model_snapshot(self.embeddings)
        return snapshot

    async def _statements(self, question: str, answer: str) -> list[str]:
        from tallyjudge.plugins.metrics.rag.faithfulness import (
            StatementGeneratorInput,
            StatementGeneratorOutput,
            StatementGeneratorPrompt,
        )

        message = StatementGeneratorPrompt().render(
            StatementGeneratorInput(question=question, answer=answer)
        )
        return (await self._ask(message, StatementGeneratorOutput)).statements

    async def ascore(
        self, user_input: str, response: str, reference: str
    ) -> MetricResult:
        answer = await self._statements(user_input, response)
        truth = await self._statements(user_input, reference)
        factuality = 1.0
        if answer or truth:
            classifier = prompt()
            item = classifier.input_model(
                question=user_input, answer=answer, ground_truth=truth
            )
            classes = await self._ask(classifier.render(item), classifier.output_model)
            factuality = _f_beta(
                len(classes.TP), len(classes.FP), len(classes.FN), self.beta
            )
        if self.embeddings is None:
            return MetricResult(factuality)
        semantic = await embedding_cosine(self.embeddings, response, reference)
        wf, ws = self._shares
        return MetricResult(wf * factuality + ws * semantic)

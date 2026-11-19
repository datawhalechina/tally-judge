"""AnswerCorrectness from Ragas collections/answer_correctness (ec00339, Apache-2.0).

AnswerCorrectnessMetric retains the judge prompts and signed cosine. Undefined F-beta
denominators and zero-norm embeddings have explicit reasons instead of fallback scores.
"""

import math
from typing import Any

from tallyjudge.core.errors import UndefinedScore
from tallyjudge.core.results import MetricResult
from tallyjudge.core.usage import mark_usage_source, run_scoped_score
from tallyjudge.interfaces.metrics import MetricBase, MetricSpec
from tallyjudge.interfaces.models import EmbeddingModel, LanguageModel
from tallyjudge.plugins.metrics._cosine import embedding_cosine
from tallyjudge.plugins.metrics.rag._correctness_prompt import prompt
from tallyjudge.registry.specs import EmptyConfig

type Weights = tuple[float, float] | list[float]


class AnswerCorrectnessConfig(EmptyConfig):
    model_config = {"allow_inf_nan": False}

    weights: Weights = (0.75, 0.25)
    # Keep integers distinguishable: the archived constructor specifically requires a float.
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
    # F-beta = TP / (TP + (FP + beta² FN) / (1 + beta²)).
    # Normalize the coefficients before squaring so every finite beta is supported.
    norm = math.hypot(1.0, beta)
    return tp / (tp + (1.0 / norm) ** 2 * fp + (beta / norm) ** 2 * fn)


class AnswerCorrectnessMetric(MetricBase):
    """Combine statement-level factuality and answer-to-reference embedding similarity."""

    @staticmethod
    def spec_from_config(
        *, weights: Weights = (0.75, 0.25), beta: float = 1.0
    ) -> MetricSpec:
        """Describe the signed numeric range implied by the two weights."""
        _, semantic = _weight_shares(weights)
        return MetricSpec(
            "metric.rag.answer_correctness",
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
        self.llm = llm
        self.embeddings = embeddings if config.weights[1] > 0 else None
        self.weights = (config.weights[0], config.weights[1])
        self.beta = config.beta
        self.spec = self.spec_from_config(weights=self.weights)
        self._shares = _weight_shares(self.weights)

    def config_snapshot(self) -> dict[str, Any]:
        snapshot = {
            "weights": list(self.weights),
            "beta": self.beta,
            "prompt_revision": "1",
            "llm": {
                "plugin_id": self.llm.plugin_id,
                "config": self.llm.config_snapshot(),
            },
        }
        if self.embeddings is not None:
            snapshot["embeddings"] = {
                "plugin_id": self.embeddings.plugin_id,
                "config": self.embeddings.config_snapshot(),
            }
        return snapshot

    async def ascore(
        self, user_input: str, response: str, reference: str
    ) -> MetricResult:
        """Score a system response against the reference for the same question."""
        return await self.ascore_inputs(
            {"user_input": user_input, "response": response, "reference": reference}
        )

    async def _statements(self, question: str, answer: str) -> list[str]:
        from tallyjudge.plugins.metrics.rag.faithfulness import (
            StatementGeneratorInput,
            StatementGeneratorOutput,
            StatementGeneratorPrompt,
        )

        message = StatementGeneratorPrompt().render(
            StatementGeneratorInput(question=question, answer=answer)
        )
        mark_usage_source("generation", self.llm.plugin_id, self.llm.usage_tracking)
        return (
            await self.llm.agenerate(message, StatementGeneratorOutput)
        ).data.statements

    async def _ascore_validated(
        self, user_input: str, response: str, reference: str
    ) -> MetricResult:
        return await run_scoped_score(
            self._correctness(user_input, response, reference)
        )

    async def _correctness(
        self, question: str, response: str, reference: str
    ) -> MetricResult:
        answer = await self._statements(question, response)
        truth = await self._statements(question, reference)
        factuality = 1.0
        if answer or truth:
            classifier = prompt()
            item = classifier.input_model(
                question=question, answer=answer, ground_truth=truth
            )
            mark_usage_source("generation", self.llm.plugin_id, self.llm.usage_tracking)
            classes = (
                await self.llm.agenerate(
                    classifier.render(item), classifier.output_model
                )
            ).data
            factuality = _f_beta(
                len(classes.TP), len(classes.FP), len(classes.FN), self.beta
            )
        if self.embeddings is None:
            return MetricResult(factuality)
        semantic = await embedding_cosine(self.embeddings, response, reference)
        wf, ws = self._shares
        return MetricResult(wf * factuality + ws * semantic)

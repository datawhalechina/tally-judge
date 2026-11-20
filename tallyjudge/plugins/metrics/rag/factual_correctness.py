"""FactualCorrectness from Ragas ec00339 (Apache-2.0), renamed FactualCorrectnessMetric.

Evaluate only the directions required by the scoring mode. Empty claims and
verdicts are undefined. The denominator counts the returned binary judgments.
"""

from typing import Any, Literal

from tallyjudge.core.errors import UndefinedScore
from tallyjudge.core.results import MetricResult
from tallyjudge.interfaces.metrics import MetricSpec
from tallyjudge.interfaces.models import LanguageModel
from tallyjudge.plugins.metrics.rag._factual_prompts import prompts
from tallyjudge.plugins.metrics.rag._judge import (
    RagJudge,
    binary_verdict,
    required_text_inputs,
)
from tallyjudge.registry.specs import EmptyConfig

type Mode = Literal["precision", "recall", "f1"]
type Granularity = Literal["low", "high"]


class FactualCorrectnessConfig(EmptyConfig):
    """Choose a scoring direction, F-beta weight and claim decomposition examples."""

    model_config = {"allow_inf_nan": False, "frozen": True}

    mode: Mode = "f1"
    beta: float = 1.0
    atomicity: Granularity = "low"
    coverage: Granularity = "low"


def _f_beta(precision: float, recall: float, beta: float) -> float:
    if precision == 0:
        return 0.0
    try:
        weight = beta**2
    except OverflowError:
        # At this magnitude F-beta rounds to recall for positive precision.
        return recall
    denominator = weight * precision + recall
    if denominator == 0:
        raise UndefinedScore("zero_fbeta_denominator")
    return (1 + weight) * (precision * recall) / denominator


class FactualCorrectnessMetric(RagJudge):
    """Verify factual claims in the answer against the reference and vice versa."""

    spec = MetricSpec(
        "metric.rag.factual_correctness",
        "1",
        required_text_inputs("response", "reference"),
        value_range=(0.0, 1.0),
    )

    def __init__(
        self,
        llm: LanguageModel,
        *,
        mode: Mode = "f1",
        beta: float = 1.0,
        atomicity: Granularity = "low",
        coverage: Granularity = "low",
    ):
        super().__init__(llm)
        self.settings = FactualCorrectnessConfig(
            mode=mode, beta=beta, atomicity=atomicity, coverage=coverage
        )
        claim_prompt, self._nli = prompts()
        self._decomposition = claim_prompt(
            self.settings.atomicity, self.settings.coverage
        )

    def config_snapshot(self) -> dict[str, Any]:
        """Include the scoring mode, beta and both claim decomposition settings."""
        return {**super().config_snapshot(), **self.settings.model_dump()}

    async def ascore(self, response: str, reference: str) -> MetricResult:
        """Score precision, recall or F-beta, rounded to two decimal places."""
        return await self.ascore_inputs({"response": response, "reference": reference})

    async def _support_rate(self, text: str, evidence: str, empty_claims: str) -> float:
        template = self._decomposition
        sample = template.input_model(
            response=text,
            atomicity=self.settings.atomicity,
            coverage=self.settings.coverage,
        )
        claims = (
            await self._ask(template.render(sample), template.output_model)
        ).claims
        if not claims:
            raise UndefinedScore(empty_claims)
        item = self._nli.input_model(context=evidence, statements=claims)
        reply = await self._ask(self._nli.render(item), self._nli.output_model)
        if not reply.statements:
            raise UndefinedScore("no_verdicts")
        supported = sum(binary_verdict(item.verdict) for item in reply.statements)
        return supported / (len(reply.statements) + 1e-8)

    async def _score(self, response: str, reference: str) -> MetricResult:
        mode = self.settings.mode
        if mode == "recall":
            score = await self._support_rate(
                reference, response, "no_reference_statements"
            )
        else:
            score = await self._support_rate(
                response, reference, "no_response_statements"
            )
            if mode == "f1":
                recall = await self._support_rate(
                    reference, response, "no_reference_statements"
                )
                score = _f_beta(score, recall, self.settings.beta)
        # The archived scalar uses NumPy's scale, round-to-even, then unscale.
        return MetricResult(round(score * 100) / 100)

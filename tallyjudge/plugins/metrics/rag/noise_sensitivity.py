"""NoiseSensitivity from Ragas ec00339 (Apache-2.0), renamed NoiseSensitivityMetric.

Empty statement sets are undefined. Each input statement needs one binary
verdict. Relevant support takes precedence over irrelevant support.
"""

from typing import Any, Literal

from tallyjudge.core.errors import ModelOutputError, UndefinedScore
from tallyjudge.core.prompts import PromptInput
from tallyjudge.core.results import MetricResult
from tallyjudge.interfaces.metrics import MetricSpec
from tallyjudge.interfaces.models import LanguageModel
from tallyjudge.plugins.metrics.rag._judge import (
    RagJudge,
    binary_verdict,
    required_text_inputs,
)
from tallyjudge.plugins.metrics.rag._noise_prompts import (
    nli_statement_prompt,
    output_models,
    statement_generator_prompt,
)
from tallyjudge.registry.specs import EmptyConfig

type NoiseMode = Literal["relevant", "irrelevant"]


class NoiseSensitivityConfig(EmptyConfig):
    """Choose the source of support counted for incorrect answer statements."""

    mode: NoiseMode = "relevant"


class NoiseSensitivityMetric(RagJudge):
    """Fraction of all answer statements that are incorrect and supported by the selected passages."""

    spec = MetricSpec(
        "metric.rag.noise_sensitivity",
        "1",
        required_text_inputs("user_input", "response", "reference", passages=True),
        value_range=(0.0, 1.0),
        direction="lower_is_better",
    )

    def __init__(self, llm: LanguageModel, *, mode: NoiseMode = "relevant"):
        super().__init__(llm)
        self.mode = NoiseSensitivityConfig(mode=mode).mode

    def config_snapshot(self) -> dict[str, Any]:
        """Include the source of retrieval support selected for this score."""
        return {**super().config_snapshot(), "mode": self.mode}

    async def ascore(
        self,
        user_input: str,
        response: str,
        reference: str,
        retrieved_contexts: list[str],
    ) -> MetricResult:
        """Score incorrect claims supported by relevant or exclusively irrelevant retrievals."""
        return await self.ascore_inputs(
            {
                "user_input": user_input,
                "response": response,
                "reference": reference,
                "retrieved_contexts": retrieved_contexts,
            }
        )

    async def _statements(
        self, question: str, text: str, empty_reason: str
    ) -> list[str]:
        schema, _ = output_models()
        prompt = PromptInput.from_text(statement_generator_prompt(question, text))
        statements = (await self._ask(prompt, schema)).statements
        if not statements:
            raise UndefinedScore(empty_reason)
        return statements

    async def _supported(self, evidence: str, statements: list[str]) -> set[int]:
        _, schema = output_models()
        prompt = PromptInput.from_text(nli_statement_prompt(evidence, statements))
        verdicts = (await self._ask(prompt, schema)).statements
        if len(verdicts) != len(statements):
            raise ModelOutputError("verdict_count_mismatch")
        return {
            index for index, item in enumerate(verdicts) if binary_verdict(item.verdict)
        }

    async def _score(
        self,
        user_input: str,
        response: str,
        reference: str,
        retrieved_contexts: list[str],
    ) -> MetricResult:
        gold = await self._statements(user_input, reference, "no_reference_statements")
        answer = await self._statements(user_input, response, "no_response_statements")
        relevant: set[int] = set()
        irrelevant: set[int] = set()
        for passage in retrieved_contexts:
            supports_gold = await self._supported(passage, gold)
            supported_answer = await self._supported(passage, answer)
            if supports_gold:
                relevant.update(supported_answer)
            else:
                irrelevant.update(supported_answer)
        correct = await self._supported(reference, answer)
        attributed = relevant if self.mode == "relevant" else irrelevant - relevant
        return MetricResult(len(attributed - correct) / len(answer))

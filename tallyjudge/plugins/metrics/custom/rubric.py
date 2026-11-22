"""Single-turn rubric scoring from Ragas ec00339 (Apache-2.0).

The two fixed metrics select the archived reference-based or reference-free
default rubric. The caller's rubric order is part of the rendered prompt.
"""

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from tallyjudge.core.results import MetricResult
from tallyjudge.core.usage import mark_usage_source, run_scoped_score
from tallyjudge.interfaces.metrics import MetricBase, MetricSpec
from tallyjudge.interfaces.models import LanguageModel
from tallyjudge.plugins.metrics.custom._model import judge_snapshot, rule_digest
from tallyjudge.plugins.metrics.custom._rubric_prompts import (
    DEFAULT_REFERENCE_FREE_RUBRICS,
    DEFAULT_WITH_REFERENCE_RUBRICS,
    InstanceRubricScoreInput,
    InstanceRubricScoreOutput,
    InstanceRubricScorePrompt,
    RubricScoreInput,
    RubricScoreOutput,
    RubricScorePrompt,
)

_INPUTS = {
    "user_input": str | None,
    "response": str | None,
    "retrieved_contexts": list[str] | None,
    "reference_contexts": list[str] | None,
    "reference": str | None,
}
_DEFAULTS = dict.fromkeys(_INPUTS)


class RubricConfig(BaseModel):
    model_config = ConfigDict(strict=True)
    rubrics: dict[str, str] | None = None


class _FixedRubric(MetricBase):
    default_rubrics: dict[str, str]

    def __init__(self, llm: LanguageModel, rubrics: dict[str, str] | None = None):
        self.llm = llm
        supplied = RubricConfig(rubrics=rubrics).rubrics
        rules = self.default_rubrics if supplied is None else supplied
        rule_text = "\n".join(f"{key}: {value}" for key, value in rules.items())
        self.prompt = RubricScorePrompt()
        self.prompt.instruction += f"\n\nScoring Rubrics:\n{rule_text}\n"
        self._digest = rule_digest(rule_text)

    def config_snapshot(self) -> dict[str, Any]:
        return {
            **judge_snapshot(self.llm, prompt_revision=self.spec.revision),
            "rubrics_sha256": self._digest,
        }

    async def ascore(
        self,
        response: str | None = None,
        *,
        user_input: str | None = None,
        retrieved_contexts: list[str] | None = None,
        reference_contexts: list[str] | None = None,
        reference: str | None = None,
    ) -> MetricResult:
        """Apply the configured rubric to the supplied sample fields."""
        return await self.ascore_inputs(
            dict(
                user_input=user_input,
                response=response,
                retrieved_contexts=retrieved_contexts,
                reference_contexts=reference_contexts,
                reference=reference,
            )
        )

    async def _ascore_validated(self, **inputs: Any) -> MetricResult:
        prompt = self.prompt.render(RubricScoreInput.model_construct(**inputs))
        mark_usage_source("generation", self.llm.plugin_id, self.llm.usage_tracking)
        verdict = (
            await run_scoped_score(self.llm.agenerate(prompt, RubricScoreOutput))
        ).data
        return MetricResult(float(verdict.score), verdict.feedback)


class RubricWithoutReferenceMetric(_FixedRubric):
    """Grade a sample using the reference-free default rubric or supplied criteria."""

    spec = MetricSpec(
        "metric.custom.rubric_without_reference",
        "3",
        _INPUTS,
        defaults=_DEFAULTS,
        value_range=(1, 5),
        levels=(1, 2, 3, 4, 5),
    )
    default_rubrics = DEFAULT_REFERENCE_FREE_RUBRICS


class RubricWithReferenceMetric(_FixedRubric):
    """Grade a sample using the reference-based default rubric or supplied criteria."""

    spec = MetricSpec(
        "metric.custom.rubric_with_reference",
        "3",
        _INPUTS,
        defaults=_DEFAULTS,
        value_range=(1, 5),
        levels=(1, 2, 3, 4, 5),
    )
    default_rubrics = DEFAULT_WITH_REFERENCE_RUBRICS

    async def ascore(
        self,
        response: str | None = None,
        reference: str | None = None,
        *,
        user_input: str | None = None,
        retrieved_contexts: list[str] | None = None,
        reference_contexts: list[str] | None = None,
    ) -> MetricResult:
        """Grade one answer with its optional positional reference."""
        return await super().ascore(
            response,
            reference=reference,
            user_input=user_input,
            retrieved_contexts=retrieved_contexts,
            reference_contexts=reference_contexts,
        )


class InstanceRubricMetric(MetricBase):
    """Apply the rubric supplied with each sample."""

    spec = MetricSpec(
        "metric.custom.instance_rubric",
        "2",
        {**_INPUTS, "rubrics": Annotated[dict[str, str], Field(min_length=1)]},
        defaults=_DEFAULTS,
        value_range=(1, 5),
        levels=(1, 2, 3, 4, 5),
    )

    def __init__(self, llm: LanguageModel):
        self.llm = llm
        self.prompt = InstanceRubricScorePrompt()

    def config_snapshot(self) -> dict[str, Any]:
        return judge_snapshot(self.llm, prompt_revision=self.spec.revision)

    async def ascore(
        self,
        response: str | None = None,
        rubrics: dict[str, str] | None = None,
        *,
        reference: str | None = None,
        user_input: str | None = None,
        retrieved_contexts: list[str] | None = None,
        reference_contexts: list[str] | None = None,
    ) -> MetricResult:
        """Grade one sample using its own nonempty rubric."""
        return await self.ascore_inputs(
            dict(
                response=response,
                rubrics=rubrics,
                reference=reference,
                user_input=user_input,
                retrieved_contexts=retrieved_contexts,
                reference_contexts=reference_contexts,
            )
        )

    async def _ascore_validated(self, **inputs: Any) -> MetricResult:
        prompt = self.prompt.render(InstanceRubricScoreInput.model_construct(**inputs))
        mark_usage_source("generation", self.llm.plugin_id, self.llm.usage_tracking)
        verdict = (
            await run_scoped_score(
                self.llm.agenerate(prompt, InstanceRubricScoreOutput)
            )
        ).data
        return MetricResult(float(verdict.score), verdict.feedback)

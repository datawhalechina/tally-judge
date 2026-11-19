"""Fixed and per-row single-turn rubrics migrated from Ragas ec00339 (Apache-2.0)."""

from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from tallyjudge.core.prompts import PromptInput
from tallyjudge.core.results import MetricResult
from tallyjudge.core.usage import mark_usage_source, run_scoped_score
from tallyjudge.interfaces.metrics import MetricBase, MetricSpec
from tallyjudge.interfaces.models import LanguageModel
from tallyjudge.plugins.metrics.custom._model import judge_snapshot, rule_digest
from tallyjudge.plugins.metrics.custom._rubric_prompts import (
    DEFAULT_REFERENCE_FREE_RUBRICS,
    DEFAULT_WITH_REFERENCE_RUBRICS,
    FixedRubricPrompt,
    InstanceRubricInput,
    InstanceRubricPrompt,
    RubricInput,
    RubricVerdict,
)

_LEVELS = tuple(f"score{i}_description" for i in range(1, 6))
_CONTEXT = {
    "user_input": str | None,
    "retrieved_contexts": list[str] | None,
    "reference_contexts": list[str] | None,
}
_DEFAULTS = dict.fromkeys(_CONTEXT)


def _ordered_levels(rules: dict[str, str]) -> dict[str, str]:
    if rules.keys() != set(_LEVELS):
        raise ValueError("A rubric must describe every level from 1 to 5")
    return {key: rules[key] for key in _LEVELS}


Rubrics = Annotated[
    dict[str, Annotated[str, Field(min_length=1)]], AfterValidator(_ordered_levels)
]


class RubricConfig(BaseModel):
    model_config = ConfigDict(strict=True)
    rubrics: Rubrics | None = None


async def _score(llm: LanguageModel, prompt: PromptInput) -> MetricResult:
    mark_usage_source("generation", llm.plugin_id, llm.usage_tracking)
    verdict = (await run_scoped_score(llm.agenerate(prompt, RubricVerdict))).data
    return MetricResult(float(verdict.score), verdict.feedback)


class _FixedRubric(MetricBase):
    default_rubrics: dict[str, str]

    def __init__(self, llm: LanguageModel, rubrics: dict[str, str] | None = None):
        self.llm = llm
        rules = RubricConfig(rubrics=rubrics).rubrics or self.default_rubrics
        self.prompt = FixedRubricPrompt(
            rules, use_default_examples=rules == self.default_rubrics
        )
        self._digest = rule_digest(rules)

    def config_snapshot(self) -> dict[str, Any]:
        return {
            **judge_snapshot(self.llm, prompt_revision=self.spec.revision),
            "rubrics_sha256": self._digest,
        }

    async def _ascore_validated(self, **inputs: Any) -> MetricResult:
        return await _score(
            self.llm, self.prompt.render(RubricInput.model_construct(**inputs))
        )


class RubricWithoutReferenceMetric(_FixedRubric):
    """Score an answer with a fixed five-level rubric and optional supporting contexts."""

    spec = MetricSpec(
        "metric.custom.rubric_without_reference",
        "2",
        {"response": str, **_CONTEXT},
        defaults=_DEFAULTS,
        value_range=(1, 5),
        levels=(1, 2, 3, 4, 5),
    )
    default_rubrics = DEFAULT_REFERENCE_FREE_RUBRICS

    async def ascore(
        self,
        response: str,
        *,
        user_input: str | None = None,
        retrieved_contexts: list[str] | None = None,
        reference_contexts: list[str] | None = None,
    ) -> MetricResult:
        """Grade one answer without a reference answer."""
        return await self.ascore_inputs(
            {
                "response": response,
                "user_input": user_input,
                "retrieved_contexts": retrieved_contexts,
                "reference_contexts": reference_contexts,
            }
        )


class RubricWithReferenceMetric(_FixedRubric):
    """Score an answer against a required reference and a fixed five-level rubric."""

    spec = MetricSpec(
        "metric.custom.rubric_with_reference",
        "2",
        {"response": str, "reference": str, **_CONTEXT},
        defaults=_DEFAULTS,
        value_range=(1, 5),
        levels=(1, 2, 3, 4, 5),
    )
    default_rubrics = DEFAULT_WITH_REFERENCE_RUBRICS

    async def ascore(
        self,
        response: str,
        reference: str,
        *,
        user_input: str | None = None,
        retrieved_contexts: list[str] | None = None,
        reference_contexts: list[str] | None = None,
    ) -> MetricResult:
        """Grade one answer using its reference."""
        return await self.ascore_inputs(
            {
                "response": response,
                "reference": reference,
                "user_input": user_input,
                "retrieved_contexts": retrieved_contexts,
                "reference_contexts": reference_contexts,
            }
        )


class InstanceRubricMetric(MetricBase):
    """Apply the five level descriptions supplied with each answer."""

    spec = MetricSpec(
        "metric.custom.instance_rubric",
        "1",
        {"response": str, "rubrics": Rubrics, "reference": str | None, **_CONTEXT},
        defaults={"reference": None, **_DEFAULTS},
        value_range=(1, 5),
        levels=(1, 2, 3, 4, 5),
    )

    def __init__(self, llm: LanguageModel):
        self.llm = llm
        self.prompt = InstanceRubricPrompt()

    def config_snapshot(self) -> dict[str, Any]:
        return judge_snapshot(self.llm)

    async def ascore(
        self,
        response: str,
        rubrics: dict[str, str],
        *,
        reference: str | None = None,
        user_input: str | None = None,
        retrieved_contexts: list[str] | None = None,
        reference_contexts: list[str] | None = None,
    ) -> MetricResult:
        """Grade one answer using the level descriptions on that sample."""
        return await self.ascore_inputs(
            {
                "response": response,
                "rubrics": rubrics,
                "reference": reference,
                "user_input": user_input,
                "retrieved_contexts": retrieved_contexts,
                "reference_contexts": reference_contexts,
            }
        )

    async def _ascore_validated(self, **inputs: Any) -> MetricResult:
        return await _score(
            self.llm, self.prompt.render(InstanceRubricInput.model_construct(**inputs))
        )

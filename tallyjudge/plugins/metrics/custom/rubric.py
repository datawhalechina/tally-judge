"""Single-turn rubric scoring from Ragas ec00339 (Apache-2.0).

The two fixed metrics select the archived reference-based or reference-free
default rubric. The caller's rubric order is part of the rendered prompt.
"""

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from tallyjudge.plugins.metrics._judge import ModelJudge
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
from tallyjudge.sdk.dataset import canonical_sha256
from tallyjudge.sdk.metrics import MetricSpec
from tallyjudge.sdk.models import LanguageModel
from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.results import MetricResult

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


class _Rubric(ModelJudge):
    @property
    def prompt_revision(self) -> str:
        return self.spec.revision


class _FixedRubric(_Rubric):
    default_rubrics: dict[str, str]

    def __init__(self, llm: LanguageModel, rubrics: dict[str, str] | None = None):
        super().__init__(llm)
        supplied = RubricConfig(rubrics=rubrics).rubrics
        rules = self.default_rubrics if supplied is None else supplied
        rule_text = "\n".join(f"{key}: {value}" for key, value in rules.items())
        self.prompt = RubricScorePrompt()
        self.prompt.instruction += f"\n\nScoring Rubrics:\n{rule_text}\n"
        self._digest = canonical_sha256(rule_text)

    def config_snapshot(self) -> dict[str, Any]:
        return {**super().config_snapshot(), "rubrics_sha256": self._digest}

    async def ascore(self, **inputs: Any) -> MetricResult:
        prompt = self.prompt.render(RubricScoreInput.model_construct(**inputs))
        verdict = await self._ask(prompt, RubricScoreOutput)
        return MetricResult(float(verdict.score), verdict.feedback)


class RubricWithoutReferenceMetric(_FixedRubric):
    plugin = Plugin("metric", dependencies={"llm": "language_model"})

    Config = RubricConfig

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
    plugin = Plugin("metric", dependencies={"llm": "language_model"})

    Config = RubricConfig

    spec = MetricSpec(
        "metric.custom.rubric_with_reference",
        "3",
        _INPUTS,
        defaults=_DEFAULTS,
        value_range=(1, 5),
        levels=(1, 2, 3, 4, 5),
    )
    default_rubrics = DEFAULT_WITH_REFERENCE_RUBRICS


class InstanceRubricMetric(_Rubric):
    plugin = Plugin("metric", dependencies={"llm": "language_model"})

    spec = MetricSpec(
        "metric.custom.instance_rubric",
        "2",
        {**_INPUTS, "rubrics": Annotated[dict[str, str], Field(min_length=1)]},
        defaults=_DEFAULTS,
        value_range=(1, 5),
        levels=(1, 2, 3, 4, 5),
    )

    def __init__(self, llm: LanguageModel):
        super().__init__(llm)
        self.prompt = InstanceRubricScorePrompt()

    async def ascore(self, **inputs: Any) -> MetricResult:
        prompt = self.prompt.render(InstanceRubricScoreInput.model_construct(**inputs))
        verdict = await self._ask(prompt, InstanceRubricScoreOutput)
        return MetricResult(float(verdict.score), verdict.feedback)

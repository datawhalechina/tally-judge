"""Single-turn judges derived from Ragas numeric, _aspect_critic and _simple_criteria (ec00339).

Revision 1 uses a typed verdict and one independent generation per vote.
"""

from collections import Counter
from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    create_model,
    field_validator,
    model_validator,
)

from tallyjudge.core.prompts import PromptInput, StructuredPrompt
from tallyjudge.core.results import MetricResult
from tallyjudge.core.usage import mark_usage_source, run_scoped_score
from tallyjudge.interfaces.metrics import MetricBase, MetricSpec
from tallyjudge.interfaces.models import LanguageModel
from tallyjudge.plugins.metrics.custom._model import judge_snapshot, rule_digest


class JudgeConfig(BaseModel):
    model_config = ConfigDict(strict=True)
    criteria: str = Field(min_length=1)
    strictness: int = Field(default=1, gt=0)

    @field_validator("strictness")
    @classmethod
    def odd_vote_count(cls, count: int) -> int:
        return count if count % 2 else count + 1


class NumericConfig(JudgeConfig):
    levels: list[int] | None = Field(default=None, min_length=1)
    value_range: tuple[FiniteFloat, FiniteFloat] | None = None

    @field_validator("value_range", mode="before")
    @classmethod
    def accept_json_range(cls, bounds: Any) -> Any:
        return tuple(bounds) if isinstance(bounds, list) else bounds

    @model_validator(mode="after")
    def check_scale(self) -> Self:
        if self.levels is not None and self.value_range is not None:
            raise ValueError("Use integer levels or a continuous value_range")
        if self.levels is None and self.strictness != 1:
            raise ValueError("Continuous scores require one vote")
        return self


class CategoricalConfig(JudgeConfig):
    labels: list[str] = Field(min_length=1)


class JudgeInput(BaseModel):
    response: str
    user_input: str | None = None
    reference: str | None = None
    retrieved_contexts: list[str] | None = None


_INPUTS = {
    "response": str,
    "user_input": str | None,
    "reference": str | None,
    "retrieved_contexts": list[str] | None,
}
_DEFAULTS = {"user_input": None, "reference": None, "retrieved_contexts": None}


class Vote(BaseModel):
    model_config = ConfigDict(strict=True)
    value: int | float | str
    reason: str


def _choice_vote(value_type: type, choices: tuple) -> type[Vote]:
    # Include the same choices in the prompt schema and the structured retry validator.
    def declared_value(value: int | str) -> int | str:
        if value not in choices:
            raise ValueError("Choose a declared verdict")
        return value

    return create_model(
        "ChoiceVote",
        __base__=Vote,
        __validators__={"declared_value": field_validator("value")(declared_value)},
        value=(value_type, Field(json_schema_extra={"enum": list(choices)})),
    )


class _JudgePrompt(StructuredPrompt[JudgeInput, Vote]):
    input_model = JudgeInput
    examples: list[tuple[JudgeInput, Vote]] = []

    def __init__(self, criteria: str, output: type[Vote]):
        self.instruction = f"Evaluate the response using these criteria and explain the verdict.\n{criteria}"
        self.output_model = output


class _JudgeMetric(MetricBase):
    def __init__(
        self, llm: LanguageModel, criteria: str, strictness: int, output: type[Vote]
    ):
        self.llm = llm
        self.strictness = strictness
        self._criteria_digest = rule_digest(criteria)
        self.prompt = _JudgePrompt(criteria, output)

    def config_snapshot(self) -> dict[str, Any]:
        return {
            **judge_snapshot(self.llm),
            "criteria_sha256": self._criteria_digest,
            "strictness": self.strictness,
        }

    async def ascore(
        self,
        response: str,
        *,
        user_input: str | None = None,
        reference: str | None = None,
        retrieved_contexts: list[str] | None = None,
    ) -> MetricResult:
        """Judge one response using the supplied question, reference and retrieved evidence."""
        return await self.ascore_inputs(
            {
                "response": response,
                "user_input": user_input,
                "reference": reference,
                "retrieved_contexts": retrieved_contexts,
            }
        )

    async def _ascore_validated(self, **inputs: Any) -> MetricResult:
        prompt = self.prompt.render(JudgeInput.model_construct(**inputs))
        return await run_scoped_score(self._vote(prompt))

    async def _vote(self, prompt: PromptInput) -> MetricResult:
        votes = []
        for _ in range(self.strictness):
            mark_usage_source("generation", self.llm.plugin_id, self.llm.usage_tracking)
            votes.append(
                (await self.llm.agenerate(prompt, self.prompt.output_model)).data
            )
        # Counter retains insertion order for ties; use the first winning reason as well.
        winner = Counter(vote.value for vote in votes).most_common(1)[0][0]
        reason = next(vote.reason for vote in votes if vote.value == winner)
        return MetricResult(
            float(winner) if self.spec.output_type is float else str(winner), reason
        )


class NumericMetric(_JudgeMetric):
    """Return a bounded continuous score or vote among explicit integer levels."""

    @property
    def levels(self) -> tuple[int, ...] | None:
        """Return the discrete grades, or None for a continuous score."""
        return self.spec.levels or None

    @staticmethod
    def spec_from_config(
        *,
        levels: list[int] | None = None,
        value_range: tuple[float, float] | None = None,
        **settings: Any,
    ) -> MetricSpec:
        bounds = (
            (min(levels), max(levels))
            if levels is not None
            else value_range or (0.0, 1.0)
        )
        return MetricSpec(
            "metric.custom.numeric",
            "1",
            _INPUTS,
            defaults=_DEFAULTS,
            value_range=bounds,
            levels=tuple(levels or ()),
        )

    def __init__(
        self,
        *,
        llm: LanguageModel,
        criteria: str,
        levels: list[int] | None = None,
        value_range: tuple[float, float] | None = None,
        strictness: int = 1,
    ):
        config = NumericConfig(
            criteria=criteria,
            levels=levels,
            value_range=value_range,
            strictness=strictness,
        )
        self.spec = self.spec_from_config(
            levels=config.levels, value_range=config.value_range
        )
        if not self.spec.levels:
            low, high = self.spec.value_range
            output = create_model(
                "NumericVote",
                __base__=Vote,
                value=(float, Field(ge=low, le=high, allow_inf_nan=False)),
            )
        else:
            output = _choice_vote(int, self.spec.levels)
        super().__init__(llm, config.criteria, config.strictness, output)

    def config_snapshot(self) -> dict[str, Any]:
        result = super().config_snapshot()
        result["value_range"] = list(self.spec.value_range)
        if self.spec.levels:
            result["levels"] = list(self.spec.levels)
        return result


class CategoricalMetric(_JudgeMetric):
    """Vote among the configured labels and preserve the winning label as text."""

    @staticmethod
    def spec_from_config(*, labels: list[str], **settings: Any) -> MetricSpec:
        return MetricSpec(
            "metric.custom.categorical",
            "1",
            _INPUTS,
            defaults=_DEFAULTS,
            output_type=str,
            labels=tuple(labels),
        )

    def __init__(
        self,
        *,
        llm: LanguageModel,
        criteria: str,
        labels: list[str],
        strictness: int = 1,
    ):
        config = CategoricalConfig(
            criteria=criteria, labels=labels, strictness=strictness
        )
        self.spec = self.spec_from_config(labels=config.labels)
        super().__init__(
            llm, config.criteria, config.strictness, _choice_vote(str, self.spec.labels)
        )

    def config_snapshot(self) -> dict[str, Any]:
        return {**super().config_snapshot(), "labels": list(self.spec.labels)}

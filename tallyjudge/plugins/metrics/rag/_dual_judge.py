"""Paired normalized ratings migrated from Ragas ec00339 (Apache-2.0).

Revision 3 propagates model output failures so the adapter owns the structured
retry budget. Each judge repeats its prompt only for a rating outside its scale.
"""

from typing import Any, ClassVar

from pydantic import Field

from tallyjudge.plugins._prompting import StructuredPrompt
from tallyjudge.plugins.metrics._judge import ModelJudge
from tallyjudge.sdk.config import EmptyConfig
from tallyjudge.sdk.errors import UndefinedScore
from tallyjudge.sdk.models import LanguageModel
from tallyjudge.sdk.results import MetricResult


class DualJudgeConfig(EmptyConfig):
    """Number of rating attempts per judge, including its first attempt."""

    max_retries: int = Field(default=5, ge=0)


class DualJudge(ModelJudge):
    grades: ClassVar[tuple[int, ...]]

    def __init__(self, llm: LanguageModel, max_retries: int = 5):
        super().__init__(llm)
        self.max_retries = DualJudgeConfig(max_retries=max_retries).max_retries

    def config_snapshot(self) -> dict[str, Any]:
        return {**super().config_snapshot(), "max_retries": self.max_retries}

    async def _rate_pair(
        self,
        prompts: tuple[StructuredPrompt[Any, Any], StructuredPrompt[Any, Any]],
        first: dict[str, str],
        second: dict[str, str],
    ) -> MetricResult:
        ratings = []
        for template, fields in ((prompts[0], first), (prompts[1], second)):
            request = template.render(template.input_model(**fields))
            for _ in range(self.max_retries):
                rating = (await self._ask(request, template.output_model)).rating
                if rating in self.grades:
                    ratings.append(rating)
                    break
        if not ratings:
            raise UndefinedScore("no_valid_ratings")
        return MetricResult(sum(ratings) / (len(ratings) * max(self.grades)))

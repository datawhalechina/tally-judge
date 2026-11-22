"""Paired normalized ratings migrated from Ragas ec00339 (Apache-2.0)."""

from typing import Any, ClassVar

from pydantic import Field

from tallyjudge.core.errors import ModelOutputError, UndefinedScore
from tallyjudge.core.prompts import StructuredPrompt
from tallyjudge.core.results import MetricResult
from tallyjudge.interfaces.models import LanguageModel
from tallyjudge.plugins.metrics.rag._judge import RagJudge
from tallyjudge.registry.specs import EmptyConfig


class DualJudgeConfig(EmptyConfig):
    """Number of rating attempts per judge, including its first attempt."""

    max_retries: int = Field(default=5, ge=0)


class DualJudge(RagJudge):
    """Normalize the available ratings after the archived per-judge attempt budget."""

    grades: ClassVar[tuple[int, ...]]

    def __init__(self, llm: LanguageModel, max_retries: int = 5):
        super().__init__(llm)
        self.max_retries = DualJudgeConfig(max_retries=max_retries).max_retries

    def config_snapshot(self) -> dict[str, Any]:
        """Include the attempt budget because a later valid rating can affect the score."""
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
                try:
                    rating = (await self._ask(request, template.output_model)).rating
                except ModelOutputError:
                    continue
                if rating in self.grades:
                    ratings.append(rating)
                    break
        if not ratings:
            raise UndefinedScore("no_valid_ratings")
        return MetricResult(sum(ratings) / (len(ratings) * max(self.grades)))

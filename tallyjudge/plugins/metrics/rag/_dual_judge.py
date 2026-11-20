"""Paired normalized ratings migrated from Ragas ec00339 (Apache-2.0)."""

from typing import Any, ClassVar

from tallyjudge.core.errors import ModelOutputError
from tallyjudge.core.prompts import StructuredPrompt
from tallyjudge.core.results import MetricResult
from tallyjudge.plugins.metrics.rag._judge import RagJudge


class DualJudge(RagJudge):
    """Average two valid judge ratings on the same scale in their original order."""

    grades: ClassVar[tuple[int, ...]]

    async def _rate_pair(
        self,
        prompts: tuple[StructuredPrompt[Any, Any], StructuredPrompt[Any, Any]],
        first: dict[str, str],
        second: dict[str, str],
    ) -> MetricResult:
        total = 0
        for template, fields in ((prompts[0], first), (prompts[1], second)):
            request = template.render(template.input_model(**fields))
            rating = (await self._ask(request, template.output_model)).rating
            if rating not in self.grades:
                raise ModelOutputError("invalid_rating")
            total += rating
        return MetricResult(total / (2 * max(self.grades)))

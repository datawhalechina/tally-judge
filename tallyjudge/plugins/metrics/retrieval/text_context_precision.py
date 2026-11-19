"""Ragas _context_precision.py (ec00339, Apache-2.0), with real candidates required for a hit."""

from tallyjudge.core.results import MetricResult
from tallyjudge.interfaces.metrics import MetricSpec
from tallyjudge.plugins.metrics.retrieval._text_context import (
    CONTEXT_INPUTS,
    TextContextMetric,
)


class TextContextPrecisionMetric(TextContextMetric):
    """Average the prefix precision at each matching retrieval rank, retaining duplicates."""

    spec = MetricSpec(
        "metric.retrieval.text_context_precision",
        "1",
        CONTEXT_INPUTS,
        value_range=(0.0, 1.0),
    )

    async def _ascore_validated(
        self, retrieved_contexts: list[str], reference_contexts: list[str]
    ) -> MetricResult:
        matched = await self.distance.matches(
            retrieved_contexts, reference_contexts, self.threshold
        )
        hits = 0
        total = 0.0
        for rank, hit in enumerate(matched, 1):
            if hit:
                hits += 1
                total += hits / rank
        return MetricResult(total / hits if hits else 0.0)

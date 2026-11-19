"""Ragas _context_recall.py (ec00339, Apache-2.0), with explicit empty-reference semantics."""

from tallyjudge.core.errors import UndefinedScore
from tallyjudge.core.results import MetricResult
from tallyjudge.interfaces.metrics import MetricSpec
from tallyjudge.plugins.metrics.retrieval._text_context import (
    CONTEXT_INPUTS,
    TextContextMetric,
)


class TextContextRecallMetric(TextContextMetric):
    """Fraction of reference passages matched by at least one retrieved passage."""

    spec = MetricSpec(
        "metric.retrieval.text_context_recall",
        "1",
        CONTEXT_INPUTS,
        value_range=(0.0, 1.0),
    )

    async def _ascore_validated(
        self, retrieved_contexts: list[str], reference_contexts: list[str]
    ) -> MetricResult:
        if not reference_contexts:
            raise UndefinedScore("no_reference_contexts")
        matched = await self.distance.matches(
            reference_contexts, retrieved_contexts, self.threshold
        )
        return MetricResult(sum(matched) / len(matched))

"""Ragas _context_recall.py (ec00339, Apache-2.0), with explicit empty-reference semantics."""

from tallyjudge.plugins.metrics.retrieval._text_context import (
    CONTEXT_INPUTS,
    TextContextConfig,
    TextContextMetric,
)
from tallyjudge.sdk.errors import UndefinedScore
from tallyjudge.sdk.metrics import MetricSpec
from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.results import MetricResult


class TextContextRecallMetric(TextContextMetric):
    plugin = Plugin("metric", requires=("rapidfuzz",), extra="text-distance")

    Config = TextContextConfig

    spec = MetricSpec(
        "metric.retrieval.text_context_recall",
        "1",
        CONTEXT_INPUTS,
        value_range=(0.0, 1.0),
    )

    async def ascore(
        self, retrieved_contexts: list[str], reference_contexts: list[str]
    ) -> MetricResult:
        if not reference_contexts:
            raise UndefinedScore("no_reference_contexts")
        matched = await self.distance.matches(
            reference_contexts, retrieved_contexts, self.threshold
        )
        return MetricResult(sum(matched) / len(matched))

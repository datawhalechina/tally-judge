"""Migrate IDBasedContextPrecision from Ragas _context_precision.py (ec00339, Apache-2.0).

IdContextPrecisionMetric reports an empty retrieval set as undefined instead of NaN.
"""

from tallyjudge.sdk.errors import UndefinedScore
from tallyjudge.sdk.metrics import MetricBase, MetricSpec
from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.results import MetricResult


class IdContextPrecisionMetric(MetricBase):
    plugin = Plugin("metric")

    spec = MetricSpec(
        "metric.retrieval.id_context_precision",
        "1",
        {
            "retrieved_context_ids": list[str | int],
            "reference_context_ids": list[str | int],
        },
        value_range=(0.0, 1.0),
    )

    async def ascore(
        self,
        retrieved_context_ids: list[str | int],
        reference_context_ids: list[str | int],
    ) -> MetricResult:
        retrieved = set(map(str, retrieved_context_ids))
        if not retrieved:
            raise UndefinedScore("no_retrieved_ids")
        hits = retrieved.intersection(map(str, reference_context_ids))
        return MetricResult(len(hits) / len(retrieved))

"""Migrate IDBasedContextRecall from Ragas _context_recall.py (ec00339, Apache-2.0).

IdContextRecallMetric reports an empty reference set as undefined instead of NaN.
"""

from tallyjudge.sdk.errors import UndefinedScore
from tallyjudge.sdk.metrics import MetricBase, MetricSpec
from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.results import MetricResult


class IdContextRecallMetric(MetricBase):
    plugin = Plugin("metric")

    spec = MetricSpec(
        "metric.retrieval.id_context_recall",
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
        reference = set(map(str, reference_context_ids))
        if not reference:
            raise UndefinedScore("no_reference_ids")
        hits = reference.intersection(map(str, retrieved_context_ids))
        return MetricResult(len(hits) / len(reference))

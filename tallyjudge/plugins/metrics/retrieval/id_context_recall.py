"""Migrate IDBasedContextRecall from Ragas _context_recall.py (ec00339, Apache-2.0).

IdContextRecallMetric reports an empty reference set as undefined instead of NaN.
"""

from tallyjudge.core.errors import UndefinedScore
from tallyjudge.core.results import MetricResult
from tallyjudge.interfaces.metrics import MetricBase, MetricSpec


class IdContextRecallMetric(MetricBase):
    """Fraction of distinct reference IDs found among the retrieved documents."""

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
        """Score the two document ID lists, matching integers by their string form."""
        return await self.ascore_inputs(
            {
                "retrieved_context_ids": retrieved_context_ids,
                "reference_context_ids": reference_context_ids,
            }
        )

    async def _ascore_validated(
        self,
        retrieved_context_ids: list[str | int],
        reference_context_ids: list[str | int],
    ) -> MetricResult:
        reference = set(map(str, reference_context_ids))
        if not reference:
            raise UndefinedScore("no_reference_ids")
        hits = reference.intersection(map(str, retrieved_context_ids))
        return MetricResult(len(hits) / len(reference))

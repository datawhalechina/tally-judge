"""Migrate IDBasedContextPrecision from Ragas _context_precision.py (ec00339, Apache-2.0).

IdContextPrecisionMetric reports an empty retrieval set as undefined instead of NaN.
"""

from tallyjudge.core.errors import UndefinedScore
from tallyjudge.core.results import MetricResult
from tallyjudge.interfaces.metrics import MetricBase, MetricSpec


class IdContextPrecisionMetric(MetricBase):
    """Fraction of distinct retrieved IDs found among the references."""

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
        retrieved = set(map(str, retrieved_context_ids))
        if not retrieved:
            raise UndefinedScore("no_retrieved_ids")
        hits = retrieved.intersection(map(str, reference_context_ids))
        return MetricResult(len(hits) / len(retrieved))

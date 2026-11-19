"""String similarity migrated from Ragas collections/_string.py (ec00339, Apache-2.0)."""

from tallyjudge.core.results import MetricResult
from tallyjudge.interfaces.metrics import MetricBase, MetricSpec
from tallyjudge.plugins.metrics._text_distance import DistanceMeasure, TextDistance

__all__ = ["DistanceMeasure", "StringSimilarityMetric"]


class StringSimilarityMetric(MetricBase):
    """Score two strings with one minus their normalized distance."""

    spec = MetricSpec(
        "metric.text.string_similarity",
        "1",
        {"reference": str, "response": str},
        value_range=(0.0, 1.0),
    )

    def __init__(self, distance_measure: str = DistanceMeasure.LEVENSHTEIN):
        self.distance = TextDistance(distance_measure)

    def config_snapshot(self) -> dict[str, str]:
        return {"distance_measure": self.distance.measure.value}

    async def ascore(self, reference: str, response: str) -> MetricResult:
        """Compare both strings as supplied, including case and whitespace."""
        return await self.ascore_inputs({"reference": reference, "response": response})

    async def _ascore_validated(self, reference: str, response: str) -> MetricResult:
        return MetricResult(await self.distance.asimilarity(reference, response))

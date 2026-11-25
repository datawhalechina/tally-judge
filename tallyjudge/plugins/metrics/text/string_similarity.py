"""String similarity migrated from Ragas collections/_string.py (ec00339, Apache-2.0)."""

from tallyjudge.plugins.metrics._text_distance import (
    DistanceConfig,
    DistanceMeasure,
    TextDistance,
)
from tallyjudge.sdk.metrics import MetricBase, MetricSpec
from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.results import MetricResult

__all__ = ["DistanceMeasure", "StringSimilarityMetric"]


class StringSimilarityMetric(MetricBase):
    plugin = Plugin("metric", requires=("rapidfuzz",), extra="text-distance")

    Config = DistanceConfig

    spec = MetricSpec(
        "metric.text.string_similarity",
        "1",
        {"reference": str, "response": str},
        value_range=(0.0, 1.0),
    )

    def __init__(self, distance_measure: str = DistanceMeasure.LEVENSHTEIN):
        config = DistanceConfig.model_validate({"distance_measure": distance_measure})
        self.distance = TextDistance(config.distance_measure)

    def config_snapshot(self) -> dict[str, str]:
        return {"distance_measure": self.distance.measure.value}

    async def ascore(self, reference: str, response: str) -> MetricResult:
        return MetricResult(await self.distance.asimilarity(reference, response))

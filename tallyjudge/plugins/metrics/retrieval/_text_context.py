"""Text-distance settings and row-level execution shared by retrieval metrics."""

from abc import abstractmethod

from pydantic import Field

from tallyjudge.core.results import MetricResult
from tallyjudge.interfaces.metrics import MetricBase
from tallyjudge.plugins.metrics._text_distance import (
    DistanceConfig,
    DistanceMeasure,
    TextDistance,
)

CONTEXT_INPUTS = {"retrieved_contexts": list[str], "reference_contexts": list[str]}


class TextContextConfig(DistanceConfig):
    threshold: float = Field(default=0.5, ge=0, le=1)


class TextContextMetric(MetricBase):
    """Share retrieval inputs and distance settings across precision and recall."""

    def __init__(
        self,
        distance_measure: str = DistanceMeasure.LEVENSHTEIN,
        threshold: float = 0.5,
    ):
        config = TextContextConfig.model_validate(
            {"distance_measure": distance_measure, "threshold": threshold}
        )
        self.distance = TextDistance(config.distance_measure)
        self.threshold = config.threshold

    def config_snapshot(self) -> dict[str, str | float]:
        return {
            "distance_measure": self.distance.measure.value,
            "threshold": self.threshold,
        }

    async def ascore(
        self, retrieved_contexts: list[str], reference_contexts: list[str]
    ) -> MetricResult:
        """Score the ranked retrieval and reference passage lists."""
        return await self.ascore_inputs(
            {
                "retrieved_contexts": retrieved_contexts,
                "reference_contexts": reference_contexts,
            }
        )

    @abstractmethod
    async def _ascore_validated(
        self, retrieved_contexts: list[str], reference_contexts: list[str]
    ) -> MetricResult: ...

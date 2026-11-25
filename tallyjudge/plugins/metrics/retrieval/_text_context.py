from pydantic import Field

from tallyjudge.plugins.metrics._text_distance import (
    DistanceConfig,
    DistanceMeasure,
    TextDistance,
)
from tallyjudge.sdk.metrics import MetricBase

CONTEXT_INPUTS = {"retrieved_contexts": list[str], "reference_contexts": list[str]}


class TextContextConfig(DistanceConfig):
    threshold: float = Field(default=0.5, ge=0, le=1)


class TextContextMetric(MetricBase):
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

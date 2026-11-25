from pydantic import BaseModel, ConfigDict, Field

from tallyjudge.sdk import MetricBase, MetricResult, MetricSpec, Plugin


class LengthRatioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    cap: float | None = Field(default=None, gt=0)


class LengthRatioMetric(MetricBase):
    plugin = Plugin("metric")
    Config = LengthRatioConfig
    spec = MetricSpec(
        "metric.length_ratio.response_to_reference",
        "1",
        {"response": str, "reference": str},
        value_range=(0.0, None),
    )

    def __init__(self, cap: float | None = None):
        self.cap = LengthRatioConfig(cap=cap).cap

    def config_snapshot(self):
        return {"cap": self.cap}

    async def ascore(self, response: str, reference: str) -> MetricResult:
        ratio = len(response) / max(len(reference), 1)
        return MetricResult(ratio if self.cap is None else min(ratio, self.cap))

"""Typed function metrics. Decoration creates a factory; calling it creates a metric."""

import inspect
from collections.abc import Callable
from typing import Any

from tallyjudge.core.results import MetricResult
from tallyjudge.interfaces.metrics import MetricBase, MetricSpec


class FunctionMetric(MetricBase):
    def __init__(self, spec: MetricSpec, function: Callable[..., Any]) -> None:
        self.spec = spec
        self._function = function

    async def ascore(self, **inputs: Any) -> MetricResult:
        value = self._function(**inputs)
        if inspect.isawaitable(value):
            value = await value
        return value if isinstance(value, MetricResult) else MetricResult(value)

    def config_snapshot(self) -> dict[str, Any]:
        if self.spec.output_type is str:
            return {"labels": list(self.spec.labels)}
        if self.spec.output_type is list:
            return {"ranking_length": self.spec.ranking_length}
        return {"value_range": list(self.spec.value_range)}

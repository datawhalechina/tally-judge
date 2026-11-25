import inspect
from collections.abc import Callable
from typing import Any

from tallyjudge.sdk.metrics import MetricBase, MetricSpec
from tallyjudge.sdk.results import MetricResult


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
        function = getattr(
            self._function, "__qualname__", type(self._function).__name__
        )
        snapshot: dict[str, Any] = {
            "function": f"{getattr(self._function, '__module__', None)}.{function}"
        }
        if self.spec.output_type is str:
            snapshot["labels"] = list(self.spec.labels)
        elif self.spec.output_type is list:
            snapshot["ranking_length"] = self.spec.ranking_length
        else:
            snapshot["value_range"] = list(self.spec.value_range)
        return snapshot

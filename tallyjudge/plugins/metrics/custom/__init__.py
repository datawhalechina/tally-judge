"""Typed function metrics. Decoration creates a factory; calling it creates a metric."""

import inspect
from collections.abc import Callable
from typing import Any

from tallyjudge.core.results import MetricResult
from tallyjudge.interfaces.metrics import MetricBase, MetricSpec


class FunctionMetric(MetricBase):
    """Score with a caller's typed function under a declared plugin ID and revision."""

    def __init__(self, spec: MetricSpec, function: Callable[..., Any]) -> None:
        self.spec = spec
        self._function = function

    async def ascore(self, **inputs: Any) -> MetricResult:
        value = self._function(**inputs)
        if inspect.isawaitable(value):
            value = await value
        return value if isinstance(value, MetricResult) else MetricResult(value)

    def config_snapshot(self) -> dict[str, Any]:
        """Describe the output contract and which function computes the score.

        The function path is part of cache and comparison identity, so swapping the
        decorated function under one ID is visible. Editing a function body in place
        still requires a new revision or cache namespace.
        """
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

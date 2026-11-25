import inspect
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, get_type_hints

from tallyjudge.sdk.metrics import MetricSpec

if TYPE_CHECKING:
    from tallyjudge.plugins.metrics.custom import FunctionMetric


class FunctionMetricFactory:
    def __init__(self, function: Callable[..., Any], spec: MetricSpec) -> None:
        self.function = function
        self.spec = spec

    def __call__(self) -> "FunctionMetric":
        from tallyjudge.plugins.metrics.custom import FunctionMetric

        return FunctionMetric(self.spec, self.function)


def _decorator(
    plugin_id: str, revision: str, **output: Any
) -> Callable[[Callable[..., Any]], FunctionMetricFactory]:
    def decorate(function: Callable[..., Any]) -> FunctionMetricFactory:
        hints = get_type_hints(function, include_extras=True)
        inputs = {}
        defaults = {}
        for name, param in inspect.signature(function).parameters.items():
            if name not in hints or param.kind not in (
                param.POSITIONAL_OR_KEYWORD,
                param.KEYWORD_ONLY,
            ):
                raise TypeError(
                    f"Metric parameter {name!r} needs a type and must accept a keyword"
                )
            inputs[name] = hints[name]
            if param.default is not param.empty:
                defaults[name] = param.default
        return FunctionMetricFactory(
            function,
            MetricSpec(plugin_id, revision, inputs, defaults=defaults, **output),
        )

    return decorate


def numeric_metric(
    plugin_id: str,
    *,
    revision: str,
    value_range: tuple[float | None, float | None] = (None, None),
) -> Callable[[Callable[..., Any]], FunctionMetricFactory]:
    """Declare a numeric function metric with a finite score and optional bounds."""
    return _decorator(plugin_id, revision, value_range=value_range)


def categorical_metric(
    plugin_id: str, *, revision: str, labels: Sequence[str]
) -> Callable[[Callable[..., Any]], FunctionMetricFactory]:
    """Declare a string-valued function metric with a fixed set of labels."""
    return _decorator(plugin_id, revision, output_type=str, labels=tuple(labels))


def ranking_metric(
    plugin_id: str, *, revision: str, length: int
) -> Callable[[Callable[..., Any]], FunctionMetricFactory]:
    """Declare an ordered JSON-list metric with a fixed output length."""
    return _decorator(plugin_id, revision, output_type=list, ranking_length=length)

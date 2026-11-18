"""Metric contracts, immutable bindings and direct scoring helpers."""

import math
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Protocol

from pydantic import TypeAdapter, ValidationError

from tallyjudge.core.errors import InputValidationError
from tallyjudge.core.results import MetricResult


@dataclass(frozen=True)
class MetricSpec:
    id: str
    revision: str
    inputs: Mapping[str, Any]
    output_type: type = float
    value_range: tuple[float | None, float | None] = (None, None)
    direction: Literal["higher_is_better", "lower_is_better"] = "higher_is_better"

    _adapters: Mapping[str, TypeAdapter[Any]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.id, str)
            or not self.id
            or not isinstance(self.revision, str)
            or not self.revision
        ):
            raise ValueError("Metric id and revision must be non-empty")
        if any(not isinstance(key, str) or not key for key in self.inputs):
            raise ValueError("Metric inputs must map parameter names to types")
        if self.output_type is not float:
            raise ValueError("Phase 1 supports float metric outputs only")
        if self.direction not in ("higher_is_better", "lower_is_better"):
            raise ValueError("Unknown metric direction")
        low, high = self.value_range
        if any(
            value is not None
            and (type(value) not in (int, float) or not math.isfinite(value))
            for value in (low, high)
        ):
            raise ValueError("Metric range bounds must be finite numbers or None")
        if low is not None and high is not None and low > high:
            raise ValueError("Metric value range is reversed")
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))
        adapters = {
            name: TypeAdapter(annotation) for name, annotation in self.inputs.items()
        }
        object.__setattr__(self, "_adapters", MappingProxyType(adapters))


class Metric(Protocol):
    """A mapping-based metric usable by the evaluator, independent of its implementation."""

    @property
    def spec(self) -> MetricSpec: ...

    async def ascore_inputs(self, inputs: Mapping[str, Any]) -> MetricResult:
        """Score the declared parameters supplied by the binding."""
        ...

    def config_snapshot(self) -> dict[str, Any]: ...


def validate_inputs(spec: MetricSpec, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Return the strictly validated parameter values that the scorer will receive."""
    values = {}
    for name, annotation in spec.inputs.items():
        expected = (
            annotation.__name__ if isinstance(annotation, type) else str(annotation)
        )
        if name not in inputs:
            raise InputValidationError(
                "Missing metric input field", field=name, expected=expected
            )
        try:
            values[name] = spec._adapters[name].validate_python(
                inputs[name], strict=True
            )
        except ValidationError as exc:
            raise InputValidationError(
                "Metric input type does not match its specification",
                field=name,
                expected=expected,
            ) from exc
    if set(inputs) - set(spec.inputs):
        raise InputValidationError("Unexpected metric input fields")
    return values


class MetricBase(ABC):
    """Implement ascore with the named parameters declared in spec."""

    spec: MetricSpec

    @abstractmethod
    async def ascore(self, *args: Any, **kwargs: Any) -> MetricResult:
        """Compute one score from typed arguments."""

    async def ascore_inputs(self, inputs: Mapping[str, Any]) -> MetricResult:
        """Validate a mapping against spec and pass the resulting values to ascore."""
        return await self.ascore(**validate_inputs(self.spec, inputs))

    def config_snapshot(self) -> dict[str, Any]:
        return {}

    def score(self, **inputs: Any) -> MetricResult:
        """Score checked inputs synchronously. In an active loop, await ascore_inputs(inputs)."""
        from tallyjudge.core.execution import run_sync

        return run_sync(lambda: self.ascore_inputs(inputs), "ascore_inputs")


type _ScoreInputs = Callable[[Mapping[str, Any]], Awaitable[MetricResult]]


def _checked_scorer(metric: Metric) -> _ScoreInputs:
    # The named-argument helper already owns validation at its mapping entry point.
    if (
        isinstance(metric, MetricBase)
        and type(metric).ascore_inputs is MetricBase.ascore_inputs
    ):
        return metric.ascore_inputs

    async def score(inputs: Mapping[str, Any]) -> MetricResult:
        return await metric.ascore_inputs(validate_inputs(metric.spec, inputs))

    return score


@dataclass(frozen=True, init=False)
class MetricBinding:
    """Bind metric parameter names to dataset columns under a unique result key."""

    metric: Metric
    key: str
    input_map: Mapping[str, str]
    _score_inputs: _ScoreInputs = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        metric: Metric,
        key: str | None = None,
        input_map: Mapping[str, str] | None = None,
    ) -> None:
        key = metric.spec.id if key is None else key
        if not isinstance(key, str) or not key:
            raise ValueError("Binding key must be a non-empty string")
        mapping = {} if input_map is None else dict(input_map)
        if set(mapping) - set(metric.spec.inputs):
            raise ValueError("input_map contains an unknown metric parameter")
        if any(not isinstance(value, str) or not value for value in mapping.values()):
            raise ValueError("input_map fields must be non-empty strings")
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "key", key)
        object.__setattr__(
            self,
            "input_map",
            MappingProxyType(
                {name: mapping.get(name, name) for name in metric.spec.inputs}
            ),
        )
        object.__setattr__(self, "_score_inputs", _checked_scorer(metric))

    async def ascore_row(self, row: Mapping[str, Any]) -> MetricResult:
        """Select this binding's columns and pass checked parameters to the metric."""
        inputs = {
            name: row[column]
            for name, column in self.input_map.items()
            if column in row
        }
        return await self._score_inputs(inputs)

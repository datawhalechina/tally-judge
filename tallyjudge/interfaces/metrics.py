"""Metric contracts, immutable bindings and direct scoring helpers."""

import json
import math
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Annotated, Any, Literal, Protocol, get_origin

from pydantic import TypeAdapter, ValidationError
from pydantic.fields import FieldInfo

from tallyjudge.core.dataset import canonical_json
from tallyjudge.core.errors import InputValidationError
from tallyjudge.core.results import MetricResult


@dataclass(frozen=True)
class MetricSpec:
    """Declare a metric's identity, typed parameters and output contract.

    ``defaults`` holds values for optional parameters. Binding validation applies
    them when a row lacks the column, so batch scoring, direct ``ascore_inputs``
    calls and score cache keys all see the same effective arguments.
    Numeric ``levels`` restrict successful scores to the declared integer grades.
    """

    id: str
    revision: str
    inputs: Mapping[str, Any]
    output_type: type = float
    value_range: tuple[float | None, float | None] = (None, None)
    direction: Literal["higher_is_better", "lower_is_better"] = "higher_is_better"

    labels: tuple[str, ...] = ()
    ranking_length: int | None = None
    defaults: Mapping[str, Any] = field(default_factory=dict)
    levels: tuple[int, ...] = ()

    @property
    def output_kind(self) -> Literal["numeric", "categorical", "ranking"]:
        if self.output_type is str:
            return "categorical"
        return "ranking" if self.output_type is list else "numeric"

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
        if self.output_type not in (float, str, list):
            raise ValueError("Metric output_type must be float, str or list")
        if self.output_type is str:
            if (
                not self.labels
                or not all(isinstance(label, str) for label in self.labels)
                or len(set(self.labels)) != len(self.labels)
            ):
                raise ValueError("Categorical metrics need distinct string labels")
            object.__setattr__(self, "labels", tuple(self.labels))
        if self.output_type is list and (
            type(self.ranking_length) is not int or self.ranking_length < 1
        ):
            raise ValueError("Ranking metrics need a positive ranking_length")
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
        if self.levels:
            if (
                self.output_type is not float
                or any(
                    type(n) is not int
                    or (low is not None and n < low)
                    or (high is not None and n > high)
                    for n in self.levels
                )
                or len(set(self.levels)) != len(self.levels)
            ):
                raise ValueError(
                    "Numeric levels must be distinct integers within the declared range"
                )
            object.__setattr__(self, "levels", tuple(self.levels))
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))
        if set(self.defaults) - set(self.inputs):
            raise ValueError("Metric defaults must name declared inputs")
        object.__setattr__(self, "defaults", MappingProxyType(dict(self.defaults)))
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


def _expected(annotation: Any) -> str:
    prefix = ""
    if get_origin(annotation) is Annotated:
        field = FieldInfo.from_annotation(annotation)
        annotation = field.annotation
        if any(getattr(item, "min_length", None) == 1 for item in field.metadata):
            prefix = "non-empty "
    return prefix + (
        annotation.__name__ if isinstance(annotation, type) else str(annotation)
    )


def validate_inputs(spec: MetricSpec, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Return the strictly validated parameter values that the scorer will receive.

    A missing parameter takes its declared default; defaults are validated too.
    """
    values = {}
    for name, annotation in spec.inputs.items():
        if name in inputs:
            value = inputs[name]
        elif name in spec.defaults:
            value = spec.defaults[name]
        else:
            raise InputValidationError(
                "Missing metric input field", field=name, expected=_expected(annotation)
            )
        try:
            values[name] = spec._adapters[name].validate_python(value, strict=True)
        except ValidationError as exc:
            raise InputValidationError(
                "Metric input type does not match its specification",
                field=name,
                expected=_expected(annotation),
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
        """Validate a mapping once before scoring its named parameters."""
        return await self._ascore_validated(**validate_inputs(self.spec, inputs))

    async def _ascore_validated(self, *args: Any, **kwargs: Any) -> MetricResult:
        """Default to the typed scorer for plugins that implement ascore directly."""
        return await self.ascore(*args, **kwargs)

    def config_snapshot(self) -> dict[str, Any]:
        return {}

    def score(self, **inputs: Any) -> MetricResult:
        """Score checked inputs synchronously. In an active loop, await ascore_inputs(inputs)."""
        from tallyjudge.core.execution import run_sync

        return run_sync(lambda: self.ascore_inputs(inputs), "ascore_inputs")


type _ScoreInputs = Callable[[Mapping[str, Any]], Awaitable[MetricResult]]


def _validated_scorer(metric: Metric) -> _ScoreInputs:
    if (
        isinstance(metric, MetricBase)
        and type(metric).ascore_inputs is MetricBase.ascore_inputs
    ):

        async def score(inputs: Mapping[str, Any]) -> MetricResult:
            return await metric._ascore_validated(**inputs)

        return score
    return metric.ascore_inputs


@dataclass(frozen=True, init=False)
class MetricBinding:
    """Bind metric parameter names to dataset columns under a unique result key."""

    metric: Metric
    key: str
    input_map: Mapping[str, str]
    ascore_validated: _ScoreInputs = field(init=False, repr=False, compare=False)

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
        object.__setattr__(self, "ascore_validated", _validated_scorer(metric))

    def snapshot(self) -> dict[str, Any]:
        """Describe this binding's score and settings independently of its live plugin."""
        spec = self.metric.spec
        config = self.metric.config_snapshot()
        if not isinstance(config, dict):
            raise ValueError("Metric config_snapshot must return a JSON object")
        metadata = {
            "id": spec.id,
            "revision": spec.revision,
            "output_kind": spec.output_kind,
            "input_map": dict(self.input_map),
            "config_snapshot": json.loads(canonical_json(config)),
        }
        if spec.output_type is float:
            metadata["value_range"] = list(spec.value_range)
            if spec.direction == "lower_is_better":
                metadata["direction"] = spec.direction
            if spec.levels:
                metadata["levels"] = list(spec.levels)
        elif spec.output_type is str:
            metadata["labels"] = list(spec.labels)
        elif spec.output_type is list:
            metadata["ranking_length"] = spec.ranking_length
        return metadata

    def inputs_for(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Select and strictly validate the parameters used for scoring and caching."""
        inputs = {
            name: row[column]
            for name, column in self.input_map.items()
            if column in row
        }
        return validate_inputs(self.metric.spec, inputs)

    async def ascore_row(self, row: Mapping[str, Any]) -> MetricResult:
        """Score the bound columns of one input row."""
        return await self.ascore_validated(self.inputs_for(row))

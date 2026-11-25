import json
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Annotated, Any, Literal, Protocol, get_origin

from pydantic import TypeAdapter, ValidationError
from pydantic.fields import FieldInfo

from tallyjudge.sdk.dataset import canonical_json
from tallyjudge.sdk.errors import InputValidationError
from tallyjudge.sdk.results import (
    ErrorInfo,
    MetricResult,
    ScoreResult,
    numeric_score_is_valid,
)
from tallyjudge.sdk.sync import run_sync

type Control = Callable[[str, str], bool]


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
        ...

    async def ascore_inputs(self, inputs: Mapping[str, Any]) -> MetricResult:
        """Validate a mapping once before scoring its named parameters."""
        return await self.ascore(**validate_inputs(self.spec, inputs))

    def config_snapshot(self) -> dict[str, Any]:
        return {}

    def score(self, **inputs: Any) -> MetricResult:
        """Score checked inputs synchronously. In an active loop, await ascore_inputs(inputs)."""
        return run_sync(lambda: self.ascore_inputs(inputs), "ascore_inputs")


@dataclass(frozen=True, init=False)
class MetricBinding:
    """Bind metric parameter names to dataset columns under a unique result key."""

    metric: Metric
    key: str
    input_map: Mapping[str, str]

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

    def inputs_for(
        self,
        row: Mapping[str, Any],
        *,
        output: Mapping[str, Any] | None = None,
        controls: Control | None = None,
        shared_row: bool = False,
    ) -> dict[str, Any]:
        """Resolve bound inputs. An experiment output overrides a dataset column unless ``controls`` binds it to the dataset."""
        inputs = {}
        for name, column in self.input_map.items():
            if (
                output is not None
                and column in output
                and not (controls is not None and controls(name, column))
            ):
                inputs[name] = output[column]
            elif column in row:
                inputs[name] = row[column]
        return validate_inputs(
            self.metric.spec, deepcopy(inputs) if shared_row else inputs
        )

    async def ascore_row(self, row: Mapping[str, Any]) -> MetricResult:
        """Score the bound columns of one input row."""
        return await self.ascore_validated(self.inputs_for(row))

    async def ascore_validated(self, inputs: Mapping[str, Any]) -> MetricResult:
        if isinstance(self.metric, MetricBase):
            return await self.metric.ascore(**inputs)
        return await self.metric.ascore_inputs(inputs)


def validate_output(output: Any, spec: MetricSpec) -> ScoreResult:
    invalid = ScoreResult("invalid_output", error=ErrorInfo("invalid_output"))
    if not isinstance(output, MetricResult) or (
        output.reason is not None and not isinstance(output.reason, str)
    ):
        return invalid
    value = output.value
    if spec.output_type is float:
        if not numeric_score_is_valid(value, spec.value_range, spec.levels):
            return invalid
        value = float(value)
    elif spec.output_type is str:
        if not isinstance(value, str) or value not in spec.labels:
            return invalid
    else:
        if not isinstance(value, list) or len(value) != spec.ranking_length:
            return invalid
        try:
            value = json.loads(canonical_json(value))
        except ValueError:
            return invalid
    details = output.details
    if details is not None:
        if not isinstance(details, dict):
            return invalid
        try:
            details = json.loads(canonical_json(details))
        except ValueError:
            return invalid
    return ScoreResult("success", value, output.reason, details=details)

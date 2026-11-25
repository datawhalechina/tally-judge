import copy
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from statistics import mean
from typing import TYPE_CHECKING, Any, Literal, TypeGuard, cast, get_args

from pydantic import TypeAdapter

from tallyjudge.sdk._run_migrations import SCHEMA_VERSION, upgrade
from tallyjudge.sdk.dataset import canonical_sha256
from tallyjudge.sdk.usage import ScoreUsage

if TYPE_CHECKING:
    from tallyjudge.sdk.storage import StorageBackend

type ScoreValue = float | str | list[Any]

Status = Literal[
    "not_started",
    "running",
    "success",
    "error",
    "invalid_output",
    "undefined",
    "timeout",
    "cancelled",
]
RunStatus = Literal["running", "completed", "aborted", "cancelled", "stopped"]


@dataclass(frozen=True)
class MetricResult:
    value: ScoreValue
    reason: str | None = None
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class ErrorInfo:
    category: str
    exception_type: str | None = None
    field: str | None = None
    expected: str | None = None
    code: str | None = None


@dataclass(frozen=True)
class ScoreResult:
    status: Status
    value: ScoreValue | None = None
    reason: str | None = None
    error: ErrorInfo | None = None
    usage: ScoreUsage | None = None
    cache_hit: bool = False
    inputs_sha256: str | None = None
    input_sha256_by_name: dict[str, str | None] | None = None
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class RowTaskResult:
    status: Literal[
        "not_started", "running", "success", "skipped", "error", "timeout", "cancelled"
    ]
    output: dict[str, Any] | None = None
    error: ErrorInfo | None = None
    usage: ScoreUsage | None = None


@dataclass
class RowResult:
    case_id: str
    input_index: int
    metrics: dict[str, ScoreResult]
    task: RowTaskResult | None = None


def aggregate(
    rows: Iterable[RowResult], keys: Iterable[str]
) -> dict[str, dict[str, Any]]:
    rows = tuple(rows)
    summaries = {}
    for key in keys:
        counts = dict.fromkeys(get_args(Status), 0)
        values = []
        usages = []
        for row in rows:
            score = row.metrics[key]
            counts[score.status] += 1
            if score.usage is not None:
                usages.append(score.usage)
            if score.status == "success" and score.value is not None:
                values.append(score.value)
        meta = keys[key] if isinstance(keys, Mapping) else {}
        kind = meta.get("output_kind", "numeric")
        average = (
            mean(cast(list[float], values)) if kind == "numeric" and values else None
        )
        usage = ScoreUsage.aggregate(usages)
        summaries[key] = {
            "usage": asdict(usage) if usage is not None else None,
            "total": len(rows),
            "status_counts": counts,
            "valid_count": len(values),
            "coverage": len(values) / len(rows) if rows else None,
            "mean": average,
        }
        if kind == "categorical":
            counts_by_label = dict.fromkeys(meta["labels"], 0)
            counts_by_label.update(Counter(cast(list[str], values)))
            summaries[key]["label_counts"] = counts_by_label
    return summaries


@dataclass(frozen=True)
class CleanupWarning:
    case_id: str
    input_index: int
    metric_key: str | None
    category: str = "cleanup_timeout"


@dataclass
class RunResult:
    run_id: str
    manifest: list[dict[str, Any]]
    metrics: dict[str, dict[str, Any]]
    execution_config: dict[str, Any]
    rows: list[RowResult]
    inputs: list[dict[str, Any]] | None = None
    run_status: RunStatus = "running"
    stop_reason: str | None = None
    schema_version: int = SCHEMA_VERSION
    cleanup_warnings: list[CleanupWarning] = field(default_factory=list)
    events_complete: bool | None = None
    metadata: dict[str, Any] | None = None

    @property
    def summary(self) -> dict[str, dict[str, Any]]:
        return aggregate(self.rows, self.metrics)

    @property
    def usage(self) -> ScoreUsage | None:
        usages = []
        for row in self.rows:
            if row.task is not None and row.task.usage is not None:
                usages.append(row.task.usage)
            usages.extend(s.usage for s in row.metrics.values() if s.usage is not None)
        return ScoreUsage.aggregate(usages)

    def row_for(self, case_id: str | int) -> RowResult:
        key = str(case_id)
        for row in self.rows:
            if row.case_id == key:
                return row
        raise KeyError(key)

    def input_for(self, case_id: str | int) -> dict[str, Any]:
        row = self.row_for(case_id)
        if self.inputs is None:
            raise LookupError("Original inputs were not saved")
        return copy.deepcopy(self.inputs[row.input_index])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunResult":
        result = _RUN_SCHEMA.validate_json(
            json.dumps(upgrade(data), allow_nan=False), strict=True
        )
        output_types = {}
        for key, metadata in result.metrics.items():
            kind = metadata.get("output_kind")
            if kind not in ("numeric", "categorical", "ranking"):
                raise ValueError(f"Unknown output kind for binding {key}")
            output_types[key] = {"numeric": float, "categorical": str, "ranking": list}[
                kind
            ]
        seen = set()
        if result.inputs is not None and len(result.inputs) != len(result.rows):
            raise ValueError("Input count does not match the rows")
        for index, (entry, row) in enumerate(
            zip(result.manifest, result.rows, strict=True)
        ):
            if row.case_id in seen:
                raise ValueError(f"Duplicate case_id: {row.case_id}")
            seen.add(row.case_id)
            if (row.case_id, row.input_index) != (
                entry["case_id"],
                entry["input_index"],
            ) or row.input_index != index:
                raise ValueError(f"Manifest identity differs at row {index}")
            if (
                entry["id_source"] not in ("provided", "generated")
                or row.metrics.keys() != result.metrics.keys()
            ):
                raise ValueError(f"Invalid ID source or metric bindings at row {index}")
            for key, score in row.metrics.items():
                if score.status == "success" and not _stored_value_is_valid(
                    score.value, output_types[key], result.metrics[key]
                ):
                    raise ValueError(
                        f"Invalid stored score at row {index}, binding {key}"
                    )
            if (
                result.inputs is not None
                and canonical_sha256(result.inputs[index]) != entry["sha256"]
            ):
                raise ValueError(f"Input digest differs at row {index}")
        return result

    def save(self, *, backend: "StorageBackend", include_inputs: bool = True) -> None:
        backend.save_run(self, include_inputs=include_inputs)

    def _payload(self, *, include_inputs: bool) -> dict[str, Any]:
        data = asdict(self if include_inputs else replace(self, inputs=None))
        if data["inputs"] is None:
            del data["inputs"]
        for row in data["rows"]:
            for score in row["metrics"].values():
                for optional in ("input_sha256_by_name", "details"):
                    if score[optional] is None:
                        del score[optional]
        return data

    def to_dict(self, *, include_inputs: bool = True) -> dict[str, Any]:
        data = self._payload(include_inputs=include_inputs)
        usage = self.usage
        data["summary"] = self.summary
        data["usage"] = asdict(usage) if usage is not None else None
        return data

    def to_json(self, *, include_inputs: bool = True) -> str:
        return json.dumps(
            self.to_dict(include_inputs=include_inputs),
            ensure_ascii=False,
            allow_nan=False,
        )


_RUN_SCHEMA = TypeAdapter(RunResult)


def numeric_score_is_valid(
    value: object, bounds: Sequence[float | None], levels: Sequence[int]
) -> TypeGuard[float]:
    low, high = bounds
    return (
        isinstance(value, float)
        and math.isfinite(value)
        and (low is None or value >= low)
        and (high is None or value <= high)
        and (not levels or value in levels)
    )


def _stored_value_is_valid(
    value: Any, output_type: type, binding: dict[str, Any]
) -> bool:
    if output_type is float:
        return numeric_score_is_valid(
            value,
            binding.get("value_range") or (None, None),
            binding.get("levels", ()),
        )
    if not isinstance(value, output_type):
        return False
    if output_type is str:
        labels = binding.get("labels")
        return isinstance(labels, list) and value in labels
    length = binding.get("ranking_length")
    return type(length) is int and len(value) == length


class RunCollector:
    def __init__(self) -> None:
        self._result: RunResult | None = None
        self._active = False
        self.cleanup_deadline: float | None = None

    @property
    def run_id(self) -> str | None:
        return None if self._result is None else self._result.run_id

    def begin(self, result: RunResult) -> None:
        if self._active:
            raise ValueError("RunCollector is already collecting another run")
        self._result = result
        self._active = True
        self.cleanup_deadline = None

    def end(self) -> None:
        self._active = False

    def snapshot(self) -> RunResult:
        if self._result is None:
            raise RuntimeError("No run has been accepted by this collector")
        return copy.deepcopy(self._result)

"""Portable run results, sample identities and partial-result collection."""

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from pydantic import TypeAdapter

from tallyjudge.core.aggregation import aggregate
from tallyjudge.core.dataset import canonical_json
from tallyjudge.core.usage import ScoreUsage

if TYPE_CHECKING:
    from tallyjudge.interfaces.storage import StorageBackend

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


@dataclass(frozen=True)
class RowTaskResult:
    """One experiment function invocation, before its dependent metrics."""

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
    schema_version: int = 2
    cleanup_warnings: list[CleanupWarning] = field(default_factory=list)
    events_complete: bool | None = None

    @property
    def summary(self) -> dict[str, dict[str, Any]]:
        return aggregate(self.rows, self.metrics)

    @property
    def usage(self) -> ScoreUsage | None:
        """Sum sealed experiment and metric usage, once per invocation."""
        usages = []
        for row in self.rows:
            if row.task is not None and row.task.usage is not None:
                usages.append(row.task.usage)
            usages.extend(s.usage for s in row.metrics.values() if s.usage is not None)
        return ScoreUsage.aggregate(usages)

    def row_for(self, case_id: str | int) -> RowResult:
        """Look up a sample by its normalized dataset ID."""
        key = str(case_id)
        for row in self.rows:
            if row.case_id == key:
                return row
        raise KeyError(key)

    def input_for(self, case_id: str | int) -> dict[str, Any]:
        """Return the original sample, raising LookupError for a redacted run."""
        row = self.row_for(case_id)
        if self.inputs is None:
            raise LookupError("Original inputs were not saved")
        return copy.deepcopy(self.inputs[row.input_index])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunResult":
        """Read a v1 or v2 export without loading the scoring plugins.

        Derived summaries are recomputed. The manifest binds each result and
        retained input to the same ID and position, even with a custom ID field.
        """
        version = data.get("schema_version")
        if type(version) is not int or version not in (1, 2):
            raise ValueError(f"Unsupported run schema version: {version!r}")
        result = _RUN_SCHEMA.validate_json(
            json.dumps(data, allow_nan=False), strict=True
        )
        output_types = {}
        for key, metadata in result.metrics.items():
            kind = "numeric" if version == 1 else metadata.get("output_kind")
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
                if score.status == "success" and not isinstance(
                    score.value, output_types[key]
                ):
                    raise ValueError(
                        f"Invalid stored score at row {index}, binding {key}"
                    )
            digest = entry["sha256"]
            if result.inputs is not None:
                encoded = canonical_json(result.inputs[index]).encode("utf-8")
                if hashlib.sha256(encoded).hexdigest() != digest:
                    raise ValueError(f"Input digest differs at row {index}")
        return result

    def save(self, *, backend: "StorageBackend", include_inputs: bool = True) -> None:
        """Persist a terminal run using the caller's backend."""
        backend.save_run(self, include_inputs=include_inputs)

    def _payload(self, *, include_inputs: bool) -> dict[str, Any]:
        data = asdict(self if include_inputs else replace(self, inputs=None))
        if data["inputs"] is None:
            del data["inputs"]
        if self.schema_version == 1:
            del data["events_complete"]
            for row in data["rows"]:
                del row["task"]
                for score in row["metrics"].values():
                    del score["cache_hit"]
        return data

    def to_dict(self, *, include_inputs: bool = True) -> dict[str, Any]:
        """Export the stored fields plus summaries recomputed from the individual steps."""
        data = self._payload(include_inputs=include_inputs)
        data["summary"] = self.summary
        if self.schema_version == 2:
            usage = self.usage
            data["usage"] = asdict(usage) if usage is not None else None
        return data

    def to_json(self, *, include_inputs: bool = True) -> str:
        return json.dumps(
            self.to_dict(include_inputs=include_inputs),
            ensure_ascii=False,
            allow_nan=False,
        )


_RUN_SCHEMA = TypeAdapter(RunResult)


class RunCollector:
    """One active run at a time. Snapshots are detached from live execution."""

    def __init__(self) -> None:
        self._result: RunResult | None = None
        self._active = False

    def _start(self, result: RunResult) -> None:
        if self._active:
            raise ValueError("RunCollector is already collecting another run")
        self._result = result
        self._active = True

    def _finish(self) -> None:
        self._active = False

    def snapshot(self) -> RunResult:
        """Return a detached copy of the current or most recent run.

        Raises:
            RuntimeError: If this collector has never accepted a run.
        """
        if self._result is None:
            raise RuntimeError("No run has been accepted by this collector")
        return copy.deepcopy(self._result)

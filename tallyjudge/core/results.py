"""Schema v1 results and explicit partial-result collection."""

import copy
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from tallyjudge.core.aggregation import aggregate
from tallyjudge.core.usage import ScoreUsage

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
    value: float
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
    value: float | None = None
    reason: str | None = None
    error: ErrorInfo | None = None
    usage: ScoreUsage | None = None


@dataclass
class RowResult:
    case_id: str
    input_index: int
    metrics: dict[str, ScoreResult]


@dataclass(frozen=True)
class CleanupWarning:
    case_id: str
    input_index: int
    metric_key: str
    category: str = "cleanup_timeout"


@dataclass
class RunResult:
    run_id: str
    manifest: list[dict[str, Any]]
    metrics: dict[str, dict[str, Any]]
    execution_config: dict[str, Any]
    rows: list[RowResult]
    inputs: list[dict[str, Any]]
    run_status: RunStatus = "running"
    stop_reason: str | None = None
    schema_version: int = 1
    cleanup_warnings: list[CleanupWarning] = field(default_factory=list)

    @property
    def summary(self) -> dict[str, dict[str, Any]]:
        return aggregate(self.rows, self.metrics)

    def to_dict(self, *, include_inputs: bool = True) -> dict[str, Any]:
        data = asdict(self)
        if not include_inputs:
            del data["inputs"]
        data["summary"] = self.summary
        return data

    def to_json(self, *, include_inputs: bool = True) -> str:
        return json.dumps(
            self.to_dict(include_inputs=include_inputs),
            ensure_ascii=False,
            allow_nan=False,
        )


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

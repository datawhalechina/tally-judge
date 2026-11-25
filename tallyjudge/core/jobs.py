"""One scheduled unit of work and its accepted terminal result."""

import asyncio
import inspect
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from tallyjudge.core.dataset import (
    arguments_sha256,
    canonical_json,
    encode_json,
    json_sha256,
)
from tallyjudge.core.errors import (
    ClientLoopError,
    InputValidationError,
    ModelError,
    ModelOutputError,
    UndefinedScore,
    UsageScopeClosed,
)
from tallyjudge.core.events import RunEvents
from tallyjudge.core.results import (
    ErrorInfo,
    MetricResult,
    RowResult,
    RowTaskResult,
    ScoreResult,
    numeric_score_is_valid,
)
from tallyjudge.core.usage import UsageScope
from tallyjudge.interfaces.cache import ScoreCache
from tallyjudge.interfaces.metrics import MetricBinding, MetricSpec

# Keep row diagnostics on the evaluator's public logging channel.
logger = logging.getLogger("tallyjudge.core.execution")
type RowFunction = Callable[
    [dict[str, Any]], dict[str, Any] | None | Awaitable[dict[str, Any] | None]
]
type StepStatus = Literal["running", "error", "timeout", "cancelled"]


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
    return ScoreResult("success", value, output.reason)


@dataclass
class Job[T: (ScoreResult, RowTaskResult)](ABC):
    row: RowResult
    deadline: float | None
    events: RunEvents
    started: bool = field(default=False, init=False)
    finished_at: float | None = field(default=None, init=False)
    usage: UsageScope = field(init=False)
    key: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.usage = UsageScope(
            events=self.events, case_id=self.row.case_id, metric_key=self.key
        )

    @abstractmethod
    def state(self, status: StepStatus, error: ErrorInfo | None = None) -> T: ...

    @abstractmethod
    def store(self, outcome: T) -> None: ...

    @abstractmethod
    async def perform(self, record: dict[str, Any]) -> T: ...

    async def run(self, record: dict[str, Any]) -> T:
        clock = asyncio.get_running_loop().time
        self.started = True
        self.store(self.state("running"))
        self.events.emit(
            "step_started",
            case_id=self.row.case_id,
            metric_key=self.key,
            status="running",
        )
        try:
            return await self.perform(record)
        finally:
            self.finished_at = clock()

    def record(self, outcome: T) -> str:
        self.store(replace(outcome, usage=self.usage.seal()))
        self.events.emit(
            "step_finished",
            case_id=self.row.case_id,
            metric_key=self.key,
            status=outcome.status,
        )
        return outcome.status

    def cancel(self) -> None:
        if self.started:
            self.record(self.state("cancelled"))

    def timeout(self) -> str:
        return self.record(self.state("timeout", ErrorInfo("timeout")))

    def finish(self, task: asyncio.Task[T]) -> str:
        if task.cancelled():
            self.cancel()
            return "cancelled" if self.started else "not_started"
        try:
            outcome = task.result()
        except BaseException as exc:
            self.record(
                self.state("error", ErrorInfo("internal_error", type(exc).__name__))
            )
            raise
        if (
            self.deadline is not None
            and self.finished_at is not None
            and self.finished_at > self.deadline
        ):
            return self.timeout()
        return self.record(outcome)


@dataclass
class MetricJob(Job[ScoreResult]):
    binding: MetricBinding
    metadata: dict[str, Any]
    cache: ScoreCache | None
    cache_key: str | None = field(default=None, init=False)
    inputs_sha256: str | None = field(default=None, init=False)
    input_sha256_by_name: dict[str, str | None] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.key = self.binding.key
        super().__post_init__()

    def state(self, status: StepStatus, error: ErrorInfo | None = None) -> ScoreResult:
        return ScoreResult(status, error=error)

    def store(self, outcome: ScoreResult) -> None:
        self.row.metrics[self.binding.key] = replace(
            outcome,
            inputs_sha256=self.inputs_sha256,
            input_sha256_by_name=self.input_sha256_by_name,
        )
        if outcome.cache_hit:
            self.events.emit(
                "cache_hit",
                case_id=self.row.case_id,
                metric_key=self.key,
                status="success",
            )

    def cache_entry(self) -> tuple[ScoreCache, str, MetricResult] | None:
        """Return the write for an accepted, freshly computed success, if caching applies.

        The scheduler calls this only after it records the terminal state, so a worker
        that outlived its deadline can never populate the cache.
        """
        score = self.row.metrics[self.binding.key]
        if (
            self.cache is None
            or self.cache_key is None
            or score.status != "success"
            or score.cache_hit
            or score.value is None
        ):
            return None
        return self.cache, self.cache_key, MetricResult(score.value, score.reason)

    async def _cached_score(self, cache: ScoreCache) -> ScoreResult | None:
        if self.inputs_sha256 is None:
            return None
        # The binding key, column mapping and case ID do not change a score, so they
        # are excluded: renaming a binding or re-keying a dataset still reuses it.
        # Archive bounds validate cached values on read. Keep earlier cache keys usable.
        scoring = {
            k: v
            for k, v in self.metadata.items()
            if k not in ("input_map", "value_range")
        }
        key = json_sha256(
            [cache.namespace, scoring, self.inputs_sha256], sort_keys=True
        )
        self.cache_key = key
        if key is None:
            return None
        try:
            # Disk caches may wait on file locks; keep the event loop responsive.
            hit = await asyncio.to_thread(cache.get, key)
        except Exception:
            logger.warning(
                "Could not read score cache for %s, binding %s",
                self.row.case_id,
                self.key,
                exc_info=True,
            )
            self.cache_key = None
            return None
        if hit is not None:
            score = validate_output(hit, self.binding.metric.spec)
            if score.status == "success":
                return replace(score, cache_hit=True)
        return None

    async def perform(self, record: dict[str, Any]) -> ScoreResult:
        spec = self.binding.metric.spec
        try:
            output = self.row.task.output if self.row.task is not None else None
            inputs = self.binding.inputs_for(record, output=output, shared_row=True)
            self.inputs_sha256 = arguments_sha256(inputs)
            if self.row.task is not None:
                self.input_sha256_by_name = {
                    name: json_sha256(value, sort_keys=False)
                    for name, value in inputs.items()
                }
            if self.cache is not None:
                hit = await self._cached_score(self.cache)
                if hit is not None:
                    return hit
            output = await self.binding.ascore_validated(inputs)
            return validate_output(output, spec)
        except InputValidationError as exc:
            logger.warning(
                "Metric input validation failed: plugin=%s binding=%s input_index=%d",
                spec.id,
                self.key,
                self.row.input_index,
            )
            logger.debug("Metric input validation details", exc_info=True)
            return ScoreResult(
                "error",
                error=ErrorInfo(
                    "validation_error", field=exc.field, expected=exc.expected
                ),
            )
        except UndefinedScore as exc:
            return ScoreResult(
                "undefined", error=ErrorInfo("undefined_score", code=exc.code)
            )
        except (ModelError, ModelOutputError) as exc:
            logger.exception(
                "Model scoring failed: plugin=%s binding=%s input_index=%d",
                spec.id,
                self.key,
                self.row.input_index,
            )
            if isinstance(exc, ModelOutputError):
                return ScoreResult(
                    "error", error=ErrorInfo("model_output_error", code=exc.code)
                )
            return ScoreResult(
                "error",
                error=ErrorInfo("model_error", type(exc.__cause__ or exc).__name__),
            )
        except (UsageScopeClosed, ClientLoopError):
            # A sealed score must stop here; a loop-bound client is a run-level error.
            raise
        except Exception as exc:
            logger.exception(
                "Metric execution failed: plugin=%s binding=%s input_index=%d",
                spec.id,
                self.key,
                self.row.input_index,
            )
            return ScoreResult(
                "error", error=ErrorInfo("metric_error", type(exc).__name__)
            )


@dataclass
class ExperimentJob(Job[RowTaskResult]):
    function: RowFunction

    def state(
        self, status: StepStatus, error: ErrorInfo | None = None
    ) -> RowTaskResult:
        return RowTaskResult(status, error=error)

    def store(self, outcome: RowTaskResult) -> None:
        self.row.task = outcome

    async def perform(self, record: dict[str, Any]) -> RowTaskResult:
        try:
            output = self.function(record)
            if inspect.isawaitable(output):
                output = await output
            if output is None:
                return RowTaskResult("skipped")
            if not isinstance(output, dict):
                raise TypeError("An experiment must return a JSON object or None")
            return RowTaskResult("success", json.loads(encode_json(output)))
        except (UsageScopeClosed, ClientLoopError):
            raise
        except Exception as exc:
            logger.exception(
                "Experiment function failed: input_index=%d", self.row.input_index
            )
            return RowTaskResult(
                "error", error=ErrorInfo("experiment_error", type(exc).__name__)
            )

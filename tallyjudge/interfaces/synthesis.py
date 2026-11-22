"""Contracts for generating evaluation samples with traceable evidence."""

import asyncio
import copy
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from tallyjudge.core.dataset import Dataset
from tallyjudge.core.results import ErrorInfo
from tallyjudge.core.usage import ScoreUsage, UsageScope, UsageSummary


@dataclass(frozen=True)
class SourceDocument:
    """Source text with a stable document ID, optional entity labels and page locator."""

    document_id: str
    text: str
    entities: tuple[str, ...] = ()
    page: int | str | None = None


@dataclass
class SynthesisCandidate:
    """One attempted sample, with its source and the outcome of generation."""

    case_id: str
    chunk_id: str
    document_id: str
    status: Literal["running", "success", "error", "cancelled"] = "running"
    record: dict[str, Any] | None = None
    error: ErrorInfo | None = None
    usage: ScoreUsage | None = None
    evidence_ids: tuple[str, ...] = ()
    document_ids: tuple[str, ...] = ()


@dataclass
class SynthesisResult:
    """Generated samples and failed attempts in selection order, including partial runs."""

    plugin_id: str
    config_snapshot: dict[str, Any]
    testset_size: int
    candidates: list[SynthesisCandidate] = field(default_factory=list)

    planning_usage: ScoreUsage | None = None

    @property
    def attempted_count(self) -> int:
        """Count candidates that started, including a currently running candidate."""
        return len(self.candidates)

    @property
    def generated_count(self) -> int:
        """Count successfully generated records available for evaluation."""
        return sum(candidate.status == "success" for candidate in self.candidates)

    @property
    def usage(self) -> ScoreUsage:
        """Sum observed requests, tokens and untracked sources across all attempts."""
        usage = [c.usage for c in self.candidates if c.usage is not None]
        if self.planning_usage is not None:
            usage.append(self.planning_usage)
        return ScoreUsage.aggregate(usage) or ScoreUsage(
            UsageSummary(), UsageSummary(), []
        )

    def to_records(self) -> list[dict[str, Any]]:
        """Return detached, JSON-compatible records for successful candidates."""
        return [
            copy.deepcopy(c.record) for c in self.candidates if c.record is not None
        ]

    def to_dataset(self) -> Dataset:
        """Build an evaluation dataset preserving the generated case IDs."""
        return Dataset.from_records(self.to_records())


class SynthesisCollector:
    """Expose detached snapshots of one active or most recent synthesis run."""

    def __init__(self) -> None:
        self._result: SynthesisResult | None = None
        self._active = False
        self._scope: UsageScope | None = None
        self._planning_scope: UsageScope | None = None

    @contextmanager
    def collect(self, result: SynthesisResult) -> Iterator[None]:
        """Publish the result while a synthesizer processes its candidates."""
        if self._active:
            raise ValueError("SynthesisCollector is already collecting another run")
        self._result = result
        self._active = True
        try:
            yield
        finally:
            self._active = False

    @contextmanager
    def plan(self) -> Iterator[None]:
        """Capture graph preparation and scenario-selection requests before candidates exist."""
        if self._result is None or not self._active:
            raise RuntimeError("Start collecting a synthesis run before planning")
        scope = self._planning_scope = UsageScope()
        try:
            with scope.activate():
                yield
        finally:
            self._result.planning_usage = scope.seal()
            self._planning_scope = None

    @contextmanager
    def attempt(self, candidate: SynthesisCandidate) -> Iterator[None]:
        """Append a candidate to the active run and capture its usage up to termination."""
        if self._result is None or not self._active:
            raise RuntimeError(
                "Start collecting a synthesis run before recording an attempt"
            )
        self._result.candidates.append(candidate)
        scope = self._scope = UsageScope()
        try:
            with scope.activate():
                yield
        except asyncio.CancelledError:
            candidate.status = "cancelled"
            raise
        except Exception as exc:
            candidate.status = "error"
            candidate.error = ErrorInfo("internal_error", type(exc).__name__)
            raise
        finally:
            candidate.usage = scope.seal()
            self._scope = None

    def snapshot(self) -> SynthesisResult:
        """Return a detached result, including observed usage on an in-flight request."""
        if self._result is None:
            raise RuntimeError("No synthesis run has been accepted by this collector")
        result = copy.deepcopy(self._result)
        if self._planning_scope is not None:
            result.planning_usage = self._planning_scope.snapshot()
        if self._scope is not None:
            result.candidates[-1].usage = self._scope.snapshot()
        return result


@runtime_checkable
class Synthesizer(Protocol):
    """Generate evaluation samples from records with chunk_id, document_id and text."""

    @property
    def plugin_id(self) -> str: ...
    def config_snapshot(self) -> dict[str, Any]:
        """Return the JSON settings that identify this generation policy."""
        ...

    async def arun(
        self,
        chunks: Iterable[Mapping[str, Any]],
        *,
        testset_size: int,
        collector: SynthesisCollector | None = None,
    ) -> SynthesisResult:
        """Generate using the selected policy’s quotas and propagate external cancellation."""
        ...

    def run(
        self,
        chunks: Iterable[Mapping[str, Any]],
        *,
        testset_size: int,
        collector: SynthesisCollector | None = None,
    ) -> SynthesisResult:
        """Generate samples synchronously outside a running event loop."""
        ...

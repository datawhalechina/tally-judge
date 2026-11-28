import asyncio
from collections.abc import Coroutine, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from tallyjudge.sdk._tasks import detach_task
from tallyjudge.sdk.errors import UsageScopeClosed
from tallyjudge.sdk.events import EventKind, RunEvents
from tallyjudge.sdk.usage import (
    RequestKind,
    RequestRecord,
    ScoreUsage,
    Tracking,
    UsageSummary,
)


@dataclass
class UsageScope:
    records: dict[RequestKind, list[RequestRecord]] = field(
        default_factory=lambda: {"generation": [], "embedding": []}
    )
    untracked: dict[RequestKind, set[str]] = field(
        default_factory=lambda: {"generation": set(), "embedding": set()}
    )
    sealed: bool = False
    events: RunEvents | None = None
    case_id: str | None = None
    metric_key: str | None = None

    @contextmanager
    def activate(self) -> Iterator[None]:
        token = _scope.set(self)
        try:
            yield
        finally:
            _scope.reset(token)

    @property
    def observed(self) -> bool:
        return self.events is not None and self.events.sink is not None

    def seal(self) -> ScoreUsage | None:
        self.sealed = True
        return self.snapshot()

    def snapshot(self) -> ScoreUsage | None:
        if not any(self.records.values()) and not any(self.untracked.values()):
            return None
        return ScoreUsage(
            UsageSummary.from_records(
                self.records["generation"], tracked=not self.untracked["generation"]
            ),
            UsageSummary.from_records(
                self.records["embedding"], tracked=not self.untracked["embedding"]
            ),
            sorted(self.untracked["generation"] | self.untracked["embedding"]),
        )


_scope: ContextVar[UsageScope | None] = ContextVar("tallyjudge_usage", default=None)


def current_case_id() -> str | None:
    """Return the active evaluation case ID, if this is a row-scoped call."""
    scope = _scope.get()
    return scope.case_id if scope is not None else None


class RequestLimitReached(RuntimeError): ...


class RequestBudget:
    def __init__(self, maximum: int):
        self._parent = _budget.get()
        self.limit = maximum
        self.requests = 0
        self.exhausted = False
        self.closed = False

    @contextmanager
    def activate(self) -> Iterator[None]:
        token = _budget.set(self)
        try:
            yield
        finally:
            _budget.reset(token)

    def claim(self) -> None:
        if self.closed or self.requests >= self.limit:
            self.exhausted = True
            raise RequestLimitReached("Request budget exhausted")
        if self._parent is not None:
            try:
                self._parent.claim()
            except RequestLimitReached:
                self.exhausted = True
                raise
        self.requests += 1


_budget: ContextVar[RequestBudget | None] = ContextVar(
    "tallyjudge_request_budget", default=None
)


class RequestCall(list[RequestRecord]):
    """All actual requests made by a single generation or embedding invocation."""

    def __init__(self) -> None:
        super().__init__()
        scope = _scope.get()
        self.call_id = str(uuid4()) if scope and scope.observed else None


async def _in_scope[T](work: Coroutine[Any, Any, T]) -> T:
    if _scope.get() is not None:
        return await work
    scope = UsageScope()
    with scope.activate():
        task = asyncio.create_task(work)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        scope.sealed = True
        task.cancel()
        detach_task(task)
        raise


def _open_scope() -> UsageScope | None:
    scope = _scope.get()
    if scope is not None and scope.sealed:
        raise UsageScopeClosed("The score has already reached its terminal state")
    return scope


def _enter(kind: RequestKind, plugin_id: str, tracking: Tracking) -> None:
    if tracking == "untracked" and _budget.get() is not None:
        raise ValueError("A request budget requires per-request usage tracking")
    scope = _open_scope()
    if scope is not None and tracking == "untracked":
        scope.untracked[kind].add(plugin_id)


class GatedModel:
    def __init__(self, model: Any) -> None:
        self.model = model

    @property
    def plugin_id(self) -> str:
        return self.model.plugin_id

    @property
    def usage_tracking(self) -> Tracking:
        return self.model.usage_tracking

    def config_snapshot(self) -> dict[str, Any]:
        return self.model.config_snapshot()

    async def agenerate(self, prompt: Any, response_model: Any) -> Any:
        _enter("generation", self.plugin_id, self.usage_tracking)
        return await _in_scope(self.model.agenerate(prompt, response_model))

    async def aembed_texts(self, texts: Any) -> Any:
        _enter("embedding", self.plugin_id, self.usage_tracking)
        return await _in_scope(self.model.aembed_texts(texts))


def gated(model: Any) -> GatedModel:
    """Wrap a model so its calls are accounted for; wrapping twice has no effect."""
    return model if isinstance(model, GatedModel) else GatedModel(model)


@contextmanager
def record_request(
    kind: RequestKind, call: list[RequestRecord]
) -> Iterator[RequestRecord]:
    """Enter immediately before transport. A direct call and its score share one record."""
    scope = _open_scope()
    budget = _budget.get()
    if budget is not None:
        budget.claim()
    record = RequestRecord()
    call.append(record)
    if scope is not None:
        scope.records[kind].append(record)
    attempt_id = str(uuid4()) if scope is not None and scope.observed else None
    call_id = call.call_id if isinstance(call, RequestCall) else attempt_id

    def emit(event_kind: EventKind, status: str) -> None:
        if scope is not None and scope.events is not None and not scope.sealed:
            scope.events.emit(
                event_kind,
                case_id=scope.case_id,
                metric_key=scope.metric_key,
                call_id=call_id,
                attempt_id=attempt_id,
                request_kind=kind,
                status=status,
                usage=record.usage,
            )

    emit("request_started", "running")
    status = "success"
    try:
        yield record
    except asyncio.CancelledError:
        status = "cancelled"
        raise
    except Exception:
        status = "error"
        raise
    finally:
        record.finished = True
        emit("request_finished", status)

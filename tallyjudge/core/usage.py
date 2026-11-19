"""Count actual requests and freeze the observed token usage at a score's deadline."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Literal

type RequestKind = Literal["generation", "embedding"]
type Tracking = Literal["per_request", "untracked"]


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class RequestRecord:
    usage: Usage | None = None
    finished: bool = False


@dataclass
class UsageSummary:
    requests: int = 0
    requests_without_usage: int = 0
    in_flight_requests: int = 0
    complete: bool = True
    known_tokens: dict[str, int | None] = field(
        default_factory=lambda: {"input_tokens": None, "output_tokens": None}
    )
    known_counts: dict[str, int] = field(
        default_factory=lambda: {"input_tokens": 0, "output_tokens": 0}
    )

    @classmethod
    def from_records(
        cls, records: list[RequestRecord], *, tracked: bool = True
    ) -> "UsageSummary":
        summary = cls(requests=len(records))
        for record in records:
            summary.in_flight_requests += int(not record.finished)
            if record.usage is None:
                summary.requests_without_usage += 1
                continue
            for name in summary.known_counts:
                value = getattr(record.usage, name)
                if value is not None:
                    summary.known_tokens[name] = (
                        summary.known_tokens[name] or 0
                    ) + value
                    summary.known_counts[name] += 1
        summary.complete = tracked and not (
            summary.in_flight_requests or summary.requests_without_usage
        )
        return summary

    def add(self, other: "UsageSummary") -> None:
        self.requests += other.requests
        self.requests_without_usage += other.requests_without_usage
        self.in_flight_requests += other.in_flight_requests
        self.complete &= other.complete
        for name, value in other.known_tokens.items():
            if value is not None:
                self.known_tokens[name] = (self.known_tokens[name] or 0) + value
                self.known_counts[name] += other.known_counts[name]


@dataclass
class ScoreUsage:
    generation: UsageSummary
    embedding: UsageSummary
    untracked_sources: list[str]

    @classmethod
    def aggregate(cls, scores: Sequence["ScoreUsage"]) -> "ScoreUsage | None":
        if not scores:
            return None
        generation, embedding = UsageSummary(), UsageSummary()
        sources: set[str] = set()
        for score in scores:
            generation.add(score.generation)
            embedding.add(score.embedding)
            sources.update(score.untracked_sources)
        return cls(generation, embedding, sorted(sources))


@dataclass
class UsageScope:
    records: dict[RequestKind, list[RequestRecord]] = field(
        default_factory=lambda: {"generation": [], "embedding": []}
    )
    untracked: dict[RequestKind, set[str]] = field(
        default_factory=lambda: {"generation": set(), "embedding": set()}
    )
    sealed: bool = False

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Child tasks inherit this scope; the scheduler immediately restores its own."""
        token = _scope.set(self)
        try:
            yield
        finally:
            _scope.reset(token)

    def seal(self) -> ScoreUsage | None:
        self.sealed = True
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


class UsageScopeClosed(RuntimeError):
    """A finished score can no longer start a model request."""


def mark_usage_source(kind: RequestKind, plugin_id: str, tracking: Tracking) -> None:
    scope = _scope.get()
    if scope is not None and tracking == "untracked" and not scope.sealed:
        scope.untracked[kind].add(plugin_id)


@contextmanager
def record_request(
    kind: RequestKind, call: list[RequestRecord]
) -> Iterator[RequestRecord]:
    """Enter immediately before transport. A direct call and its score share one record."""
    scope = _scope.get()
    if scope is not None and scope.sealed:
        raise UsageScopeClosed("The score has already reached its terminal state")
    record = RequestRecord()
    call.append(record)
    if scope is not None:
        scope.records[kind].append(record)
    try:
        yield record
    finally:
        record.finished = True

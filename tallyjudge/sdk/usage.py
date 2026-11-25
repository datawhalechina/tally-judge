from collections.abc import Sequence
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

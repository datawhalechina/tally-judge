"""Opt-in cache for complete, successful binding scores."""

from typing import Protocol, runtime_checkable

from tallyjudge.core.results import MetricResult


@runtime_checkable
class ScoreCache(Protocol):
    @property
    def namespace(self) -> str:
        """Deployment or implementation identity chosen by the caller."""
        ...

    def get(self, key: str) -> MetricResult | None: ...

    def put(self, key: str, score: MetricResult) -> None: ...

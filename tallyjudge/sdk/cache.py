from typing import Protocol, runtime_checkable

from tallyjudge.sdk.results import MetricResult


@runtime_checkable
class ScoreCache(Protocol):
    """Reuse scores for identical inputs, settings and namespace."""

    @property
    def namespace(self) -> str:
        """Deployment or implementation identity chosen by the caller."""
        ...

    def get(self, key: str) -> MetricResult | None: ...

    def put(self, key: str, score: MetricResult) -> None: ...

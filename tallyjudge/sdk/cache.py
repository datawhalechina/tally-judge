from typing import Protocol, runtime_checkable

from tallyjudge.sdk.results import MetricResult


@runtime_checkable
class ScoreCache(Protocol):
    @property
    def namespace(self) -> str: ...

    def get(self, key: str) -> MetricResult | None: ...

    def put(self, key: str, score: MetricResult) -> None: ...

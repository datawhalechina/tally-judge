from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from tallyjudge.sdk.results import RunResult


@runtime_checkable
class Analyzer(Protocol):
    def analyze(self, runs: Sequence[RunResult]) -> dict[str, Any]: ...

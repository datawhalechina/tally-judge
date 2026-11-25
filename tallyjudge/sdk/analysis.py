from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from tallyjudge.sdk.results import RunResult


@runtime_checkable
class Analyzer(Protocol):
    """Derive a JSON report from finished runs without changing them."""

    def analyze(self, runs: Sequence[RunResult]) -> dict[str, Any]: ...

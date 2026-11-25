import math
from dataclasses import dataclass
from typing import Any, Literal


def _positive(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


@dataclass(frozen=True)
class ExecutionConfig:
    max_concurrency: int = 8
    timeout: float | None = None
    failure_policy: Literal["record", "raise"] = "record"
    cleanup_timeout: float = 5.0

    def __post_init__(self) -> None:
        if type(self.max_concurrency) is not int or self.max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")
        if self.timeout is not None and not _positive(self.timeout):
            raise ValueError("timeout must be a finite positive number or None")
        if not _positive(self.cleanup_timeout):
            raise ValueError("cleanup_timeout must be a finite positive number")
        if self.failure_policy not in ("record", "raise"):
            raise ValueError("failure_policy must be 'record' or 'raise'")

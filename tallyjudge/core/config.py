"""Run-scoped execution settings; provider retries belong to plugins."""

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ExecutionConfig:
    max_concurrency: int = 8
    timeout: float | None = None
    failure_policy: Literal["record", "raise"] = "record"
    cleanup_timeout: float = 5.0

    def __post_init__(self) -> None:
        if type(self.max_concurrency) is not int or self.max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")
        if self.timeout is not None and (
            type(self.timeout) not in (int, float)
            or not math.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise ValueError("timeout must be a finite positive number or None")
        if (
            type(self.cleanup_timeout) not in (int, float)
            or not math.isfinite(self.cleanup_timeout)
            or self.cleanup_timeout <= 0
        ):
            raise ValueError("cleanup_timeout must be a finite positive number")
        if self.failure_policy not in ("record", "raise"):
            raise ValueError("failure_policy must be 'record' or 'raise'")

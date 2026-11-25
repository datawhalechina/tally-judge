from collections.abc import Awaitable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Runner(Protocol):
    def __call__(
        self, row: dict[str, Any]
    ) -> dict[str, Any] | None | Awaitable[dict[str, Any] | None]: ...

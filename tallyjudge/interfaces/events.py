"""Observer contract for run and request lifecycle events."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from tallyjudge.core.events import RunEvent


@runtime_checkable
class EventSink(Protocol):
    def emit(self, event: "RunEvent") -> None:
        """Record one event in emission order."""
        ...


@runtime_checkable
class BufferedEventSink(EventSink, Protocol):
    async def aflush_run(self, run_id: str) -> None:
        """Wait for this run's terminal event to reach storage."""
        ...

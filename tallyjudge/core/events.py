"""Run-scoped lifecycle events with no sample, prompt or response content."""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tallyjudge.interfaces.events import EventSink

if TYPE_CHECKING:
    from tallyjudge.core.usage import RequestKind, Usage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunEvent:
    run_id: str
    sequence: int
    kind: str
    case_id: str | None = None
    metric_key: str | None = None
    status: str | None = None
    call_id: str | None = None
    attempt_id: str | None = None
    request_kind: "RequestKind | None" = None
    usage: "Usage | None" = None


class RunEvents:
    def __init__(self, run_id: str, sink: EventSink | None) -> None:
        self.run_id = run_id
        self.sink = sink
        self.complete = None if sink is None else True
        self.sequence = 0

    def emit(
        self,
        kind: str,
        *,
        case_id: str | None = None,
        metric_key: str | None = None,
        status: str | None = None,
        call_id: str | None = None,
        attempt_id: str | None = None,
        request_kind: "RequestKind | None" = None,
        usage: "Usage | None" = None,
    ) -> None:
        if self.sink is None:
            return
        self.sequence += 1
        event = RunEvent(
            self.run_id,
            self.sequence,
            kind,
            case_id,
            metric_key,
            status,
            call_id,
            attempt_id,
            request_kind,
            usage,
        )
        try:
            self.sink.emit(event)
        except Exception:
            logger.exception(
                "Event sink failed for run %s at event %d", self.run_id, self.sequence
            )
            self.complete = False
            self.sink = None

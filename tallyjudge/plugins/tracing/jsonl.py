"""Local append-only events, one JSONL stream per run."""

from dataclasses import asdict
from pathlib import Path
from typing import TextIO

from tallyjudge.core.dataset import canonical_json
from tallyjudge.core.events import RunEvent
from tallyjudge.plugins._local import run_file


class JsonlEventSink:
    """Append each run's events to ``events-<run_id>.jsonl`` under ``root``.

    The scheduler emits events on the event loop, so each run keeps one
    line-buffered handle open instead of reopening the file per event. The handle
    is closed on that run's ``run_finished`` event; ``close`` releases any handle
    left open by a run that never finished in this process.
    """

    def __init__(self, *, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._streams: dict[str, TextIO] = {}

    def emit(self, event: RunEvent) -> None:
        stream = self._streams.get(event.run_id)
        if stream is None:
            stream = run_file(self.root, "events", event.run_id).open(
                "a", encoding="utf-8", newline="\n", buffering=1
            )
            self._streams[event.run_id] = stream
        stream.write(canonical_json(asdict(event)) + "\n")
        if event.kind == "run_finished":
            self._streams.pop(event.run_id).close()

    def close(self) -> None:
        """Close the files of runs that did not emit ``run_finished``."""
        while self._streams:
            self._streams.popitem()[1].close()

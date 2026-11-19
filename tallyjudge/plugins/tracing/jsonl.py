"""Local append-only events, one JSONL stream per run."""

import asyncio
from concurrent.futures import Future, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from queue import SimpleQueue
from threading import Thread

from tallyjudge.core.dataset import canonical_json
from tallyjudge.core.events import RunEvent
from tallyjudge.plugins._local import run_file


@dataclass
class _Stream:
    queue: SimpleQueue[RunEvent | None] = field(default_factory=SimpleQueue)
    done: Future[None] = field(default_factory=Future)

    def write(self, path: Path) -> None:
        try:
            with path.open("a", encoding="utf-8", newline="\n", buffering=1) as stream:
                while (event := self.queue.get()) is not None:
                    stream.write(canonical_json(asdict(event)) + "\n")
                    if event.kind == "run_finished":
                        break
        except Exception as exc:
            self.done.set_exception(exc)
        else:
            self.done.set_result(None)


class JsonlEventSink:
    """Write each run's ordered events on one worker with its own JSONL file."""

    def __init__(self, *, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._streams: dict[str, _Stream] = {}

    def emit(self, event: RunEvent) -> None:
        """Queue one event without waiting for disk I/O on the event loop."""
        stream = self._streams.get(event.run_id)
        if stream is None:
            path = run_file(self.root, "events", event.run_id)
            stream = _Stream()
            stream.done.set_running_or_notify_cancel()
            Thread(target=stream.write, args=(path,), daemon=True).start()
            self._streams[event.run_id] = stream
        if not stream.done.done():
            stream.queue.put(event)

    async def aflush_run(self, run_id: str) -> None:
        """Wait for queued writes and propagate the writer's error to the evaluator."""
        stream = self._streams.pop(run_id)
        await asyncio.wrap_future(stream.done)

    def close(self) -> None:
        """Finish and join any writers used outside an evaluator."""
        streams = tuple(self._streams.values())
        self._streams.clear()
        for stream in streams:
            stream.queue.put(None)
        wait(stream.done for stream in streams)
        for stream in streams:
            stream.done.result()

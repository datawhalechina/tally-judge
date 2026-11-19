"""Local append-only events, one JSONL stream per run."""

from dataclasses import asdict
from pathlib import Path

from tallyjudge.core.dataset import canonical_json
from tallyjudge.core.events import RunEvent
from tallyjudge.plugins._local import run_file


class JsonlEventSink:
    def __init__(self, *, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def emit(self, event: RunEvent) -> None:
        with run_file(self.root, "events", event.run_id).open(
            "a", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(canonical_json(asdict(event)) + "\n")

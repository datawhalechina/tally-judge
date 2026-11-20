"""Explicit, run-scoped export of content-free evaluation events to MLflow."""

import asyncio
import json
from dataclasses import asdict
from importlib import import_module

from tallyjudge.core.events import RunEvent
from tallyjudge.registry.specs import EmptyConfig


class MlflowConfig(EmptyConfig):
    """Identify an existing MLflow experiment and its tracking server."""

    tracking_uri: str
    experiment_id: str


class MlflowEventSink:
    """Upload one ordered event artifact per evaluation run on a worker thread."""

    def __init__(self, *, tracking_uri: str, experiment_id: str):
        self.client = import_module("mlflow.tracking").MlflowClient(
            tracking_uri=tracking_uri
        )
        self.experiment_id = experiment_id
        self._pending: dict[str, list[RunEvent]] = {}

    def emit(self, event: RunEvent) -> None:
        """Buffer lifecycle metadata until the evaluator finishes this run."""
        self._pending.setdefault(event.run_id, []).append(event)

    async def aflush_run(self, run_id: str) -> None:
        """Upload events in sequence, propagating delivery failure to events_complete."""
        events = self._pending.pop(run_id)
        await asyncio.to_thread(self._export, run_id, events)

    def _export(self, run_id: str, events: list[RunEvent]) -> None:
        run = self.client.create_run(
            self.experiment_id, tags={"tallyjudge.run_id": run_id}
        )
        remote = run.info.run_id
        try:
            stream = "".join(
                json.dumps(asdict(event), separators=(",", ":")) + "\n"
                for event in events
            )
            self.client.log_text(remote, stream, "tallyjudge/events.jsonl")
        except Exception:
            self.client.set_terminated(remote, status="FAILED")
            raise
        status = {"completed": "FINISHED", "cancelled": "KILLED"}.get(
            events[-1].status or "", "FAILED"
        )
        self.client.set_terminated(remote, status=status)

import json
from threading import Lock

from tallyjudge.sdk.dataset import canonical_json
from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.results import RunResult
from tallyjudge.sdk.storage import storage_snapshot


class MemoryStorageBackend:
    plugin = Plugin("storage")
    plugin_id = "storage.local.memory"

    def __init__(self) -> None:
        self._runs: dict[str, str] = {}
        self._lock = Lock()

    def save_run(self, result: RunResult, *, include_inputs: bool = True) -> None:
        data = storage_snapshot(result, include_inputs=include_inputs)
        payload = json.dumps(data, ensure_ascii=False, allow_nan=False)
        with self._lock:
            previous = self._runs.setdefault(result.run_id, payload)
        if previous != payload and canonical_json(
            json.loads(previous)
        ) != canonical_json(data):
            raise FileExistsError(
                f"Run already exists with different content: {result.run_id}"
            )

    def load_run(self, run_id: str) -> RunResult:
        try:
            payload = self._runs[run_id]
        except KeyError as exc:
            raise FileNotFoundError(run_id) from exc
        return RunResult.from_dict(json.loads(payload))

from typing import Any, Protocol, runtime_checkable

from tallyjudge.sdk.results import RunResult


@runtime_checkable
class StorageBackend(Protocol):
    def save_run(self, result: RunResult, *, include_inputs: bool = True) -> None: ...

    def load_run(self, run_id: str) -> RunResult: ...


def storage_snapshot(result: RunResult, *, include_inputs: bool) -> dict[str, Any]:
    if result.run_status == "running":
        raise ValueError("Only terminal runs can be saved")
    return result._payload(include_inputs=include_inputs)

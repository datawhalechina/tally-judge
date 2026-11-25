import filecmp
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from tallyjudge.plugins._local import DirectoryConfig, directory, run_file
from tallyjudge.sdk.dataset import canonical_json
from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.results import RunResult
from tallyjudge.sdk.storage import storage_snapshot


class JsonlStorageBackend:
    plugin = Plugin("storage")
    plugin_id = "storage.local.jsonl"

    Config = DirectoryConfig

    def __init__(self, *, root: str | Path) -> None:
        self.root = directory(root)

    def save_run(self, result: RunResult, *, include_inputs: bool = True) -> None:
        target = run_file(self.root, "run", result.run_id)
        header = storage_snapshot(result, include_inputs=include_inputs)
        rows, manifest = header.pop("rows"), header.pop("manifest")
        inputs = header.pop("inputs", None)
        header.update(row_count=len(rows), includes_inputs=inputs is not None)
        self.root.mkdir(parents=True, exist_ok=True)
        stream = NamedTemporaryFile(
            dir=self.root, delete=False, mode="w", encoding="utf-8", newline="\n"
        )
        temporary = Path(stream.name)
        try:
            with stream:
                stream.write(
                    json.dumps(header, ensure_ascii=False, allow_nan=False) + "\n"
                )
                for index, (entry, row) in enumerate(zip(manifest, rows, strict=True)):
                    record = {"manifest": entry, "row": row}
                    if inputs is not None:
                        record["input"] = inputs[index]
                    stream.write(
                        json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                    )
            try:
                os.link(temporary, target)
            except FileExistsError:
                if not filecmp.cmp(temporary, target, shallow=False):
                    stored = self.load_run(result.run_id)
                    previous = storage_snapshot(stored, include_inputs=True)
                    current = storage_snapshot(result, include_inputs=include_inputs)
                    if canonical_json(previous) != canonical_json(current):
                        raise FileExistsError(
                            f"Run already exists with different content: {target}"
                        )
        finally:
            temporary.unlink()

    def load_run(self, run_id: str) -> RunResult:
        target = run_file(self.root, "run", run_id)
        with target.open(encoding="utf-8") as stream:
            try:
                data = json.loads(stream.readline())
                if data["run_id"] != run_id:
                    raise ValueError("Run ID differs from the file name")
                count = data.pop("row_count")
                keep_inputs = data.pop("includes_inputs")
                data["rows"], data["manifest"] = [], []
                if keep_inputs:
                    data["inputs"] = []
                for line in stream:
                    record = json.loads(line)
                    data["rows"].append(record["row"])
                    data["manifest"].append(record["manifest"])
                    if keep_inputs:
                        data["inputs"].append(record["input"])
                if len(data["rows"]) != count:
                    raise ValueError(
                        f"Expected {count} rows, found {len(data['rows'])}"
                    )
                return RunResult.from_dict(data)
            except (TypeError, KeyError, ValueError) as exc:
                raise ValueError(f"Invalid run file {target}: {exc}") from exc

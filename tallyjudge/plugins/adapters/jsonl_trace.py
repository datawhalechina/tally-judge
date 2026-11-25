import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.workflow import WorkflowTrace


class JsonlTraceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    run_id: str | None = None


class JsonlTrace:
    plugin = Plugin("importer", produces="trace")
    plugin_id = "importer.jsonl.trace"
    Config = JsonlTraceConfig

    def __init__(self, run_id: str | None = None):
        self.run_id = JsonlTraceConfig(run_id=run_id).run_id

    def convert(self, source: str | Path) -> WorkflowTrace:
        path = Path(source)
        with path.open(encoding="utf-8") as stream:
            steps = [json.loads(line) for line in stream if line.strip()]
        return WorkflowTrace.model_validate(
            {"run_id": self.run_id or path.stem, "steps": steps}
        )

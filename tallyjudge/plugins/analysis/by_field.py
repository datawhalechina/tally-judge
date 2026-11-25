from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tallyjudge.sdk.dataset import canonical_json
from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.results import RowResult, RunResult, aggregate


class ByFieldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    field: str = Field(min_length=1)


class ByFieldAnalyzer:
    plugin = Plugin("analyzer")
    plugin_id = "analyzer.summary.by_field"
    Config = ByFieldConfig

    def __init__(self, *, field: str):
        self.field = ByFieldConfig(field=field).field

    def _groups(self, run: RunResult) -> dict[str, Any]:
        if run.inputs is None:
            raise ValueError(f"Run {run.run_id} was saved without its inputs")
        groups: dict[str, list[RowResult]] = {}
        for row in run.rows:
            value = run.inputs[row.input_index].get(self.field)
            name = value if isinstance(value, str) else canonical_json(value)
            groups.setdefault(name, []).append(row)
        return {name: aggregate(rows, run.metrics) for name, rows in groups.items()}

    def analyze(self, runs: Sequence[RunResult]) -> dict[str, Any]:
        return {
            "field": self.field,
            "runs": [
                {"run_id": run.run_id, "groups": self._groups(run)} for run in runs
            ],
        }

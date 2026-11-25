from collections.abc import Sequence
from statistics import mean
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.results import RunResult


class AgreementConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    metric: str = Field(min_length=1)
    field: str = Field(min_length=1)
    tolerance: float = Field(default=0.0, ge=0)


class AgreementAnalyzer:
    plugin = Plugin("analyzer")
    plugin_id = "analyzer.agreement.with_labels"
    Config = AgreementConfig

    def __init__(self, **settings: Any):
        self.config = AgreementConfig(**settings)

    def _compare(self, run: RunResult) -> dict[str, Any]:
        if run.inputs is None:
            raise ValueError(f"Run {run.run_id} was saved without its inputs")
        errors = {}
        for row in run.rows:
            score = row.metrics[self.config.metric]
            label = run.inputs[row.input_index].get(self.config.field)
            if (
                score.status == "success"
                and isinstance(score.value, float)
                and isinstance(label, int | float)
                and not isinstance(label, bool)
            ):
                errors[row.case_id] = abs(score.value - label)
        within = [
            case for case, error in errors.items() if error <= self.config.tolerance
        ]
        return {
            "run_id": run.run_id,
            "compared": len(errors),
            "agreed": len(within),
            "agreement": len(within) / len(errors) if errors else None,
            "mean_absolute_error": mean(errors.values()) if errors else None,
            "disagreements": sorted(set(errors) - set(within)),
        }

    def analyze(self, runs: Sequence[RunResult]) -> dict[str, Any]:
        return {
            "metric": self.config.metric,
            "field": self.config.field,
            "tolerance": self.config.tolerance,
            "runs": [self._compare(run) for run in runs],
        }

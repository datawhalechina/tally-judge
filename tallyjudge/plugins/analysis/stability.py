from collections.abc import Sequence
from statistics import mean
from typing import Any

from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.results import RunResult


class StabilityAnalyzer:
    plugin = Plugin("analyzer")
    plugin_id = "analyzer.stability.repeats"

    def analyze(self, runs: Sequence[RunResult]) -> dict[str, Any]:
        if len(runs) < 2:
            raise ValueError("Stability needs at least two runs")
        metrics = {}
        for key in set.intersection(*(set(run.metrics) for run in runs)):
            ranges = {}
            for case_id in set.intersection(
                *({row.case_id for row in run.rows} for run in runs)
            ):
                scores = [run.row_for(case_id).metrics[key] for run in runs]
                values = [
                    s.value
                    for s in scores
                    if s.status == "success" and isinstance(s.value, float)
                ]
                if len(values) == len(runs):
                    ranges[case_id] = max(values) - min(values)
            metrics[key] = {
                "compared": len(ranges),
                "identical": sum(spread == 0 for spread in ranges.values()),
                "mean_range": mean(ranges.values()) if ranges else None,
                "max_range": max(ranges.values(), default=None),
                "unstable_cases": sorted(c for c, spread in ranges.items() if spread),
            }
        return {"runs": [run.run_id for run in runs], "metrics": metrics}

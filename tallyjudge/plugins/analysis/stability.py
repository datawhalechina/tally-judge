from collections.abc import Sequence
from statistics import mean
from typing import Any

from tallyjudge.sdk.comparison import binding_differences
from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.results import RunResult


class StabilityAnalyzer:
    plugin = Plugin("analyzer")
    plugin_id = "analyzer.stability.repeats"

    def analyze(self, runs: Sequence[RunResult]) -> dict[str, Any]:
        """Measure score variation for matching samples and metric settings."""
        if len(runs) < 2:
            raise ValueError("Stability needs at least two runs")
        rows = [{row.case_id: row for row in run.rows} for run in runs]
        shared_cases = set.intersection(*(set(index) for index in rows))
        for case_id in shared_cases:
            fingerprints = {
                run.manifest[index[case_id].input_index]["sha256"]
                for run, index in zip(runs, rows, strict=True)
            }
            if len(fingerprints) > 1:
                raise ValueError(f"Input changed for sample {case_id!r}")

        metrics = {}
        for key in set.intersection(*(set(run.metrics) for run in runs)):
            for run in runs[1:]:
                changes = binding_differences(runs[0].metrics[key], run.metrics[key])
                if changes:
                    raise ValueError(f"Metric {key!r} changed: {', '.join(changes)}")
            ranges = {}
            for case_id in shared_cases:
                scores = [index[case_id].metrics[key] for index in rows]
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

from collections.abc import Iterable

from tallyjudge.api.controls import dataset_bound
from tallyjudge.sdk import comparison
from tallyjudge.sdk.comparison import (
    CaseComparison,
    ExperimentComparison,
    PairedScores,
    RunComparison,
    ScoreComparison,
    compare_runs,
)
from tallyjudge.sdk.results import RunResult

APPLICATION_OUTPUT_FIELDS = ("response", "retrieved_contexts", "retrieved_context_ids")


def compare_experiments(
    left: RunResult,
    right: RunResult,
    *,
    output_fields: Iterable[str] = APPLICATION_OUTPUT_FIELDS,
) -> ExperimentComparison:
    """Pair successful scores by ID and compare numeric scores across application outputs."""
    return comparison.compare_experiments(
        left, right, output_fields=output_fields, controls=dataset_bound
    )


__all__ = [
    "CaseComparison",
    "ExperimentComparison",
    "PairedScores",
    "RunComparison",
    "ScoreComparison",
    "compare_experiments",
    "compare_runs",
]

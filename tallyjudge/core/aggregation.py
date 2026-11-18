"""Coverage-aware numerical summaries; failed scores are never zero-filled."""

from collections.abc import Iterable
from statistics import mean
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tallyjudge.core.results import RowResult

STATUSES = (
    "not_started",
    "running",
    "success",
    "error",
    "invalid_output",
    "timeout",
    "cancelled",
)


def aggregate(
    rows: Iterable["RowResult"], keys: Iterable[str]
) -> dict[str, dict[str, Any]]:
    rows = tuple(rows)
    summaries = {}
    for key in keys:
        counts = dict.fromkeys(STATUSES, 0)
        values = []
        for row in rows:
            score = row.metrics[key]
            counts[score.status] += 1
            if score.status == "success" and score.value is not None:
                values.append(score.value)
        # statistics.mean avoids overflow and cancellation in intermediate sums.
        average = mean(values) if values else None
        summaries[key] = {
            "total": len(rows),
            "status_counts": counts,
            "valid_count": len(values),
            "coverage": len(values) / len(rows) if rows else None,
            "mean": average,
        }
    return summaries

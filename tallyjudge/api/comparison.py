"""Align saved runs by sample identity and metric binding."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from tallyjudge.core.dataset import canonical_json
from tallyjudge.core.results import RowResult, RunResult, ScoreResult


@dataclass(frozen=True)
class ScoreComparison:
    left: ScoreResult | None
    right: ScoreResult | None
    changes: tuple[str, ...]
    delta: float | None


@dataclass(frozen=True)
class CaseComparison:
    case_id: str
    changes: tuple[str, ...]
    metrics: dict[str, ScoreComparison]


@dataclass(frozen=True)
class RunComparison:
    rows: list[CaseComparison]
    warnings: tuple[str, ...]


def _experiment_inputs(
    row: RowResult, columns: Iterable[str], original: dict[str, Any]
) -> dict[str, Any]:
    output = row.task.output if row.task and row.task.output else {}
    return {
        key: output[key] if key in output else original[key]
        for key in columns
        if key in output or key in original
    }


def compare_runs(left: RunResult, right: RunResult) -> RunComparison:
    """Compare right minus left for successful numeric scores with the same inputs and settings."""
    generated = any(
        entry["id_source"] == "generated"
        for run in (left, right)
        for entry in run.manifest
    )
    if generated and left.manifest != right.manifest:
        raise ValueError("Generated case IDs require identical ordered manifests")
    warnings = ("Sample IDs were generated from row positions",) if generated else ()
    sides = [
        {
            row.case_id: (row, item["sha256"])
            for row, item in zip(run.rows, run.manifest, strict=True)
        }
        for run in (left, right)
    ]
    before, after = sides
    keys = list(dict.fromkeys((*left.metrics, *right.metrics)))
    binding_changes = {}
    for key in keys:
        if key not in left.metrics or key not in right.metrics:
            binding_changes[key] = (
                "binding_missing_left"
                if key not in left.metrics
                else "binding_missing_right",
            )
            continue
        a, b = left.metrics[key], right.metrics[key]
        changes = [
            name
            for name in ("id", "revision", "config_snapshot", "input_map")
            if a[name] != b[name]
        ]
        for name, default in (
            ("output_kind", "numeric"),
            ("labels", None),
            ("ranking_length", None),
        ):
            if a.get(name, default) != b.get(name, default):
                changes.append(name)
        binding_changes[key] = tuple(changes)
    rows = []
    for case_id in dict.fromkeys((*before, *after)):
        lrow, lhash = before.get(case_id, (None, None))
        rrow, rhash = after.get(case_id, (None, None))
        if lrow is None:
            changes = ("row_missing_left",)
        elif rrow is None:
            changes = ("row_missing_right",)
        elif lhash != rhash:
            changes = ("input_changed",)
        else:
            changes = ()
        metrics = {}
        for key in keys:
            a = lrow.metrics.get(key) if lrow else None
            b = rrow.metrics.get(key) if rrow else None
            settings = binding_changes[key]
            if not (changes or settings) and lrow and rrow and (lrow.task or rrow.task):
                columns = left.metrics[key]["input_map"].values()
                original = {}
                if left.inputs is not None:
                    original = left.inputs[lrow.input_index]
                elif right.inputs is not None:
                    original = right.inputs[rrow.input_index]
                if canonical_json(
                    _experiment_inputs(lrow, columns, original)
                ) != canonical_json(_experiment_inputs(rrow, columns, original)):
                    settings += ("experiment_input_changed",)
            delta = None
            if (
                not (changes or settings)
                and a
                and b
                and a.status == b.status == "success"
                and isinstance(a.value, float)
                and isinstance(b.value, float)
            ):
                delta = b.value - a.value
            metrics[key] = ScoreComparison(a, b, settings, delta)
        rows.append(CaseComparison(case_id, changes, metrics))
    return RunComparison(rows, warnings)

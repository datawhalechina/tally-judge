"""Align saved runs by sample identity and metric binding."""

from collections.abc import Iterable
from dataclasses import dataclass
from statistics import mean
from typing import Any

from tallyjudge.core.dataset import arguments_sha256, canonical_json, json_sha256
from tallyjudge.core.results import (
    RowResult,
    RowTaskResult,
    RunResult,
    ScoreResult,
    stored_value_range,
)
from tallyjudge.core.usage import ScoreUsage


@dataclass(frozen=True)
class ScoreComparison:
    """Both scores for a binding and any change to the comparison criteria."""

    left: ScoreResult | None
    right: ScoreResult | None
    changes: tuple[str, ...]
    delta: float | None


@dataclass(frozen=True)
class CaseComparison:
    """Align the application outputs and bound scores for one case ID."""

    case_id: str
    changes: tuple[str, ...]
    metrics: dict[str, ScoreComparison]
    left_task: RowTaskResult | None = None
    right_task: RowTaskResult | None = None


@dataclass(frozen=True)
class RunComparison:
    """Aligned cases and warnings about their identity."""

    rows: list[CaseComparison]
    warnings: tuple[str, ...]


def _same(a: Any, b: Any) -> bool:
    """Compare saved settings as JSON, so True and 1 or 1 and 1.0 stay distinct."""
    return canonical_json(a) == canonical_json(b)


def _argument_hashes(
    row: RowResult,
    key: str,
    input_map: dict[str, str],
    original: dict[str, Any] | None,
) -> dict[str, str | None]:
    """Recover old per-argument identities only when the actual scorer digest confirms them."""
    score = row.metrics[key]
    if score.input_sha256_by_name is not None:
        return score.input_sha256_by_name
    if score.inputs_sha256 is None:
        return {}
    source = original or {}
    output = row.task.output if row.task and row.task.output else {}
    arguments = {
        name: output[column] if column in output else source[column]
        for name, column in input_map.items()
        if column in output or column in source
    }
    # Raw columns may omit a default or precede a custom validator. The whole
    # digest also verifies any original borrowed from the other side of a redacted run.
    if arguments_sha256(arguments) != score.inputs_sha256:
        return {}
    return {
        name: json_sha256(value, sort_keys=False) for name, value in arguments.items()
    }


def _experiment_input_change(
    left: RunResult,
    right: RunResult,
    lrow: RowResult,
    rrow: RowResult,
    key: str,
    outputs: frozenset[str] | None,
) -> str | None:
    """Compare all scored arguments, or the controls retained across business outputs."""
    a, b = lrow.metrics[key], rrow.metrics[key]
    if a.inputs_sha256 is not None and b.inputs_sha256 is not None:
        if a.inputs_sha256 == b.inputs_sha256:
            return None
        if outputs is None:
            return "experiment_input_changed"
    input_map = left.metrics[key]["input_map"]
    controls = [
        name
        for name, column in input_map.items()
        if outputs is None
        or column not in outputs
        or name in ("user_input", "reference")
        or name.startswith("reference_")
    ]
    if not controls:
        return None
    linput = left.inputs[lrow.input_index] if left.inputs is not None else None
    rinput = right.inputs[rrow.input_index] if right.inputs is not None else None
    before = _argument_hashes(
        lrow, key, input_map, linput if linput is not None else rinput
    )
    after = _argument_hashes(
        rrow, key, input_map, rinput if rinput is not None else linput
    )
    kind = "experiment_input" if outputs is None else "control_input"
    unknown = False
    for name in controls:
        lvalue, rvalue = before.get(name), after.get(name)
        if lvalue is None or rvalue is None:
            unknown = True
        elif lvalue != rvalue:
            return f"{kind}_changed"
    return f"{kind}_unverified" if unknown else None


def compare_runs(left: RunResult, right: RunResult) -> RunComparison:
    """Compare right minus left for successful numeric scores with the same inputs and settings."""
    return _compare(left, right, None)


def _compare(
    left: RunResult, right: RunResult, outputs: frozenset[str] | None
) -> RunComparison:
    generated = any(
        entry["id_source"] == "generated"
        for run in (left, right)
        for entry in run.manifest
    )
    if generated and outputs is not None:
        raise ValueError("Paired experiments require provided case IDs")
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
            if not _same(a[name], b[name])
        ]
        for name, default in (
            ("output_kind", "numeric"),
            ("direction", "higher_is_better"),
            ("labels", None),
            ("levels", None),
            ("ranking_length", None),
        ):
            if not _same(a.get(name, default), b.get(name, default)):
                changes.append(name)
        left_range, right_range = stored_value_range(a), stored_value_range(b)
        if (
            left_range is not None
            and right_range is not None
            and left_range != right_range
        ):
            changes.append("value_range")
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
                change = _experiment_input_change(left, right, lrow, rrow, key, outputs)
                if change is not None:
                    settings += (change,)
            elif (
                not (changes or settings)
                and a
                and b
                and a.inputs_sha256
                and b.inputs_sha256
                and a.inputs_sha256 != b.inputs_sha256
            ):
                settings += ("metric_input_changed",)
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
        rows.append(
            CaseComparison(
                case_id,
                changes,
                metrics,
                lrow.task if lrow else None,
                rrow.task if rrow else None,
            )
        )
    return RunComparison(rows, warnings)


@dataclass(frozen=True)
class PairedScores:
    """Per-side coverage for any score kind, with numeric outcomes on comparable pairs."""

    left_valid: int
    right_valid: int
    left_coverage: float | None
    right_coverage: float | None
    paired: int
    wins: int
    ties: int
    losses: int
    mean_delta: float | None


@dataclass(frozen=True)
class ExperimentComparison:
    """Aligned experiment answers, individual scores, metadata and measured usage."""

    rows: list[CaseComparison]
    metrics: dict[str, PairedScores]
    left_metadata: dict[str, Any] | None
    right_metadata: dict[str, Any] | None
    left_usage: ScoreUsage | None
    right_usage: ScoreUsage | None


def compare_experiments(
    left: RunResult,
    right: RunResult,
    *,
    output_fields: Iterable[str] = (
        "response",
        "retrieved_contexts",
        "retrieved_context_ids",
    ),
) -> ExperimentComparison:
    """Pair successful scores by ID and compare numeric scores across application outputs."""
    rows = _compare(left, right, frozenset(output_fields)).rows
    summaries = {}
    for key in dict.fromkeys((*left.metrics, *right.metrics)):
        left_valid = right_valid = paired = 0
        deltas = []
        for row in rows:
            scores = row.metrics[key]
            left_ok = scores.left is not None and scores.left.status == "success"
            right_ok = scores.right is not None and scores.right.status == "success"
            left_valid += left_ok
            right_valid += right_ok
            if left_ok and right_ok and not row.changes and not scores.changes:
                paired += 1
            if scores.delta is not None:
                deltas.append(scores.delta)
        sign = (
            -1 if left.metrics.get(key, {}).get("direction") == "lower_is_better" else 1
        )
        summaries[key] = PairedScores(
            left_valid,
            right_valid,
            left_valid / len(left.rows) if left.rows else None,
            right_valid / len(right.rows) if right.rows else None,
            paired,
            sum(sign * d > 0 for d in deltas),
            deltas.count(0),
            sum(sign * d < 0 for d in deltas),
            mean(deltas) if deltas else None,
        )
    return ExperimentComparison(
        rows, summaries, left.metadata, right.metadata, left.usage, right.usage
    )

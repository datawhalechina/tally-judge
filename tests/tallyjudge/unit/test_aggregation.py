import json

from tallyjudge import Dataset, Evaluator, MetricBinding, create_metric
from tallyjudge.core.aggregation import aggregate
from tallyjudge.core.results import RowResult, ScoreResult


def test_failures_are_not_zeroes():
    statuses = ["success", "success", "error", "timeout", "cancelled", "invalid_output"]
    rows = [
        RowResult(
            str(index),
            index,
            {"m": ScoreResult(status, float(index) if status == "success" else None)},
        )
        for index, status in enumerate(statuses)
    ]
    summary = aggregate(rows, ["m"])["m"]
    assert summary["total"] == 6
    assert summary["valid_count"] == 2
    assert summary["coverage"] == 2 / 6
    assert summary["mean"] == 0.5
    assert sum(summary["status_counts"].values()) == 6
    assert aggregate(rows[2:], ["m"])["m"]["mean"] is None


def test_empty_and_private_json_roundtrip():
    metric = create_metric("metric.text.exact_match")
    empty = Evaluator([metric]).run(Dataset.from_records([]))
    assert empty.rows == []
    assert empty.summary[metric.spec.id]["mean"] is None
    assert empty.summary[metric.spec.id]["coverage"] is None
    dataset = Dataset.from_records(
        [
            {"case_id": "a", "answer": "敏感文本", "reference": "敏感文本"},
            {"case_id": "b"},
        ]
    )
    report = Evaluator(
        [MetricBinding(metric, key="answer_match", input_map={"response": "answer"})]
    ).run(dataset)
    for include_inputs in (True, False):
        data = json.loads(report.to_json(include_inputs=include_inputs))
        assert data == report.to_dict(include_inputs=include_inputs)
        assert ("inputs" in data) == include_inputs
        assert data["schema_version"] == 1
        assert data["manifest"] == dataset.manifest
        assert data["metrics"]["answer_match"] == {
            "id": metric.spec.id,
            "revision": "1",
            "input_map": {"response": "answer", "reference": "reference"},
            "config_snapshot": {},
        }
        assert [row["case_id"] for row in data["rows"]] == ["a", "b"]
        assert data["rows"][1]["metrics"]["answer_match"]["value"] is None
    assert "敏感文本" not in report.to_json(include_inputs=False)
    assert "NaN" not in report.to_json()


def test_mean_preserves_extreme_finite_scores():
    import sys

    values = [sys.float_info.max] * 3
    rows = [
        RowResult(str(index), index, {"m": ScoreResult("success", value)})
        for index, value in enumerate(values)
    ]
    assert aggregate(rows, ["m"])["m"]["mean"] == sys.float_info.max

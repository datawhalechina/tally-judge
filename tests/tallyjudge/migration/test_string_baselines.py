import json
from pathlib import Path

import pytest

from tallyjudge import Dataset, Evaluator, create_metric

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/string_metrics.json").read_text(encoding="utf-8")
)
CASES = [
    (plugin_id, case)
    for plugin_id, metric in FIXTURE["metrics"].items()
    for case in metric["cases"]
]


@pytest.mark.parametrize(
    "plugin_id,case",
    CASES,
    ids=[f"{plugin_id}:{case['name']}" for plugin_id, case in CASES],
)
def test_legacy_fixture(plugin_id, case):
    metric = create_metric(plugin_id)
    result = Evaluator([metric]).run(Dataset.from_records([case["inputs"]]))
    score = result.rows[0].metrics[plugin_id]
    assert metric.spec.revision == FIXTURE["metrics"][plugin_id]["new_revision"]
    if case["intentional_difference"] is None:
        assert score.status == "success"
        assert score.value == case["old"]["value"]
        assert type(score.value).__name__ == case["old"]["value_type"] == "float"
    else:
        assert score.status == "error"
        assert score.error.category == "validation_error"
        assert score.value is None
        if case["intentional_difference"] == "missing_field_validation":
            assert case["old"]["exception_type"] == "TypeError"
        elif plugin_id == "metric.text.string_presence":
            assert case["old"]["exception_type"] == "AssertionError"
        else:
            assert "value" in case["old"]


def test_fixture_identity():
    assert FIXTURE["source_commit"] == "ec003395bb37c223fc981e6567162513aec52a71"
    assert FIXTURE["optimized"] is False
    assert len(FIXTURE["source_sha256"]) == 64

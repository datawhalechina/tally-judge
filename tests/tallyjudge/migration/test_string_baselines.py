"""Text scoring behavior verified against Ragas ec00339."""

import pytest

from tallyjudge import Dataset, Evaluator, create_metric
from tallyjudge.registry import registry

EXACT = "metric.text.exact_match"
PRESENCE = "metric.text.string_presence"


@pytest.fixture(params=[EXACT, PRESENCE])
def metric(request):
    return create_metric(request.param)


def test_text_metric_revisions():
    assert [registry.describe(key).revision for key in (EXACT, PRESENCE)] == ["1", "1"]


@pytest.mark.parametrize(
    "reference,response,exact,presence",
    [
        ("Paris", "Paris", 1.0, 1.0),
        ("Paris", "Lyon", 0.0, 0.0),
        ("cat", "the cat sat", 0.0, 1.0),
        ("Cat", "cat", 0.0, 0.0),
        (" cat", "cat", 0.0, 0.0),
        ("", "", 1.0, 1.0),
        ("", "text", 0.0, 1.0),
        ("text", "", 0.0, 0.0),
        ("北京", "我在北京", 0.0, 1.0),
        ("a\nb", "a\nb", 1.0, 1.0),
    ],
)
def test_upstream_text_scores(metric, reference, response, exact, presence):
    dataset = Dataset.from_records([{"reference": reference, "response": response}])
    score = Evaluator([metric]).run(dataset).rows[0].metrics[metric.spec.id]
    assert score.status == "success"
    assert type(score.value) is float
    assert score.value == (exact if metric.spec.id == EXACT else presence)


@pytest.mark.parametrize(
    "record,field",
    [
        ({"reference": 1, "response": "1"}, "reference"),
        ({"reference": 1, "response": 1}, "reference"),
        ({"reference": True, "response": True}, "reference"),
        ({"reference": ["a"], "response": ["a"]}, "reference"),
        ({"reference": None, "response": None}, "reference"),
        ({"response": "a"}, "reference"),
        ({"reference": "a"}, "response"),
    ],
)
def test_migration_rejects_missing_or_nonstring_fields(metric, record, field):
    dataset = Dataset.from_records([record])
    score = Evaluator([metric]).run(dataset).rows[0].metrics[metric.spec.id]
    assert (score.status, score.value) == ("error", None)
    assert (score.error.category, score.error.field) == ("validation_error", field)

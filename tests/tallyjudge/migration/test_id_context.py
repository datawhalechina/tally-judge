"""ID overlap scores captured from Ragas ec00339 (Apache-2.0)."""

import pytest

from tallyjudge import Dataset, Evaluator, create_metric

PRECISION = "metric.retrieval.id_context_precision"
RECALL = "metric.retrieval.id_context_recall"


@pytest.mark.parametrize(
    ("retrieved", "reference", "precision", "recall"),
    [
        (["B", "A", "A"], ["A", "C", "D"], 1 / 2, 1 / 3),
        (["A", "B", "A"], ["D", "A", "C", "C"], 1 / 2, 1 / 3),
        ([1, "1", "01", 2], ["1", 2, "2"], 2 / 3, 1.0),
        (["A"], ["B"], 0.0, 0.0),
        ([], ["A"], "no_retrieved_ids", 0.0),
        (["A"], [], 0.0, "no_reference_ids"),
        ([], [], "no_retrieved_ids", "no_reference_ids"),
    ],
)
def test_archived_id_scores(retrieved, reference, precision, recall):
    row = {"retrieved_context_ids": retrieved, "reference_context_ids": reference}
    run = Evaluator([create_metric(PRECISION), create_metric(RECALL)]).run(
        Dataset.from_records([row])
    )
    for key, expected in ((PRECISION, precision), (RECALL, recall)):
        score = run.rows[0].metrics[key]
        if isinstance(expected, str):
            assert (
                score.status,
                score.value,
                score.error.category,
                score.error.code,
            ) == ("undefined", None, "undefined_score", expected)
        else:
            assert (score.status, score.value) == ("success", pytest.approx(expected))
        assert run.summary[key]["coverage"] == float(not isinstance(expected, str))

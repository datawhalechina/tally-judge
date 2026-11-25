from types import SimpleNamespace

import pytest

from tallyjudge import Dataset, Evaluator, create_metric
from tallyjudge.registry import registry
from tallyjudge.sdk.errors import PluginError

KEY = "metric.text.semantic_similarity"


class Embeddings:
    plugin_id = "embedding.test.deterministic"
    usage_tracking = "untracked"

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def config_snapshot(self):
        return {}

    async def aembed_texts(self, texts):
        self.calls.append(list(texts))
        return SimpleNamespace(vectors=next(self.outputs))


@pytest.mark.parametrize("threshold,bounds", [(None, (-1.0, 1.0)), (0.0, (0.0, 1.0))])
def test_semantic_threshold_changes_described_and_archived_range(threshold, bounds):
    embeddings = Embeddings([[[1e300, 1e300]], [[-1e300, -1e300]]])
    metric = create_metric(KEY, threshold=threshold, embeddings=embeddings)
    assert (
        registry.describe(KEY, threshold=threshold).value_range
        == metric.spec.value_range
        == bounds
    )
    run = Evaluator([metric]).run(
        Dataset.from_records([{"reference": "", "response": "b"}])
    )
    assert run.rows[0].metrics[KEY].value == pytest.approx(
        -1.0 if threshold is None else 0.0
    )
    assert run.metrics[KEY]["config_snapshot"]["threshold"] == threshold
    assert embeddings.calls == [[" "], ["b"]]


@pytest.mark.parametrize("malformed", [[], [[]], [[float("inf")]], [[1.0], [2.0]]])
def test_malformed_embedding_is_a_model_output_error(malformed):
    metric = create_metric(KEY, embeddings=Embeddings([[[1.0]], malformed]))
    run = Evaluator([metric]).run(
        Dataset.from_records([{"reference": "r", "response": "a"}])
    )
    error = run.rows[0].metrics[KEY].error
    assert (error.category, error.code) == ("model_output_error", "invalid_embedding")


def test_nonfinite_threshold_rejected_when_describing_and_constructing():
    with pytest.raises(PluginError):
        registry.describe(KEY, threshold=float("nan"))
    with pytest.raises(PluginError):
        create_metric(KEY, embeddings=Embeddings([]), threshold=float("nan"))

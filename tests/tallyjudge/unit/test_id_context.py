"""Public inputs and model-free registration of the document ID metrics."""

import pytest

from tallyjudge import Dataset, Evaluator, MetricBinding, create_metric
from tallyjudge.core.errors import InputValidationError
from tallyjudge.registry import registry

IDS = ("metric.retrieval.id_context_precision", "metric.retrieval.id_context_recall")


@pytest.mark.parametrize("plugin_id", IDS)
@pytest.mark.parametrize("invalid", [None, "A", [True], [1.5], ["A", None]])
@pytest.mark.asyncio
async def test_id_fields_use_strict_types_in_direct_and_bound_calls(plugin_id, invalid):
    metric = create_metric(plugin_id)
    fields = {"retrieved_context_ids": invalid, "reference_context_ids": ["A"]}
    with pytest.raises(InputValidationError) as failure:
        await metric.ascore(**fields)
    assert failure.value.field == "retrieved_context_ids"
    run = await Evaluator([metric]).arun(Dataset.from_records([fields]))
    score = run.rows[0].metrics[plugin_id]
    assert (score.status, score.error.field) == ("error", "retrieved_context_ids")


def test_id_metrics_use_remapped_columns_without_model_requests():
    bindings = [
        MetricBinding(
            create_metric(key),
            input_map={
                "retrieved_context_ids": "hits",
                "reference_context_ids": "gold",
            },
        )
        for key in IDS
    ]
    run = Evaluator(bindings).run(
        Dataset.from_records([{"hits": [1, 2], "gold": ["2"]}, {"hits": [1]}])
    )
    for key, expected in zip(IDS, (0.5, 1.0), strict=True):
        assert registry.describe(key).revision == "1"
        assert run.rows[0].metrics[key].value == expected
        assert run.rows[1].metrics[key].error.field == "reference_context_ids"
    assert run.usage is None

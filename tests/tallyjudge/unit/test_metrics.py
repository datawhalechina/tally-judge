import pytest

from tallyjudge import Dataset, Evaluator, MetricResult
from tallyjudge.plugins.metrics.text.exact_match import ExactMatchMetric
from tallyjudge.plugins.metrics.text.string_presence import StringPresenceMetric
from tallyjudge.sdk.errors import InputValidationError
from tallyjudge.sdk.metrics import MetricBase, MetricSpec


def test_sync_score_and_empty_snapshot():
    for metric in (ExactMatchMetric(), StringPresenceMetric()):
        assert metric.score(reference="a", response="a").value == 1.0
        assert metric.config_snapshot() == {}


@pytest.mark.asyncio
async def test_named_async_and_validation():
    for metric in (ExactMatchMetric(), StringPresenceMetric()):
        assert (await metric.ascore(reference="a", response="a")).value == 1.0
        for inputs in (
            {"reference": 1, "response": "1"},
            {"reference": "a"},
            {"reference": "a", "response": "a", "extra": "b"},
        ):
            with pytest.raises(InputValidationError):
                await metric.ascore_inputs(inputs)
        with pytest.raises(RuntimeError, match="await ascore_inputs"):
            metric.score(reference="a", response="a")


@pytest.mark.asyncio
async def test_generic_inputs_validate_nested_values_strictly():
    class ContextMetric(MetricBase):
        spec = MetricSpec(
            "metric.test.contexts",
            "1",
            {"retrieved_contexts": list[str], "options": dict[str, list[int]]},
        )

        async def ascore(self, retrieved_contexts, options):
            return MetricResult(float(len(retrieved_contexts)))

    metric = ContextMetric()
    records = [
        {"retrieved_contexts": ["a", "b"], "options": {"ids": [1, 2]}},
        {"retrieved_contexts": "a", "options": {"ids": [1]}},
        {"retrieved_contexts": ["a", 1], "options": {"ids": [1]}},
        {"retrieved_contexts": ["a"], "options": {"ids": ["1"]}},
    ]
    result = await Evaluator([metric]).arun(Dataset.from_records(records))
    assert [row.metrics[metric.spec.id].status for row in result.rows] == [
        "success",
        "error",
        "error",
        "error",
    ]
    assert result.rows[0].metrics[metric.spec.id].value == 2.0
    with pytest.raises(InputValidationError):
        await metric.ascore_inputs(
            {"retrieved_contexts": ("a",), "options": {"ids": [1]}}
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value,valid",
    [
        (1.0, True),
        (True, False),
        (1, False),
    ],
)
async def test_float_subclass_is_normalized_but_non_floats_are_rejected(value, valid):
    class FloatSubclass(float):
        pass

    class OutputMetric(MetricBase):
        spec = MetricSpec("metric.test.output", "1", {})

        async def ascore(self):
            return MetricResult(
                FloatSubclass(value) if isinstance(value, float) else value
            )

    score = (
        (await Evaluator([OutputMetric()]).arun(Dataset.from_records([{}])))
        .rows[0]
        .metrics["metric.test.output"]
    )
    assert score.status == ("success" if valid else "invalid_output")
    if valid:
        assert type(score.value) is float


@pytest.mark.asyncio
async def test_numpy_float64_normalization():
    np = pytest.importorskip("numpy")

    class NumpyMetric(MetricBase):
        spec = MetricSpec("metric.test.numpy", "1", {})

        async def ascore(self):
            return MetricResult(np.float64(0.5))

    result = await Evaluator([NumpyMetric()]).arun(Dataset.from_records([{}]))
    score = result.rows[0].metrics["metric.test.numpy"]
    assert score.status == "success"
    assert type(score.value) is float
    assert score.value == 0.5
    assert result.to_json()


@pytest.mark.asyncio
async def test_validated_values_reach_direct_and_bound_metrics_once():
    from typing import Annotated

    from pydantic import AfterValidator

    seen = []

    def normalize(value: float) -> float:
        seen.append(value)
        return value + 0.5

    number_spec = MetricSpec(
        "metric.test.number",
        "1",
        {"number": Annotated[float, AfterValidator(normalize)]},
    )

    class NamedMetric(MetricBase):
        spec = number_spec

        async def ascore(self, number: float):
            return MetricResult(float.fromhex(number.hex()))

    class MappingMetric:
        spec = number_spec

        async def ascore_inputs(self, inputs):
            number = inputs["number"]
            return MetricResult(float.fromhex(number.hex()))

        def config_snapshot(self):
            return {}

    assert (await NamedMetric().ascore_inputs({"number": 1})).value == 1.5
    for metric in (NamedMetric(), MappingMetric()):
        result = await Evaluator([metric]).arun(
            Dataset.from_records([{"number": 1}, {"number": "1"}])
        )
        valid, invalid = [row.metrics[number_spec.id] for row in result.rows]
        assert (valid.status, valid.value) == ("success", 1.5)
        assert (invalid.status, invalid.error.category) == ("error", "validation_error")
    assert seen == [1.0, 1.0, 1.0]

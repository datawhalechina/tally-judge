import pytest
from pydantic import BaseModel

from tallyjudge import Dataset, Evaluator, create_metric
from tallyjudge.plugins.metrics.custom.judge import NumericMetric
from tallyjudge.registry import PluginRegistry, PluginSpec
from tallyjudge.sdk import (
    GenerationResult,
    MetricBase,
    MetricResult,
    MetricSpec,
    PromptInput,
    UsageSummary,
)


class Verdict(BaseModel):
    score: float


class JudgedMetric(MetricBase):
    spec = MetricSpec("metric.example.judged", "1", {"response": str})

    def __init__(self, llm):
        self.llm = llm

    async def ascore(self, response):
        reply = await self.llm.agenerate(PromptInput.from_text(response), Verdict)
        return MetricResult(reply.data.score)


class FixedModel:
    plugin_id = "model.example.fixed"
    usage_tracking = "untracked"

    def config_snapshot(self):
        return {}

    async def agenerate(self, prompt, response_model):
        return GenerationResult(response_model(score=1.0), UsageSummary())


@pytest.mark.asyncio
async def test_sdk_only_metric_reports_its_untracked_model():
    registry = PluginRegistry(
        [
            PluginSpec(
                JudgedMetric.spec.id,
                "metric",
                1,
                JudgedMetric,
                dependencies={"llm": "language_model"},
            )
        ]
    )
    metric = registry.create(JudgedMetric.spec.id, llm=FixedModel())
    run = await Evaluator([metric]).arun(Dataset.from_records([{"response": "ok"}]))
    score = run.rows[0].metrics[JudgedMetric.spec.id]
    assert score.value == 1.0
    assert score.usage.untracked_sources == ["model.example.fixed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("through_registry", [True, False])
async def test_builtin_judge_reports_an_untracked_model_however_it_is_built(
    through_registry,
):
    class Voter(FixedModel):
        async def agenerate(self, prompt, response_model):
            return GenerationResult(
                response_model(value=1.0, reason="r"), UsageSummary()
            )

    settings = {"llm": Voter(), "criteria": "Is it complete?"}
    metric = (
        create_metric("metric.custom.numeric", **settings)
        if through_registry
        else NumericMetric(**settings)
    )
    run = await Evaluator([metric]).arun(Dataset.from_records([{"response": "ok"}]))
    usage = run.rows[0].metrics[metric.spec.id].usage
    assert usage.untracked_sources == ["model.example.fixed"]
    assert not usage.generation.complete

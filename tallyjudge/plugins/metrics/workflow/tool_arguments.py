from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, TypeAdapter

from tallyjudge.plugins.metrics.workflow._trace import Trace, tool_calls
from tallyjudge.sdk.errors import UndefinedScore
from tallyjudge.sdk.metrics import MetricBase, MetricSpec
from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.results import MetricResult


class ExpectedCall(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(min_length=1)
    arguments: dict[str, Any]


_CALLS = TypeAdapter(list[ExpectedCall])


def _calls(value: list) -> list[dict[str, Any]]:
    return [call.model_dump() for call in _CALLS.validate_python(value)]


class ToolArgumentsMetric(MetricBase):
    plugin = Plugin("metric")
    spec = MetricSpec(
        "metric.workflow.tool_arguments",
        "1",
        {"trace": Trace, "expected_calls": Annotated[list, AfterValidator(_calls)]},
        value_range=(0.0, 1.0),
    )

    async def ascore(
        self, trace: dict, expected_calls: list[dict[str, Any]]
    ) -> MetricResult:
        if not expected_calls:
            raise UndefinedScore("no_expected_calls")
        recorded = tool_calls(trace)
        if any(step["arguments"] is None for step in recorded):
            raise UndefinedScore("arguments_not_recorded")
        remaining = [(step["name"], step["arguments"]) for step in recorded]
        unmatched = []
        for call in expected_calls:
            wanted = (call["name"], call["arguments"])
            if wanted in remaining:
                remaining.remove(wanted)
            else:
                unmatched.append(call["name"])
        return MetricResult(
            (len(expected_calls) - len(unmatched)) / len(expected_calls),
            details={"unmatched": unmatched},
        )

from tallyjudge.plugins.metrics.workflow._trace import EXPECTED_TOOLS, tool_calls
from tallyjudge.sdk.metrics import MetricBase, MetricSpec
from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.results import MetricResult


class ToolSequenceMetric(MetricBase):
    plugin = Plugin("metric")

    spec = MetricSpec(
        "metric.workflow.tool_sequence",
        "1",
        EXPECTED_TOOLS,
        value_range=(0.0, 1.0),
    )

    async def ascore(self, trace: dict, expected_tools: list[str]) -> MetricResult:
        tools = [step["name"] for step in tool_calls(trace)]
        return MetricResult(float(tools == expected_tools))

from tallyjudge.plugins.metrics.workflow._trace import EXPECTED_TOOLS, tool_calls
from tallyjudge.sdk.errors import UndefinedScore
from tallyjudge.sdk.metrics import MetricBase, MetricSpec
from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.results import MetricResult


class ToolRecallMetric(MetricBase):
    plugin = Plugin("metric")
    spec = MetricSpec(
        "metric.workflow.tool_recall", "1", EXPECTED_TOOLS, value_range=(0.0, 1.0)
    )

    async def ascore(self, trace: dict, expected_tools: list[str]) -> MetricResult:
        expected = set(expected_tools)
        if not expected:
            raise UndefinedScore("no_expected_tools")
        used = {step["name"] for step in tool_calls(trace)}
        missing = sorted(expected - used)
        return MetricResult(
            len(expected & used) / len(expected), details={"missing": missing}
        )


class ToolPrecisionMetric(MetricBase):
    plugin = Plugin("metric")
    spec = MetricSpec(
        "metric.workflow.tool_precision", "1", EXPECTED_TOOLS, value_range=(0.0, 1.0)
    )

    async def ascore(self, trace: dict, expected_tools: list[str]) -> MetricResult:
        calls = [step["name"] for step in tool_calls(trace)]
        if not calls:
            raise UndefinedScore("no_tool_calls")
        unexpected = [name for name in calls if name not in expected_tools]
        return MetricResult(
            (len(calls) - len(unexpected)) / len(calls),
            details={"unexpected": unexpected},
        )

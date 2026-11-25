from tallyjudge.plugins.metrics.workflow._trace import EXPECTED_TOOLS, tool_attempts
from tallyjudge.sdk.errors import UndefinedScore
from tallyjudge.sdk.metrics import MetricBase, MetricSpec
from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.results import MetricResult


class StepEfficiencyMetric(MetricBase):
    plugin = Plugin("metric")
    spec = MetricSpec(
        "metric.workflow.step_efficiency",
        "1",
        EXPECTED_TOOLS,
        value_range=(0.0, 1.0),
    )

    async def ascore(self, trace: dict, expected_tools: list[str]) -> MetricResult:
        attempts = len(tool_attempts(trace))
        if not attempts:
            raise UndefinedScore("no_tool_attempts")
        return MetricResult(
            min(1.0, len(expected_tools) / attempts), details={"attempts": attempts}
        )

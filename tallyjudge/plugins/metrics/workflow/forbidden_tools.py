from tallyjudge.plugins.metrics.workflow._trace import Trace, tool_attempts
from tallyjudge.sdk.metrics import MetricBase, MetricSpec
from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.results import MetricResult


class ForbiddenToolsMetric(MetricBase):
    plugin = Plugin("metric")
    spec = MetricSpec(
        "metric.workflow.forbidden_tools",
        "1",
        {"trace": Trace, "forbidden_tools": list[str]},
        output_type=str,
        labels=("compliant", "violated"),
    )

    async def ascore(self, trace: dict, forbidden_tools: list[str]) -> MetricResult:
        attempted = sorted(
            {step["name"] for step in tool_attempts(trace)} & set(forbidden_tools)
        )
        return MetricResult(
            "violated" if attempted else "compliant", details={"attempted": attempted}
        )

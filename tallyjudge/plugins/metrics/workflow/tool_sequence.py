"""Exact ordered tool execution from validated neutral workflow traces."""

from typing import Annotated

from pydantic import AfterValidator

from tallyjudge.core.errors import UndefinedScore
from tallyjudge.core.results import MetricResult
from tallyjudge.interfaces.metrics import MetricBase, MetricSpec
from tallyjudge.interfaces.workflow import WorkflowTrace


def _trace_data(value: dict) -> dict:
    return WorkflowTrace.model_validate(value).model_dump()


class ToolSequenceMetric(MetricBase):
    """Compare the names of successful tool attempts in start order."""

    spec = MetricSpec(
        "metric.workflow.tool_sequence",
        "1",
        {
            "trace": Annotated[dict, AfterValidator(_trace_data)],
            "expected_tools": list[str],
        },
        value_range=(0.0, 1.0),
    )

    async def ascore(self, trace: dict, expected_tools: list[str]) -> MetricResult:
        """Score a recorded workflow and reject unfinished attempts as undefined."""
        return await self.ascore_inputs(
            {"trace": trace, "expected_tools": expected_tools}
        )

    async def _ascore_validated(
        self, trace: dict, expected_tools: list[str]
    ) -> MetricResult:
        if any(
            step["status"] in ("incomplete", "cancelled") for step in trace["steps"]
        ):
            raise UndefinedScore("incomplete_trace")
        tools = [
            step["name"]
            for step in trace["steps"]
            if step["kind"] == "tool" and step["status"] == "success"
        ]
        return MetricResult(float(tools == expected_tools))

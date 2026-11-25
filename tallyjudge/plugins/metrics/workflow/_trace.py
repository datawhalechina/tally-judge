from typing import Annotated, Any

from pydantic import AfterValidator

from tallyjudge.sdk.errors import UndefinedScore
from tallyjudge.sdk.workflow import WorkflowTrace


def _trace_data(value: dict) -> dict:
    return WorkflowTrace.model_validate(value).model_dump()


Trace = Annotated[dict, AfterValidator(_trace_data)]
EXPECTED_TOOLS = {"trace": Trace, "expected_tools": list[str]}


def finished(trace: dict[str, Any]) -> list[dict[str, Any]]:
    steps = trace["steps"]
    if any(step["status"] in ("incomplete", "cancelled") for step in steps):
        raise UndefinedScore("incomplete_trace")
    return steps


def tool_attempts(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [step for step in finished(trace) if step["kind"] == "tool"]


def tool_calls(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [step for step in tool_attempts(trace) if step["status"] == "success"]

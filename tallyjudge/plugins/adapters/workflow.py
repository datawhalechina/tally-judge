"""Convert LangChain/LangGraph v2 stream events into neutral workflow attempts."""

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from tallyjudge.interfaces.workflow import WorkflowStep, WorkflowTrace


def langchain_trace(
    events: Iterable[Mapping[str, Any]], *, run_id: str, include_content: bool = False
) -> WorkflowTrace:
    """Pair tool and chain events by invocation ID, retaining failed and unfinished attempts."""
    steps: dict[str, WorkflowStep] = {}
    kinds: dict[str, Literal["workflow", "tool", "model"]] = {
        "chain": "workflow",
        "tool": "tool",
        "chat_model": "model",
        "llm": "model",
    }
    stream_root = None
    for event in events:
        label = event["event"]
        if not label.startswith("on_"):
            continue
        source, _, action = label[3:].rpartition("_")
        if source not in kinds or action not in ("start", "end", "error"):
            continue
        key = str(event["run_id"])
        ancestors = event.get("parent_ids", ())
        root = str(ancestors[0]) if ancestors else key
        if stream_root is not None and stream_root != root:
            raise ValueError("Workflow events belong to different root runs")
        stream_root = root
        data = event.get("data", {})
        if action == "start":
            if key in steps:
                raise ValueError("Duplicate start for workflow attempt")
            parent = next(
                (str(p) for p in reversed(ancestors) if str(p) in steps),
                None,
            )
            args = data.get("input")
            call = event.get("metadata", {}).get("tool_call_id")
            if isinstance(args, dict) and args.get("type") == "tool_call":
                call, args = args.get("id"), args.get("args")
            steps[key] = WorkflowStep(
                step_id=key,
                parent_id=parent,
                name=event["name"],
                kind=kinds[source],
                call_id=call,
                status="incomplete",
                arguments=args
                if source == "tool" and include_content and isinstance(args, dict)
                else None,
            )
            continue
        step = steps[key]
        if step.status != "incomplete":
            raise ValueError("Duplicate terminal event for workflow attempt")
        output = data.get("output")
        step.status = "error" if action == "error" else "success"
        if source == "tool":
            if isinstance(output, Mapping):
                message = (
                    output
                    if output.get("type") == "tool" and "tool_call_id" in output
                    else None
                )
            elif getattr(output, "type", None) == "tool":
                message = output.model_dump(
                    include={"tool_call_id", "status", "content"}
                )
            else:
                message = None
            if message is not None:
                call = message["tool_call_id"]
                if step.call_id and call != step.call_id:
                    raise ValueError("Tool result belongs to a different call")
                step.call_id = call
                if message.get("status") == "error":
                    step.status = "error"
                output = message["content"]
            if include_content:
                step.result = output
    return WorkflowTrace(run_id=run_id, steps=list(steps.values()))

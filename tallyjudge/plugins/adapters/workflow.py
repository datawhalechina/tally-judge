from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.workflow import WorkflowStep, WorkflowTrace


class LangChainTraceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    include_content: bool = False
    run_id: str | None = None


class LangChainTrace:
    plugin = Plugin("importer", produces="trace")

    Config = LangChainTraceConfig
    plugin_id = "importer.langchain.trace"

    def __init__(self, include_content: bool = False, run_id: str | None = None):
        config = LangChainTraceConfig(include_content=include_content, run_id=run_id)
        self.include_content = config.include_content
        self.run_id = config.run_id

    def convert(self, source: Iterable[Mapping[str, Any]]) -> WorkflowTrace:
        steps: dict[str, WorkflowStep] = {}
        kinds: dict[str, Literal["workflow", "tool", "model"]] = {
            "chain": "workflow",
            "tool": "tool",
            "chat_model": "model",
            "llm": "model",
        }
        stream_root = None
        for event in source:
            label = event["event"]
            if not label.startswith("on_"):
                continue
            subject, _, action = label[3:].rpartition("_")
            if subject not in kinds or action not in ("start", "end", "error"):
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
                metadata = event.get("metadata", {})
                call = metadata.get("tool_call_id")
                if isinstance(args, dict) and args.get("type") == "tool_call":
                    call, args = args.get("id"), args.get("args")
                steps[key] = WorkflowStep(
                    step_id=key,
                    parent_id=parent,
                    name=event["name"],
                    kind=kinds[subject],
                    call_id=call,
                    status="incomplete",
                    attributes={"langgraph.node": metadata["langgraph_node"]}
                    if "langgraph_node" in metadata
                    else {},
                    arguments=args
                    if subject == "tool"
                    and self.include_content
                    and isinstance(args, dict)
                    else None,
                )
                continue
            step = steps[key]
            if step.status != "incomplete":
                raise ValueError("Duplicate terminal event for workflow attempt")
            output = data.get("output")
            step.status = "error" if action == "error" else "success"
            if subject == "tool":
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
                if self.include_content:
                    step.result = output
        return WorkflowTrace(
            run_id=self.run_id or str(stream_root), steps=list(steps.values())
        )

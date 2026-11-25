import json
from collections.abc import Collection
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.workflow import WorkflowStep, WorkflowTrace

_KINDS: dict[str, Literal["workflow", "tool", "model"]] = {
    "TOOL": "tool",
    "LLM": "model",
}
_ERROR = (2, "STATUS_CODE_ERROR")


def _attributes(span: dict[str, Any]) -> dict[str, Any]:
    values = {}
    for item in span.get("attributes", ()):
        value = item["value"]
        if "intValue" in value:
            values[item["key"]] = int(value["intValue"])
        else:
            for name in ("stringValue", "doubleValue", "boolValue"):
                if name in value:
                    values[item["key"]] = value[name]
    return values


class OpenInferenceTraceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    include_content: bool = False
    run_id: str | None = None


class OpenInferenceTrace:
    plugin = Plugin("importer", produces="trace")
    plugin_id = "importer.otlp.openinference"
    Config = OpenInferenceTraceConfig

    def __init__(self, include_content: bool = False, run_id: str | None = None):
        config = OpenInferenceTraceConfig(
            include_content=include_content, run_id=run_id
        )
        self.include_content = config.include_content
        self.run_id = config.run_id

    def _step(self, span: dict[str, Any], known: Collection[str]) -> WorkflowStep:
        attributes = _attributes(span)
        kind = attributes.get("openinference.span.kind", "UNKNOWN")
        arguments = None
        if self.include_content and kind == "TOOL" and "input.value" in attributes:
            parsed = json.loads(attributes["input.value"])
            arguments = parsed if isinstance(parsed, dict) else None
        parent = span.get("parentSpanId")
        return WorkflowStep(
            step_id=span["spanId"],
            parent_id=parent if parent in known else None,
            name=attributes.get("tool.name", span["name"]),
            kind=_KINDS.get(kind, "workflow"),
            status="error"
            if span.get("status", {}).get("code") in _ERROR
            else "success",
            arguments=arguments,
            result=attributes.get("output.value") if self.include_content else None,
            agent_id=attributes.get("agent.name"),
            started_at=int(span["startTimeUnixNano"]) / 1e9,
            ended_at=int(span["endTimeUnixNano"]) / 1e9,
            input_tokens=attributes.get("llm.token_count.prompt"),
            output_tokens=attributes.get("llm.token_count.completion"),
            attributes={
                key: attributes[key]
                for key in ("openinference.span.kind", "graph.node.id")
                if key in attributes
            },
        )

    def convert(self, source: dict[str, Any]) -> WorkflowTrace:
        spans = [
            span
            for resource in source["resourceSpans"]
            for scope in resource["scopeSpans"]
            for span in scope["spans"]
        ]
        traces = {span["traceId"] for span in spans}
        if len(traces) != 1:
            raise ValueError("The export must contain the spans of exactly one trace")
        by_id = {span["spanId"]: span for span in spans}
        depth: dict[str, int] = {}

        def level(span: dict[str, Any]) -> int:
            key = span["spanId"]
            if key not in depth:
                parent = by_id.get(span.get("parentSpanId"))
                depth[key] = 0 if parent is None else level(parent) + 1
            return depth[key]

        spans.sort(key=lambda span: (int(span["startTimeUnixNano"]), level(span)))
        root = next(span for span in spans if level(span) == 0)
        return WorkflowTrace(
            run_id=self.run_id or traces.pop(),
            steps=[self._step(span, by_id.keys()) for span in spans],
            output=_attributes(root).get("output.value")
            if self.include_content
            else None,
        )

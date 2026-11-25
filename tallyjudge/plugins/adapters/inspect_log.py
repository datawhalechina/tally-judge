from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from tallyjudge.plugins.adapters._source import json_document, seconds, text
from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.workflow import WorkflowStep, WorkflowTrace


class InspectLogConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    include_content: bool = False


class InspectLog:
    plugin = Plugin("importer", produces="records")
    plugin_id = "importer.inspect.log"
    Config = InspectLogConfig

    def __init__(self, include_content: bool = False):
        self.include_content = InspectLogConfig(
            include_content=include_content
        ).include_content

    def _step(self, index: int, event: dict[str, Any]) -> WorkflowStep:
        failed = event.get("error") is not None
        fields: dict[str, Any] = {
            "step_id": str(index),
            "kind": event["event"],
            "status": "error" if failed else "success",
            "started_at": seconds(event["timestamp"]),
            "ended_at": seconds(event.get("completed")),
        }
        if event["event"] == "model":
            usage = event["output"].get("usage") or {}
            fields |= {
                "name": event["model"],
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
            }
        else:
            fields |= {
                "name": event["function"],
                "call_id": event["id"],
                "agent_id": event.get("agent"),
            }
            if failed:
                fields["attributes"] = {"inspect.error_type": event["error"]["type"]}
            if self.include_content:
                fields |= {"arguments": event["arguments"], "result": event["result"]}
        return WorkflowStep(**fields)

    def _record(self, sample: dict[str, Any]) -> dict[str, Any]:
        epoch = sample["epoch"]
        case_id = str(sample["id"]) if epoch == 1 else f"{sample['id']}#{epoch}"
        calls = sorted(
            (
                event
                for event in sample["events"]
                if event["event"] in ("model", "tool")
            ),
            key=lambda event: datetime.fromisoformat(event["timestamp"]),
        )
        response = sample["output"]["completion"]
        trace = WorkflowTrace(
            run_id=sample.get("uuid") or case_id,
            steps=[self._step(index, event) for index, event in enumerate(calls)],
            output=response if self.include_content else None,
        )
        prompt = sample["input"]
        if not isinstance(prompt, str):
            prompt = text(
                next(m for m in reversed(prompt) if m["role"] == "user")["content"]
            )
        target = sample["target"]
        return {
            "case_id": case_id,
            "user_input": prompt,
            "reference" if isinstance(target, str) else "references": target,
            "response": response,
            "final_state": sample["store"],
            "trace": trace.model_dump(),
            "inspect_scores": {
                name: score["value"]
                for name, score in (sample.get("scores") or {}).items()
            },
        }

    def convert(self, source: dict[str, Any] | str | Path) -> list[dict[str, Any]]:
        return [self._record(sample) for sample in json_document(source)["samples"]]

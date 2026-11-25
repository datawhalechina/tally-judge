from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from tallyjudge.plugins.adapters._source import json_document, seconds, text
from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.workflow import WorkflowStep, WorkflowTrace


class AtifTrajectoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    include_content: bool = False
    run_id: str | None = None


class AtifTrajectory:
    plugin = Plugin("importer", produces="trace")
    plugin_id = "importer.atif.trajectory"
    Config = AtifTrajectoryConfig

    def __init__(self, include_content: bool = False, run_id: str | None = None):
        config = AtifTrajectoryConfig(include_content=include_content, run_id=run_id)
        self.include_content = config.include_content
        self.run_id = config.run_id

    def convert(self, source: dict[str, Any] | str | Path) -> WorkflowTrace:
        document = json_document(source)
        agent = document["agent"]
        steps: list[WorkflowStep] = []
        output = None
        for turn in document["steps"]:
            if turn["source"] != "agent" or turn.get("is_copied_context"):
                continue
            output = text(turn["message"])
            parent = None
            if turn.get("llm_call_count") != 0:
                parent = str(turn["step_id"])
                usage = turn.get("metrics") or {}
                steps.append(
                    WorkflowStep(
                        step_id=parent,
                        name=turn.get("model_name")
                        or agent.get("model_name")
                        or agent["name"],
                        kind="model",
                        status="success",
                        agent_id=agent["name"],
                        started_at=seconds(turn.get("timestamp")),
                        input_tokens=usage.get("prompt_tokens"),
                        output_tokens=usage.get("completion_tokens"),
                    )
                )
            observed = (turn.get("observation") or {}).get("results", ())
            results = {item["source_call_id"]: item.get("content") for item in observed}
            for call in turn.get("tool_calls") or ():
                key = call["tool_call_id"]
                steps.append(
                    WorkflowStep(
                        step_id=f"{turn['step_id']}.{key}",
                        parent_id=parent,
                        call_id=key,
                        name=call["function_name"],
                        kind="tool",
                        status="success",
                        agent_id=agent["name"],
                        arguments=call["arguments"] if self.include_content else None,
                        result=results.get(key) if self.include_content else None,
                    )
                )
        run_id = (
            self.run_id or document.get("trajectory_id") or document.get("session_id")
        )
        if run_id is None:
            raise ValueError(
                "The trajectory has no trajectory_id or session_id; pass run_id"
            )
        return WorkflowTrace(
            run_id=run_id,
            steps=steps,
            output=output if self.include_content else None,
            attributes={"atif.schema_version": document["schema_version"]},
        )

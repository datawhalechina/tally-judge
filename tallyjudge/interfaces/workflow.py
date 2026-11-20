"""Ordered application traces for evaluating workflow execution."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WorkflowStep(BaseModel):
    """One invocation, ordered by start and identified independently of its retries."""

    model_config = ConfigDict(extra="forbid", strict=True)
    step_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    kind: Literal["workflow", "tool", "model"]
    status: Literal["success", "error", "skipped", "cancelled", "incomplete"]
    parent_id: str | None = None
    call_id: str | None = None
    arguments: dict[str, Any] | None = None
    result: Any = None


class WorkflowTrace(BaseModel):
    """A single workflow run with parent-before-child invocation order."""

    model_config = ConfigDict(extra="forbid", strict=True)
    run_id: str = Field(min_length=1)
    steps: list[WorkflowStep]

    @model_validator(mode="after")
    def validate_tree(self):
        """Reject ambiguous attempts or parent links in a recorded trace."""
        seen = set()
        for step in self.steps:
            if step.step_id in seen or (
                step.parent_id is not None and step.parent_id not in seen
            ):
                raise ValueError("Step IDs must be unique and follow their parents")
            seen.add(step.step_id)
        return self

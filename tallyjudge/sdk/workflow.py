from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

type Attributes = dict[Annotated[str, Field(pattern=r"^[a-z0-9_]+\.[a-z0-9_.]+$")], Any]


class WorkflowStep(BaseModel):
    """One invocation, ordered by start and identified independently of its retries.

    Times are seconds on the recorder's clock. ``attributes`` carries framework-specific
    facts under ``namespace.key`` names; TallyJudge stores them and never interprets them.
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    step_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    kind: Literal["workflow", "tool", "model"]
    status: Literal["success", "error", "skipped", "cancelled", "incomplete"]
    parent_id: str | None = None
    call_id: str | None = None
    arguments: dict[str, Any] | None = None
    result: Any = None
    agent_id: str | None = None
    started_at: float | None = None
    ended_at: float | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    attributes: Attributes = {}


class WorkflowTrace(BaseModel):
    """A single workflow run with parent-before-child invocation order."""

    model_config = ConfigDict(extra="forbid", strict=True)
    run_id: str = Field(min_length=1)
    steps: list[WorkflowStep]
    output: Any = None
    attributes: Attributes = {}

    @model_validator(mode="after")
    def validate_tree(self):
        seen = set()
        for step in self.steps:
            if step.step_id in seen or (
                step.parent_id is not None and step.parent_id not in seen
            ):
                raise ValueError("Step IDs must be unique and follow their parents")
            seen.add(step.step_id)
        return self

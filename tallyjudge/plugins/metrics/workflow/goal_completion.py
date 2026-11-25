import json
from typing import Any

from pydantic import BaseModel, Field

from tallyjudge.plugins._prompting import StructuredPrompt
from tallyjudge.plugins.metrics._judge import ModelJudge, binary_verdict
from tallyjudge.plugins.metrics.workflow._trace import Trace, tool_calls
from tallyjudge.sdk.errors import UndefinedScore
from tallyjudge.sdk.metrics import MetricSpec
from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.results import MetricResult


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


class ToolEvidence(BaseModel):
    """One successful tool call and what it returned."""

    tool: str = Field(..., description="The name of the tool that was called")
    arguments: str = Field(..., description="The arguments of the call as JSON")
    result: str = Field(..., description="What the tool returned")


class GoalCompletionInput(BaseModel):
    """The goal, the recorded tool results and the workflow's own final statement."""

    goal: str = Field(..., description="What the user asked the workflow to achieve")
    evidence: list[ToolEvidence] = Field(
        ..., description="Successful tool calls in the order they ran"
    )
    final_output: str = Field(
        ..., description="The workflow's final statement to the user"
    )


class GoalCompletionOutput(BaseModel):
    """The verdict and the tool result it rests on."""

    reason: str = Field(..., description="Which tool results decide the verdict")
    verdict: int = Field(
        ..., ge=0, le=1, description="1 if the goal was achieved, else 0"
    )


class GoalCompletionPrompt(StructuredPrompt[GoalCompletionInput, GoalCompletionOutput]):
    input_model = GoalCompletionInput
    output_model = GoalCompletionOutput

    instruction = """Decide whether a workflow achieved the user's goal. Base the verdict only on what the tool calls returned. The final output is the workflow's own claim: return 0 when the tool results do not confirm it, even if it says the goal was achieved."""

    examples = [
        (
            GoalCompletionInput(
                goal="Refund order 1042.",
                evidence=[
                    ToolEvidence(
                        tool="lookup_order",
                        arguments='{"order_id": "1042"}',
                        result='{"status": "delivered", "refundable": true}',
                    )
                ],
                final_output="Your refund for order 1042 has been issued.",
            ),
            GoalCompletionOutput(
                reason="Only lookup_order ran. No tool result shows that a refund was created.",
                verdict=0,
            ),
        ),
    ]


class GoalCompletionMetric(ModelJudge):
    plugin = Plugin("metric", dependencies={"llm": "language_model"})
    spec = MetricSpec(
        "metric.workflow.goal_completion",
        "1",
        {"user_input": str, "trace": Trace},
        value_range=(0.0, 1.0),
        levels=(0, 1),
    )

    async def ascore(self, user_input: str, trace: dict) -> MetricResult:
        evidence = [
            ToolEvidence(
                tool=step["name"],
                arguments=_text(step["arguments"]),
                result=_text(step["result"]),
            )
            for step in tool_calls(trace)
            if step["result"] is not None
        ]
        if not evidence:
            raise UndefinedScore("results_not_recorded")
        prompt = GoalCompletionPrompt().render(
            GoalCompletionInput(
                goal=user_input, evidence=evidence, final_output=_text(trace["output"])
            )
        )
        judged = await self._ask(prompt, GoalCompletionOutput)
        return MetricResult(float(binary_verdict(judged.verdict)), judged.reason)

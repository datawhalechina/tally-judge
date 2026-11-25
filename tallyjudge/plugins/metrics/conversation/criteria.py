from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, TypeAdapter

from tallyjudge.plugins._prompting import StructuredPrompt
from tallyjudge.plugins.metrics._judge import ModelJudge, binary_verdict
from tallyjudge.sdk.metrics import MetricSpec
from tallyjudge.sdk.models import LanguageModel
from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.results import MetricResult


class Turn(BaseModel):
    """One message of the conversation."""

    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["system", "user", "assistant", "tool"] = Field(
        ..., description="Who sent the message"
    )
    content: str = Field(..., description="The text of the message")


_TURNS = TypeAdapter(Annotated[list[Turn], Field(min_length=1)])


def _turns(value: list) -> list[dict[str, str]]:
    return [turn.model_dump() for turn in _TURNS.validate_python(value)]


class ConversationInput(BaseModel):
    """A criterion and the conversation to check against it."""

    criterion: str = Field(..., description="What the conversation must satisfy")
    conversation: list[Turn] = Field(..., description="The messages in order")


class ConversationOutput(BaseModel):
    """The verdict and the turns it rests on."""

    reason: str = Field(..., description="Which turns decide the verdict")
    verdict: int = Field(
        ..., ge=0, le=1, description="1 if the criterion is met, else 0"
    )


class ConversationPrompt(StructuredPrompt[ConversationInput, ConversationOutput]):
    input_model = ConversationInput
    output_model = ConversationOutput

    instruction = """Decide whether the whole conversation meets the criterion. Consider every turn, not only the last one. Return verdict 1 if the criterion is met and 0 if it is not."""

    examples = [
        (
            ConversationInput(
                criterion="The assistant confirms the order number before changing the order.",
                conversation=[
                    Turn(role="user", content="Please change my delivery address."),
                    Turn(role="assistant", content="Done. Your address is updated."),
                ],
            ),
            ConversationOutput(
                reason="The assistant changed the address in its first reply without asking for or confirming an order number.",
                verdict=0,
            ),
        ),
    ]


class ConversationCriteriaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    criteria: str = Field(min_length=1)


class ConversationCriteriaMetric(ModelJudge):
    plugin = Plugin("metric", dependencies={"llm": "language_model"})
    Config = ConversationCriteriaConfig
    spec = MetricSpec(
        "metric.conversation.criteria",
        "1",
        {"messages": Annotated[list, AfterValidator(_turns)]},
        value_range=(0.0, 1.0),
        levels=(0, 1),
    )

    def __init__(self, llm: LanguageModel, **settings: Any):
        config = ConversationCriteriaConfig(**settings)
        super().__init__(llm)
        self.criteria = config.criteria

    def config_snapshot(self) -> dict[str, Any]:
        return {**super().config_snapshot(), "criteria": self.criteria}

    async def ascore(self, messages: list[dict[str, str]]) -> MetricResult:
        prompt = ConversationPrompt().render(
            ConversationInput(
                criterion=self.criteria,
                conversation=_TURNS.validate_python(messages),
            )
        )
        judged = await self._ask(prompt, ConversationOutput)
        return MetricResult(float(binary_verdict(judged.verdict)), judged.reason)

"""Model request scope shared by the single-turn RAG diagnostic judges."""

from abc import abstractmethod
from typing import TYPE_CHECKING, Annotated, Any

from tallyjudge.core.errors import ModelOutputError
from tallyjudge.core.prompts import PromptInput
from tallyjudge.core.results import MetricResult
from tallyjudge.core.usage import mark_usage_source, run_scoped_score
from tallyjudge.interfaces.metrics import MetricBase
from tallyjudge.interfaces.models import LanguageModel

if TYPE_CHECKING:
    from pydantic import BaseModel


def required_text_inputs(*names: str, passages: bool = False) -> dict[str, Any]:
    """Declare nonempty text fields and, when requested, a nonempty passage list."""
    from pydantic import Field

    inputs: dict[str, Any] = dict.fromkeys(names, Annotated[str, Field(min_length=1)])
    if passages:
        inputs["retrieved_contexts"] = Annotated[list[str], Field(min_length=1)]
    return inputs


def binary_verdict(value: int) -> int:
    """Read a model judgment on its declared 0/1 scale."""
    if value not in (0, 1):
        raise ModelOutputError("invalid_verdict")
    return value


class RagJudge(MetricBase):
    """Run one diagnostic in a shared cancellation and request-usage scope."""

    def __init__(self, llm: LanguageModel):
        self.llm = llm

    def config_snapshot(self) -> dict[str, Any]:
        """Identify the judge deployment and prompt revision used for this score."""
        return {
            "prompt_revision": "1",
            "llm": {
                "plugin_id": self.llm.plugin_id,
                "config": self.llm.config_snapshot(),
            },
        }

    async def _ask[T: BaseModel](self, prompt: PromptInput, schema: type[T]) -> T:
        mark_usage_source("generation", self.llm.plugin_id, self.llm.usage_tracking)
        return (await self.llm.agenerate(prompt, schema)).data

    async def _ascore_validated(self, **inputs: Any) -> MetricResult:
        return await run_scoped_score(self._score(**inputs))

    @abstractmethod
    async def _score(self, **inputs: Any) -> MetricResult: ...

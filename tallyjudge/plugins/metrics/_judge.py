from typing import Any

from pydantic import BaseModel

from tallyjudge.sdk.errors import ModelOutputError
from tallyjudge.sdk.gate import gated
from tallyjudge.sdk.metrics import MetricBase
from tallyjudge.sdk.models import LanguageModel, model_snapshot
from tallyjudge.sdk.prompts import PromptInput


def binary_verdict(value: int) -> int:
    if value not in (0, 1):
        raise ModelOutputError("invalid_verdict")
    return value


class ModelJudge(MetricBase):
    def __init__(self, llm: LanguageModel):
        self.llm = gated(llm)

    @property
    def prompt_revision(self) -> str:
        return "1"

    def config_snapshot(self) -> dict[str, Any]:
        return {
            "prompt_revision": self.prompt_revision,
            "llm": model_snapshot(self.llm),
        }

    async def _ask[T: BaseModel](self, prompt: PromptInput, schema: type[T]) -> T:
        return (await self.llm.agenerate(prompt, schema)).data

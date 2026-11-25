import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from tallyjudge.sdk.dataset import canonical_sha256
from tallyjudge.sdk.errors import ModelError
from tallyjudge.sdk.models import GenerationResult, LanguageModel
from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.prompts import PromptInput
from tallyjudge.sdk.usage import Tracking, UsageSummary


class ReplayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: Path | str


class ReplayLanguageModel:
    plugin = Plugin("model", optional_dependencies={"llm": "language_model"})

    Config = ReplayConfig
    plugin_id = "model.replay.jsonl"

    def __init__(self, *, path: Path | str, llm: LanguageModel | None = None):
        self.path = Path(ReplayConfig(path=path).path)
        self.llm = llm
        self.replies: dict[str, Any] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as stream:
                for line in stream:
                    entry = json.loads(line)
                    self.replies[entry["key"]] = entry["reply"]

    @property
    def usage_tracking(self) -> Tracking:
        return "untracked" if self.llm is None else self.llm.usage_tracking

    def config_snapshot(self) -> dict[str, Any]:
        return {"recording": self.path.name}

    async def agenerate[T: BaseModel](
        self, prompt: PromptInput, response_model: type[T]
    ) -> GenerationResult[T]:
        key = canonical_sha256(
            {
                "messages": [[m.role, m.content] for m in prompt.messages],
                "schema": response_model.model_json_schema(),
            }
        )
        if self.llm is None:
            if key not in self.replies:
                raise ModelError("No recorded reply for this prompt")
            return GenerationResult(
                response_model.model_validate(self.replies[key]), UsageSummary()
            )
        result = await self.llm.agenerate(prompt, response_model)
        if key not in self.replies:
            reply = self.replies[key] = result.data.model_dump(mode="json")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                print(
                    json.dumps({"key": key, "reply": reply}, ensure_ascii=False),
                    file=stream,
                )
        return result

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from tallyjudge.sdk.dataset import canonical_sha256
from tallyjudge.sdk.errors import ModelError
from tallyjudge.sdk.gate import current_case_id
from tallyjudge.sdk.models import GenerationResult, LanguageModel, model_snapshot
from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.prompts import PromptInput
from tallyjudge.sdk.usage import Tracking, UsageSummary


class ReplayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: Path | str


class ReplayLanguageModel:
    """Record successful calls or replay their replies in per-case invocation order.

    A new reader starts at the first recorded call. Direct model calls and older
    recordings use the prompt and response schema without a case ID.
    """

    plugin = Plugin("model", optional_dependencies={"llm": "language_model"})

    Config = ReplayConfig
    plugin_id = "model.replay.jsonl"

    def __init__(self, *, path: Path | str, llm: LanguageModel | None = None):
        self.path = Path(ReplayConfig(path=path).path)
        self.llm = llm
        self._replies: dict[str, dict[int, Any]] = {}
        self._positions: Counter[str] = Counter()
        digest = hashlib.sha256()
        if self.path.exists():
            with self.path.open("rb") as stream:
                for line in stream:
                    digest.update(line)
                    entry = json.loads(line)
                    key = entry["key"]
                    position = entry.get("position", self._positions[key])
                    self._positions[key] = max(self._positions[key], position + 1)
                    if llm is None:
                        self._replies.setdefault(key, {})[position] = entry["reply"]
        if llm is None:
            self._positions.clear()
            self._identity = {"sha256": digest.hexdigest()}
        else:
            # Cached scores from an earlier recorder cannot populate a new tape.
            self._identity = {"source": model_snapshot(llm)}
        self._session = uuid4().hex if llm is not None else None

    @property
    def usage_tracking(self) -> Tracking:
        return "untracked" if self.llm is None else self.llm.usage_tracking

    def config_snapshot(self) -> dict[str, Any]:
        snapshot = {"recording": self.path.name, **self._identity}
        case_id = current_case_id()
        if case_id is not None:
            snapshot["case_id"] = case_id
            if self._session is not None:
                snapshot["session"] = self._session
        return snapshot

    async def agenerate[T: BaseModel](
        self, prompt: PromptInput, response_model: type[T]
    ) -> GenerationResult[T]:
        key = canonical_sha256(
            {
                "messages": [[m.role, m.content] for m in prompt.messages],
                "schema": response_model.model_json_schema(),
            }
        )
        case_id = current_case_id()
        if case_id is not None:
            case_key = canonical_sha256([case_id, key])
            if self.llm is not None or case_key in self._replies:
                key = case_key
        # Reserve the position before awaiting the model. Concurrent calls can
        # finish in a different order from the one in which they were started.
        position = self._positions[key]
        self._positions[key] += 1
        if self.llm is None:
            try:
                reply = self._replies[key][position]
            except KeyError:
                raise ModelError(
                    "No recorded reply for this prompt occurrence"
                ) from None
            return GenerationResult(
                response_model.model_validate(reply), UsageSummary()
            )
        result = await self.llm.agenerate(prompt, response_model)
        entry = {
            "key": key,
            "position": position,
            "reply": result.data.model_dump(mode="json"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            print(json.dumps(entry, ensure_ascii=False), file=stream)
        return result

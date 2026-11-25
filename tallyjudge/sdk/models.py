import json
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError

from tallyjudge.sdk.errors import ModelOutputError
from tallyjudge.sdk.gate import RequestCall
from tallyjudge.sdk.prompts import Message, PromptInput
from tallyjudge.sdk.usage import RequestRecord, Tracking, UsageSummary


@dataclass(frozen=True)
class GenerationResult[T]:
    """A parsed response with usage from every attempt at the same prompt."""

    data: T
    usage: UsageSummary
    model: str | None = None
    call_id: str | None = None


@dataclass(frozen=True)
class EmbeddingResult:
    """Vectors in the input order and usage across transport batches."""

    vectors: list[list[float]]
    usage: UsageSummary
    model: str | None = None
    call_id: str | None = None


@runtime_checkable
class LanguageModel(Protocol):
    """Generate structured text using the plugin's fixed settings and tracking mode."""

    @property
    def plugin_id(self) -> str: ...
    @property
    def usage_tracking(self) -> Tracking: ...
    def config_snapshot(self) -> dict[str, Any]: ...
    async def agenerate[T: BaseModel](
        self, prompt: PromptInput, response_model: type[T]
    ) -> GenerationResult[T]:
        """Validate the generated text against response_model and return observed usage."""
        ...


@runtime_checkable
class EmbeddingModel(Protocol):
    """Embed a sequence of texts; the caller owns any injected client."""

    @property
    def plugin_id(self) -> str: ...
    @property
    def usage_tracking(self) -> Tracking: ...
    def config_snapshot(self) -> dict[str, Any]: ...
    async def aembed_texts(self, texts: Sequence[str]) -> EmbeddingResult:
        """Return one vector per input in order, including an empty result for no texts."""
        ...


def model_snapshot(model: LanguageModel | EmbeddingModel) -> dict[str, Any]:
    return {"plugin_id": model.plugin_id, "config": model.config_snapshot()}


class LanguageModelBase(ABC):
    """Generate a typed response, spending the retry budget solely on invalid JSON or schema."""

    usage_tracking: Literal["per_request"] = "per_request"

    @property
    @abstractmethod
    def structured_retries(self) -> int:
        """Number of additional generation attempts allowed after parsing fails."""
        ...

    @abstractmethod
    async def _acomplete(
        self, messages: Sequence[Message], records: list[RequestRecord]
    ) -> tuple[str, str | None]: ...

    async def agenerate[T: BaseModel](
        self, prompt: PromptInput, response_model: type[T]
    ) -> GenerationResult[T]:
        """Return the parsed response and usage from all attempts at this prompt."""
        messages = list(prompt.messages)
        records = RequestCall()
        retries_left = self.structured_retries
        while True:
            text, model = await self._acomplete(messages, records)
            fenced = re.fullmatch(
                r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```",
                text.strip(),
                re.DOTALL | re.IGNORECASE,
            )
            payload = fenced.group(1) if fenced else text
            try:
                data = response_model.model_validate_json(payload)
            except ValidationError as exc:
                errors = exc.errors(
                    include_url=False, include_input=False, include_context=False
                )
                code = (
                    "invalid_json"
                    if errors[0]["type"] == "json_invalid"
                    else "schema_validation"
                )
                if not retries_left:
                    raise ModelOutputError(code) from exc
                retries_left -= 1
                fields = [{"path": e["loc"], "type": e["type"]} for e in errors]
                messages.extend(
                    (
                        Message("assistant", text),
                        Message("user", "Return corrected JSON. " + json.dumps(fields)),
                    )
                )
            else:
                return GenerationResult(
                    data, UsageSummary.from_records(records), model, records.call_id
                )

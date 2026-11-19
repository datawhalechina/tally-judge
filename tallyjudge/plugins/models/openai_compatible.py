"""Structured text generation through an injected asynchronous Chat Completions client."""

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from tallyjudge.core.errors import ModelOutputError
from tallyjudge.core.prompts import Message
from tallyjudge.core.usage import RequestRecord
from tallyjudge.interfaces.models import LanguageModelBase
from tallyjudge.plugins._openai_compatible import ClientConfig, ClientTransport


class ChatConfig(ClientConfig):
    structured_output: Literal["json_object", "text"] = "json_object"
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = Field(default=None, gt=0)
    token_limit_param: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    structured_retries: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def check_sampling(self) -> Self:
        if self.temperature is not None and self.top_p is not None:
            raise ValueError("Choose either temperature or top_p")
        return self


class OpenAICompatibleLanguageModel(LanguageModelBase):
    """Read one non-streaming text choice per request on the caller's async client."""

    plugin_id = "model.openai_compatible.chat_completions"

    def __init__(self, *, client: Any, **settings: Any):
        """Validate ``ChatConfig`` settings and bind a retry-disabled client view."""
        self.config = ChatConfig(**settings)
        self._transport = ClientTransport(client, self.config, "generation")

    @property
    def structured_retries(self) -> int:
        return self.config.structured_retries

    def config_snapshot(self) -> dict[str, Any]:
        return self.config.model_dump(mode="json")

    async def _acomplete(
        self, messages: Sequence[Message], records: list[RequestRecord]
    ) -> tuple[str, str | None]:
        config = self.config
        params = config.model_dump(
            include={"model", "temperature", "top_p"}, exclude_none=True
        )
        if config.max_output_tokens is not None:
            params[config.token_limit_param] = config.max_output_tokens
        if config.structured_output == "json_object":
            params["response_format"] = {"type": "json_object"}
        response = await self._transport.send(
            records, messages=[asdict(m) for m in messages], **params
        )
        choices = getattr(response, "choices", None)
        if not isinstance(choices, list) or len(choices) != 1:
            raise ModelOutputError("unexpected_response")
        choice = choices[0]
        message = getattr(choice, "message", None)
        stop = getattr(choice, "finish_reason", None)
        if stop == "length":
            raise ModelOutputError("truncated")
        if stop == "content_filter" or getattr(message, "refusal", None):
            raise ModelOutputError("refusal")
        if stop != "stop" or message is None or getattr(message, "tool_calls", None):
            raise ModelOutputError("unexpected_response")
        text = getattr(message, "content", None)
        if text is None or isinstance(text, str) and not text.strip():
            raise ModelOutputError("empty_content")
        if not isinstance(text, str):
            raise ModelOutputError("unexpected_response")
        return text, getattr(response, "model", None)

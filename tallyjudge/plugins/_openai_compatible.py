import asyncio
import logging
import sys
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tallyjudge.sdk.errors import ClientLoopError, ModelError
from tallyjudge.sdk.gate import record_request
from tallyjudge.sdk.usage import RequestKind, RequestRecord, Usage

logger = logging.getLogger(__name__)


class ClientConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False, frozen=True
    )
    model: str = Field(min_length=1)
    endpoint_id: str | None = None
    request_timeout: float = Field(default=60.0, gt=0)


class ClientTransport:
    def __init__(self, client: Any, config: ClientConfig, kind: RequestKind):
        sync_client = getattr(sys.modules.get("openai"), "OpenAI", ())
        if isinstance(client, sync_client):
            raise TypeError("Pass AsyncOpenAI for asynchronous model requests")
        view = client.with_options(max_retries=0, timeout=config.request_timeout)
        self.kind: RequestKind = kind
        resource = view.chat.completions if kind == "generation" else view.embeddings
        self._create: Callable[..., Awaitable[Any]] = resource.create
        if not callable(self._create):
            raise TypeError("The client resource must provide an async create method")
        self._loop: asyncio.AbstractEventLoop | None = None

    async def send(self, records: list[RequestRecord], **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise ClientLoopError(
                "This model's client is bound to another event loop. Create the "
                "client and model inside one asyncio.run(main()) and await arun "
                "there, instead of calling the synchronous run() repeatedly."
            )

        with record_request(self.kind, records) as record:
            try:
                response = await self._create(**kwargs)
            except Exception as exc:
                raise ModelError("Model request failed") from exc
            record.usage = observed_usage(response)
            return response


def observed_usage(response: Any) -> Usage | None:
    raw = getattr(response, "usage", None)
    if raw is None:
        return None

    def count(name: str) -> int | None:
        value = getattr(raw, name, None)
        if value is None or type(value) is int and value >= 0:
            return value
        logger.warning("Invalid token count for %s", name)
        return None

    return Usage(count("prompt_tokens"), count("completion_tokens"))

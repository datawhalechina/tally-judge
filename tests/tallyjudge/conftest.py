from collections import deque
from types import SimpleNamespace

import pytest


class ScriptedClient:
    """Test transport with explicit responses and the caller's observed request bodies."""

    def __init__(self):
        self.responses = deque()
        self.calls = []
        self.options = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.generate))
        self.embeddings = SimpleNamespace(create=self.embed)

    def with_options(self, **options):
        self.options.append(options)
        return self

    async def generate(self, **kwargs):
        return await self.reply("generation", kwargs)

    async def embed(self, **kwargs):
        return await self.reply("embedding", kwargs)

    async def reply(self, kind, kwargs):
        self.calls.append((kind, kwargs))
        value = self.responses.popleft()
        if isinstance(value, Exception):
            raise value
        return await value() if callable(value) else value

    def chat_reply(self, content, *, tokens=None, finish="stop", refusal=None):
        message = SimpleNamespace(content=content, refusal=refusal)
        self.responses.append(
            SimpleNamespace(
                model="served-chat",
                usage=tokens,
                choices=[SimpleNamespace(message=message, finish_reason=finish)],
            )
        )

    def embedding_reply(self, entries, *, tokens=None):
        self.responses.append(
            SimpleNamespace(
                model="served-embedding",
                usage=tokens,
                data=[SimpleNamespace(index=i, embedding=v) for i, v in entries],
            )
        )


@pytest.fixture
def client():
    return ScriptedClient()

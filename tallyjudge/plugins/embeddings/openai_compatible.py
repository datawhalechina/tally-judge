"""Batched, index-preserving embeddings through a caller-owned async client."""

import math
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import Field

from tallyjudge.core.errors import ModelOutputError
from tallyjudge.core.usage import RequestRecord, UsageSummary
from tallyjudge.interfaces.models import EmbeddingResult
from tallyjudge.plugins._openai_compatible import ClientConfig, ClientTransport


class EmbeddingConfig(ClientConfig):
    batch_size: int = Field(default=256, gt=0)
    dimensions: int | None = Field(default=None, gt=0)


def _ordered_vectors(
    items: Any, size: int, dimensions: int | None
) -> list[list[float]]:
    if not isinstance(items, list) or len(items) != size:
        raise ModelOutputError("invalid_embedding")
    ordered: dict[int, list[float]] = {}
    for item in items:
        index = getattr(item, "index", None)
        vector = getattr(item, "embedding", None)
        if type(index) is not int or not 0 <= index < size or index in ordered:
            raise ModelOutputError("invalid_embedding")
        if not isinstance(vector, list) or not vector:
            raise ModelOutputError("invalid_embedding")
        dimensions = dimensions or len(vector)
        if len(vector) != dimensions or not all(
            type(v) in (int, float) and math.isfinite(v) for v in vector
        ):
            raise ModelOutputError("invalid_embedding")
        ordered[index] = [float(v) for v in vector]
    return [ordered[i] for i in range(size)]


class OpenAICompatibleEmbeddingModel:
    """Embed text in ordered batches on the caller's async client."""

    plugin_id = "embedding.openai_compatible.embeddings"
    usage_tracking: Literal["per_request"] = "per_request"

    def __init__(self, *, client: Any, config: EmbeddingConfig):
        self.config = config
        self._transport = ClientTransport(client, config, "embedding")

    def config_snapshot(self) -> dict[str, Any]:
        return self.config.model_dump(mode="json")

    async def aembed_texts(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed texts in input order. Each transport batch owns its request record."""
        if isinstance(texts, (str, bytes)) or not all(
            isinstance(text, str) for text in texts
        ):
            raise TypeError("Expected a sequence of texts")
        vectors: list[list[float]] = []
        records: list[RequestRecord] = []
        model = None
        config = self.config
        params = config.model_dump(include={"model", "dimensions"}, exclude_none=True)
        for start in range(0, len(texts), config.batch_size):
            batch = list(texts[start : start + config.batch_size])
            response = await self._transport.send(
                records, input=batch, encoding_format="float", **params
            )
            dimensions = len(vectors[0]) if vectors else config.dimensions
            vectors.extend(
                _ordered_vectors(
                    getattr(response, "data", None), len(batch), dimensions
                )
            )
            model = getattr(response, "model", None)
        return EmbeddingResult(vectors, UsageSummary.from_records(records), model)

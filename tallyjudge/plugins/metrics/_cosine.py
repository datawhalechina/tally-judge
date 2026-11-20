"""Signed cosine scoring for embedding metrics migrated from Ragas ec00339 (Apache-2.0)."""

import math

from tallyjudge.core.errors import ModelOutputError
from tallyjudge.core.usage import mark_usage_source
from tallyjudge.interfaces.models import EmbeddingModel
from tallyjudge.plugins._vectors import unit_vector


async def embedding_cosine(
    embeddings: EmbeddingModel, first: str, second: str
) -> float:
    """Embed each text separately in caller order and return its signed cosine."""
    vectors = []
    for text in (first, second):
        mark_usage_source("embedding", embeddings.plugin_id, embeddings.usage_tracking)
        result = await embeddings.aembed_texts([text])
        if len(result.vectors) != 1:
            raise ModelOutputError("invalid_embedding")
        vectors.append(result.vectors[0])
    left, right = vectors
    if len(left) != len(right):
        raise ModelOutputError("invalid_embedding")
    score = math.fsum(
        a * b for a, b in zip(unit_vector(left), unit_vector(right), strict=True)
    )
    # A dot product of unit vectors can round just outside its mathematical range.
    return min(1.0, max(-1.0, score))

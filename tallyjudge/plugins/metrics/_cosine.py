"""Signed cosine scoring for embedding metrics migrated from Ragas ec00339 (Apache-2.0)."""

import math

from tallyjudge.plugins._vectors import unit_vector
from tallyjudge.sdk.errors import ModelOutputError
from tallyjudge.sdk.models import EmbeddingModel


async def embedding_vectors(
    embeddings: EmbeddingModel, texts: list[str]
) -> list[list[float]]:
    return (await embeddings.aembed_texts(texts)).vectors


async def embedding_cosine(
    embeddings: EmbeddingModel, first: str, second: str
) -> float:
    vectors = []
    for text in (first, second):
        batch = await embedding_vectors(embeddings, [text])
        if len(batch) != 1:
            raise ModelOutputError("invalid_embedding")
        vectors.append(batch[0])
    left, right = vectors
    if len(left) != len(right):
        raise ModelOutputError("invalid_embedding")
    score = math.fsum(
        a * b for a, b in zip(unit_vector(left), unit_vector(right), strict=True)
    )
    return min(1.0, max(-1.0, score))

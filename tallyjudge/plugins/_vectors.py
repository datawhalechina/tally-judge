"""Vector normalization shared by similarity scoring and evidence selection."""

import math

from tallyjudge.core.errors import ModelOutputError, UndefinedScore


def unit_vector(vector: list[float]) -> list[float]:
    """Normalize a finite nonzero embedding without overflowing its norm."""
    if not vector or not all(math.isfinite(v) for v in vector):
        raise ModelOutputError("invalid_embedding")
    scale = max(map(abs, vector))
    if scale == 0:
        raise UndefinedScore("zero_norm_embedding")
    # Scaling before the norm also handles finite components whose norm would overflow.
    scaled = [v / scale for v in vector]
    norm = math.hypot(*scaled)
    return [v / norm for v in scaled]

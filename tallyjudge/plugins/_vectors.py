import math

from tallyjudge.sdk.errors import ModelOutputError, UndefinedScore


def unit_vector(vector: list[float]) -> list[float]:
    if not vector or not all(math.isfinite(v) for v in vector):
        raise ModelOutputError("invalid_embedding")
    scale = max(map(abs, vector))
    if scale == 0:
        raise UndefinedScore("zero_norm_embedding")
    scaled = [v / scale for v in vector]
    norm = math.hypot(*scaled)
    return [v / norm for v in scaled]

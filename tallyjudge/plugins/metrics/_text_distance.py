"""Normalized distances used by Ragas collections/_string.py (ec00339, Apache-2.0)."""

import asyncio
import json
import sys
from enum import StrEnum
from importlib import import_module
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict

# A native Levenshtein comparison can hold the GIL for its entire duration.
# Keep ordinary passages in-process and give expensive rows a cancellable worker.
_PROCESS_WORK = 256_000_000


class DistanceMeasure(StrEnum):
    """String distance used to compare an unmodified pair of passages."""

    LEVENSHTEIN = "levenshtein"
    HAMMING = "hamming"
    JARO = "jaro"
    JARO_WINKLER = "jaro_winkler"


class DistanceConfig(BaseModel):
    model_config = ConfigDict(strict=True)
    distance_measure: Annotated[
        DistanceMeasure, BeforeValidator(lambda value: DistanceMeasure(value))
    ] = DistanceMeasure.LEVENSHTEIN


class TextDistance:
    """Load RapidFuzz only when a distance-based metric is constructed."""

    def __init__(self, measure: str):
        self.measure = DistanceMeasure(measure)
        distance = import_module("rapidfuzz.distance")
        self._normalized_distance = {
            DistanceMeasure.LEVENSHTEIN: distance.Levenshtein,
            DistanceMeasure.HAMMING: distance.Hamming,
            DistanceMeasure.JARO: distance.Jaro,
            DistanceMeasure.JARO_WINKLER: distance.JaroWinkler,
        }[self.measure].normalized_distance

    def similarity(self, left: str, right: str) -> float:
        return 1.0 - self._normalized_distance(left, right)

    async def asimilarity(self, left: str, right: str) -> float:
        """Compare a pair without blocking other scores on long strings."""
        if (1 + len(left)) * (1 + len(right)) > _PROCESS_WORK:
            return await self._in_process([left], [right], None)
        return self.similarity(left, right)

    def _matches(
        self, passages: list[str], candidates: list[str], threshold: float
    ) -> list[bool]:
        return [
            any(self.similarity(passage, other) >= threshold for other in candidates)
            for passage in passages
        ]

    async def matches(
        self, passages: list[str], candidates: list[str], threshold: float
    ) -> list[bool]:
        """Return one hit per passage, preserving ranks and repeated passages."""
        work = (sum(map(len, passages)) + len(passages)) * (
            sum(map(len, candidates)) + len(candidates)
        )
        if work > _PROCESS_WORK:
            return await self._in_process(passages, candidates, threshold)
        return await asyncio.to_thread(self._matches, passages, candidates, threshold)

    async def _in_process(
        self, left: list[str], right: list[str], threshold: float | None
    ) -> Any:
        payload = json.dumps([self.measure, left, right, threshold]).encode("ascii")
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            __file__,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        transfer = asyncio.create_task(process.communicate(payload))
        try:
            output, _ = await asyncio.shield(transfer)
        except asyncio.CancelledError:
            if process.returncode is None:
                process.kill()
            await transfer
            raise
        if process.returncode:
            raise RuntimeError(
                f"Text distance worker exited with status {process.returncode}"
            )
        return json.loads(output)


if __name__ == "__main__":
    measure, left, right, threshold = json.load(sys.stdin.buffer)
    distance = TextDistance(measure)
    result = (
        distance.similarity(left[0], right[0])
        if threshold is None
        else distance._matches(left, right, threshold)
    )
    json.dump(result, sys.stdout)

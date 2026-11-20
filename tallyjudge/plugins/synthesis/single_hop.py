"""Single-hop generation from pre-chunked source text.

The output fields follow the Ragas authors' SingleHopQuerySynthesizer._generate_sample
in src/ragas/testset/synthesizers/single_hop/base.py (ec00339, Apache-2.0).
This plugin selects stable chunk IDs directly and uses its own one-passage prompt.
"""

import random
from collections.abc import Iterable, Mapping
from typing import Any

from tallyjudge.plugins.synthesis._evidence import (
    EvidenceSynthesizer,
    SamplingConfig,
    _Chunk,
)

SingleHopConfig = SamplingConfig

_INSTRUCTION = (
    "Use the supplied passage to write one single-hop question and its factual reference answer. "
    "The question must be answerable using only this passage. Use the language of the passage. "
    'Return a JSON object with exactly two string fields: "user_input" and "reference".'
)


class SingleHopSynthesizer(EvidenceSynthesizer):
    """Sample unique passages and generate one question per passage."""

    plugin_id = "synthesis.testset.single_hop"
    instruction = _INSTRUCTION

    def select(
        self, chunks: Iterable[Mapping[str, Any]], size: int
    ) -> list[tuple[_Chunk, ...]]:
        """Select without replacement, independently of input ordering."""
        by_id = {}
        for record in chunks:
            chunk = _Chunk.model_validate(record)
            if chunk.chunk_id in by_id:
                raise ValueError(f"Duplicate chunk_id: {chunk.chunk_id!r}")
            by_id[chunk.chunk_id] = chunk
        ordered = sorted(by_id)
        random.Random(self.seed).shuffle(ordered)
        return [(by_id[key],) for key in ordered[:size]]

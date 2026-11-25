"""Single-hop generation from pre-chunked source text.

The output fields follow the Ragas authors' SingleHopQuerySynthesizer._generate_sample
in src/ragas/testset/synthesizers/single_hop/base.py (ec00339, Apache-2.0).
The convenience sampler selects stable chunk IDs and renders the archived template.
"""

import random
from collections.abc import Iterable, Mapping
from typing import Any

from tallyjudge.plugins.synthesis._evidence import (
    EvidenceSynthesizer,
    SamplingConfig,
)
from tallyjudge.plugins.synthesis.scenarios import EvidenceChunk
from tallyjudge.sdk.plugin import Plugin


class SingleHopSynthesizer(EvidenceSynthesizer):
    plugin = Plugin("synthesis", dependencies={"llm": "language_model"})

    Config = SamplingConfig

    plugin_id = "synthesis.testset.single_hop"
    prompt_revision = "2"

    def select(
        self, chunks: Iterable[Mapping[str, Any]], size: int
    ) -> list[tuple[EvidenceChunk, ...]]:
        by_id = {}
        for record in chunks:
            chunk = EvidenceChunk.model_validate(record)
            if chunk.chunk_id in by_id:
                raise ValueError(f"Duplicate chunk_id: {chunk.chunk_id!r}")
            by_id[chunk.chunk_id] = chunk
        return select_passages(by_id, size, self.seed)


def select_passages(
    by_id: Mapping[str, EvidenceChunk], size: int, seed: int
) -> list[tuple[EvidenceChunk, ...]]:
    ordered = sorted(by_id)
    random.Random(seed).shuffle(ordered)
    return [(by_id[key],) for key in ordered[:size]]

"""Multi-hop scenarios from two related passages.

Based on the Ragas ec00339 (Apache-2.0) multi-hop synthesizers. References
carry the selected source IDs and the archived hop-labelled passages.
"""

import random
from collections.abc import Iterable, Mapping
from typing import Any

from tallyjudge.plugins.synthesis._evidence import EvidenceSynthesizer, SamplingConfig
from tallyjudge.plugins.synthesis.graph import EvidenceGraph, build_graph
from tallyjudge.plugins.synthesis.scenarios import EvidenceChunk
from tallyjudge.sdk.plugin import Plugin


class MultiHopSynthesizer(EvidenceSynthesizer):
    plugin = Plugin("synthesis", dependencies={"llm": "language_model"})

    Config = SamplingConfig

    plugin_id = "synthesis.testset.multi_hop"
    prompt_revision = "3"

    def select(
        self, chunks: Iterable[Mapping[str, Any]], size: int
    ) -> list[tuple[EvidenceChunk, ...]]:
        graph = chunks if isinstance(chunks, EvidenceGraph) else build_graph(chunks)
        return select_pairs(graph, size, self.seed)


def select_pairs(
    graph: EvidenceGraph, size: int, seed: int
) -> list[tuple[EvidenceChunk, ...]]:
    pairs = list(graph.relations)
    random.Random(seed).shuffle(pairs)
    return [(graph._evidence[a], graph._evidence[b]) for a, b in pairs[:size]]

"""Multi-hop scenarios from two related passages.

Based on the Ragas ec00339 (Apache-2.0) multi-hop synthesizers. References
carry the selected source IDs and the archived hop-labelled passages.
"""

import random
from collections.abc import Iterable, Mapping
from typing import Any

from tallyjudge.plugins.synthesis._evidence import EvidenceSynthesizer
from tallyjudge.plugins.synthesis.graph import EvidenceGraph, build_graph
from tallyjudge.plugins.synthesis.scenarios import EvidenceChunk


class MultiHopSynthesizer(EvidenceSynthesizer):
    """Generate a question requiring facts from two related source chunks."""

    plugin_id = "synthesis.testset.multi_hop"
    prompt_revision = "3"

    def select(
        self, chunks: Iterable[Mapping[str, Any]], size: int
    ) -> list[tuple[EvidenceChunk, ...]]:
        """Select unique evidence pairs from explicit relations using the configured seed."""
        graph = chunks if isinstance(chunks, EvidenceGraph) else build_graph(chunks)
        return select_pairs(graph, size, self.seed)


def select_pairs(
    graph: EvidenceGraph, size: int, seed: int
) -> list[tuple[EvidenceChunk, ...]]:
    """Sample distinct undirected pairs from a graph in seed order."""
    pairs = list(graph.relations)
    random.Random(seed).shuffle(pairs)
    return [(graph._evidence[a], graph._evidence[b]) for a, b in pairs[:size]]

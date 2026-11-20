"""Multi-hop scenarios from two related passages.

Based on the Ragas ec00339 (Apache-2.0) multi-hop synthesizers. References
carry both stable chunk IDs, and verbatim quotations validate both sources.
"""

import json
import random
from collections.abc import Iterable, Mapping
from typing import Any

from tallyjudge.plugins.synthesis._evidence import (
    EvidenceSynthesizer,
    _Chunk,
    _QuestionAnswer,
)
from tallyjudge.plugins.synthesis.graph import EvidenceGraph, build_graph


class _GroundedAnswer(_QuestionAnswer):
    evidence_quotes: tuple[str, str]

    def grounded(self, chunks: tuple[_Chunk, ...]) -> bool:
        return all(
            quote.strip() and quote in chunk.text
            for quote, chunk in zip(self.evidence_quotes, chunks, strict=True)
        )


class MultiHopSynthesizer(EvidenceSynthesizer):
    """Generate a question requiring facts from two related source chunks."""

    plugin_id = "synthesis.testset.multi_hop"
    prompt_revision = "2"
    instruction = (
        "Write one question that requires combining distinct facts from both passages. "
        "Return a JSON object with user_input, reference and evidence_quotes, with one verbatim quotation "
        "from each passage in order. Use the language of the passages."
    )
    output_type = _GroundedAnswer

    def select(
        self, chunks: Iterable[Mapping[str, Any]], size: int
    ) -> list[tuple[_Chunk, ...]]:
        """Select unique evidence pairs from explicit relations using the configured seed."""
        graph = chunks if isinstance(chunks, EvidenceGraph) else build_graph(chunks)
        pairs = list(graph.relations)
        random.Random(self.seed).shuffle(pairs)
        return [(graph._evidence[a], graph._evidence[b]) for a, b in pairs[:size]]

    def passage(self, chunks: tuple[_Chunk, ...]) -> str:
        return json.dumps([chunk.text for chunk in chunks], ensure_ascii=False)

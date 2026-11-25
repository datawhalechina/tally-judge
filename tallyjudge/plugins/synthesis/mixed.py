"""Generate single- and two-passage questions under one candidate budget.

Ragas ec00339 (Apache-2.0) distributes scenarios across query synthesizers.
Here one total bounds both types, with redistribution only before generation.
"""

from collections.abc import Iterable, Mapping
from fractions import Fraction
from typing import Any

from pydantic import Field

from tallyjudge.plugins.synthesis._evidence import (
    EvidenceSynthesizer,
    SamplingConfig,
)
from tallyjudge.plugins.synthesis.graph import EvidenceGraph, build_graph
from tallyjudge.plugins.synthesis.multi_hop import MultiHopSynthesizer, select_pairs
from tallyjudge.plugins.synthesis.scenarios import EvidenceChunk
from tallyjudge.plugins.synthesis.single_hop import (
    SingleHopSynthesizer,
    select_passages,
)
from tallyjudge.sdk.models import LanguageModel
from tallyjudge.sdk.plugin import Plugin


class MixedConfig(SamplingConfig):
    """Share of requested candidates that should use two passages."""

    multi_hop_fraction: float = Field(default=0.5, ge=0, le=1)


class MixedHopSynthesizer(EvidenceSynthesizer):
    plugin = Plugin("synthesis", dependencies={"llm": "language_model"})

    Config = MixedConfig

    plugin_id = "synthesis.testset.mixed"

    def __init__(
        self,
        *,
        llm: LanguageModel,
        seed: int = 0,
        personas: list[dict[str, str]] | None = None,
        multi_hop_fraction: float = 0.5,
    ):
        config = MixedConfig.model_validate(
            {
                "seed": seed,
                "personas": [] if personas is None else personas,
                "multi_hop_fraction": multi_hop_fraction,
            }
        )
        self._configure(llm, config)
        self.multi_hop_fraction = config.multi_hop_fraction

    def config_snapshot(self) -> dict[str, Any]:
        return {
            **super().config_snapshot(),
            "multi_hop_fraction": self.multi_hop_fraction,
            "component_prompt_revisions": {
                SingleHopSynthesizer.plugin_id: SingleHopSynthesizer.prompt_revision,
                MultiHopSynthesizer.plugin_id: MultiHopSynthesizer.prompt_revision,
            },
        }

    def select(
        self, chunks: Iterable[Mapping[str, Any]], size: int
    ) -> list[tuple[EvidenceChunk, ...]]:
        if isinstance(chunks, EvidenceGraph):
            graph = chunks
        elif self.multi_hop_fraction:
            graph = build_graph(chunks)
        else:
            graph = EvidenceGraph(chunks, ())
        if self.multi_hop_fraction == 0:
            return select_passages(graph._evidence, size, self.seed)
        if self.multi_hop_fraction == 1:
            return select_pairs(graph, size, self.seed)

        target_pairs = int(
            Fraction(str(self.multi_hop_fraction)) * size + Fraction(1, 2)
        )
        available_pairs = min(target_pairs, len(graph.relations))
        singles = select_passages(graph._evidence, size - available_pairs, self.seed)
        return singles + select_pairs(graph, size - len(singles), self.seed)

    @classmethod
    def _question_style(
        cls, chunks: tuple[EvidenceChunk, ...]
    ) -> type[EvidenceSynthesizer]:
        return SingleHopSynthesizer if len(chunks) == 1 else MultiHopSynthesizer

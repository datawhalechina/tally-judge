"""The default text RAG testset workflow from Ragas ec00339 (Apache-2.0).

Three graph-driven strategies target independent quotas. Case IDs and separate
planning usage let the same generated dataset feed paired TallyJudge experiments.
"""

from collections.abc import Iterable, Mapping
from hashlib import sha256
from random import Random
from typing import Any

from pydantic import Field

from tallyjudge.core.dataset import canonical_json
from tallyjudge.core.execution import run_sync
from tallyjudge.interfaces.models import EmbeddingModel, LanguageModel
from tallyjudge.interfaces.synthesis import (
    SynthesisCandidate,
    SynthesisCollector,
    SynthesisResult,
)
from tallyjudge.plugins.synthesis._evidence import (
    SamplingConfig,
    cancellable,
    generate_candidate,
)
from tallyjudge.plugins.synthesis._strategies import Scenario, ScenarioPlanner
from tallyjudge.plugins.synthesis.graph import EvidenceGraph
from tallyjudge.plugins.synthesis.transforms import build_chunk_graph


class RagTestsetConfig(SamplingConfig):
    """Number of graph-derived personas and optional guidance for question generation."""

    num_personas: int = Field(default=3, gt=0)
    llm_context: str | None = None


class RagTestsetSynthesizer:
    """Generate single-hop specific, multi-hop abstract and multi-hop specific cases.

    Supply a typed EvidenceGraph, or inject embeddings to build one from prechunked
    records. Each eligible strategy rounds its quota up independently. Save the
    resulting dataset when comparing retrievers on exactly the same questions.
    """

    plugin_id = "synthesis.testset.rag"
    prompt_revision = "1"

    def __init__(
        self,
        *,
        llm: LanguageModel,
        embeddings: EmbeddingModel | None = None,
        **settings: Any,
    ):
        self.llm = llm
        self.embeddings = embeddings
        self.config = RagTestsetConfig(**settings)

    def config_snapshot(self) -> dict[str, Any]:
        """Identify planning settings, model configuration and prompt revision."""
        settings = self.config.model_dump()
        settings.update(
            prompt_revision=self.prompt_revision,
            llm={"plugin_id": self.llm.plugin_id, "config": self.llm.config_snapshot()},
        )
        if self.embeddings is not None:
            settings["embeddings"] = {
                "plugin_id": self.embeddings.plugin_id,
                "config": self.embeddings.config_snapshot(),
            }
        return settings

    async def _plan(
        self, chunks: Iterable[Mapping[str, Any]], size: int
    ) -> list[tuple[str, Scenario]]:
        if isinstance(chunks, EvidenceGraph):
            graph = chunks
        elif self.embeddings is not None:
            graph = await build_chunk_graph(
                chunks, llm=self.llm, embeddings=self.embeddings
            )
        else:
            raise ValueError(
                "Supply embeddings for prechunked records, or pass a graph from build_chunk_graph or build_document_graph"
            )
        planner = ScenarioPlanner(graph, self.llm, Random(self.config.seed))
        return await planner.select(
            size, self.config.personas, self.config.num_personas
        )

    async def arun(
        self,
        chunks: Iterable[Mapping[str, Any]],
        *,
        testset_size: int,
        collector: SynthesisCollector | None = None,
    ) -> SynthesisResult:
        """Plan eligible scenarios, generate references, and retain planning and per-case usage."""
        if type(testset_size) is not int or testset_size < 0:
            raise ValueError("testset_size must be a nonnegative integer")
        result = SynthesisResult(self.plugin_id, self.config_snapshot(), testset_size)
        progress = collector if collector is not None else SynthesisCollector()
        with progress.collect(result):
            if testset_size == 0:
                return result
            with progress.plan():
                scenarios = await cancellable(self._plan(chunks, testset_size))
            for ordinal, (strategy, scenario) in enumerate(scenarios):
                identity = canonical_json(
                    [
                        self.plugin_id,
                        result.config_snapshot,
                        strategy,
                        scenario.model_dump(mode="json"),
                        ordinal,
                    ]
                )
                ids = tuple(node.chunk_id for node in scenario.nodes)
                documents = tuple(node.document_id for node in scenario.nodes)
                candidate = SynthesisCandidate(
                    sha256(identity.encode()).hexdigest(),
                    ids[0],
                    documents[0],
                    evidence_ids=ids,
                    document_ids=documents,
                )
                with progress.attempt(candidate):
                    await generate_candidate(
                        scenario,
                        self.llm,
                        candidate,
                        llm_context=self.config.llm_context,
                    )
                    if candidate.record is not None:
                        candidate.record["synthesizer_name"] = strategy
        return result

    def run(
        self,
        chunks: Iterable[Mapping[str, Any]],
        *,
        testset_size: int,
        collector: SynthesisCollector | None = None,
    ) -> SynthesisResult:
        """Run from synchronous code outside an active event loop."""
        return run_sync(
            lambda: self.arun(chunks, testset_size=testset_size, collector=collector),
            "arun",
        )

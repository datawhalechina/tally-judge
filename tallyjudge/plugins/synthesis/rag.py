"""The default text RAG testset workflow from Ragas ec00339 (Apache-2.0).

Three graph-driven strategies target independent quotas. Case IDs and separate
planning usage let the same generated dataset feed paired TallyJudge experiments.
"""

import asyncio
from collections.abc import Iterable, Mapping
from random import Random
from typing import Any

from pydantic import Field

from tallyjudge.plugins.synthesis._evidence import (
    SamplingConfig,
    cancellable,
    generate_candidate,
)
from tallyjudge.plugins.synthesis._strategies import Scenario, ScenarioPlanner
from tallyjudge.plugins.synthesis.contract import (
    SynthesisCandidate,
    SynthesisCollector,
    SynthesisResult,
    SynthesizerBase,
)
from tallyjudge.plugins.synthesis.graph import EvidenceGraph
from tallyjudge.plugins.synthesis.transforms import build_chunk_graph
from tallyjudge.sdk.dataset import canonical_sha256
from tallyjudge.sdk.gate import gated
from tallyjudge.sdk.models import EmbeddingModel, LanguageModel, model_snapshot
from tallyjudge.sdk.plugin import Plugin


class RagTestsetConfig(SamplingConfig):
    """Number of graph-derived personas and optional guidance for question generation."""

    num_personas: int = Field(default=3, gt=0)
    llm_context: str | None = None


class RagTestsetSynthesizer(SynthesizerBase):
    plugin = Plugin(
        "synthesis",
        dependencies={"llm": "language_model"},
        optional_dependencies={"embeddings": "embedding_model"},
    )

    Config = RagTestsetConfig

    plugin_id = "synthesis.testset.rag"
    prompt_revision = "1"

    def __init__(
        self,
        *,
        llm: LanguageModel,
        embeddings: EmbeddingModel | None = None,
        **settings: Any,
    ):
        self.llm = gated(llm)
        self.embeddings = embeddings and gated(embeddings)
        self.config = RagTestsetConfig(**settings)

    def config_snapshot(self) -> dict[str, Any]:
        settings = self.config.model_dump()
        settings.update(
            prompt_revision=self.prompt_revision,
            llm=model_snapshot(self.llm),
        )
        if self.embeddings is not None:
            settings["embeddings"] = model_snapshot(self.embeddings)
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
        planner = await asyncio.to_thread(
            ScenarioPlanner, graph, self.llm, Random(self.config.seed)
        )
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
                case_id = canonical_sha256(
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
                    case_id,
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

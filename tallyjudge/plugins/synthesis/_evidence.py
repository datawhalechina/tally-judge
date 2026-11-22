"""Shared evidence provenance, generation and cancellation handling."""

import asyncio
import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Coroutine, Iterable, Mapping
from typing import Any

from pydantic import ConfigDict

from tallyjudge.core._tasks import detach_task
from tallyjudge.core.dataset import canonical_json
from tallyjudge.core.errors import ModelError, ModelOutputError
from tallyjudge.core.execution import run_sync
from tallyjudge.core.results import ErrorInfo
from tallyjudge.interfaces.models import LanguageModel
from tallyjudge.interfaces.synthesis import (
    SynthesisCandidate,
    SynthesisCollector,
    SynthesisResult,
)
from tallyjudge.plugins.synthesis._prompts import Persona
from tallyjudge.plugins.synthesis.scenarios import (
    EvidenceChunk,
    MultiHopScenario,
    QueryLength,
    QueryStyle,
    SingleHopScenario,
    generate_sample,
)
from tallyjudge.registry.specs import EmptyConfig


class ConfiguredPersona(Persona):
    model_config = ConfigDict(extra="forbid", strict=True)


class SamplingConfig(EmptyConfig):
    """The seed used to sample unique evidence chunks."""

    seed: int = 0
    personas: list[ConfiguredPersona] = []


class EvidenceSynthesizer(ABC):
    """Generate one candidate per selected evidence group, preserving its provenance."""

    plugin_id: str
    prompt_revision = "1"

    @abstractmethod
    def select(
        self, chunks: Iterable[Mapping[str, Any]], size: int
    ) -> list[tuple[EvidenceChunk, ...]]:
        """Return the evidence groups selected under this sampling policy."""

    def __init__(
        self,
        *,
        llm: LanguageModel,
        seed: int = 0,
        personas: list[dict[str, str]] | None = None,
    ):
        config = SamplingConfig.model_validate(
            {"seed": seed, "personas": [] if personas is None else personas}
        )
        self._configure(llm, config)

    def _configure(self, llm: LanguageModel, config: SamplingConfig) -> None:
        self.llm = llm
        self.seed, self.personas = config.seed, config.personas

    def config_snapshot(self) -> dict[str, Any]:
        """Identify the selection seed, prompt revision and caller-supplied model."""
        return {
            "seed": self.seed,
            **(
                {"personas": [p.model_dump() for p in self.personas]}
                if self.personas
                else {}
            ),
            "prompt_revision": self.prompt_revision,
            "llm": {
                "plugin_id": self.llm.plugin_id,
                "config": json.loads(canonical_json(self.llm.config_snapshot())),
            },
        }

    def run(
        self,
        chunks: Iterable[Mapping[str, Any]],
        *,
        testset_size: int,
        collector: SynthesisCollector | None = None,
    ) -> SynthesisResult:
        """Generate synchronously outside an event loop. Use arun with an async client."""
        return run_sync(
            lambda: self.arun(chunks, testset_size=testset_size, collector=collector),
            "arun",
        )

    async def arun(
        self,
        chunks: Iterable[Mapping[str, Any]],
        *,
        testset_size: int,
        collector: SynthesisCollector | None = None,
    ) -> SynthesisResult:
        """Select without replacement and generate sequentially. Failed candidates keep their usage."""
        if type(testset_size) is not int or testset_size < 0:
            raise ValueError("testset_size must be a nonnegative integer")
        selected = self.select(chunks, testset_size) if testset_size else []
        result = SynthesisResult(self.plugin_id, self.config_snapshot(), testset_size)
        progress = collector if collector is not None else SynthesisCollector()
        with progress.collect(result):
            for ordinal, evidence in enumerate(selected):
                chunk = evidence[0]
                ids = tuple(item.chunk_id for item in evidence)
                documents = tuple(item.document_id for item in evidence)
                if len(evidence) == 1:
                    topics = evidence[0].entities[:1]
                else:
                    topics = list(
                        dict.fromkeys(
                            label for node in evidence for label in node.entities
                        )
                    )
                identity = canonical_json(
                    [
                        self.plugin_id,
                        result.config_snapshot,
                        [
                            [node.chunk_id, node.document_id, node.text]
                            for node in evidence
                        ],
                        topics,
                        ordinal,
                    ]
                )
                case_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
                candidate = SynthesisCandidate(
                    case_id,
                    chunk.chunk_id,
                    chunk.document_id,
                    evidence_ids=ids,
                    document_ids=documents,
                )
                with progress.attempt(candidate):
                    await self._generate(evidence, topics, candidate)
        return result

    @classmethod
    def _question_style(
        cls, chunks: tuple[EvidenceChunk, ...]
    ) -> type["EvidenceSynthesizer"]:
        """Select the plugin whose prompt and output schema apply to this evidence."""
        return cls

    async def _generate(
        self,
        chunks: tuple[EvidenceChunk, ...],
        topics: list[str],
        candidate: SynthesisCandidate,
    ) -> None:
        style = self._question_style(chunks)
        persona = (
            self.personas[int(candidate.case_id, 16) % len(self.personas)]
            if self.personas
            else Persona(name="Reader", role_description="Reads the source documents.")
        )
        if len(chunks) == 1:
            scenario = SingleHopScenario(
                nodes=list(chunks),
                persona=persona,
                style=QueryStyle.PERFECT_GRAMMAR,
                length=QueryLength.MEDIUM,
                term=next(iter(topics), ""),
            )
        else:
            scenario = MultiHopScenario(
                nodes=list(chunks),
                persona=persona,
                style=QueryStyle.PERFECT_GRAMMAR,
                length=QueryLength.MEDIUM,
                combinations=topics,
            )
        await generate_candidate(scenario, self.llm, candidate)
        if candidate.record is not None and style.plugin_id != self.plugin_id:
            candidate.record["synthesizer_id"] = style.plugin_id


async def cancellable[T](work: Coroutine[Any, Any, T]) -> T:
    """Release the caller if a model transport outlives cancellation."""
    task = asyncio.create_task(work)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        task.cancel()
        detach_task(task)
        raise


async def generate_candidate(
    scenario: SingleHopScenario | MultiHopScenario,
    llm: LanguageModel,
    candidate: SynthesisCandidate,
    *,
    llm_context: str | None = None,
) -> None:
    """Record the generated sample or its model failure on the selected candidate."""
    try:
        record = await cancellable(
            generate_sample(scenario, llm=llm, llm_context=llm_context)
        )
    except ModelOutputError as exc:
        candidate.status = "error"
        candidate.error = ErrorInfo("model_output_error", code=exc.code)
    except ModelError as exc:
        candidate.status = "error"
        candidate.error = ErrorInfo("model_error", type(exc.__cause__ or exc).__name__)
    else:
        candidate.record = {"case_id": candidate.case_id, **record}
        candidate.status = "success"

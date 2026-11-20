"""Shared evidence provenance, generation and cancellation handling."""

import asyncio
import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any

from tallyjudge.core._tasks import detach_task
from tallyjudge.core.dataset import canonical_json
from tallyjudge.core.errors import ModelError, ModelOutputError
from tallyjudge.core.execution import run_sync
from tallyjudge.core.prompts import Message, PromptInput
from tallyjudge.core.results import ErrorInfo
from tallyjudge.core.usage import mark_usage_source
from tallyjudge.interfaces.models import LanguageModel
from tallyjudge.interfaces.synthesis import (
    SynthesisCandidate,
    SynthesisCollector,
    SynthesisResult,
)
from tallyjudge.registry.specs import EmptyConfig


class Persona(EmptyConfig):
    """A reader identity for generating questions from a particular perspective."""

    name: str
    role_description: str


class SamplingConfig(EmptyConfig):
    """The seed used to sample unique evidence chunks."""

    seed: int = 0
    personas: list[Persona] = []


class _Chunk(EmptyConfig):
    model_config = {"extra": "ignore", "frozen": True}
    chunk_id: str
    document_id: str
    text: str

    def model_post_init(self, context: Any) -> None:
        for name in ("chunk_id", "document_id", "text"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must contain text")


class _QuestionAnswer(EmptyConfig):
    def grounded(self, chunks: tuple[_Chunk, ...]) -> bool:
        return True

    user_input: str
    reference: str

    def model_post_init(self, context: Any) -> None:
        if not self.user_input.strip() or not self.reference.strip():
            raise ValueError(
                "The generated question and reference must both contain text"
            )


class EvidenceSynthesizer(ABC):
    """Generate one candidate per selected evidence group, preserving its provenance."""

    plugin_id: str
    instruction: str
    prompt_revision = "1"
    output_type: type[_QuestionAnswer] = _QuestionAnswer

    @abstractmethod
    def select(
        self, chunks: Iterable[Mapping[str, Any]], size: int
    ) -> list[tuple[_Chunk, ...]]:
        """Return the evidence groups selected under this sampling policy."""

    def passage(self, chunks: tuple[_Chunk, ...]) -> str:
        """Render the evidence as the model's user message."""
        return chunks[0].text

    def __init__(
        self,
        *,
        llm: LanguageModel,
        seed: int = 0,
        personas: list[dict[str, str]] | None = None,
    ):
        self.llm = llm
        config = SamplingConfig.model_validate(
            {"seed": seed, "personas": [] if personas is None else personas}
        )
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
                identity = canonical_json(
                    [
                        self.plugin_id,
                        result.config_snapshot,
                        ids[0] if len(ids) == 1 else list(ids),
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
                    await self._generate(evidence, candidate)
        return result

    async def _generate(
        self, chunks: tuple[_Chunk, ...], candidate: SynthesisCandidate
    ) -> None:
        instruction = self.instruction
        if self.personas:
            persona = self.personas[int(candidate.case_id, 16) % len(self.personas)]
            instruction += f" Write for {persona.name}: {persona.role_description}"
        prompt = PromptInput(
            (Message("system", instruction), Message("user", self.passage(chunks)))
        )
        mark_usage_source("generation", self.llm.plugin_id, self.llm.usage_tracking)
        request = asyncio.create_task(self.llm.agenerate(prompt, self.output_type))
        try:
            response = await asyncio.shield(request)
        except asyncio.CancelledError:
            # A late transport response must never publish a sample or start another candidate.
            request.cancel()
            detach_task(request)
            raise
        except ModelOutputError as exc:
            candidate.status = "error"
            candidate.error = ErrorInfo("model_output_error", code=exc.code)
        except ModelError as exc:
            candidate.status = "error"
            candidate.error = ErrorInfo(
                "model_error", type(exc.__cause__ or exc).__name__
            )
        else:
            if not response.data.grounded(chunks):
                candidate.status = "error"
                candidate.error = ErrorInfo(
                    "model_output_error", code="unsupported_evidence"
                )
                return
            candidate.record = {
                "case_id": candidate.case_id,
                "user_input": response.data.user_input,
                "reference": response.data.reference,
                "reference_contexts": [item.text for item in chunks],
                "reference_context_ids": list(candidate.evidence_ids),
                "reference_document_ids": list(candidate.document_ids),
            }
            candidate.status = "success"

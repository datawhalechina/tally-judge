"""Generate samples from explicit Ragas ec00339 scenarios (Apache-2.0).

The node adapter carries TallyJudge chunk IDs. Prompt inputs, hop labels and
sample metadata follow SingleHopQuerySynthesizer and MultiHopQuerySynthesizer.
"""

from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Any, Self

from pydantic import BaseModel, Field

from tallyjudge.core.usage import mark_usage_source
from tallyjudge.interfaces.models import LanguageModel
from tallyjudge.plugins.synthesis._prompts import (
    GeneratedQueryAnswer,
    MultiHopPrompt,
    Persona,
    QueryCondition,
    QueryConditions,
    SingleHopPrompt,
)
from tallyjudge.registry.specs import EmptyConfig


class EvidenceChunk(EmptyConfig):
    """An identified source passage and its optional extracted entities."""

    model_config = {"extra": "ignore", "frozen": True}
    chunk_id: str
    document_id: str
    text: str
    entities: list[str] = []

    def model_post_init(self, context: Any) -> None:
        for name in ("chunk_id", "document_id", "text"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must contain text")


class QueryStyle(str, Enum):
    """Question styles supported by the archived scenario generator."""

    MISSPELLED = "Misspelled queries"
    PERFECT_GRAMMAR = "Perfect grammar"
    POOR_GRAMMAR = "Poor grammar"
    WEB_SEARCH_LIKE = "Web search like queries"


class QueryLength(str, Enum):
    """Question lengths supported by the archived scenario generator."""

    LONG = "long"
    MEDIUM = "medium"
    SHORT = "short"


class _Scenario(BaseModel):
    nodes: list[EvidenceChunk] = Field(min_length=1)
    persona: Persona
    style: QueryStyle
    length: QueryLength

    @classmethod
    def from_ragas(
        cls, data: Mapping[str, Any], chunks: Iterable[Mapping[str, Any]]
    ) -> Self:
        """Resolve a serialized Ragas scenario's ordered node IDs against its source graph."""
        by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        return cls.model_validate(
            {**data, "nodes": [by_id[str(node["id"])] for node in data["nodes"]]}
        )


class SingleHopScenario(_Scenario):
    """One selected source passage and the term used in the question."""

    term: str


class MultiHopScenario(_Scenario):
    """Ordered source passages and the selected combination of themes."""

    combinations: list[str]


async def generate_sample(
    scenario: SingleHopScenario | MultiHopScenario,
    *,
    llm: LanguageModel,
    llm_context: str | None = None,
) -> dict[str, Any]:
    """Generate one archived-format sample from a selected scenario and its source IDs."""
    if isinstance(scenario, SingleHopScenario):
        nodes = [scenario.nodes[0]]
        contexts = [nodes[0].text]
        prompt = SingleHopPrompt().render(
            QueryCondition(
                persona=scenario.persona,
                term=scenario.term,
                context=contexts[0],
                query_style=scenario.style.value,
                query_length=scenario.length.value,
                llm_context=llm_context,
            )
        )
    else:
        nodes = scenario.nodes
        contexts = [f"<{i}-hop>\n\n{node.text}" for i, node in enumerate(nodes, 1)]
        prompt = MultiHopPrompt().render(
            QueryConditions(
                persona=scenario.persona,
                themes=scenario.combinations,
                context=contexts,
                query_style=scenario.style.value,
                query_length=scenario.length.value,
                llm_context=llm_context,
            )
        )
    mark_usage_source("generation", llm.plugin_id, llm.usage_tracking)
    answer = (await llm.agenerate(prompt, GeneratedQueryAnswer)).data
    return {
        "user_input": answer.query,
        "reference": answer.answer,
        "reference_contexts": contexts,
        "persona_name": scenario.persona.name,
        "query_style": scenario.style.name,
        "query_length": scenario.length.name,
        "reference_context_ids": [node.chunk_id for node in nodes],
        "reference_document_ids": [node.document_id for node in nodes],
    }

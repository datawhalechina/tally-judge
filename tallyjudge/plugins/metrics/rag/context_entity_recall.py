"""ContextEntityRecall from Ragas ec00339 (Apache-2.0), renamed ContextEntityRecallMetric.

An empty reference entity set is undefined and needs no passage extraction.
Defined scores retain the archived epsilon.
"""

from tallyjudge.core.errors import UndefinedScore
from tallyjudge.core.results import MetricResult
from tallyjudge.interfaces.metrics import MetricSpec
from tallyjudge.plugins.metrics.rag._entity_prompt import prompt
from tallyjudge.plugins.metrics.rag._judge import RagJudge


class ContextEntityRecallMetric(RagJudge):
    """Fraction of distinct reference entities also found in the retrieved passages."""

    spec = MetricSpec(
        "metric.rag.context_entity_recall",
        "1",
        {"reference": str, "retrieved_contexts": list[str]},
        value_range=(0.0, 1.0),
    )

    async def ascore(
        self, reference: str, retrieved_contexts: list[str]
    ) -> MetricResult:
        """Extract entities from the reference and from the joined passages, in order."""
        return await self.ascore_inputs(
            {"reference": reference, "retrieved_contexts": retrieved_contexts}
        )

    async def _entities(self, text: str) -> set[str]:
        template = prompt()
        reply = await self._ask(
            template.render(template.input_model(text=text)), template.output_model
        )
        return set(reply.entities)

    async def _score(
        self, reference: str, retrieved_contexts: list[str]
    ) -> MetricResult:
        reference_entities = await self._entities(reference)
        if not reference_entities:
            raise UndefinedScore("no_reference_entities")
        found = await self._entities("\n".join(retrieved_contexts))
        return MetricResult(
            len(reference_entities & found) / (len(reference_entities) + 1e-8)
        )

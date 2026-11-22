"""Migrate Ragas ContextRecall to ContextRecallMetric (ec00339, Apache-2.0).

Revision 2 rejects nonbinary attributions. The denominator is the number of
returned classifications. Empty classifications are undefined.
"""

from tallyjudge.core.errors import UndefinedScore
from tallyjudge.core.results import MetricResult
from tallyjudge.core.usage import mark_usage_source, run_scoped_score
from tallyjudge.plugins.metrics.rag._context import ContextJudge
from tallyjudge.plugins.metrics.rag._judge import binary_verdict
from tallyjudge.plugins.metrics.rag._recall_prompt import prompt


class ContextRecallMetric(ContextJudge):
    """Fraction of reference-answer statements attributed to the retrieved passages."""

    plugin_id = "metric.rag.context_recall"
    answer_field = "reference"

    async def ascore(
        self, user_input: str, retrieved_contexts: list[str], reference: str
    ) -> MetricResult:
        """Judge the reference answer against all retrieved passages in a single request."""
        return await self.ascore_inputs(
            {
                "user_input": user_input,
                "retrieved_contexts": retrieved_contexts,
                "reference": reference,
            }
        )

    async def _ascore_validated(
        self, user_input: str, retrieved_contexts: list[str], reference: str
    ) -> MetricResult:
        template = prompt()
        item = template.input_model(
            question=user_input, context="\n".join(retrieved_contexts), answer=reference
        )
        mark_usage_source("generation", self.llm.plugin_id, self.llm.usage_tracking)
        result = await run_scoped_score(
            self.llm.agenerate(template.render(item), template.output_model)
        )
        records = result.data.classifications
        if not records:
            raise UndefinedScore("no_classifications")
        attributed = sum(binary_verdict(record.attributed) for record in records)
        return MetricResult(attributed / len(records))

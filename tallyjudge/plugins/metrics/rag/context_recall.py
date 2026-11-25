"""Migrate Ragas ContextRecall to ContextRecallMetric (ec00339, Apache-2.0).

Revision 2 rejects nonbinary attributions. The denominator is the number of
returned classifications. Empty classifications are undefined.
"""

from tallyjudge.plugins.metrics._judge import binary_verdict
from tallyjudge.plugins.metrics.rag._context import ContextJudge
from tallyjudge.plugins.metrics.rag._recall_prompt import prompt
from tallyjudge.sdk.errors import UndefinedScore
from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.results import MetricResult


class ContextRecallMetric(ContextJudge):
    plugin = Plugin("metric", dependencies={"llm": "language_model"})

    plugin_id = "metric.rag.context_recall"
    answer_field = "reference"

    async def ascore(
        self, user_input: str, retrieved_contexts: list[str], reference: str
    ) -> MetricResult:
        template = prompt()
        item = template.input_model(
            question=user_input, context="\n".join(retrieved_contexts), answer=reference
        )
        result = await self.llm.agenerate(template.render(item), template.output_model)
        records = result.data.classifications
        if not records:
            raise UndefinedScore("no_classifications")
        attributed = sum(binary_verdict(record.attributed) for record in records)
        return MetricResult(attributed / len(records))

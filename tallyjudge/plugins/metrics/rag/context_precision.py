"""ContextPrecisionWithReference and ContextPrecisionWithoutReference from Ragas ec00339 (Apache-2.0).

The two renamed metrics preserve ranked average precision. No relevant verdicts make the score undefined.
"""

from tallyjudge.core.errors import UndefinedScore
from tallyjudge.core.results import MetricResult
from tallyjudge.core.usage import mark_usage_source, run_scoped_score
from tallyjudge.plugins.metrics.rag._context import ContextJudge
from tallyjudge.plugins.metrics.rag._precision_prompt import prompt


class _ContextPrecision(ContextJudge):
    async def _ascore_validated(
        self, user_input: str, retrieved_contexts: list[str], **answers: str
    ) -> MetricResult:
        return await run_scoped_score(
            self._average_precision(
                user_input, answers[self.answer_field], retrieved_contexts
            )
        )

    async def _average_precision(
        self, question: str, answer: str, contexts: list[str]
    ) -> MetricResult:
        template = prompt()
        hits = 0
        total = 0.0
        for rank, context in enumerate(contexts, 1):
            item = template.input_model(
                question=question, context=context, answer=answer
            )
            mark_usage_source("generation", self.llm.plugin_id, self.llm.usage_tracking)
            judged = await self.llm.agenerate(
                template.render(item), template.output_model
            )
            verdict = judged.data.verdict
            hits += verdict
            if verdict:
                total += hits / rank
        if not hits:
            raise UndefinedScore("no_relevant_contexts")
        return MetricResult(total / hits)


class ContextPrecisionWithReferenceMetric(_ContextPrecision):
    """Average precision of ranked passages judged against the reference answer."""

    plugin_id = "metric.rag.context_precision_with_reference"
    answer_field = "reference"

    async def ascore(
        self, user_input: str, reference: str, retrieved_contexts: list[str]
    ) -> MetricResult:
        """Judge each retrieved passage against the reference answer in retrieval order."""
        return await self.ascore_inputs(
            {
                "user_input": user_input,
                "reference": reference,
                "retrieved_contexts": retrieved_contexts,
            }
        )


class ContextPrecisionWithoutReferenceMetric(_ContextPrecision):
    """Average precision of ranked passages judged against the system response."""

    plugin_id = "metric.rag.context_precision_without_reference"
    answer_field = "response"

    async def ascore(
        self, user_input: str, response: str, retrieved_contexts: list[str]
    ) -> MetricResult:
        """Judge each retrieved passage against the generated response in retrieval order."""
        return await self.ascore_inputs(
            {
                "user_input": user_input,
                "response": response,
                "retrieved_contexts": retrieved_contexts,
            }
        )

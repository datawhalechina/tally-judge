"""ContextPrecisionWithReference and ContextPrecisionWithoutReference from Ragas ec00339 (Apache-2.0).

Revision 2 rejects nonbinary verdicts before computing ranked average precision.
A list with no relevant verdicts makes the score undefined.
"""

import asyncio

from pydantic import BaseModel, ConfigDict, Field

from tallyjudge.plugins.metrics._judge import binary_verdict
from tallyjudge.plugins.metrics.rag._context import ContextJudge
from tallyjudge.plugins.metrics.rag._precision_prompt import prompt
from tallyjudge.sdk.errors import UndefinedScore
from tallyjudge.sdk.models import LanguageModel
from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.results import MetricResult


class ContextPrecisionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    concurrency: int = Field(default=4, ge=1)


class _ContextPrecision(ContextJudge):
    Config = ContextPrecisionConfig

    def __init__(self, llm: LanguageModel, concurrency: int = 4):
        super().__init__(llm)
        self.concurrency = ContextPrecisionConfig(concurrency=concurrency).concurrency

    async def ascore(
        self, user_input: str, retrieved_contexts: list[str], **answers: str
    ) -> MetricResult:
        template = prompt()
        answer = answers[self.answer_field]

        async def judge(context: str) -> int:
            item = template.input_model(
                question=user_input, context=context, answer=answer
            )
            judged = await self.llm.agenerate(
                template.render(item), template.output_model
            )
            return binary_verdict(judged.data.verdict)

        verdicts: list[int] = []
        for start in range(0, len(retrieved_contexts), self.concurrency):
            batch = retrieved_contexts[start : start + self.concurrency]
            try:
                async with asyncio.TaskGroup() as group:
                    tasks = [group.create_task(judge(context)) for context in batch]
            except ExceptionGroup as failures:
                raise failures.exceptions[0] from None
            verdicts += [task.result() for task in tasks]
        hits = 0
        total = 0.0
        for rank, verdict in enumerate(verdicts, 1):
            hits += verdict
            if verdict:
                total += hits / rank
        if not hits:
            raise UndefinedScore("no_relevant_contexts")
        return MetricResult(total / hits)


class ContextPrecisionWithReferenceMetric(_ContextPrecision):
    plugin = Plugin("metric", dependencies={"llm": "language_model"})

    plugin_id = "metric.rag.context_precision_with_reference"
    answer_field = "reference"


class ContextPrecisionWithoutReferenceMetric(_ContextPrecision):
    plugin = Plugin("metric", dependencies={"llm": "language_model"})

    plugin_id = "metric.rag.context_precision_without_reference"
    answer_field = "response"

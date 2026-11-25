"""Faithfulness scoring and original prompts from Ragas ec00339 (Apache-2.0)."""

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tallyjudge.plugins._prompting import StructuredPrompt
from tallyjudge.plugins.metrics._judge import ModelJudge
from tallyjudge.plugins.metrics.rag._inputs import required_text_inputs
from tallyjudge.sdk.errors import ModelOutputError, UndefinedScore
from tallyjudge.sdk.metrics import MetricSpec
from tallyjudge.sdk.models import LanguageModel
from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.results import MetricResult

logger = logging.getLogger(__name__)


class StatementGeneratorInput(BaseModel):
    """Input model for statement generation."""

    question: str = Field(..., description="The question being answered")
    answer: str = Field(
        ..., description="The answer text to break down into statements"
    )


class StatementGeneratorOutput(BaseModel):
    """Structured output for statement generation."""

    statements: list[str] = Field(
        ..., description="The generated statements from the answer"
    )


class StatementGeneratorPrompt(
    StructuredPrompt[StatementGeneratorInput, StatementGeneratorOutput]
):
    input_model = StatementGeneratorInput
    output_model = StatementGeneratorOutput

    instruction = """Given a question and an answer, analyze the complexity of each sentence in the answer. Break down each sentence into one or more fully understandable statements. Ensure that no pronouns are used in any statement."""

    examples = [
        (
            StatementGeneratorInput(
                question="Who was Albert Einstein and what is he best known for?",
                answer="He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",
            ),
            StatementGeneratorOutput(
                statements=[
                    "Albert Einstein was a German-born theoretical physicist.",
                    "Albert Einstein is recognized as one of the greatest and most influential physicists of all time.",
                    "Albert Einstein was best known for developing the theory of relativity.",
                    "Albert Einstein made important contributions to the development of the theory of quantum mechanics.",
                ]
            ),
        ),
    ]


class StatementFaithfulnessAnswer(BaseModel):
    """Individual statement with reason and verdict for NLI evaluation."""

    statement: str = Field(..., description="the original statement, word-by-word")
    reason: str = Field(..., description="the reason of the verdict")
    verdict: int = Field(
        ..., ge=0, le=1, description="the verdict(0/1) of the faithfulness"
    )


class NLIStatementInput(BaseModel):
    """Input model for NLI statement evaluation."""

    context: str = Field(..., description="The context to evaluate statements against")
    statements: list[str] = Field(
        ..., description="The statements to judge for faithfulness"
    )


class NLIStatementOutput(BaseModel):
    """Structured output for NLI statement evaluation."""

    statements: list[StatementFaithfulnessAnswer] = Field(
        ..., description="Evaluated statements with verdicts"
    )


class NLIStatementPrompt(StructuredPrompt[NLIStatementInput, NLIStatementOutput]):
    input_model = NLIStatementInput
    output_model = NLIStatementOutput

    instruction = """Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as 1 if the statement can be directly inferred based on the context or 0 if the statement can not be directly inferred based on the context."""

    examples = [
        (
            NLIStatementInput(
                context="John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.",
                statements=[
                    "John is majoring in Biology.",
                    "John is taking a course on Artificial Intelligence.",
                    "John is a dedicated student.",
                    "John has a part-time job.",
                ],
            ),
            NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="John is majoring in Biology.",
                        reason="John's major is explicitly stated as Computer Science, not Biology.",
                        verdict=0,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="John is taking a course on Artificial Intelligence.",
                        reason="The context mentions courses in Data Structures, Algorithms, and Database Management, but does not mention Artificial Intelligence.",
                        verdict=0,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="John is a dedicated student.",
                        reason="The context states that John is a diligent student who spends a significant amount of time studying and completing assignments.",
                        verdict=1,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="John has a part-time job.",
                        reason="There is no information in the context about John having a part-time job.",
                        verdict=0,
                    ),
                ]
            ),
        ),
    ]


class FaithfulnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    revision: Literal["1", "2"] = "2"


class FaithfulnessMetric(ModelJudge):
    plugin = Plugin("metric", dependencies={"llm": "language_model"})
    plugin_id = "metric.rag.faithfulness"
    Config = FaithfulnessConfig

    @staticmethod
    def spec_from_config(*, revision: str = "2") -> MetricSpec:
        return MetricSpec(
            FaithfulnessMetric.plugin_id,
            revision,
            required_text_inputs("user_input", "response", passages=True),
            value_range=(0.0, 1.0),
        )

    def __init__(self, llm: LanguageModel, revision: str = "2"):
        super().__init__(llm)
        config = FaithfulnessConfig.model_validate({"revision": revision})
        self.spec = self.spec_from_config(revision=config.revision)

    async def ascore(
        self, user_input: str, response: str, retrieved_contexts: list[str]
    ) -> MetricResult:
        prompt = StatementGeneratorPrompt().render(
            StatementGeneratorInput(question=user_input, answer=response)
        )
        statements = (await self._ask(prompt, StatementGeneratorOutput)).statements
        if not statements:
            raise UndefinedScore("no_statements")
        prompt = NLIStatementPrompt().render(
            NLIStatementInput(
                context="\n".join(retrieved_contexts), statements=statements
            )
        )
        verdicts = (await self._ask(prompt, NLIStatementOutput)).statements
        if not verdicts:
            raise UndefinedScore("no_verdicts")
        score = sum(item.verdict for item in verdicts) / len(verdicts)
        if self.spec.revision == "1":
            return MetricResult(score)
        if len(verdicts) != len(statements):
            raise ModelOutputError("verdict_count")
        return MetricResult(
            score,
            details={
                "statements": [
                    {
                        "statement": statement,
                        "supported": bool(item.verdict),
                        "reason": item.reason,
                    }
                    for statement, item in zip(statements, verdicts, strict=True)
                ]
            },
        )

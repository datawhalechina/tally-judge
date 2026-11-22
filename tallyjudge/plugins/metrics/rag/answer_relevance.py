"""Answer relevance scoring and original prompts from Ragas ec00339 (Apache-2.0)."""

import math
from typing import Annotated, Any

from pydantic import BaseModel, Field

from tallyjudge.core.errors import ModelOutputError
from tallyjudge.core.prompts import StructuredPrompt
from tallyjudge.core.results import MetricResult
from tallyjudge.interfaces.metrics import MetricSpec
from tallyjudge.interfaces.models import EmbeddingModel, LanguageModel
from tallyjudge.plugins._vectors import unit_vector
from tallyjudge.plugins.metrics._cosine import embedding_vectors
from tallyjudge.plugins.metrics.rag._judge import RagJudge


class AnswerRelevanceInput(BaseModel):
    """Input model for answer relevance evaluation."""

    response: str = Field(
        ..., description="The response/answer to generate questions from"
    )


class AnswerRelevanceOutput(BaseModel):
    """Structured output for answer relevance question generation."""

    question: str = Field(
        ..., description="Question that can be answered from the response"
    )
    noncommittal: int = Field(
        ...,
        description="1 if the response is evasive/vague, 0 if it is substantive",
    )


class AnswerRelevancePrompt(
    StructuredPrompt[AnswerRelevanceInput, AnswerRelevanceOutput]
):
    """Answer relevance evaluation prompt with structured input/output."""

    input_model = AnswerRelevanceInput
    output_model = AnswerRelevanceOutput

    instruction = """Generate a question for the given answer and identify if the answer is noncommittal.
Give noncommittal as 1 if the answer is noncommittal (evasive, vague, or ambiguous) and 0 if the answer is substantive.
Examples of noncommittal answers: "I don't know", "I'm not sure", "It depends"."""

    examples = [
        (
            AnswerRelevanceInput(response="Albert Einstein was born in Germany."),
            AnswerRelevanceOutput(
                question="Where was Albert Einstein born?",
                noncommittal=0,
            ),
        ),
        (
            AnswerRelevanceInput(
                response="The capital of France is Paris, a city known for its architecture and culture."
            ),
            AnswerRelevanceOutput(
                question="What is the capital of France?",
                noncommittal=0,
            ),
        ),
        (
            AnswerRelevanceInput(
                response="I don't know about the groundbreaking feature of the smartphone invented in 2023 as I am unaware of information beyond 2022."
            ),
            AnswerRelevanceOutput(
                question="What was the groundbreaking feature of the smartphone invented in 2023?",
                noncommittal=1,
            ),
        ),
    ]


class AnswerRelevanceConfig(BaseModel):
    strictness: int = Field(default=3, gt=0)


class AnswerRelevanceMetric(RagJudge):
    """Compare the original question with questions inferred from the answer."""

    spec = MetricSpec(
        "metric.rag.answer_relevance",
        "1",
        {
            "user_input": Annotated[str, Field(min_length=1)],
            "response": Annotated[str, Field(min_length=1)],
        },
        value_range=(-1.0, 1.0),
    )

    def __init__(
        self, llm: LanguageModel, embeddings: EmbeddingModel, strictness: int = 3
    ):
        if type(strictness) is not int or strictness < 1:
            raise ValueError("strictness must be a positive integer")
        super().__init__(llm)
        self.embeddings = embeddings
        self.strictness = strictness

    def config_snapshot(self) -> dict[str, Any]:
        return {
            **super().config_snapshot(),
            "strictness": self.strictness,
            "embeddings": {
                "plugin_id": self.embeddings.plugin_id,
                "config": self.embeddings.config_snapshot(),
            },
        }

    async def ascore(self, user_input: str, response: str) -> MetricResult:
        """Score one answer against the original question."""
        return await self.ascore_inputs(
            {"user_input": user_input, "response": response}
        )

    async def _score(self, user_input: str, response: str) -> MetricResult:
        prompt = AnswerRelevancePrompt().render(AnswerRelevanceInput(response=response))
        questions = []
        flags = []
        for _ in range(self.strictness):
            result = await self._ask(prompt, AnswerRelevanceOutput)
            if result.question:
                questions.append(result.question)
                flags.append(result.noncommittal)
        if not questions:
            return MetricResult(0.0)
        original = await embedding_vectors(self.embeddings, [user_input])
        generated = await embedding_vectors(self.embeddings, questions)
        if (
            len(original) != 1
            or len(generated) != len(questions)
            or any(len(v) != len(original[0]) for v in generated)
        ):
            raise ModelOutputError("invalid_embedding")
        direction = unit_vector(original[0])
        similarities = [
            math.fsum(a * b for a, b in zip(direction, unit_vector(v), strict=True))
            for v in generated
        ]
        value = math.fsum(similarities) / len(similarities) * int(not all(flags))
        if 1.0 < abs(value) <= 1.0 + 1e-12:
            value = math.copysign(1.0, value)
        return MetricResult(value)

"""Answer relevance scoring and original prompts from Ragas ec00339 (Apache-2.0)."""

import math
from typing import Any

from pydantic import BaseModel, Field

from tallyjudge.plugins._prompting import StructuredPrompt
from tallyjudge.plugins._vectors import unit_vector
from tallyjudge.plugins.metrics._cosine import embedding_vectors
from tallyjudge.plugins.metrics._judge import ModelJudge
from tallyjudge.plugins.metrics.rag._inputs import required_text_inputs
from tallyjudge.sdk.errors import ModelOutputError
from tallyjudge.sdk.gate import gated
from tallyjudge.sdk.metrics import MetricSpec
from tallyjudge.sdk.models import EmbeddingModel, LanguageModel, model_snapshot
from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.results import MetricResult


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


class AnswerRelevanceMetric(ModelJudge):
    plugin = Plugin(
        "metric",
        dependencies={"llm": "language_model", "embeddings": "embedding_model"},
    )

    Config = AnswerRelevanceConfig

    spec = MetricSpec(
        "metric.rag.answer_relevance",
        "1",
        required_text_inputs("user_input", "response"),
        value_range=(-1.0, 1.0),
    )

    def __init__(
        self, llm: LanguageModel, embeddings: EmbeddingModel, strictness: int = 3
    ):
        config = AnswerRelevanceConfig.model_validate(
            {"strictness": strictness}, strict=True
        )
        super().__init__(llm)
        self.embeddings = gated(embeddings)
        self.strictness = config.strictness

    def config_snapshot(self) -> dict[str, Any]:
        return {
            **super().config_snapshot(),
            "strictness": self.strictness,
            "embeddings": model_snapshot(self.embeddings),
        }

    async def ascore(self, user_input: str, response: str) -> MetricResult:
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

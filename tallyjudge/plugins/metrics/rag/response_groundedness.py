"""ResponseGroundedness from Ragas ec00339 (Apache-2.0), renamed ResponseGroundednessMetric.

Available ratings are normalized as in the archive. An empty set of ratings is undefined."""

from functools import cache

from tallyjudge.core.prompts import StructuredPrompt
from tallyjudge.core.results import MetricResult
from tallyjudge.interfaces.metrics import MetricSpec
from tallyjudge.plugins.metrics.rag._dual_judge import DualJudge
from tallyjudge.plugins.metrics.rag._judge import required_text_inputs


@cache
def _prompts():
    """Build both archived rating prompts."""
    from pydantic import BaseModel, Field

    class ResponseGroundednessInput(BaseModel):
        """Input model for response groundedness evaluation."""

        response: str = Field(..., description="The response/assertion to evaluate")
        context: str = Field(..., description="The context to evaluate against")

    class ResponseGroundednessOutput(BaseModel):
        """Structured output for response groundedness evaluation."""

        rating: int = Field(..., description="Groundedness rating (0, 1, or 2)")

    class ResponseGroundednessJudge1Prompt(
        StructuredPrompt[ResponseGroundednessInput, ResponseGroundednessOutput]
    ):
        """First judge prompt for response groundedness evaluation."""

        input_model = ResponseGroundednessInput
        output_model = ResponseGroundednessOutput

        instruction = """You are a world class expert designed to evaluate the groundedness of an assertion.
You will be provided with an assertion and a context.
Your task is to determine if the assertion is supported by the context.
Follow the instructions below:
A. If there is no context or no assertion or context is empty or assertion is empty, say 0.
B. If the assertion is not supported by the context, say 0.
C. If the assertion is partially supported by the context, say 1.
D. If the assertion is fully supported by the context, say 2.
You must provide a rating of 0, 1, or 2, nothing else.
Return your response as JSON in this format: {"rating": X} where X is 0, 1, or 2."""

        examples = [
            (
                ResponseGroundednessInput(
                    response="Albert Einstein was born in Germany.",
                    context="Albert Einstein was born March 14, 1879 at Ulm, in Württemberg, Germany.",
                ),
                ResponseGroundednessOutput(rating=2),
            ),
            (
                ResponseGroundednessInput(
                    response="Einstein was a chemist who invented gunpowder.",
                    context="Albert Einstein was a theoretical physicist known for his theory of relativity.",
                ),
                ResponseGroundednessOutput(rating=0),
            ),
            (
                ResponseGroundednessInput(
                    response="Einstein received the Nobel Prize.",
                    context="Albert Einstein received the 1921 Nobel Prize in Physics for his services to theoretical physics.",
                ),
                ResponseGroundednessOutput(rating=2),
            ),
        ]

    class ResponseGroundednessJudge2Prompt(
        StructuredPrompt[ResponseGroundednessInput, ResponseGroundednessOutput]
    ):
        """Second judge prompt for response groundedness evaluation."""

        input_model = ResponseGroundednessInput
        output_model = ResponseGroundednessOutput

        instruction = """As a specialist in assessing the strength of connections between statements and their given contexts, I will evaluate the level of support an assertion receives from the provided context. Follow these guidelines:

* If the assertion is not supported or context is empty or assertion is empty, assign a score of 0.
* If the assertion is partially supported, assign a score of 1.
* If the assertion is fully supported, assign a score of 2.

I will provide a rating of 0, 1, or 2, without any additional information.
Return your response as JSON in this format: {"rating": X} where X is 0, 1, or 2."""

        examples = [
            (
                ResponseGroundednessInput(
                    response="Albert Einstein was a scientist.",
                    context="Albert Einstein was a German-born theoretical physicist widely held to be one of the greatest and most influential scientists of all time.",
                ),
                ResponseGroundednessOutput(rating=2),
            ),
            (
                ResponseGroundednessInput(
                    response="Einstein invented television.",
                    context="Albert Einstein developed the theory of relativity.",
                ),
                ResponseGroundednessOutput(rating=0),
            ),
            (
                ResponseGroundednessInput(
                    response="Einstein won a Nobel Prize.",
                    context="Albert Einstein received the 1921 Nobel Prize in Physics.",
                ),
                ResponseGroundednessOutput(rating=2),
            ),
        ]

    return ResponseGroundednessJudge1Prompt(), ResponseGroundednessJudge2Prompt()


class ResponseGroundednessMetric(DualJudge):
    """Rate how well the retrieved passages support the entire response."""

    grades = (0, 1, 2)
    spec = MetricSpec(
        "metric.rag.response_groundedness",
        "3",
        required_text_inputs("response", passages=True),
        value_range=(0.0, 1.0),
    )

    async def ascore(
        self, response: str, retrieved_contexts: list[str]
    ) -> MetricResult:
        """Score the response against the joined passages with two groundedness prompts."""
        return await self.ascore_inputs(
            {"response": response, "retrieved_contexts": retrieved_contexts}
        )

    async def _score(
        self, response: str, retrieved_contexts: list[str]
    ) -> MetricResult:
        context = "\n".join(retrieved_contexts)
        if not response.strip() or not context.strip():
            return MetricResult(0.0)
        fields = {"response": response, "context": context}
        return await self._rate_pair(_prompts(), fields, fields)

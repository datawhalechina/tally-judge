"""Claim decomposition and NLI prompts from Ragas ec00339 (Apache-2.0)."""

from functools import cache

from tallyjudge.core.prompts import StructuredPrompt


@cache
def prompts():
    """Build the archived schemas and per-granularity example sets."""
    from pydantic import BaseModel, Field

    class ClaimDecompositionInput(BaseModel):
        """Input for claim decomposition."""

        response: str = Field(
            ..., description="The response text to decompose into claims"
        )
        atomicity: str = Field(
            default="low", description="Atomicity level: 'low' or 'high'"
        )
        coverage: str = Field(
            default="low", description="Coverage level: 'low' or 'high'"
        )

    class ClaimDecompositionOutput(BaseModel):
        """Output from claim decomposition."""

        claims: list[str] = Field(..., description="Decomposed claims")

    class ClaimDecompositionPrompt(
        StructuredPrompt[ClaimDecompositionInput, ClaimDecompositionOutput]
    ):
        """Prompt for decomposing text into claims with configurable atomicity and coverage."""

        input_model = ClaimDecompositionInput
        output_model = ClaimDecompositionOutput

        instruction = """Decompose and break down each of the input sentences into one or more standalone statements. Each statement should be a standalone claim that can be independently verified.
Follow the level of atomicity and coverage as shown in the examples."""

        _all_examples: dict[
            tuple[str, str],
            list[tuple[ClaimDecompositionInput, ClaimDecompositionOutput]],
        ] = {
            ("low", "low"): [
                (
                    ClaimDecompositionInput(
                        response="Charles Babbage was a French mathematician, philosopher, and food critic.",
                        atomicity="low",
                        coverage="low",
                    ),
                    ClaimDecompositionOutput(
                        claims=["Charles Babbage was a mathematician and philosopher."]
                    ),
                ),
                (
                    ClaimDecompositionInput(
                        response="Albert Einstein was a German theoretical physicist. He developed the theory of relativity and also contributed to the development of quantum mechanics.",
                        atomicity="low",
                        coverage="low",
                    ),
                    ClaimDecompositionOutput(
                        claims=[
                            "Albert Einstein was a German physicist.",
                            "Albert Einstein developed relativity and contributed to quantum mechanics.",
                        ]
                    ),
                ),
            ],
            ("low", "high"): [
                (
                    ClaimDecompositionInput(
                        response="Charles Babbage was a French mathematician, philosopher, and food critic.",
                        atomicity="low",
                        coverage="high",
                    ),
                    ClaimDecompositionOutput(
                        claims=[
                            "Charles Babbage was a French mathematician, philosopher, and food critic."
                        ]
                    ),
                ),
                (
                    ClaimDecompositionInput(
                        response="Albert Einstein was a German theoretical physicist. He developed the theory of relativity and also contributed to the development of quantum mechanics.",
                        atomicity="low",
                        coverage="high",
                    ),
                    ClaimDecompositionOutput(
                        claims=[
                            "Albert Einstein was a German theoretical physicist.",
                            "Albert Einstein developed the theory of relativity and also contributed to the development of quantum mechanics.",
                        ]
                    ),
                ),
            ],
            ("high", "low"): [
                (
                    ClaimDecompositionInput(
                        response="Charles Babbage was a French mathematician, philosopher, and food critic.",
                        atomicity="high",
                        coverage="low",
                    ),
                    ClaimDecompositionOutput(
                        claims=[
                            "Charles Babbage was a mathematician.",
                            "Charles Babbage was a philosopher.",
                        ]
                    ),
                ),
                (
                    ClaimDecompositionInput(
                        response="Albert Einstein was a German theoretical physicist. He developed the theory of relativity and also contributed to the development of quantum mechanics.",
                        atomicity="high",
                        coverage="low",
                    ),
                    ClaimDecompositionOutput(
                        claims=[
                            "Albert Einstein was a German theoretical physicist.",
                            "Albert Einstein developed the theory of relativity.",
                        ]
                    ),
                ),
            ],
            ("high", "high"): [
                (
                    ClaimDecompositionInput(
                        response="Charles Babbage was a French mathematician, philosopher, and food critic.",
                        atomicity="high",
                        coverage="high",
                    ),
                    ClaimDecompositionOutput(
                        claims=[
                            "Charles Babbage was a mathematician.",
                            "Charles Babbage was a philosopher.",
                            "Charles Babbage was a food critic.",
                            "Charles Babbage was French.",
                        ]
                    ),
                ),
                (
                    ClaimDecompositionInput(
                        response="Albert Einstein was a German theoretical physicist. He developed the theory of relativity and also contributed to the development of quantum mechanics.",
                        atomicity="high",
                        coverage="high",
                    ),
                    ClaimDecompositionOutput(
                        claims=[
                            "Albert Einstein was a German theoretical physicist.",
                            "Albert Einstein developed the theory of relativity.",
                            "Albert Einstein contributed to the development of quantum mechanics.",
                        ]
                    ),
                ),
            ],
        }

        def __init__(self, atomicity: str, coverage: str):
            self.examples = self._all_examples[(atomicity, coverage)]

    class NLIStatementInput(BaseModel):
        """Input for NLI statement evaluation."""

        context: str = Field(
            ..., description="The context to evaluate statements against"
        )
        statements: list[str] = Field(
            ..., description="The statements to judge for faithfulness"
        )

    class StatementFaithfulnessAnswer(BaseModel):
        """Individual statement with reason and verdict for NLI evaluation."""

        statement: str = Field(..., description="the original statement, word-by-word")
        reason: str = Field(..., description="the reason of the verdict")
        verdict: int = Field(..., description="the verdict(0/1) of the faithfulness.")

    class NLIStatementOutput(BaseModel):
        """Structured output for NLI statement evaluation."""

        statements: list[StatementFaithfulnessAnswer]

    class NLIStatementPrompt(StructuredPrompt[NLIStatementInput, NLIStatementOutput]):
        """Prompt for evaluating statement faithfulness using NLI."""

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

    return ClaimDecompositionPrompt, NLIStatementPrompt()

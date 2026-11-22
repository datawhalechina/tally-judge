"""Graph transform and scenario planning prompts from Ragas ec00339 (Apache-2.0).

Prompt schemas, examples and instructions are part of the migration contract.
"""

import typing as t

from pydantic import BaseModel, Field

from tallyjudge.core.prompts import StructuredPrompt
from tallyjudge.plugins.synthesis._prompts import Persona, TextWithExtractionLimit


class StringIO(BaseModel):
    text: str


DEFAULT_RUBRICS = {
    "score1_description": "The page content is irrelevant or does not align with the main themes or topics of the document summary.",
    "score2_description": "The page content partially aligns with the document summary, but it includes unrelated details or lacks critical information related to the document's main themes.",
    "score3_description": "The page content generally reflects the document summary but may miss key details or lack depth in addressing the main themes.",
    "score4_description": "The page content aligns well with the document summary, covering the main themes and topics with minor gaps or minimal unrelated information.",
    "score5_description": "The page content is highly relevant, accurate, and directly reflects the main themes of the document summary, covering all important details and adding depth to the understanding of the document's topics.",
}


class SummaryExtractorPrompt(StructuredPrompt[StringIO, StringIO]):
    instruction: str = "Summarize the given text in less than 10 sentences."
    input_model: t.Type[StringIO] = StringIO
    output_model: t.Type[StringIO] = StringIO
    examples: t.List[t.Tuple[StringIO, StringIO]] = [
        (
            StringIO(
                text="Artificial intelligence\n\nArtificial intelligence is transforming various industries by automating tasks that previously required human intelligence. From healthcare to finance, AI is being used to analyze vast amounts of data quickly and accurately. This technology is also driving innovations in areas like self-driving cars and personalized recommendations."
            ),
            StringIO(
                text="AI is revolutionizing industries by automating tasks, analyzing data, and driving innovations like self-driving cars and personalized recommendations."
            ),
        )
    ]


class Headlines(BaseModel):
    headlines: t.List[str]


class HeadlinesExtractorPrompt(StructuredPrompt[TextWithExtractionLimit, Headlines]):
    instruction: str = (
        "Extract the most important max_num headlines from the given text that can be used to split the text into independent sections."
        "Focus on Level 2 and Level 3 headings."
    )

    input_model: t.Type[TextWithExtractionLimit] = TextWithExtractionLimit
    output_model: t.Type[Headlines] = Headlines
    examples: t.List[t.Tuple[TextWithExtractionLimit, Headlines]] = [
        (
            TextWithExtractionLimit(
                text="""\
                Introduction
                Overview of the topic...

                Main Concepts
                Explanation of core ideas...

                Detailed Analysis
                Techniques and methods for analysis...

                Subsection: Specialized Techniques
                Further details on specialized techniques...

                Future Directions
                Insights into upcoming trends...

                Subsection: Next Steps in Research
                Discussion of new areas of study...

                Conclusion
                Final remarks and summary.
                """,
                max_num=6,
            ),
            Headlines(
                headlines=[
                    "Introduction",
                    "Main Concepts",
                    "Detailed Analysis",
                    "Subsection: Specialized Techniques",
                    "Future Directions",
                    "Conclusion",
                ],
            ),
        ),
    ]


class ThemesAndConcepts(BaseModel):
    output: t.List[str]


class ThemesAndConceptsExtractorPrompt(
    StructuredPrompt[TextWithExtractionLimit, ThemesAndConcepts]
):
    instruction: str = "Extract the main themes and concepts from the given text."
    input_model: t.Type[TextWithExtractionLimit] = TextWithExtractionLimit
    output_model: t.Type[ThemesAndConcepts] = ThemesAndConcepts
    examples: t.List[t.Tuple[TextWithExtractionLimit, ThemesAndConcepts]] = [
        (
            TextWithExtractionLimit(
                text="Artificial intelligence is transforming industries by automating tasks requiring human intelligence. AI analyzes vast data quickly and accurately, driving innovations like self-driving cars and personalized recommendations.",
                max_num=10,
            ),
            ThemesAndConcepts(
                output=[
                    "Artificial intelligence",
                    "Automation",
                    "Data analysis",
                    "Innovation",
                    "Self-driving cars",
                    "Personalized recommendations",
                ]
            ),
        )
    ]


class QuestionPotentialInput(BaseModel):
    document_summary: str = Field(
        ...,
        description="The summary of the document to provide context for evaluating the node.",
    )
    node_content: str = Field(
        ...,
        description="The content of the node to evaluate for question generation potential.",
    )
    rubrics: t.Dict[str, str] = Field(..., description="The rubric")


class QuestionPotentialOutput(BaseModel):
    score: int = Field(
        ...,
        description="1 to 5 score",
    )


class QuestionPotentialPrompt(
    StructuredPrompt[QuestionPotentialInput, QuestionPotentialOutput]
):
    instruction = (
        "Given a document summary and node content, score the content of the node in 1 to 5 range."
        ""
    )
    input_model = QuestionPotentialInput
    output_model = QuestionPotentialOutput
    examples: list[tuple[QuestionPotentialInput, QuestionPotentialOutput]] = []
    _empty_example_prefix = ""


class PersonaGenerationPrompt(StructuredPrompt[StringIO, Persona]):
    instruction: str = (
        "Using the provided summary, generate a single persona who would likely "
        "interact with or benefit from the content. Include a unique name and a "
        "concise role description of who they are."
    )
    input_model: t.Type[StringIO] = StringIO
    output_model: t.Type[Persona] = Persona
    examples: t.List[t.Tuple[StringIO, Persona]] = [
        (
            StringIO(
                text="Guide to Digital Marketing explains strategies for engaging audiences across various online platforms."
            ),
            Persona(
                name="Digital Marketing Specialist",
                role_description="Focuses on engaging audiences and growing the brand online.",
            ),
        )
    ]


class ThemesPersonasInput(BaseModel):
    themes: t.List[str]
    personas: t.List[Persona]


class PersonaThemesMapping(BaseModel):
    mapping: t.Dict[str, t.List[str]]


class ThemesPersonasMatchingPrompt(
    StructuredPrompt[ThemesPersonasInput, PersonaThemesMapping]
):
    instruction: str = (
        "Given a list of themes and personas with their roles, "
        "associate each persona with relevant themes based on their role description."
    )
    input_model: t.Type[ThemesPersonasInput] = ThemesPersonasInput
    output_model: t.Type[PersonaThemesMapping] = PersonaThemesMapping
    examples: t.List[t.Tuple[ThemesPersonasInput, PersonaThemesMapping]] = [
        (
            ThemesPersonasInput(
                themes=["Empathy", "Inclusivity", "Remote work"],
                personas=[
                    Persona(
                        name="HR Manager",
                        role_description="Focuses on inclusivity and employee support.",
                    ),
                    Persona(
                        name="Remote Team Lead",
                        role_description="Manages remote team communication.",
                    ),
                ],
            ),
            PersonaThemesMapping(
                mapping={
                    "HR Manager": ["Inclusivity", "Empathy"],
                    "Remote Team Lead": ["Remote work", "Empathy"],
                }
            ),
        )
    ]


class ConceptsList(BaseModel):
    lists_of_concepts: t.List[t.List[str]] = Field(
        description="A list containing lists of concepts from each node"
    )
    max_combinations: int = Field(
        description="The maximum number of concept combinations to generate", default=5
    )


class ConceptCombinations(BaseModel):
    combinations: t.List[t.List[str]]


class ConceptCombinationPrompt(StructuredPrompt[ConceptsList, ConceptCombinations]):
    instruction: str = (
        "Form combinations by pairing concepts from at least two different lists.\n"
        "**Instructions:**\n"
        "- Review the concepts from each node.\n"
        "- Identify concepts that can logically be connected or contrasted.\n"
        "- Form combinations that involve concepts from different nodes.\n"
        "- Each combination should include at least one concept from two or more nodes.\n"
        "- List the combinations clearly and concisely.\n"
        "- Do not repeat the same combination more than once."
    )
    input_model: t.Type[ConceptsList] = (
        ConceptsList  # Contains lists of concepts from each node
    )
    output_model: t.Type[ConceptCombinations] = (
        ConceptCombinations  # Contains list of concept combinations
    )
    examples: t.List[t.Tuple[ConceptsList, ConceptCombinations]] = [
        (
            ConceptsList(
                lists_of_concepts=[
                    ["Artificial intelligence", "Automation"],  # Concepts from Node 1
                    ["Healthcare", "Data privacy"],  # Concepts from Node 2
                ],
                max_combinations=2,
            ),
            ConceptCombinations(
                combinations=[
                    ["Artificial intelligence", "Healthcare"],
                    ["Automation", "Data privacy"],
                ]
            ),
        )
    ]

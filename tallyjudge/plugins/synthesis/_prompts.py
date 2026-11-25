"""Ragas ec00339 testset creation and NER prompts (Apache-2.0).

SingleHopPrompt and MultiHopPrompt are the two upstream QueryAnswerGenerationPrompt
classes. Output model names, instructions, examples and their order are preserved.
"""

import typing as t

from pydantic import BaseModel

from tallyjudge.plugins._prompting import StructuredPrompt


class Persona(BaseModel):
    name: str
    role_description: str


class QueryCondition(BaseModel):
    persona: Persona
    term: str
    query_style: str
    query_length: str
    context: str
    llm_context: t.Optional[str] = None


class GeneratedQueryAnswer(BaseModel):
    query: str
    answer: str


class SingleHopPrompt(StructuredPrompt[QueryCondition, GeneratedQueryAnswer]):
    instruction: str = (
        "Generate a single-hop query and answer based on the specified conditions (persona, term, style, length) "
        "and the provided context. Ensure the answer is entirely faithful to the context, using only the information "
        "directly from the provided context."
        "### Instructions:\n"
        "1. **Generate a Query**: Based on the context, persona, term, style, and length, create a question "
        "that aligns with the persona's perspective and incorporates the term.\n"
        "2. **Generate an Answer**: Using only the content from the provided context, construct a detailed answer "
        "to the query. Do not add any information not included in or inferable from the context.\n"
        "3. **Additional Context** (if provided): If llm_context is provided, use it as guidance for "
        "what type of question to generate (e.g., comparison questions, how-to questions, application-based questions) "
        "and how to structure the answer accordingly. Still ensure the content comes only from the provided context.\n"
    )
    input_model: t.Type[QueryCondition] = QueryCondition
    output_model: t.Type[GeneratedQueryAnswer] = GeneratedQueryAnswer
    examples: t.List[t.Tuple[QueryCondition, GeneratedQueryAnswer]] = [
        (
            QueryCondition(
                persona=Persona(
                    name="Software Engineer",
                    role_description="Focuses on coding best practices and system design.",
                ),
                term="microservices",
                query_style="Formal",
                query_length="Medium",
                context="Microservices are an architectural style where applications are structured as a collection of loosely coupled services. "
                "Each service is fine-grained and focuses on a single functionality.",
            ),
            GeneratedQueryAnswer(
                query="What is the purpose of microservices in software architecture?",
                answer="Microservices are designed to structure applications as a collection of loosely coupled services, each focusing on a single functionality.",
            ),
        ),
    ]


class QueryConditions(BaseModel):
    persona: Persona
    themes: t.List[str]
    query_style: str
    query_length: str
    context: t.List[str]
    llm_context: t.Optional[str] = None


class MultiHopPrompt(StructuredPrompt[QueryConditions, GeneratedQueryAnswer]):
    instruction: str = (
        "Generate a multi-hop query and answer based on the specified conditions (persona, themes, style, length) "
        "and the provided context. The themes represent a set of phrases either extracted or generated from the "
        "context, which highlight the suitability of the selected context for multi-hop query creation. Ensure the query "
        "explicitly incorporates these themes."
        "### Instructions:\n"
        "1. **Generate a Multi-Hop Query**: Use the provided context segments and themes to form a query that requires combining "
        "information from multiple segments (e.g., `<1-hop>` and `<2-hop>`). Ensure the query explicitly incorporates one or more "
        "themes and reflects their relevance to the context.\n"
        "2. **Generate an Answer**: Use only the content from the provided context to create a detailed and faithful answer to "
        "the query. Avoid adding information that is not directly present or inferable from the given context.\n"
        "3. **Multi-Hop Context Tags**:\n"
        "   - Each context segment is tagged as `<1-hop>`, `<2-hop>`, etc.\n"
        "   - Ensure the query uses information from at least two segments and connects them meaningfully.\n"
        "4. **Additional Context** (if provided): If llm_context is provided, use it as guidance for "
        "what type of question to generate (e.g., comparison questions, cause-effect questions, application-based questions) "
        "and how to structure the answer accordingly. Still ensure the content comes only from the provided context."
    )
    input_model: t.Type[QueryConditions] = QueryConditions
    output_model: t.Type[GeneratedQueryAnswer] = GeneratedQueryAnswer
    examples: t.List[t.Tuple[QueryConditions, GeneratedQueryAnswer]] = [
        (
            QueryConditions(
                persona=Persona(
                    name="Historian",
                    role_description="Focuses on major scientific milestones and their global impact.",
                ),
                themes=["Theory of Relativity", "Experimental Validation"],
                query_style="Formal",
                query_length="Medium",
                context=[
                    "<1-hop> Albert Einstein developed the theory of relativity, introducing the concept of spacetime.",
                    "<2-hop> The bending of light by gravity was confirmed during the 1919 solar eclipse, supporting Einstein’s theory.",
                ],
            ),
            GeneratedQueryAnswer(
                query="How was the experimental validation of the theory of relativity achieved during the 1919 solar eclipse?",
                answer=(
                    "The experimental validation of the theory of relativity was achieved during the 1919 solar eclipse by confirming "
                    "the bending of light by gravity, which supported Einstein’s concept of spacetime as proposed in the theory."
                ),
            ),
        ),
    ]


class TextWithExtractionLimit(BaseModel):
    text: str
    max_num: int = 10


class NEROutput(BaseModel):
    entities: t.List[str]


class NERPrompt(StructuredPrompt[TextWithExtractionLimit, NEROutput]):
    instruction: str = (
        "Extract the named entities from the given text, limiting the output to the top entities. "
        "Ensure the number of entities does not exceed the specified maximum."
    )
    input_model: t.Type[TextWithExtractionLimit] = TextWithExtractionLimit
    output_model: t.Type[NEROutput] = NEROutput
    examples: t.List[t.Tuple[TextWithExtractionLimit, NEROutput]] = [
        (
            TextWithExtractionLimit(
                text="""Elon Musk, the CEO of Tesla and SpaceX, announced plans to expand operations to new locations in Europe and Asia.
                This expansion is expected to create thousands of jobs, particularly in cities like Berlin and Shanghai.""",
                max_num=10,
            ),
            NEROutput(
                entities=[
                    "Elon Musk",
                    "Tesla",
                    "SpaceX",
                    "Europe",
                    "Asia",
                    "Berlin",
                    "Shanghai",
                ]
            ),
        ),
    ]

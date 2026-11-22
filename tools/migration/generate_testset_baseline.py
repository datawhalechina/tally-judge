"""Replay the Ragas ec00339 graph and automatic text RAG testset offline."""

import argparse
import asyncio
import hashlib
import inspect
import json
import random
import subprocess
from unittest.mock import patch
from collections import Counter
from pathlib import Path
from uuid import UUID

from generate_prompt_baselines import CharacterTokenizer, ScenarioJudge
from generate_rag_baselines import BASE, Embeddings
from langchain_core.documents import Document
from ragas.prompt import StringIO
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.persona import (
    Persona,
    PersonaGenerationPrompt,
    generate_personas_from_kg,
)
from ragas.testset.synthesizers import default_query_distribution
from ragas.testset.synthesizers.multi_hop.prompts import (
    ConceptCombinationPrompt,
    ConceptsList,
)
from ragas.testset.synthesizers.prompts import (
    ThemesPersonasInput,
    ThemesPersonasMatchingPrompt,
)
from ragas.testset.synthesizers.utils import calculate_split_values
from ragas.testset.transforms.default import (
    default_transforms,
    default_transforms_for_prechunked,
)
from ragas.testset.transforms.engine import apply_transforms
from ragas.testset.transforms.extractors.llm_based import (
    HeadlinesExtractorPrompt,
    SummaryExtractorPrompt,
    TextWithExtractionLimit,
    ThemesAndConceptsExtractorPrompt,
)
from ragas.testset.transforms.filters import (
    DEFAULT_RUBRICS,
    QuestionPotentialInput,
    QuestionPotentialPrompt,
)


class CorpusJudge(ScenarioJudge):
    def __init__(self):
        self.calls = Counter()

    async def agenerate(self, prompt, response_model, **kwargs):
        name = response_model.__name__
        self.calls[name] += 1
        data = json.loads(prompt.rsplit("input: ", 1)[1].removesuffix("\nOutput: "))
        if name == "StringIO":
            values = {"text": data["text"].split(".")[0] + "."}
        elif name == "Headlines":
            values = {"headlines": ["First", "Second"]}
        elif name == "QuestionPotentialOutput":
            values = {"score": 2 if data["node_content"].startswith("Discount.") else 3}
        elif name == "ThemesAndConcepts":
            values = {"output": ["Design" if "Ada" in data["text"] else "City"]}
        elif name == "NEROutput":
            values = {"entities": ["Guide", "Kit"]}
        elif name == "Persona":
            values = {"name": "Reader", "role_description": "Researches Kit"}
        elif name == "PersonaThemesMapping":
            values = {"mapping": {"Reader": data["themes"]}}
        elif name == "ConceptCombinations":
            values = {"combinations": [["Design", "City"]]}
        elif name == "GeneratedQueryAnswer":
            values = (
                (
                    {"query": "Who designed Kit?", "answer": "Ada"}
                    if "Ada" in data["context"]
                    else {"query": "Where does Kit live?", "answer": "Rome"}
                )
                if isinstance(data["context"], str)
                else {"query": "Where is the object designed by Ada?", "answer": "Rome"}
            )
        else:
            raise AssertionError(name)
        return response_model.model_validate(values)


class LocalEmbeddings(Embeddings):
    is_async = True


def apply_locally(graph, transforms):
    for step in transforms:
        for participant in getattr(step, "transformations", [step]):
            if hasattr(participant, "tokenizer"):
                participant.tokenizer = CharacterTokenizer()
    apply_transforms(graph, transforms)


def raw_documents():
    examples = {
        "medium": [
            "Ada designed Kit." + " pad" * 110,
            "Kit lives in Rome." + " pad" * 110,
        ],
        "long": ["First Ada. " + "alpha " * 500 + "Second Rome. " + "beta " * 500],
        "short": ["Too short."],
    }
    reports = {}
    for name, texts in examples.items():
        graph = KnowledgeGraph(
            nodes=[
                Node(
                    id=UUID(int=i),
                    type=NodeType.DOCUMENT,
                    properties={"page_content": text},
                )
                for i, text in enumerate(texts, 1)
            ]
        )
        model = CorpusJudge()
        embed = LocalEmbeddings(
            {
                "Ada designed Kit.": [1.0, 0.0],
                "Kit lives in Rome.": [0.6, 0.8],
                "First Ada.": [1.0, 0.0],
            },
            [],
        )
        # Both runtimes receive the same offline word counter at the length decision and splitter.
        count = lambda text: len(text.split())
        with (
            patch("ragas.testset.transforms.default.num_tokens_from_string", count),
            patch(
                "ragas.testset.transforms.splitters.headline.num_tokens_from_string",
                count,
            ),
        ):
            try:
                transforms = default_transforms(
                    [Document(page_content=t) for t in texts], model, embed
                )
            except ValueError as exc:
                reports[name] = {
                    "error": type(exc).__name__,
                    "calls": dict(model.calls),
                }
                continue
            apply_locally(graph, transforms)
        reports[name] = {
            "nodes": [
                [
                    n.type.value,
                    hashlib.sha256(n.properties["page_content"].encode()).hexdigest(),
                    n.properties.get("summary"),
                ]
                for n in graph.nodes
            ],
            "links": sorted(e.type for e in graph.relationships),
            "calls": dict(model.calls),
            "embedded": embed.calls,
        }
    return reports


def capture(source_root):
    installed = Path(inspect.getfile(KnowledgeGraph)).parents[1]
    sources = (
        "graph.py",
        "graph_queries.py",
        "persona.py",
        "transforms/default.py",
        "transforms/filters.py",
        "transforms/base.py",
        "transforms/engine.py",
        "transforms/splitters/headline.py",
        "transforms/extractors/llm_based.py",
        "transforms/extractors/embeddings.py",
        "transforms/relationship_builders/cosine.py",
        "transforms/relationship_builders/traditional.py",
        "synthesizers/__init__.py",
        "synthesizers/single_hop/base.py",
        "synthesizers/single_hop/specific.py",
        "synthesizers/multi_hop/base.py",
        "synthesizers/multi_hop/specific.py",
        "synthesizers/multi_hop/abstract.py",
        "synthesizers/prompts.py",
        "synthesizers/multi_hop/prompts.py",
        "synthesizers/utils.py",
    )
    for path in sources:
        expected = subprocess.check_output(
            ["git", "show", f"{BASE}:src/ragas/testset/{path}"], cwd=source_root
        ).decode()
        if expected != (installed / "testset" / path).read_text(encoding="utf-8"):
            raise ValueError(f"Wrong archived source for {path}")
    persona = Persona(name="Reader", role_description="Researches Kit")
    passage = "Ada designed Kit."
    template_inputs = [
        (HeadlinesExtractorPrompt(), TextWithExtractionLimit(text=passage, max_num=5)),
        (SummaryExtractorPrompt(), StringIO(text=passage)),
        (
            QuestionPotentialPrompt(),
            QuestionPotentialInput(
                document_summary=passage, node_content=passage, rubrics=DEFAULT_RUBRICS
            ),
        ),
        (
            ThemesAndConceptsExtractorPrompt(),
            TextWithExtractionLimit(text=passage, max_num=10),
        ),
        (PersonaGenerationPrompt(), StringIO(text=passage)),
        (
            ThemesPersonasMatchingPrompt(),
            ThemesPersonasInput(themes=["Kit"], personas=[persona]),
        ),
        (
            ConceptCombinationPrompt(),
            ConceptsList(lists_of_concepts=[["Design"], ["City"]], max_combinations=1),
        ),
    ]
    prompts = {}
    for template, sample in template_inputs:
        raw = template.to_string(sample)
        prefix, rest = raw.split("as specified in JSON Schema:\n", 1)
        schema, suffix = rest.split("Do not use single quotes", 1)
        norm = json.dumps(
            [prefix, json.loads(schema), suffix],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        prompts[type(template).__name__] = [
            template.output_model.__name__,
            hashlib.sha256(raw.encode()).hexdigest(),
            hashlib.sha256(norm.encode()).hexdigest(),
        ]
    texts = [passage, "Kit lives in Rome.", "Discount."]
    graph = KnowledgeGraph(
        nodes=[
            Node(id=UUID(int=i), type=NodeType.CHUNK, properties={"page_content": text})
            for i, text in enumerate(texts, 1)
        ]
    )
    llm = CorpusJudge()
    embeds = LocalEmbeddings({text: [1.0, 0.0] for text in texts}, [])
    transforms = default_transforms_for_prechunked(llm, embeds)
    apply_locally(graph, transforms)
    personas = generate_personas_from_kg(graph, llm, num_personas=1)
    distribution = default_query_distribution(llm, graph)
    splits, _ = calculate_split_values([weight for _, weight in distribution], 4)

    async def samples():
        result = []
        for (plugin, _), count in zip(distribution, splits, strict=True):
            scenarios = await plugin.generate_scenarios(count, graph, personas)
            for scenario in scenarios:
                row = await plugin.generate_sample(scenario)
                result.append(
                    {
                        "strategy": plugin.name,
                        "evidence": [n.id.int for n in scenario.nodes],
                        "question": row.user_input,
                        "reference": row.reference,
                        "contexts": row.reference_contexts,
                        "persona": row.persona_name,
                        "style": row.query_style,
                        "length": row.query_length,
                    }
                )
        return result

    random.seed(0)
    rows = asyncio.run(samples())
    return {
        "prompts": prompts,
        "sources": len(sources),
        "nodes": [n.properties for n in graph.nodes],
        "links": [
            [e.type, e.source.id.int, e.target.id.int, e.properties]
            for e in graph.relationships
        ],
        "raw_documents": raw_documents(),
        "splits": splits,
        "rows": rows,
        "calls": llm.calls,
        "embedded": embeds.calls,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = capture(args.source_root)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))

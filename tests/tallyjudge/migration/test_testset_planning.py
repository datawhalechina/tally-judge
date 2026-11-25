"""Ragas ec00339 (Apache-2.0) testset prompt and typed graph replay."""

import hashlib
import json

import pydantic
import pytest

from tallyjudge import create_synthesizer
from tallyjudge.plugins.synthesis._planning_prompts import (
    DEFAULT_RUBRICS,
    ConceptCombinationPrompt,
    ConceptsList,
    HeadlinesExtractorPrompt,
    PersonaGenerationPrompt,
    QuestionPotentialInput,
    QuestionPotentialPrompt,
    StringIO,
    SummaryExtractorPrompt,
    ThemesAndConceptsExtractorPrompt,
    ThemesPersonasInput,
    ThemesPersonasMatchingPrompt,
)
from tallyjudge.plugins.synthesis._prompts import Persona, TextWithExtractionLimit
from tallyjudge.plugins.synthesis.graph import EvidenceGraph
from tallyjudge.sdk.models import GenerationResult
from tallyjudge.sdk.usage import UsageSummary

ARCHIVED = {
    "HeadlinesExtractorPrompt": [
        "Headlines",
        "0cf7287dc0154d6bdb6d4e7c4847fb500aaec6512c805f51dbe6a8eeba782644",
        "953aeac23f5efd73f8f74f91b7adccefd14cbad1b204191c7ce50e4bab39ed06",
    ],
    "SummaryExtractorPrompt": [
        "StringIO",
        "de793d20fe317ed8a36c082ce1c7af84357a38700f1370ed7f2b9b4ff5180a37",
        "70c5e0bc2cf0fd029fecb8de8d8e71edc62b8a0590f31271a5661629de4f8850",
    ],
    "QuestionPotentialPrompt": [
        "QuestionPotentialOutput",
        "20c5f9654b1a8b24ede8b5be04f2f32e9b231c595ee132dc74fa51439c138ea9",
        "41d7ee48bae21be2297533569679d5cc28376a985a65ccab7730f8dd05468a49",
    ],
    "ThemesAndConceptsExtractorPrompt": [
        "ThemesAndConcepts",
        "33b36e8ed879672565496ef49ee2c1359baaeeebae0521395fcfa62c1157814e",
        "66a0a544b5fcbcbcfd92f1e35982830663a36df783b24512df356edd502d24d4",
    ],
    "PersonaGenerationPrompt": [
        "Persona",
        "a5c45685226ece9c739150c918df94f17b520d3422fba389b1850c59be651ea8",
        "606395a354e77065ea596e795a8bcf8e2dc40eb4bc9c4137610f69a847529dd2",
    ],
    "ThemesPersonasMatchingPrompt": [
        "PersonaThemesMapping",
        "d4034e45601e871a0a40038b8ffcb1f5be5a5731693a2bb1fb66a7499d9fdf38",
        "c306c694248ed7c9d12c95445e975adfc20da6df9f08dca86edfaa23740aa221",
    ],
    "ConceptCombinationPrompt": [
        "ConceptCombinations",
        "4c2514f31599e3cf0730feabe8f7c9ba403004f10387ae84d3a89bdf568b1cad",
        "8d9f5b37967e12929d2d89b11d9cb440e772f12745510b8eabaf056ca0bf117b",
    ],
}
PASSAGE = "Ada designed Kit."
READER = Persona(name="Reader", role_description="Researches Kit")


@pytest.mark.parametrize(
    "prompt,data",
    [
        (HeadlinesExtractorPrompt(), TextWithExtractionLimit(text=PASSAGE, max_num=5)),
        (SummaryExtractorPrompt(), StringIO(text=PASSAGE)),
        (
            QuestionPotentialPrompt(),
            QuestionPotentialInput(
                document_summary=PASSAGE, node_content=PASSAGE, rubrics=DEFAULT_RUBRICS
            ),
        ),
        (
            ThemesAndConceptsExtractorPrompt(),
            TextWithExtractionLimit(text=PASSAGE, max_num=10),
        ),
        (PersonaGenerationPrompt(), StringIO(text=PASSAGE)),
        (
            ThemesPersonasMatchingPrompt(),
            ThemesPersonasInput(themes=["Kit"], personas=[READER]),
        ),
        (
            ConceptCombinationPrompt(),
            ConceptsList(lists_of_concepts=[["Design"], ["City"]], max_combinations=1),
        ),
    ],
)
def test_planning_request_matches_archived_bytes(prompt, data):
    schema, raw_hash, normalized_hash = ARCHIVED[type(prompt).__name__]
    [message] = prompt.render(data).messages
    assert message.role == "user" and prompt.output_model.__name__ == schema
    if pydantic.__version__ == "2.14.3":
        assert hashlib.sha256(message.content.encode()).hexdigest() == raw_hash
    prefix, rest = message.content.split("as specified in JSON Schema:\n", 1)
    schema, suffix = rest.split("Do not use single quotes", 1)
    normalized = json.dumps(
        [prefix, json.loads(schema), suffix],
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=False,
    )
    assert hashlib.sha256(normalized.encode()).hexdigest() == normalized_hash


def test_imported_summary_path_expands_children_and_keeps_parallel_edge_types():
    nodes = [
        {
            "id": s,
            "type": "document",
            "properties": {
                "page_content": f"Document {s}",
                "summary": s,
                "summary_embedding": [1.0],
            },
        }
        for s in ["a", "b", "c"]
    ]
    nodes += [
        {
            "id": s + "1",
            "type": "chunk",
            "properties": {
                "page_content": s.upper(),
                "themes": [s],
                "entities": ["Kit"],
            },
        }
        for s in ["a", "b", "c"]
    ]

    def link(source, target, kind, properties):
        return {
            "source": source,
            "target": target,
            "type": kind,
            "bidirectional": kind != "child",
            "properties": properties,
        }

    edges = [link(s, s + "1", "child", {}) for s in ["a", "b", "c"]]
    edges += [
        link("a", "b", "summary_similarity", {"summary_similarity": 0.8}),
        link("b", "c", "summary_similarity", {"summary_similarity": 0.9}),
        link("a1", "b1", "entities_overlap", {"overlapped_items": [["Kit", "Kit"]]}),
        link("a1", "b1", "summary_similarity", {"summary_similarity": 0.7}),
    ]
    graph = EvidenceGraph.from_ragas({"nodes": nodes, "relationships": edges})
    restored = EvidenceGraph.from_dict(json.loads(json.dumps(graph.to_dict())))
    assert len(restored.relationships) == len(edges)
    assert {
        r.type for r in restored.relationships if (r.source, r.target) == ("a1", "b1")
    } == {"summary_similarity", "entities_overlap"}

    class PlannerModel:
        plugin_id = "model.fixture.plan"
        usage_tracking = "untracked"

        def __init__(self):
            self.groups = []

        def config_snapshot(self):
            return {}

        async def agenerate(self, prompt, response_model):
            data = json.loads(
                prompt.messages[0]
                .content.rsplit("input: ", 1)[1]
                .removesuffix("\nOutput: ")
            )
            if response_model.__name__ == "ConceptCombinations":
                self.groups.append(data["lists_of_concepts"])
                reply = {
                    "combinations": [
                        [s for row in data["lists_of_concepts"] for s in row]
                    ]
                }
            elif response_model.__name__ == "PersonaThemesMapping":
                reply = {"mapping": {"Reader": data["themes"]}}
            else:
                assert response_model.__name__ == "GeneratedQueryAnswer"
                reply = {"query": "Question", "answer": "Answer"}
            return GenerationResult(
                response_model.model_validate(reply), UsageSummary()
            )

    model = PlannerModel()
    run = create_synthesizer(
        "synthesis.testset.rag", llm=model, personas=[READER.model_dump()]
    ).run(restored, testset_size=12)
    assert [["a"], ["b"], ["c"]] in model.groups
    records = run.to_records()
    assert any(row["reference_context_ids"] == ["a1", "b1", "c1"] for row in records)
    assert all(row["persona_name"] == "Reader" for row in records)
    assert not run.usage.generation.complete


def test_version_one_graph_can_be_reopened_without_guessing_typed_edges():
    plain = EvidenceGraph(
        [
            {"chunk_id": "a", "document_id": "docs", "text": "Alpha"},
            {"chunk_id": "b", "document_id": "docs", "text": "Beta"},
        ],
        [("b", "a")],
    ).to_dict()
    del plain["relationships"]
    plain["schema_version"] = 1
    restored = EvidenceGraph.from_dict(plain)
    assert restored.relations == (("a", "b"),) and restored.relationships == ()

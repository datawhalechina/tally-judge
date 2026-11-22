"""Replay the Ragas ec00339 synthesis, entity-graph and rubric contracts offline."""

import argparse
import asyncio
import inspect
import json
import subprocess
from pathlib import Path
from uuid import UUID

from generate_rag_baselines import BASE, Judge
from ragas.metrics.collections.domain_specific_rubrics.metric import (
    DomainSpecificRubrics,
)
from ragas.metrics.collections.instance_specific_rubrics.metric import (
    InstanceSpecificRubrics,
)
from ragas.prompt.pydantic_prompt import PydanticPrompt
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.base import QueryLength, QueryStyle
from ragas.testset.synthesizers.multi_hop.base import MultiHopScenario
from ragas.testset.synthesizers.multi_hop.specific import (
    MultiHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.single_hop.base import SingleHopScenario
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset.transforms.extractors.llm_based import NERExtractor
from ragas.testset.transforms.relationship_builders.traditional import (
    OverlapScoreBuilder,
)


class ScenarioJudge(Judge):
    """The archived testset prompt selects the async instructor path via this flag."""

    is_async = True


class CharacterTokenizer:
    """Use a deterministic local split to exercise the archive's window boundaries."""

    def encode(self, text):
        return list(text.encode("ascii"))

    def decode(self, tokens):
        return bytes(tokens).decode("ascii")


async def capture(source_root):
    report = {}
    sources = (
        "testset/synthesizers/single_hop/base.py",
        "testset/synthesizers/single_hop/prompts.py",
        "testset/synthesizers/multi_hop/base.py",
        "testset/synthesizers/multi_hop/prompts.py",
        "testset/transforms/extractors/llm_based.py",
        "testset/transforms/relationship_builders/traditional.py",
        "metrics/collections/domain_specific_rubrics/metric.py",
        "metrics/collections/domain_specific_rubrics/util.py",
        "metrics/collections/instance_specific_rubrics/metric.py",
        "metrics/collections/instance_specific_rubrics/util.py",
        "prompt/pydantic_prompt.py",
        "prompt/metrics/base_prompt.py",
    )
    installed = Path(inspect.getfile(PydanticPrompt)).parents[1]
    for path in sources:
        original = subprocess.check_output(
            ["git", "show", f"{BASE}:src/ragas/{path}"], cwd=source_root
        )
        if original.decode() != (installed / path).read_text(encoding="utf-8"):
            raise RuntimeError(f"Installed archive differs at {path}")

    def model(key, replies):
        entry = {"calls": [], "prompts": {}}
        report[key] = entry
        return ScenarioJudge(replies, entry["calls"], entry["prompts"])

    persona = Persona(name="Librarian", role_description="Checks provenance")
    properties = [
        {"page_content": "Ada designed Kit.", "entities": ["Guide", "Kit", "Martha"]},
        {"page_content": "Kit is in Rome.", "entities": ["Guide", "Kit", "Marhta"]},
        {"page_content": "Rye costs five coins.", "entities": ["Guide", "Rye"]},
    ]
    nodes = [
        Node(id=UUID(int=i), type=NodeType.CHUNK, properties=p)
        for i, p in enumerate(properties, 1)
    ]
    sample_reply = {"query": "Who designed Kit?", "answer": "Ada"}
    for key, plugin, scenario, guidance in [
        (
            "single",
            SingleHopSpecificQuerySynthesizer,
            SingleHopScenario(
                nodes=nodes[:1],
                persona=persona,
                style=QueryStyle.PERFECT_GRAMMAR,
                length=QueryLength.MEDIUM,
                term="Kit",
            ),
            None,
        ),
        (
            "multi",
            MultiHopSpecificQuerySynthesizer,
            MultiHopScenario(
                nodes=nodes[:2],
                persona=persona,
                style=QueryStyle.PERFECT_GRAMMAR,
                length=QueryLength.MEDIUM,
                combinations=["Kit"],
            ),
            None,
        ),
        (
            "multi_order",
            MultiHopSpecificQuerySynthesizer,
            MultiHopScenario(
                nodes=nodes[::-1],
                persona=persona,
                style=QueryStyle.WEB_SEARCH_LIKE,
                length=QueryLength.SHORT,
                combinations=["Rome", "Kit"],
            ),
            "Compare",
        ),
    ]:
        llm = model(key, [sample_reply])
        answer = await plugin(llm=llm, llm_context=guidance).generate_sample(scenario)
        report[key]["sample"] = answer.model_dump(exclude_none=True)

    for key, text, size, responses in [
        ("ner", "Ada designed Kit.", 32000, [["Ada", "Kit", "Alias"]]),
        ("ner_windows", "1234567", 3, [["x"], ["x", "y"], ["z"]]),
    ]:
        llm = model(key, [{"entities": v} for v in responses])
        extractor = NERExtractor(
            llm=llm, tokenizer=CharacterTokenizer(), max_token_limit=size
        )
        _, entities = await extractor.extract(
            Node(type=NodeType.CHUNK, properties={"page_content": text})
        )
        report[key]["entities"] = entities

    # A second pair can only be connected by fuzzy matching after the most frequent label is removed.
    graph_nodes = [
        Node(id=UUID(int=i), type=NodeType.CHUNK, properties={"entities": labels})
        for i, labels in enumerate(
            [["Guide", "Martha"], ["Guide", "Marhta"], ["Guide", "Rye"]], 1
        )
    ]
    links = await OverlapScoreBuilder().transform(KnowledgeGraph(nodes=graph_nodes))
    report["edges"] = [[r.source.id.int, r.target.id.int] for r in links]

    for key, settings, inputs in [
        ("rubric_free", {}, {"user_input": "Who?", "response": "Ada"}),
        (
            "rubric_reference",
            {"with_reference": True},
            {"user_input": "Who?", "response": "Ada", "reference": "Ada"},
        ),
        (
            "rubric_custom",
            {"rubrics": {"second": "B", "first": "A"}},
            {"response": "Ada"},
        ),
        (
            "rubric_instance",
            None,
            {"response": "Ada", "rubrics": {"second": "B", "first": "A"}},
        ),
    ]:
        llm = model(key, [{"feedback": "Archived feedback", "score": "4"}])
        metric = (
            InstanceSpecificRubrics(llm=llm)
            if settings is None
            else DomainSpecificRubrics(llm=llm, **settings)
        )
        score = await metric.ascore(**inputs)
        report[key]["score"] = [score.value, score.reason]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = asyncio.run(capture(args.source_root))
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for key, entry in result.items():
        if key == "edges":
            print(key, entry)
        else:
            print(key, [(call["generate"], call["schema"]) for call in entry["calls"]])

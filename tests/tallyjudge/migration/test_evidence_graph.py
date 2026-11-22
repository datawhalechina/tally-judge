"""Ragas ec00339, Apache-2.0: archived nodes and edges keep their original IDs."""

import json
from types import SimpleNamespace
from uuid import UUID

import pytest

from tallyjudge import create_synthesizer
from tallyjudge.plugins.synthesis.graph import EvidenceGraph


def test_archived_edge_ids_and_source_survive_conversion():
    parent, a, b = (str(UUID(int=i)) for i in (1, 2, 3))

    def node(key, kind, text, **props):
        return {"id": key, "type": kind, "properties": {"page_content": text, **props}}

    nodes = [
        node(parent, "document", "Manual", document_metadata={"source": "manual"}),
        node(a, "chunk", "Ada built Kit."),
        node(b, "chunk", "Kit lives in Rome."),
    ]

    def rel(source, target, kind):
        return {
            "id": str(UUID(int=4)),
            "source": source,
            "target": target,
            "type": kind,
            "bidirectional": kind == "entities_overlap",
            "properties": {},
        }

    graph = EvidenceGraph.from_ragas(
        {
            "nodes": nodes,
            "relationships": [
                rel(parent, a, "child"),
                rel(parent, b, "child"),
                rel(a, b, "entities_overlap"),
            ],
        }
    )
    assert graph.relations == ((a, b),)
    chunks = {c["chunk_id"]: c for c in graph}
    assert [chunks[key]["text"] for key in (a, b)] == [
        "Ada built Kit.",
        "Kit lives in Rome.",
    ]
    assert [chunks[key]["document_id"] for key in (a, b)] == ["manual", "manual"]
    reloaded = EvidenceGraph.from_dict(json.loads(json.dumps(graph.to_dict())))
    assert reloaded.chunks == graph.chunks and reloaded.relations == graph.relations
    assert not reloaded.generation_usage.complete
    assert reloaded.embedding_usage is None
    local = EvidenceGraph(graph.chunks, graph.relations)
    roundtrip = EvidenceGraph.from_dict(local.to_dict())
    assert (
        roundtrip.generation_usage.complete and roundtrip.generation_usage.requests == 0
    )


@pytest.mark.parametrize(
    "suffix, revision, original_id",
    [
        (
            "single_hop",
            "2",
            "0dec5663ae75ff2edee33606df38f6104ac099589c3e241bbd17b61bfb1fc1f1",
        ),
        (
            "multi_hop",
            "3",
            "32de39b627789b91e3118026fa552425b0ed33ac81faff7718c07565124bd36d",
        ),
    ],
)
def test_restoring_archived_prompts_changes_pre_migration_ids(
    suffix, revision, original_id
):
    """Changing the model task gives new samples a distinct reproducible identity."""

    class Judge:
        plugin_id = "model.fixture.phase7"
        usage_tracking = "untracked"

        def config_snapshot(self):
            return {}

        async def agenerate(self, prompt, response_model):
            return SimpleNamespace(data=response_model(query="Q", answer="A"))

    graph = EvidenceGraph(
        [
            {"chunk_id": key, "document_id": key, "text": text}
            for key, text in [("a", "First"), ("b", "Second")]
        ],
        [("a", "b")],
    )
    old = create_synthesizer(f"synthesis.testset.{suffix}", llm=Judge(), seed=7)
    result = old.run(graph, testset_size=1)
    assert result.config_snapshot == {
        "seed": 7,
        "prompt_revision": revision,
        "llm": {"plugin_id": Judge.plugin_id, "config": {}},
    }
    assert result.to_records()[0]["case_id"] != original_id
    assert old.run(graph, testset_size=1).to_records() == result.to_records()

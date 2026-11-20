"""Ragas ec00339, Apache-2.0: KnowledgeGraph.save and find_two_nodes_single_rel."""

from uuid import UUID

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

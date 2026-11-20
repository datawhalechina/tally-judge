"""Public raw-document, prechunked and graph generation paths."""

import json

import pytest

from tallyjudge import create_language_model, create_synthesizer
from tallyjudge.plugins.adapters.documents import (
    langchain_documents,
    llamaindex_documents,
)
from tallyjudge.plugins.synthesis.documents import SourceDocument, split_documents
from tallyjudge.plugins.synthesis.graph import EvidenceGraph, build_graph

DOCS = [
    SourceDocument("bio", "Ada built Kit.", ("Kit",)),
    SourceDocument("place", "Kit lives in Rome.", ("kit",)),
]
QA = {
    "user_input": "Where does Ada's creation live?",
    "reference": "Rome",
    "evidence_quotes": ["Ada built Kit", "Kit lives in Rome"],
}


def multi(client):
    llm = create_language_model(
        "model.openai_compatible.chat_completions",
        client=client,
        model="fake",
        structured_retries=0,
    )
    return create_synthesizer("synthesis.testset.multi_hop", llm=llm, seed=7)


@pytest.mark.asyncio
async def test_prebuilt_graph_records_both_sources_and_rejects_wrong_quote(client):
    rows = list(split_documents(DOCS))
    a, b = (r["chunk_id"] for r in rows)
    graph = EvidenceGraph(tuple(reversed(rows)), ((b, a), (a, b)))
    plugin = multi(client)
    client.chat_reply(json.dumps(QA))
    run = await plugin.arun(graph, testset_size=4)
    (record,) = run.to_dataset().records
    assert record["reference_context_ids"] == [a, b]
    assert record["reference_document_ids"] == ["bio", "place"]
    assert record["reference_contexts"] == [r["text"] for r in rows]
    assert run.usage.generation.requests == run.generated_count == 1
    client.chat_reply(json.dumps({**QA, "evidence_quotes": ["fabricated", "Rome"]}))
    failed = await plugin.arun(graph, testset_size=1)
    assert failed.candidates[0].case_id == record["case_id"]
    assert failed.candidates[0].evidence_ids == (a, b)
    assert failed.candidates[0].error.code == "unsupported_evidence"
    empty = await plugin.arun(EvidenceGraph(tuple(rows), ()), testset_size=1)
    assert empty.generated_count == empty.usage.generation.requests == 0
    with pytest.raises(ValueError, match="unknown chunk"):
        await plugin.arun(EvidenceGraph(tuple(rows), ((a, "unknown"),)), testset_size=1)


def test_document_offsets_and_empty_sources():
    doc = SourceDocument("a", "  a bb  ddddddd ")
    chunks = list(split_documents([doc, SourceDocument("blank", "  ")], max_size=3))
    assert [r["text"] for r in chunks] == ["a", "bb", "ddd", "ddd", "d"]
    assert all(doc.text[r["start"] : r["end"]] == r["text"] for r in chunks)
    assert chunks == list(split_documents([doc], max_size=3))
    with pytest.raises(ValueError, match="unique"):
        list(split_documents([doc, doc]))


def test_framework_documents_with_real_graph_algorithms():
    pytest.importorskip("sknetwork")
    lc = pytest.importorskip("langchain_core.documents")
    li = pytest.importorskip("llama_index.core")
    langchain = [
        lc.Document(
            id=d.document_id,
            page_content=d.text,
            metadata={"entities": list(d.entities)},
        )
        for d in DOCS
    ]
    llamaindex = [
        li.Document(
            id_=d.document_id, text=d.text, metadata={"entities": list(d.entities)}
        )
        for d in DOCS
    ]
    for docs in (langchain_documents(langchain), llamaindex_documents(llamaindex)):
        chunks = tuple(split_documents(docs))
        assert chunks == tuple(split_documents(DOCS))
        assert (
            build_graph(chunks).relations
            == build_graph(reversed(chunks), indirect=True).relations
            == ((chunks[0]["chunk_id"], chunks[1]["chunk_id"]),)
        )
    assert (
        build_graph([{"chunk_id": "solo", "text": "isolated"}], indirect=True).relations
        == ()
    )


@pytest.mark.asyncio
async def test_similarity_transform_and_persona(client):
    from tallyjudge import create_embedding_model
    from tallyjudge.plugins.synthesis.graph import build_similarity_graph

    rows = list(split_documents(DOCS))
    embeddings = create_embedding_model(
        "embedding.openai_compatible.embeddings", client=client, model="embed"
    )
    client.embedding_reply([(0, [1.0, 0.0]), (1, [1.0, 0.0])])
    graph = await build_similarity_graph(rows, embeddings=embeddings, threshold=1.0)
    assert len(graph.relations) == 1 and graph.embedding_usage.requests == 1
    empty = await build_similarity_graph(rows[:1], embeddings=embeddings)
    assert not empty.relations and empty.embedding_usage is None
    llm = create_language_model(
        "model.openai_compatible.chat_completions", client=client, model="test"
    )
    personas = [{"name": "Librarian", "role_description": "Checks provenance"}]
    plugin = create_synthesizer(
        "synthesis.testset.multi_hop", llm=llm, personas=personas
    )
    client.chat_reply(json.dumps(QA))
    run = await plugin.arun(graph, testset_size=1)
    assert run.config_snapshot["personas"] == personas
    assert "Checks provenance" in client.calls[-1][1]["messages"][0]["content"]
    assert run.generated_count == 1


def test_graph_detaches_evidence_and_preserves_numeric_text_ids():
    pytest.importorskip("sknetwork")
    rows = [
        {"chunk_id": key, "document_id": key, "text": "Kit", "entities": ["Kit"]}
        for key in ("0001", "01", "1")
    ]
    graph = build_graph(rows, indirect=True)
    rows[0]["text"] = "replacement"
    rows[0]["entities"].clear()
    exposed = next(iter(graph))
    exposed["entities"].append("changed")
    assert graph.relations == (("0001", "01"), ("0001", "1"), ("01", "1"))
    assert graph.chunks[0]["text"] == "Kit" and graph.chunks[0]["entities"] == ["Kit"]


@pytest.mark.parametrize("value", [False, {}, ""])
def test_personas_require_the_declared_list_type(client, value):
    from pydantic import ValidationError

    from tallyjudge.core.errors import PluginError

    llm = create_language_model(
        "model.openai_compatible.chat_completions", client=client, model="local"
    )
    with pytest.raises(PluginError) as caught:
        create_synthesizer("synthesis.testset.single_hop", llm=llm, personas=value)
    assert isinstance(caught.value.__cause__, ValidationError)

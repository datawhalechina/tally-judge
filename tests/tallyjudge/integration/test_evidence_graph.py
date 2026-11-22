"""Public raw-document, prechunked and graph generation paths."""

import asyncio
import json
import re
from functools import partial
from types import SimpleNamespace

import pytest

from tallyjudge import (
    create_language_model,
    create_metric,
    create_synthesizer,
    experiment,
)
from tallyjudge.plugins.adapters.documents import (
    langchain_documents,
    llamaindex_documents,
)
from tallyjudge.plugins.synthesis.documents import SourceDocument, split_documents
from tallyjudge.plugins.synthesis.graph import EvidenceGraph, build_graph

DOCS = [
    SourceDocument("bio", "Guide: Ada built Kit.", ("Guide", "Kit")),
    SourceDocument("place", "Guide: Kit lives in Rome.", ("Guide", "kit")),
]
QA = {
    "query": "Where does Ada's creation live?",
    "answer": "Rome",
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
async def test_prebuilt_graph_owns_sources_and_accepts_archived_answers(client):
    rows = list(split_documents(DOCS))
    a, b = (r["chunk_id"] for r in rows)
    graph = EvidenceGraph(tuple(reversed(rows)), ((b, a), (a, b)))
    plugin = multi(client)
    client.chat_reply(json.dumps(QA))
    run = await plugin.arun(graph, testset_size=4)
    (record,) = run.to_dataset().records
    assert record["reference_context_ids"] == [a, b]
    assert record["reference_document_ids"] == ["bio", "place"]
    assert record["reference_contexts"] == [
        f"<{i}-hop>\n\n" + r["text"] for i, r in enumerate(rows, 1)
    ]
    assert run.usage.generation.requests == run.generated_count == 1
    client.chat_reply(json.dumps({**QA, "reference_context_ids": ["fabricated"]}))
    replayed = await plugin.arun(graph, testset_size=1)
    assert replayed.to_records() == run.to_records()
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


def test_entity_graph_edges_require_mentions_in_both_chunks():
    pytest.importorskip("rapidfuzz")
    chunks = list(
        split_documents(
            [
                SourceDocument(
                    "manual",
                    "Kit arrived. Kitchen opens. 欢迎来北京。",
                    ("KIT", "北京"),
                ),
                SourceDocument("catalog", "在北京A馆收藏了Kit。", ("kit", "北京")),
            ],
            max_size=14,
        )
    )
    graph = build_graph(chunks)
    assert [c["entities"] for c in chunks] == [["KIT"], [], ["北京"], ["kit", "北京"]]
    assert set(graph.relations) == {
        tuple(sorted((chunks[i]["chunk_id"], chunks[-1]["chunk_id"]))) for i in (0,)
    }


def test_paginated_langchain_source_keeps_file_identity_and_page_offsets():
    name = "guide.pdf"
    pages = [
        SimpleNamespace(
            id=None, page_content="Text", metadata={"source": name, "page": n}
        )
        for n in (0, 1)
    ]
    chunks = list(split_documents(langchain_documents(pages)))
    assert [(c["document_id"], c["chunk_id"], c["page"]) for c in chunks] == [
        (name, "guide.pdf#page=0:0-4", 0),
        (name, "guide.pdf#page=1:0-4", 1),
    ]
    assert list(split_documents(langchain_documents(reversed(pages)))) == chunks[::-1]
    with pytest.raises(ValueError, match="unique"):
        list(split_documents(langchain_documents([pages[0], pages[0]])))

    pages[0].id = "my-page"
    assert (
        list(split_documents(langchain_documents(pages)))[0]["chunk_id"]
        == "my-page:0-4"
    )


@pytest.mark.asyncio
async def test_files_to_generated_question_to_top_k_scores(client, tmp_path):
    pytest.importorskip("rapidfuzz")
    from tallyjudge.plugins.synthesis.graph import build_entity_graph

    for name, text in [
        ("bakery", "Guide. Rye costs five coins."),
        ("design", "Guide. Ada designed Kit."),
        ("location", "Rome houses Kit. Guide."),
    ]:
        (tmp_path / f"{name}.txt").write_text(text, encoding="utf-8")
    chunks = list(
        split_documents(
            SourceDocument(p.name, p.read_text(encoding="utf-8"))
            for p in sorted(tmp_path.glob("*.txt"))
        )
    )
    llm = create_language_model(
        "model.openai_compatible.chat_completions",
        client=client,
        model="fixture",
        temperature=0.0,
        structured_retries=0,
    )
    for entities in (
        ["Guide", "Rye"],
        ["Guide", "Ada", "Kit"],
        ["Guide", "Rome", "Kit"],
    ):
        client.chat_reply(json.dumps({"entities": entities}))
    graph = await build_entity_graph(chunks, llm=llm)
    design, location = (c["chunk_id"] for c in chunks[1:])
    assert graph.relations == ((design, location),)
    assert graph.generation_usage.requests == 3
    assert graph.chunks[0]["entities"] == ["Guide", "Rye"]
    assert all(c["entities"] == [] for c in chunks)

    client.chat_reply(
        json.dumps(
            {
                "query": "Which city houses what Ada designed?",
                "answer": "Rome",
            }
        )
    )
    synthesis = await create_synthesizer("synthesis.testset.multi_hop", llm=llm).arun(
        graph, testset_size=2
    )
    dataset = synthesis.to_dataset()
    [gold] = dataset.records
    assert gold["reference_context_ids"] == [design, location]
    assert synthesis.generated_count == synthesis.usage.generation.requests == 1

    def query(row, *, top_k):
        words = set(re.findall(r"\w+", row["user_input"].casefold()))

        def matches(chunk):
            return len(words.intersection(re.findall(r"\w+", chunk["text"].casefold())))

        hits = sorted(chunks, key=matches, reverse=True)[:top_k]
        passages = [c["text"] for c in hits]
        city = next(
            (s.split(" houses ")[0] for s in passages if " houses " in s), "Unknown"
        )
        return {
            "response": city,
            "retrieved_contexts": passages,
            "retrieved_context_ids": [c["chunk_id"] for c in hits],
        }

    metrics = [
        create_metric(key)
        for key in (
            "metric.retrieval.id_context_precision",
            "metric.retrieval.id_context_recall",
            "metric.text.exact_match",
        )
    ]
    for k, ids, answer, values in [
        (1, [design], "Unknown", [1.0, 0.5, 0.0]),
        (2, [design, location], "Rome", [1.0, 1.0, 1.0]),
    ]:
        result = await experiment(partial(query, top_k=k)).arun(
            dataset, metrics=metrics
        )
        row = result.row_for(gold["case_id"])
        assert row.task.output["retrieved_context_ids"] == ids
        assert row.task.output["response"] == answer
        assert [row.metrics[m.spec.id].value for m in metrics] == values
    assert len(client.calls) == 4


@pytest.mark.asyncio
async def test_entity_graph_reuses_labels_and_extracts_an_unlabelled_single_node(
    client,
):
    pytest.importorskip("rapidfuzz")
    from tallyjudge.plugins.synthesis.graph import build_entity_graph

    chunks = list(split_documents(DOCS))
    llm = create_language_model(
        "model.openai_compatible.chat_completions", client=client, model="fixture"
    )
    annotated = await build_entity_graph(chunks, llm=llm)
    assert annotated.relations == ((chunks[0]["chunk_id"], chunks[1]["chunk_id"]),)
    client.chat_reply('{"entities":["Named by the model"]}')
    solo = await build_entity_graph([dict(chunks[0], entities=[])], llm=llm)
    assert solo.relations == () and len(client.calls) == 1
    assert solo.chunks[0]["entities"] == ["Named by the model"]
    assert (
        annotated.generation_usage.requests == 0 and solo.generation_usage.requests == 1
    )


@pytest.mark.asyncio
async def test_cancelling_entity_extraction_stops_after_the_inflight_request(
    client, monkeypatch
):
    from tallyjudge.plugins.synthesis.graph import build_entity_graph

    started, returned = asyncio.Event(), asyncio.Event()
    client.chat_reply('{"entities":["Kit"]}')
    late_reply = client.responses.pop()

    async def slow():
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            returned.set()
            return late_reply

    client.responses.append(slow)
    llm = create_language_model(
        "model.openai_compatible.chat_completions",
        client=client,
        model="fixture",
    )
    # Multiple extraction windows exercise cancellation independently of fuzzy matching.
    monkeypatch.setattr(
        "tallyjudge.plugins.synthesis.graph.extraction_windows",
        lambda text, limit: (text[:5], text[5:]),
    )
    chunks = [dict(next(split_documents(DOCS)), entities=[])]
    task = asyncio.create_task(build_entity_graph(chunks, llm=llm))
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(returned.wait(), 2)
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_cancelling_similarity_build_stops_further_embedding_batches(client):
    from tallyjudge import create_embedding_model
    from tallyjudge.plugins.synthesis.graph import build_similarity_graph

    started, returned = asyncio.Event(), asyncio.Event()
    client.embedding_reply([(0, [1.0])])
    late_reply = client.responses.pop()

    async def delayed_reply():
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            returned.set()
            return late_reply

    client.responses.extend([delayed_reply, late_reply])
    embeddings = create_embedding_model(
        "embedding.openai_compatible.embeddings",
        client=client,
        model="fixture",
        batch_size=1,
    )
    task = asyncio.create_task(
        build_similarity_graph(split_documents(DOCS), embeddings=embeddings)
    )
    await asyncio.wait_for(started.wait(), 2)
    task.cancel("stop")
    with pytest.raises(asyncio.CancelledError, match="stop"):
        await task
    await asyncio.wait_for(returned.wait(), 2)
    assert len(client.calls) == 1


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
        {
            "chunk_id": key,
            "document_id": key,
            "text": "Kit",
            "entities": ["Guide", "Kit"],
        }
        for key in ("0001", "01", "1")
    ]
    graph = build_graph(rows, indirect=True)
    rows[0]["text"] = "replacement"
    rows[0]["entities"].clear()
    exposed = next(iter(graph))
    exposed["entities"].append("changed")
    assert graph.relations == (("0001", "01"), ("0001", "1"), ("01", "1"))
    assert graph.chunks[0]["text"] == "Kit" and graph.chunks[0]["entities"] == [
        "Guide",
        "Kit",
    ]


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

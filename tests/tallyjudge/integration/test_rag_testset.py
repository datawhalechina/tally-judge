"""From local text files through archived graph strategies to two retrieval runs."""

import asyncio
import hashlib
import json
import sys
import threading
from collections import Counter
from functools import partial
from uuid import UUID

import pytest

from tallyjudge import SynthesisCollector, create_metric, create_synthesizer, experiment
from tallyjudge.core.usage import Usage, UsageSummary, record_request
from tallyjudge.interfaces.models import EmbeddingResult, GenerationResult
from tallyjudge.interfaces.synthesis import SourceDocument
from tallyjudge.plugins.synthesis.graph import EvidenceGraph
from tallyjudge.plugins.synthesis.transforms import (
    build_chunk_graph,
    build_document_graph,
)

SINGLE = "single_hop_specific_query_synthesizer"
ABSTRACT = "multi_hop_abstract_query_synthesizer"
SPECIFIC = "multi_hop_specific_query_synthesizer"
PLUGIN = "synthesis.testset.rag"
TEXT = ["Ada designed Kit.", "Kit lives in Rome.", "Discount."]


class CorpusModel:
    plugin_id = "model.fixture.knowledge"
    usage_tracking = "per_request"

    def __init__(self):
        self.calls = []

    def config_snapshot(self):
        return {}

    async def agenerate(self, prompt, response_model):
        name = response_model.__name__
        data = json.loads(
            prompt.messages[0]
            .content.rsplit("input: ", 1)[1]
            .removesuffix("\nOutput: ")
        )
        self.calls.append((name, data))
        replies = []
        with record_request("generation", replies) as request:
            request.usage = Usage(3, 2)
            if name == "StringIO":
                row = {"text": data["text"].split(".")[0] + "."}
            elif name == "QuestionPotentialOutput":
                row = {
                    "score": 2 if data["node_content"].startswith("Discount.") else 3
                }
            elif name == "Headlines":
                row = {"headlines": ["First", "Second"]}
            elif name == "ThemesAndConcepts":
                row = {"output": ["Design" if "Ada" in data["text"] else "City"]}
            elif name == "NEROutput":
                row = {"entities": ["Guide", "Kit"]}
            elif name == "Persona":
                row = {"name": "Reader", "role_description": "Researches Kit"}
            elif name == "PersonaThemesMapping":
                row = {"mapping": {"Reader": data["themes"]}}
            elif name == "ConceptCombinations":
                row = {"combinations": [["Design", "City"]]}
            elif name == "GeneratedQueryAnswer":
                if isinstance(data["context"], str):
                    row = (
                        {"query": "Who designed Kit?", "answer": "Ada"}
                        if "Ada" in data["context"]
                        else {"query": "Where does Kit live?", "answer": "Rome"}
                    )
                else:
                    row = {
                        "query": "Where is the object designed by Ada?",
                        "answer": "Rome",
                    }
            else:
                raise AssertionError(name)
        return GenerationResult(
            response_model.model_validate(row), UsageSummary.from_records(replies)
        )


class Embeddings:
    plugin_id = "embedding.fixture.summaries"
    usage_tracking = "per_request"

    def __init__(self, vectors=None):
        self.calls = []
        self.vectors = vectors or {}

    def config_snapshot(self):
        return {}

    async def aembed_texts(self, texts):
        self.calls.extend(texts)
        calls = []
        with record_request("embedding", calls) as request:
            request.usage = Usage(1)
            vectors = [self.vectors.get(text, [1.0, 0.0]) for text in texts]
        return EmbeddingResult(vectors, UsageSummary.from_records(calls))


def identified(passages):
    return [
        {"chunk_id": str(UUID(int=n)), "document_id": f"manual-{n}", "text": text}
        for n, text in enumerate(passages, 1)
    ]


@pytest.mark.asyncio
async def test_files_to_three_query_strategies_to_paired_retrieval(tmp_path):
    pytest.importorskip("rapidfuzz")
    paths = []
    for key, text in enumerate(TEXT, 1):
        path = tmp_path / f"{key}.txt"
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    sources = identified(path.read_text(encoding="utf-8") for path in paths)
    llm, embeddings = CorpusModel(), Embeddings()
    graph = await build_chunk_graph(sources, llm=llm, embeddings=embeddings)
    assert embeddings.calls == TEXT[:2]
    assert graph.generation_usage.requests == 10 and graph.embedding_usage.requests == 2
    assert [r["text"] for r in graph] == TEXT[:2]
    links = {edge.type: edge.properties for edge in graph.relationships}
    assert links == {
        "summary_similarity": {"summary_similarity": 1.0},
        "entities_overlap": {
            "entities_overlap_score": 1.0,
            "overlapped_items": [["Kit", "Kit"]],
        },
    }
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(graph.to_dict()), encoding="utf-8")
    restored = EvidenceGraph.from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert restored.to_dict() == graph.to_dict()

    generator = create_synthesizer(PLUGIN, llm=llm, num_personas=1)
    result = await generator.arun(restored, testset_size=4)
    rows = result.to_records()
    assert [r["synthesizer_name"] for r in rows] == [
        SINGLE,
        SINGLE,
        ABSTRACT,
        ABSTRACT,
        SPECIFIC,
        SPECIFIC,
    ]
    assert all(r["persona_name"] == "Reader" for r in rows)
    assert [(r["query_style"], r["query_length"]) for r in rows] == [
        ("MISSPELLED", "SHORT"),
        ("PERFECT_GRAMMAR", "MEDIUM"),
        ("POOR_GRAMMAR", "LONG"),
        ("PERFECT_GRAMMAR", "MEDIUM"),
        ("PERFECT_GRAMMAR", "SHORT"),
        ("WEB_SEARCH_LIKE", "LONG"),
    ]
    assert result.testset_size == 4 and result.generated_count == 6
    assert len({r["case_id"] for r in rows}) == 6
    assert Counter(kind for kind, _ in llm.calls) == {
        "StringIO": 3,
        "QuestionPotentialOutput": 3,
        "ThemesAndConcepts": 2,
        "NEROutput": 2,
        "Persona": 1,
        "PersonaThemesMapping": 4,
        "ConceptCombinations": 1,
        "GeneratedQueryAnswer": 6,
    }
    assert [d["text"] for k, d in llm.calls if k == "Persona"] == [TEXT[1]]
    assert result.planning_usage.generation.requests == 6
    assert result.usage.generation.requests == 12
    assert all(c.usage.generation.requests == 1 for c in result.candidates)
    assert [r["reference_context_ids"] for r in rows[:2]] == [
        [sources[0]["chunk_id"]],
        [sources[1]["chunk_id"]],
    ]
    for row in rows[2:]:
        assert row["reference_context_ids"] == [s["chunk_id"] for s in sources[:2]]
        assert row["reference_contexts"] == [
            f"<{n}-hop>\n\n{s['text']}" for n, s in enumerate(sources[:2], 1)
        ]
    assert (await generator.arun(restored, testset_size=4)).to_records() == rows

    def retrieve(question, *, k):
        terms = set(question.lower().strip("?").split())
        hits = sorted(
            sources,
            key=lambda s: len(
                terms.intersection(
                    w.removesuffix("s") for w in s["text"].strip(".").lower().split()
                )
            ),
            reverse=True,
        )[:k]
        found = [s["text"] for s in hits]
        if question.startswith("Who") and TEXT[0] in found:
            answer = "Ada"
        elif question.startswith("Where does") and TEXT[1] in found:
            answer = "Rome"
        elif question.startswith("Where is") and all(s in found for s in TEXT[:2]):
            answer = "Rome"
        else:
            answer = "Unknown"
        return {
            "response": answer,
            "retrieved_context_ids": [s["chunk_id"] for s in hits],
            "retrieved_contexts": found,
        }

    def answer(row, *, k):
        return retrieve(row["user_input"], k=k)

    recall = create_metric("metric.retrieval.id_context_recall")
    exact = create_metric("metric.text.exact_match")
    for k in (1, 2):
        run = await experiment(partial(answer, k=k)).arun(
            result.to_dataset(), metrics=[recall, exact]
        )
        for row in rows:
            multi = row["synthesizer_name"] != SINGLE
            scores = run.row_for(row["case_id"]).metrics
            assert scores[recall.spec.id].value == (0.5 if k == 1 and multi else 1.0)
            assert scores[exact.spec.id].value == (0.0 if k == 1 and multi else 1.0)


@pytest.mark.asyncio
async def test_corpus_length_bins_and_summary_cosine_cutoffs():
    pytest.importorskip("rapidfuzz")
    model = CorpusModel()
    # Cosine 0.6 clears the medium threshold but falls below the prechunked threshold.
    emb = Embeddings({TEXT[1]: [0.6, 0.8]})
    docs = [
        SourceDocument(str(i), s + " pad" * 110, page=7) for i, s in enumerate(TEXT[:2])
    ]

    def count(text):
        return len(text.split())

    medium = await build_document_graph(
        docs, llm=model, embeddings=emb, count_tokens=count
    )
    assert all(r["node_type"] == "document" and r["page"] == 7 for r in medium)
    assert {e.type for e in medium.relationships} == {
        "summary_similarity",
        "entities_overlap",
    }
    assert emb.calls == TEXT[:2]
    chunks = await build_chunk_graph(identified(TEXT[:2]), llm=model, embeddings=emb)
    assert [e.type for e in chunks.relationships] == ["entities_overlap"]

    for n, expected in [(100, False), (101, True), (500, True), (501, True)]:
        docs = [
            SourceDocument("article", "x " * n),
            *(SourceDocument(str(i), "x") for i in range(3)),
        ]
        model = CorpusModel()
        if not expected:
            with pytest.raises(ValueError, match="too short"):
                await build_document_graph(
                    docs, llm=model, embeddings=emb, count_tokens=count
                )
            assert not model.calls
        else:
            await build_document_graph(
                docs, llm=model, embeddings=emb, count_tokens=count
            )
            assert ("Headlines" in {kind for kind, _ in model.calls}) == (n > 500)


@pytest.mark.asyncio
async def test_heading_split_keeps_children_direction_and_filter_uses_parent_summary():
    pytest.importorskip("rapidfuzz")
    text = "First Ada. " + "alpha " * 500 + "Second Rome. " + "beta " * 500
    model = CorpusModel()
    graph = await build_document_graph(
        [SourceDocument("book", text, page=8)],
        llm=model,
        embeddings=Embeddings(),
        count_tokens=lambda s: len(s.split()),
    )
    root, one, two = graph.chunks
    assert root["summary"] == "First Ada." and root["node_type"] == "document"
    assert [hashlib.sha256(n["text"].encode()).hexdigest() for n in graph] == [
        "f12d892675347ab863d2dd53e386a4fc2e34ffcbc067e5722732918ddedb6fec",
        "00c4616dadf56462d068023b3345092bf909fe66b334cb38e9685fbf4f1675ea",
        "f9c5cd71d22c931c5afd8c419dec9a0606fcbf84aea27d9f35e0d1a77edc514c",
    ]
    assert one["text"] + two["text"] == text
    assert all(n["document_id"] == "book" and n["page"] == 8 for n in (one, two))
    hierarchy = [
        (e.type, e.source, e.target)
        for e in graph.relationships
        if e.type in ("child", "next")
    ]
    assert hierarchy == [
        ("child", root["chunk_id"], one["chunk_id"]),
        ("child", root["chunk_id"], two["chunk_id"]),
        ("next", one["chunk_id"], two["chunk_id"]),
    ]
    assert [
        d["document_summary"] for k, d in model.calls if k == "QuestionPotentialOutput"
    ] == [root["summary"]] * 2


@pytest.mark.asyncio
async def test_direct_chunks_meter_build_and_skip_unavailable_strategies():
    llm, embed = CorpusModel(), Embeddings()
    generator = create_synthesizer(PLUGIN, llm=llm, embeddings=embed)
    result = await generator.arun(identified(TEXT[:1]), testset_size=2)
    assert result.usage.generation.requests == len(llm.calls) == 8
    assert (
        result.planning_usage.generation.requests == 6
        and result.planning_usage.embedding.requests == 1
    )
    assert all(r["synthesizer_name"] == SINGLE for r in result.to_records())
    empty = await generator.arun([], testset_size=0)
    assert empty.attempted_count == empty.usage.generation.requests == 0


@pytest.mark.asyncio
async def test_prechunked_generation_ignores_blank_pages_before_building(monkeypatch):
    monkeypatch.setitem(sys.modules, "rapidfuzz.distance", None)
    llm, embeddings = CorpusModel(), Embeddings()
    plugin = create_synthesizer(PLUGIN, llm=llm, embeddings=embeddings)
    (page,) = identified(TEXT[:1])
    expected = await plugin.arun([page], testset_size=1)
    blanks = [
        dict(page, chunk_id=f"blank-{i}", text=text)
        for i, text in enumerate(("", " \n", None))
    ]
    repeated = await plugin.arun(iter([*blanks, page]), testset_size=1)
    assert repeated.to_records() == expected.to_records()
    assert repeated.usage == expected.usage


@pytest.mark.parametrize(
    "source",
    ["chunks", "documents", "long_documents", "entities", "partially_annotated"],
)
@pytest.mark.asyncio
async def test_missing_overlap_extra_fails_before_model_requests(monkeypatch, source):
    from tallyjudge.plugins.synthesis.graph import build_entity_graph

    monkeypatch.setitem(sys.modules, "rapidfuzz.distance", None)
    llm, embeddings = CorpusModel(), Embeddings()
    with pytest.raises(ImportError, match="rapidfuzz"):
        if source == "chunks":
            plugin = create_synthesizer(PLUGIN, llm=llm, embeddings=embeddings)
            await plugin.arun(identified(TEXT[:2]), testset_size=1)
        elif source in ("entities", "partially_annotated"):
            chunks = identified(TEXT[:2])
            if source == "partially_annotated":
                chunks[0]["entities"] = ["Guide", "Kit"]
            await build_entity_graph(chunks, llm=llm)
        else:
            words = 550 if source == "long_documents" else 110
            docs = [
                SourceDocument(
                    str(i), "First " + "word " * words + "Second " + "word " * words
                )
                for i in range(2)
            ]
            await build_document_graph(
                docs,
                llm=llm,
                embeddings=embeddings,
                count_tokens=lambda s: len(s.split()),
            )
    assert not llm.calls and not embeddings.calls


@pytest.mark.asyncio
async def test_can_cancel_during_persona_grouping_before_a_model_request(monkeypatch):
    from tallyjudge.plugins.synthesis import _strategies

    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    dot = _strategies.sumprod

    def slow_dot(left, right):
        loop.call_soon_threadsafe(started.set)
        release.wait(timeout=2)
        return dot(left, right)

    monkeypatch.setattr(_strategies, "sumprod", slow_dot)
    chunks = [
        dict(
            row,
            node_type="chunk",
            summary=row["text"],
            summary_embedding=[1.0],
            entities=["Kit"],
        )
        for row in identified(TEXT[:2])
    ]
    llm = CorpusModel()
    generator = create_synthesizer(PLUGIN, llm=llm)
    task = asyncio.create_task(
        generator.arun(EvidenceGraph(chunks, ()), testset_size=1)
    )
    try:
        await asyncio.wait_for(started.wait(), 3)
        task.cancel("stop")
        with pytest.raises(asyncio.CancelledError, match="stop"):
            await task
        assert not llm.calls
    finally:
        release.set()


@pytest.mark.asyncio
async def test_cancelled_planning_records_usage_without_starting_candidates():
    started, release = asyncio.Event(), asyncio.Event()

    class Blocking(CorpusModel):
        async def agenerate(self, prompt, response_model):
            if response_model.__name__ == "Persona":
                with record_request("generation", []) as call:
                    started.set()
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        await release.wait()
                    call.usage = Usage(3, 2)
            return await super().agenerate(prompt, response_model)

    model = Blocking()
    graph = EvidenceGraph(
        [
            {
                **identified(TEXT[:1])[0],
                "node_type": "chunk",
                "entities": ["Kit"],
                "summary": TEXT[0],
                "summary_embedding": [1.0],
            }
        ],
        [],
    )
    collector = SynthesisCollector()
    task = asyncio.create_task(
        create_synthesizer(PLUGIN, llm=model).arun(
            graph, testset_size=1, collector=collector
        )
    )
    await asyncio.wait_for(started.wait(), 2)
    assert collector.snapshot().planning_usage.generation.in_flight_requests == 1
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        assert collector.snapshot().attempted_count == 0
    finally:
        release.set()
    await asyncio.sleep(0)
    assert collector.snapshot().planning_usage.generation.in_flight_requests == 1

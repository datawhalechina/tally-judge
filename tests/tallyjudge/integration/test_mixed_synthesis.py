"""Exercise mixed question generation, its request budget, and real retrieval scoring."""

import asyncio
import json
import re
import threading
from collections import Counter
from functools import partial
from itertools import combinations
from types import SimpleNamespace

import pytest

from tallyjudge import (
    SynthesisCollector,
    create_embedding_model,
    create_language_model,
    create_metric,
    create_synthesizer,
    experiment,
)
from tallyjudge.core.errors import PluginError
from tallyjudge.plugins.synthesis.documents import SourceDocument, split_documents
from tallyjudge.plugins.synthesis.graph import (
    EvidenceGraph,
    build_entity_graph,
    build_similarity_graph,
)

MIXED = "synthesis.testset.mixed"
SINGLE = "synthesis.testset.single_hop"
MULTI = "synthesis.testset.multi_hop"
TOKENS = SimpleNamespace(prompt_tokens=3, completion_tokens=2)


def judge(client):
    return create_language_model(
        "model.openai_compatible.chat_completions",
        client=client,
        model="fixture",
        temperature=0.0,
        structured_retries=0,
    )


def passages(n):
    return (
        {
            "chunk_id": str(i),
            "document_id": "guide",
            "text": f"Fact {i}.",
            "entities": ["Fact", "Sample"],
        }
        for i in range(n)
    )


def queue_questions(client, count):
    for _ in range(count):
        client.chat_reply('{"query":"What is stated?","answer":"Fact"}', tokens=TOKENS)


@pytest.mark.parametrize(
    "count, connected, size, share, expected",
    [
        (3, True, 3, 0.5, (1, 2)),  # odd budget
        (3, False, 3, 0.5, (3, 0)),  # no edges
        (2, True, 9, 0.5, (2, 1)),  # all capacities exhausted
        (4, True, 7, 0.1, (4, 3)),  # surplus pairs fill the missing singles
        (20, True, 25, 0.58, (10, 15)),  # exact decimal half
        (3, True, 8, 0, (3, 0)),
        (3, False, 4, 1, (0, 0)),
        (3, True, 4, 1, (0, 3)),
        (3, True, 0, 0.5, (0, 0)),
        (0, False, 4, 0.5, (0, 0)),
    ],
)
@pytest.mark.asyncio
async def test_public_mixed_budget(client, count, connected, size, share, expected):
    generator = create_synthesizer(MIXED, llm=judge(client), multi_hop_fraction=share)
    queue_questions(client, sum(expected))
    graph = EvidenceGraph(
        passages(count), combinations(map(str, range(count)), 2) if connected else ()
    )
    result = await generator.arun(graph, testset_size=size)
    items = Counter(row["synthesizer_id"] for row in result.to_records())
    assert (items[SINGLE], items[MULTI]) == expected
    assert result.testset_size == size
    assert result.generated_count == result.attempted_count == sum(expected)
    assert len(client.calls) == result.usage.generation.requests == sum(expected)
    assert len({c.evidence_ids for c in result.candidates}) == sum(expected)
    assert [len(c.evidence_ids) for c in result.candidates] == [1] * items[SINGLE] + [
        2
    ] * items[MULTI]


@pytest.mark.parametrize("share", [-0.1, 1.1, float("nan"), "0.5", True])
def test_mixed_rejects_invalid_share_before_generation(client, share):
    with pytest.raises(PluginError, match=MIXED):
        create_synthesizer(MIXED, llm=judge(client), multi_hop_fraction=share)
    assert not client.calls


@pytest.mark.asyncio
async def test_mixed_snapshots_streamed_rows_before_the_iterator_reuses_them(client):
    def stream():
        row = {}
        for item in passages(2):
            row.update(item)
            yield row

    plugin = create_synthesizer(MIXED, llm=judge(client), multi_hop_fraction=0.0)
    queue_questions(client, 2)
    result = await plugin.arun(stream(), testset_size=2)
    assert sorted(
        record["reference_context_ids"] for record in result.to_records()
    ) == [
        ["0"],
        ["1"],
    ]


@pytest.mark.asyncio
async def test_mixed_can_be_cancelled_while_matching_source_entities(
    client, monkeypatch
):
    from tallyjudge.plugins.synthesis import graph

    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()

    def slow_overlaps(rows):
        loop.call_soon_threadsafe(started.set)
        release.wait(timeout=2)
        return []

    monkeypatch.setattr(graph, "entity_overlaps", slow_overlaps)
    plugin = create_synthesizer(MIXED, llm=judge(client), multi_hop_fraction=1.0)
    task = asyncio.create_task(plugin.arun(passages(2), testset_size=1))
    try:
        await asyncio.wait_for(started.wait(), 3)
        task.cancel("stop")
        with pytest.raises(asyncio.CancelledError, match="stop"):
            await task
    finally:
        release.set()
    assert not client.calls


@pytest.mark.asyncio
async def test_mixed_uses_the_same_archived_prompts_and_ignores_model_source_ids(
    client,
):
    model = judge(client)
    graph = EvidenceGraph(list(passages(2)), [("0", "1")])
    settings = {"seed": 7, "personas": [{"name": "Ada", "role_description": "reader"}]}
    generator = create_synthesizer(MIXED, llm=model, **settings)
    queue_questions(client, 2)
    first = await generator.arun(graph, testset_size=2)
    mixed_requests = list(client.calls)
    assert first.config_snapshot["multi_hop_fraction"] == 0.5
    assert first.config_snapshot["component_prompt_revisions"] == {
        SINGLE: "2",
        MULTI: "3",
    }
    for index, kind in enumerate((SINGLE, MULTI)):
        queue_questions(client, 1)
        standalone = await create_synthesizer(kind, llm=model, **settings).arun(
            graph, testset_size=1
        )
        assert client.calls[-1] == mixed_requests[index]
        assert "synthesizer_id" not in standalone.to_records()[0]

    queue_questions(client, 1)
    client.chat_reply('{"query":"q","answer":"a","reference_context_ids":["invented"]}')
    second = await generator.arun(
        EvidenceGraph(reversed(graph.chunks), [("1", "0")]), testset_size=2
    )
    assert [c.case_id for c in second.candidates] == [
        c.case_id for c in first.candidates
    ]
    assert [c.status for c in second.candidates] == ["success", "success"]
    assert second.to_records()[1]["reference_context_ids"] == ["0", "1"]
    assert second.generated_count == 2 and second.usage.generation.requests == 2
    queue_questions(client, 2)
    changed = await create_synthesizer(
        MIXED, llm=model, multi_hop_fraction=0.6, **settings
    ).arun(graph, testset_size=2)
    assert {c.case_id for c in first.candidates}.isdisjoint(
        c.case_id for c in changed.candidates
    )


@pytest.mark.asyncio
async def test_cancelling_mixed_between_question_types_preserves_one_snapshot(client):
    generator = create_synthesizer(MIXED, llm=judge(client))
    queue_questions(client, 1)
    started = asyncio.Event()

    async def wait_for_cancel():
        started.set()
        await asyncio.Future()

    client.responses.append(wait_for_cancel)
    collector = SynthesisCollector()
    task = asyncio.create_task(
        generator.arun(
            EvidenceGraph(passages(3), [("0", "1"), ("1", "2")]),
            testset_size=3,
            collector=collector,
        )
    )
    async with asyncio.timeout(4):
        await started.wait()
        task.cancel("operator")
        with pytest.raises(asyncio.CancelledError, match="operator"):
            await task
    result = collector.snapshot()
    assert result.plugin_id == MIXED
    assert [c.status for c in result.candidates] == ["success", "cancelled"]
    assert [len(c.evidence_ids) for c in result.candidates] == [1, 2]
    assert result.usage.generation.requests == len(client.calls) == 2


@pytest.mark.asyncio
async def test_graph_json_preserves_embedding_usage_and_owns_its_data(client):
    embeddings = create_embedding_model(
        "embedding.openai_compatible.embeddings", client=client, model="fixture"
    )
    client.embedding_reply([(0, [1, 0]), (1, [1, 0])])
    graph = await build_similarity_graph(passages(2), embeddings=embeddings)
    saved = json.loads(json.dumps(graph.to_dict()))
    restored = EvidenceGraph.from_dict(saved)
    assert restored.to_dict() == saved
    assert (
        restored.embedding_usage.requests
        == restored.embedding_usage.requests_without_usage
        == 1
    )
    assert not restored.embedding_usage.complete
    assert restored.generation_usage.requests == 0
    saved["embedding_usage"]["known_counts"]["input_tokens"] = 42
    saved["chunks"][0]["text"] = "edited"
    assert restored.to_dict() == graph.to_dict()
    assert len(client.calls) == 1
    with pytest.raises(ValueError, match="version"):
        EvidenceGraph.from_dict({**saved, "schema_version": 3})
    with pytest.raises(ValueError, match="unknown chunk"):
        EvidenceGraph.from_dict({**saved, "relations": [["0", "missing"]]})
    with pytest.raises(ValueError, match="relations"):
        EvidenceGraph.from_dict({**saved, "relations": ["01"]})


@pytest.mark.asyncio
async def test_local_files_to_mixed_questions_to_paired_retrieval(client, tmp_path):
    pytest.importorskip("rapidfuzz")
    texts = {
        "design": "Guide: Ada designed Kit.",
        "location": "Guide: Rome houses Kit.",
        "rye": "Guide: Rye costs five coins.",
    }
    for name, text in texts.items():
        (tmp_path / f"{name}.txt").write_text(text, encoding="utf-8")
    chunks = list(
        split_documents(
            SourceDocument(p.stem, p.read_text(encoding="utf-8"), page=1)
            for p in sorted(tmp_path.glob("*.txt"))
        )
    )
    model = judge(client)
    for labels in (["Guide", "Ada", "Kit"], ["Guide", "Rome", "Kit"], ["Guide", "Rye"]):
        client.chat_reply(json.dumps({"entities": labels}), tokens=TOKENS)
    graph = await build_entity_graph(chunks, llm=model)
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(graph.to_dict(), ensure_ascii=False), encoding="utf-8")
    loaded = EvidenceGraph.from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert loaded.to_dict() == graph.to_dict()
    assert loaded.generation_usage.requests == len(client.calls) == 3
    assert loaded.generation_usage.known_tokens == {
        "input_tokens": 9,
        "output_tokens": 6,
    }
    assert loaded.embedding_usage is None
    assert [row["page"] for row in loaded] == [1, 1, 1]

    client.chat_reply(
        '{"query":"What does rye cost?","answer":"five coins"}', tokens=TOKENS
    )
    client.chat_reply(
        json.dumps(
            {
                "query": "Which city houses what Ada designed?",
                "answer": "Rome",
            }
        ),
        tokens=TOKENS,
    )
    result = await create_synthesizer(MIXED, llm=model, seed=7).arun(
        loaded, testset_size=2
    )
    data = result.to_dataset()
    single, pair = data.records
    assert [single["reference_context_ids"], pair["reference_context_ids"]] == [
        [chunks[2]["chunk_id"]],
        [chunks[0]["chunk_id"], chunks[1]["chunk_id"]],
    ]
    assert (single["synthesizer_id"], pair["synthesizer_id"]) == (SINGLE, MULTI)

    def search(question, *, top_k):
        words = set(re.findall(r"\w+", question.lower()))
        hits = sorted(
            chunks,
            key=lambda row: len(
                words.intersection(re.findall(r"\w+", row["text"].lower()))
            ),
            reverse=True,
        )[:top_k]
        found = [row["text"] for row in hits]
        if "rye" in words and texts["rye"] in found:
            answer = "five coins"
        elif texts["location"] in found:
            answer = "Rome"
        else:
            answer = "Unknown"
        return {
            "response": answer,
            "retrieved_contexts": found,
            "retrieved_context_ids": [row["chunk_id"] for row in hits],
        }

    def answer(row, *, k):
        return search(row["user_input"], top_k=k)

    recall = create_metric("metric.retrieval.id_context_recall")
    exact = create_metric("metric.text.exact_match")
    for k, pair_recall, pair_exact in [(1, 0.5, 0.0), (2, 1.0, 1.0)]:
        scored = await experiment(partial(answer, k=k)).arun(
            data, metrics=[recall, exact]
        )
        assert scored.row_for(single["case_id"]).metrics[recall.spec.id].value == 1.0
        assert scored.row_for(single["case_id"]).metrics[exact.spec.id].value == 1.0
        row = scored.row_for(pair["case_id"])
        assert row.metrics[recall.spec.id].value == pair_recall
        assert row.metrics[exact.spec.id].value == pair_exact
        assert (
            row.task.output["retrieved_context_ids"]
            == pair["reference_context_ids"][:k]
        )
    assert result.usage.generation.requests == 2 and len(client.calls) == 5

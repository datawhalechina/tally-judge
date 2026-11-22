"""Exercise the installed public API without pytest or provider packages."""

import asyncio
import json
import math
import os
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import tallyjudge
from tallyjudge import (
    Dataset,
    Evaluator,
    ExecutionConfig,
    create_embedding_model,
    create_language_model,
    create_metric,
    create_synthesizer,
)
from tallyjudge.core.errors import PluginError
from tallyjudge.registry import registry

FAITH = "metric.rag.faithfulness"
RELEVANCE = "metric.rag.answer_relevance"


class Client:
    def __init__(self):
        self.outputs = iter(
            [
                {"statements": ["s"]},
                {"statements": [{"statement": "s", "reason": "yes", "verdict": 1}]},
                {"question": "q", "noncommittal": 0},
                {"statements": []},
                {"question": "", "noncommittal": 0},
                {"value": "pass", "reason": "addressed"},
            ]
        )
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.generate))
        self.embeddings = SimpleNamespace(create=self.embed)

    def with_options(self, *, max_retries, timeout):
        assert max_retries == 0
        return self

    async def generate(self, **options):
        message = SimpleNamespace(content=json.dumps(next(self.outputs)))
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=message)]
        )

    async def embed(self, *, input, **options):
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=i, embedding=[1.0, 0.0])
                for i in range(len(input))
            ]
        )


async def check_rag():
    client = Client()
    judge = create_language_model(
        "model.openai_compatible.chat_completions",
        client=client,
        model="test",
        temperature=0.01,
    )
    embeddings = create_embedding_model(
        "embedding.openai_compatible.embeddings", client=client, model="vec"
    )
    metrics = [
        create_metric(FAITH, llm=judge),
        create_metric(RELEVANCE, llm=judge, embeddings=embeddings, strictness=1),
    ]
    row = {"user_input": "q", "response": "a", "retrieved_contexts": ["s"]}
    dataset = Dataset.from_records([row, dict(row, response=""), row])
    result = await Evaluator(metrics, config=ExecutionConfig(max_concurrency=1)).arun(
        dataset
    )
    data = json.loads(result.to_json(include_inputs=False))
    assert "inputs" not in data
    first, invalid, empty = (r["metrics"] for r in data["rows"])
    assert [first[key]["value"] for key in (FAITH, RELEVANCE)] == [1.0, 1.0]
    assert all(
        s["error"]["field"] == "response" and s["usage"] is None
        for s in invalid.values()
    )
    assert (empty[FAITH]["status"], empty[FAITH]["error"]["code"]) == (
        "undefined",
        "no_statements",
    )
    assert empty[RELEVANCE]["value"] == 0.0
    assert data["summary"][FAITH]["coverage"] == 1 / 3
    assert data["summary"][RELEVANCE]["coverage"] == 2 / 3
    assert data["summary"][RELEVANCE]["usage"]["embedding"]["requests"] == 2
    assert all(
        m["revision"] == m["config_snapshot"]["prompt_revision"] == "1"
        for m in data["metrics"].values()
    )
    metric = create_metric(
        "metric.custom.categorical",
        llm=judge,
        criteria="Complete?",
        labels=["pass", "fail"],
    )
    score = await metric.ascore_inputs({"response": "answer"})
    assert (score.value, score.reason) == ("pass", "addressed")


async def check_single_turn_rag():
    client = Client()
    claim = {"statement": "s", "reason": "supported"}
    verdicts = [1, 0, 1]
    client.outputs = iter(
        [
            {"statements": ["s"]},
            {"statements": [dict(claim, verdict=1)]},
            {"question": "q", "noncommittal": 0},
            *({"reason": "fixture", "verdict": value} for value in verdicts * 2),
            {"classifications": [dict(claim, attributed=value) for value in verdicts]},
            {"statements": ["s"]},
            {"statements": ["s"]},
            {"TP": [claim], "FP": [], "FN": []},
        ]
    )
    llm = create_language_model(
        "model.openai_compatible.chat_completions",
        model="test",
        client=client,
        temperature=0.01,
    )
    embeddings = create_embedding_model(
        "embedding.openai_compatible.embeddings", model="vec", client=client
    )
    metrics = [
        create_metric(FAITH, llm=llm),
        create_metric(RELEVANCE, llm=llm, embeddings=embeddings, strictness=1),
        create_metric("metric.retrieval.id_context_precision"),
        create_metric("metric.retrieval.id_context_recall"),
        create_metric("metric.rag.context_precision_with_reference", llm=llm),
        create_metric("metric.rag.context_precision_without_reference", llm=llm),
        create_metric("metric.rag.context_recall", llm=llm),
        create_metric("metric.text.semantic_similarity", embeddings=embeddings),
        create_metric("metric.rag.answer_correctness", llm=llm, embeddings=embeddings),
    ]
    row = {
        "case_id": "ranked",
        "user_input": "q",
        "response": "a",
        "reference": "gold",
        "retrieved_contexts": ["one", "two", "three"],
        "retrieved_context_ids": ["B", "A", "A"],
        "reference_context_ids": ["A", "C", "D"],
    }
    result = await Evaluator(metrics, config=ExecutionConfig(max_concurrency=1)).arun(
        Dataset.from_records([row])
    )
    scores = result.row_for("ranked").metrics
    assert len(scores) == 9 and result.run_status == "completed"
    for metric, expected in zip(
        metrics, [1, 1, 1 / 2, 1 / 3, 5 / 6, 5 / 6, 2 / 3, 1, 1], strict=True
    ):
        score = scores[metric.spec.id]
        assert score.status == "success" and math.isclose(
            score.value, expected, abs_tol=1e-12
        )
    assert list(client.outputs) == []


async def check_synthesis():
    from tallyjudge.plugins.synthesis.graph import EvidenceGraph, build_entity_graph

    client = Client()
    client.outputs = iter(
        [{"query": "When does the library close?", "answer": "Monday"}]
    )
    llm = create_language_model(
        "model.openai_compatible.chat_completions",
        client=client,
        model="test",
        temperature=0.0,
    )
    generator = create_synthesizer("synthesis.testset.single_hop", llm=llm, seed=7)
    [row] = (
        (
            await generator.arun(
                [
                    {
                        "chunk_id": "hours",
                        "document_id": "guide",
                        "text": "Closed on Monday.",
                    }
                ],
                testset_size=2,
            )
        )
        .to_dataset()
        .records
    )
    assert (
        row["reference"],
        row["reference_context_ids"],
        row["reference_document_ids"],
    ) == ("Monday", ["hours"], ["guide"])

    client.outputs = iter([{"entities": ["Kit"]}])
    graph = await build_entity_graph(
        [{"chunk_id": "b", "document_id": "y", "text": "Kit is in Rome."}],
        llm=llm,
    )
    assert graph.generation_usage.requests == 1 and graph.chunks[0]["entities"] == [
        "Kit"
    ]
    graph = await build_entity_graph(
        [
            {
                "chunk_id": "a",
                "document_id": "x",
                "text": "Ada built Kit.",
                "entities": ["Kit"],
            },
            *graph.chunks,
        ],
        llm=llm,
    )
    assert graph.relations == () and graph.generation_usage.requests == 0
    # An explicit archived edge is sufficient for mixed sampling in a base install.
    graph = EvidenceGraph(graph.chunks, [("a", "b")])
    restored = EvidenceGraph.from_dict(json.loads(json.dumps(graph.to_dict())))
    client.outputs = iter(
        [
            {"query": "Who built Kit?", "answer": "Ada"},
            {
                "query": "Where is Ada's creation?",
                "answer": "Rome",
            },
        ]
    )
    mixed = create_synthesizer("synthesis.testset.mixed", llm=llm, seed=7)
    examples = await mixed.arun(restored, testset_size=2)
    assert [row["synthesizer_id"] for row in examples.to_records()] == [
        "synthesis.testset.single_hop",
        "synthesis.testset.multi_hop",
    ]
    assert examples.usage.generation.requests == 2

    client.outputs = iter(
        [
            {"text": "Ada built Kit."},
            {"score": 3},
            {"output": ["Design"]},
            {"entities": ["Kit"]},
            {"name": "Reader", "role_description": "Researches Kit"},
            {"mapping": {"Reader": ["Kit"]}},
            {"query": "Who built Kit?", "answer": "Ada"},
        ]
    )
    embeddings = create_embedding_model(
        "embedding.openai_compatible.embeddings", client=client, model="local"
    )
    archived = create_synthesizer(
        "synthesis.testset.rag", llm=llm, embeddings=embeddings
    )
    result = await archived.arun(
        [{"chunk_id": "a", "document_id": "x", "text": "Ada built Kit."}],
        testset_size=1,
    )
    [sample] = result.to_records()
    assert (
        sample["reference"],
        sample["reference_context_ids"],
        sample["synthesizer_name"],
    ) == (
        "Ada",
        ["a"],
        "single_hop_specific_query_synthesizer",
    )
    assert (
        result.usage.generation.requests == 7 and result.usage.embedding.requests == 1
    )


def check_distances(installed):
    ids = (
        "metric.text.string_similarity",
        "metric.retrieval.text_context_precision",
        "metric.retrieval.text_context_recall",
    )
    # Exercise the process worker from the installed wheel outside the source checkout.
    passage = "文" * 17_000
    row = {
        "response": passage,
        "reference": passage,
        "retrieved_contexts": [passage],
        "reference_contexts": [passage],
    }
    for key in ids:
        assert registry.describe(key).id == key
        if installed:
            score = Evaluator([create_metric(key)]).run(Dataset.from_records([row]))
            assert score.rows[0].metrics[key].value == 1.0
        else:
            try:
                create_metric(key)
            except PluginError as exc:
                assert all(
                    part in str(exc) for part in (key, "rapidfuzz", "text-distance")
                )
            else:
                raise AssertionError(f"{key} unexpectedly available without its extra")


def check_installation():
    assert sys.version_info >= (3, 12)
    assert Path(tallyjudge.__file__).with_name("py.typed").is_file()
    assert {p.kind for p in registry.list()} == {
        "metric",
        "model",
        "embedding",
        "storage",
        "event",
        "synthesis",
        "optimization",
        "cache",
    }
    assert not any(name.startswith("tallyjudge.plugins.") for name in sys.modules)
    metrics = [
        create_metric(key)
        for key in ("metric.text.exact_match", "metric.text.string_presence")
    ]
    dataset = Dataset.from_records(
        [
            {"response": "cat", "reference": "cat"},
            {"response": "dog", "reference": "cat"},
        ]
    )
    result = Evaluator(metrics).run(dataset)
    assert len(result.rows) == 2
    assert all(item["mean"] == 0.5 for item in result.summary.values())
    assert json.loads(result.to_json())["schema_version"] == 3
    assert result.run_status == "completed" and result.stop_reason is None
    asyncio.run(check_rag())
    asyncio.run(check_single_turn_rag())
    asyncio.run(check_synthesis())
    assert not {
        "ragas",
        "numpy",
        "langchain",
        "langchain_core",
        "openai",
        "instructor",
        "datasets",
        "PIL",
        "networkx",
        "sknetwork",
    }.intersection(sys.modules)


if __name__ == "__main__":
    check_installation()
    if len(sys.argv) > 1:
        checkout = Path(sys.argv[1]).resolve()
        module = Path(tallyjudge.__file__).resolve()
        assert not module.is_relative_to(checkout) and not Path.cwd().is_relative_to(
            checkout
        )
        assert module.is_relative_to(Path(sys.prefix).resolve())
        installed = {
            item.metadata["Name"].lower().replace("_", "-")
            for item in metadata.distributions()
        }
        with_distance = sys.argv[2] == "text-distance"
        assert ("rapidfuzz" in installed) == with_distance
        check_distances(with_distance)
        assert not {
            "ragas",
            "numpy",
            "langchain",
            "langchain-core",
            "openai",
            "instructor",
            "datasets",
            "pillow",
            "networkx",
            "scikit-network",
            "pytest",
        }.intersection(installed)
        executable = Path(sys.executable).with_name(
            "tallyjudge.exe" if os.name == "nt" else "tallyjudge"
        )
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            sample = {"case_id": "one", "response": "match", "reference": "match"}
            source = root / "rows.jsonl"
            source.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            key = "metric.text.exact_match"

            def cli(*args):
                output = subprocess.check_output(
                    [str(executable), *map(str, args)], text=True
                )
                return json.loads(output)

            assert key in [metric["id"] for metric in cli("metrics", "list")]
            saved = cli("run", "--input", source, "--metric", key, "--store", root)
            restored = cli("runs", "show", saved["run_id"], "--store", root)
            expected = (
                Evaluator([create_metric(key)])
                .run(Dataset.from_records([sample]))
                .to_dict()
            )
            for field in ("rows", "manifest", "metrics", "summary", "inputs"):
                assert restored[field] == expected[field]
        print("Wheel installation and CLI verified:", module)

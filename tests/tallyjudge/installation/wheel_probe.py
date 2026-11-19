"""Exercise the installed public API without pytest or provider packages."""

import asyncio
import json
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
    assert json.loads(result.to_json())["schema_version"] == 2
    assert result.run_status == "completed" and result.stop_reason is None
    asyncio.run(check_rag())
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

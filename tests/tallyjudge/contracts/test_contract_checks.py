import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tallyjudge import (
    Dataset,
    Evaluator,
    create_cache_backend,
    create_importer,
    create_language_model,
    create_metric,
    create_storage_backend,
)
from tallyjudge.registry import PluginRegistry, PluginSpec
from tallyjudge.sdk import MetricBase, MetricResult, MetricSpec, PluginError
from tallyjudge.sdk.testing import (
    check_cache,
    check_importer,
    check_language_model,
    check_metric,
    check_storage,
)

ROW = {"response": "周一", "reference": "周一"}
DATA = Path(__file__).parents[1] / "integration" / "data"


@pytest.mark.parametrize(
    "plugin_id", ["metric.text.exact_match", "metric.text.string_presence"]
)
@pytest.mark.asyncio
async def test_builtin_metrics_meet_the_metric_contract(plugin_id):
    await check_metric(create_metric(plugin_id), dict(ROW))


@pytest.mark.asyncio
async def test_check_rejects_a_metric_that_breaks_its_declared_range():
    class Overflow(MetricBase):
        spec = MetricSpec(
            "metric.test.overflow", "1", {"response": str}, value_range=(0, 1)
        )

        async def ascore(self, response):
            return MetricResult(2.0)

    with pytest.raises(AssertionError, match="output violates its spec"):
        await check_metric(Overflow(), {"response": "a"})


@pytest.mark.asyncio
async def test_builtin_chat_model_meets_the_model_contract(client):
    client.chat_reply(json.dumps({"answer": "ok"}))
    await check_language_model(
        create_language_model(
            "model.openai_compatible.chat_completions", client=client, model="judge"
        )
    )


@pytest.mark.parametrize("plugin_id", ["storage.local.memory", "storage.local.jsonl"])
def test_builtin_storage_meets_the_storage_contract(plugin_id, tmp_path):
    settings = {"root": tmp_path} if plugin_id.endswith("jsonl") else {}
    result = Evaluator([create_metric("metric.text.exact_match")]).run(
        Dataset.from_records([ROW])
    )
    check_storage(create_storage_backend(plugin_id, **settings), result)


def test_builtin_cache_meets_the_cache_contract(tmp_path):
    check_cache(
        create_cache_backend(
            "cache.local.sqlite", path=tmp_path / "scores.sqlite", namespace="test"
        )
    )


@pytest.mark.parametrize(
    ("plugin_id", "source"),
    [
        (
            "importer.langchain.documents",
            [SimpleNamespace(id="doc-1", page_content="text", metadata={})],
        ),
        (
            "importer.llamaindex.documents",
            [SimpleNamespace(doc_id="doc-1", text="text", metadata={})],
        ),
        ("importer.atif.trajectory", DATA / "atif_trajectory.json"),
        ("importer.inspect.log", DATA / "inspect_mockllm_log.json"),
    ],
)
def test_builtin_importers_return_what_they_declare(plugin_id, source):
    check_importer(create_importer(plugin_id), source)


def test_an_importer_must_declare_what_it_produces():
    class Undeclared:
        plugin_id = "importer.example.undeclared"

        def convert(self, source):
            return source

    registry = PluginRegistry(
        [PluginSpec(Undeclared.plugin_id, "importer", 1, Undeclared)]
    )
    with pytest.raises(PluginError, match="must declare produces"):
        registry.create(Undeclared.plugin_id)

import subprocess
import sys

import pytest
from pydantic import BaseModel, ConfigDict

from tallyjudge import MetricResult
from tallyjudge.plugins.metrics.text.exact_match import ExactMatchMetric
from tallyjudge.plugins.metrics.text.string_presence import StringPresenceMetric
from tallyjudge.registry import PluginRegistry, PluginSpec
from tallyjudge.registry.builtins import BUILTINS
from tallyjudge.registry.registry import KINDS
from tallyjudge.sdk.errors import PluginError
from tallyjudge.sdk.metrics import Metric, MetricBase, MetricSpec

exact: Metric = ExactMatchMetric()
presence: Metric = StringPresenceMetric()


BUILTIN_PLUGINS = [spec for spec in BUILTINS if isinstance(spec, PluginSpec)]


def registry_kinds():
    return list(PluginRegistry(BUILTINS).kinds())


def test_lazy_list_describe_and_construct_in_fresh_process():
    code = """
import sys
from tallyjudge.registry import registry
assert {spec.id for spec in registry.list(kind='synthesis')} == {
    'synthesis.testset.single_hop', 'synthesis.testset.multi_hop', 'synthesis.testset.mixed', 'synthesis.testset.rag',
}
assert {spec.id for spec in registry.list(kind='storage')} == {'storage.local.memory', 'storage.local.jsonl'}
assert {spec.id for spec in registry.list(kind='cache')} == {'cache.local.sqlite'}
assert {spec.id for spec in registry.list(kind='event')} == {'event.local.jsonl', 'event.mlflow.tracking'}
assert {spec.id for spec in registry.list(kind='optimization')} == {'optimization.prompt.genetic', 'optimization.prompt.dspy_mipro'}
assert not any(name.startswith('tallyjudge.plugins.') for name in sys.modules)
settings = {'metric.custom.numeric': {'criteria': 'quality'}, 'metric.custom.categorical': {'criteria': 'quality', 'labels': ['yes', 'no']}}
for plugin in registry.list(kind='metric'):
    assert registry.describe(plugin.id, **settings.get(plugin.id, {})).id == plugin.id
    if not plugin.dependencies and not plugin.required_distributions:
        assert registry.create(plugin.id).spec.id == plugin.id
assert not {'rapidfuzz', 'ragas', 'langchain', 'langchain_core', 'openai', 'instructor', 'PIL', 'networkx', 'sknetwork'}.intersection(sys.modules)
"""
    process = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert process.returncode == 0, process.stderr


def test_registry_failures_and_explicit_replace():
    registry = PluginRegistry(BUILTINS)
    with pytest.raises(PluginError, match="Unknown plugin"):
        registry.create("unknown")
    with pytest.raises(PluginError, match="already registered"):
        registry.register(BUILTIN_PLUGINS[0])
    registry.register(BUILTIN_PLUGINS[0], replace=True)
    assert len(registry.list(kind="metric")) == len(
        [spec for spec in BUILTIN_PLUGINS if spec.kind == "metric"]
    )
    assert len(registry.list(kind="model")) == 2
    with pytest.raises(PluginError, match="configuration"):
        registry.create("metric.text.exact_match", model="unexpected")
    registry.register(PluginSpec("wrong", "metric", 1, ExactMatchMetric))
    with pytest.raises(PluginError, match="must match"):
        registry.create("wrong")
    with pytest.raises(PluginError, match="matching MetricSpec"):
        registry.describe("wrong")
    registry.register(PluginSpec("version", "metric", 99, "must_not_import:Factory"))
    with pytest.raises(PluginError, match="api_version"):
        registry.create("version")
    registry.register(PluginSpec("unloaded", "metric", 1, "must_not_import:Factory"))
    with pytest.raises(PluginError, match="Cannot load.*unloaded") as unloaded:
        registry.create("unloaded")
    assert isinstance(unloaded.value.__cause__, ModuleNotFoundError)
    registry.register(
        PluginSpec(
            "absent",
            "metric",
            1,
            ExactMatchMetric,
            ("tallyjudge-nonexistent-test-sdk",),
        )
    )
    with pytest.raises(PluginError, match="Missing optional distribution"):
        registry.create("absent")


def test_custom_schema_factory_and_snapshot():
    class Settings(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        score: float = 0.5

    class ConstantMetric(MetricBase):
        spec = MetricSpec("metric.test.constant", "1", {"response": str})

        def __init__(self, score=0.5):
            self.value = score

        async def ascore(self, response):
            return MetricResult(self.value)

        def config_snapshot(self):
            return {"score": self.value}

    def factory(**config):
        return ConstantMetric(**Settings(**config).model_dump())

    factory.spec = ConstantMetric.spec
    factory.Config = Settings
    registry = PluginRegistry(
        [PluginSpec(ConstantMetric.spec.id, "metric", 1, factory)]
    )
    assert registry.describe(ConstantMetric.spec.id) is ConstantMetric.spec
    metric = registry.create(ConstantMetric.spec.id, score=0.75)
    assert metric.config_snapshot() == {"score": 0.75}
    with pytest.raises(PluginError, match="Invalid plugin configuration"):
        registry.create(ConstantMetric.spec.id, score="not a number")


def test_construction_errors_preserve_cause_and_plugin_id():
    original = ValueError("API key is missing")

    def broken_factory():
        raise original

    registry = PluginRegistry(
        [PluginSpec("metric.test.constructor", "metric", 1, broken_factory)]
    )
    with pytest.raises(PluginError, match="metric.test.constructor") as caught:
        registry.create("metric.test.constructor")
    assert caught.value.__cause__ is original


def test_missing_named_scorer_fails_at_construction():
    class Incomplete(MetricBase):
        spec = MetricSpec("metric.test.incomplete", "1", {})

    registry = PluginRegistry([PluginSpec(Incomplete.spec.id, "metric", 1, Incomplete)])
    with pytest.raises(PluginError, match="metric.test.incomplete") as caught:
        registry.create(Incomplete.spec.id)
    assert isinstance(caught.value.__cause__, TypeError)


def test_builtin_manifest_matches_the_plugin_class_declarations():
    from tallyjudge.sdk.plugin import declared_kinds, declared_plugins

    assert list(declared_plugins("tallyjudge.plugins")) == BUILTIN_PLUGINS
    assert [kind.kind for kind in declared_kinds("tallyjudge.plugins")] == [
        "optimization",
        "synthesis",
    ]
    assert registry_kinds() == [*KINDS, "optimization", "synthesis"]

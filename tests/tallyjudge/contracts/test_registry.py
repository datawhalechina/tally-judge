import subprocess
import sys

import pytest
from pydantic import BaseModel, ConfigDict

from tallyjudge import MetricResult
from tallyjudge.core.errors import PluginError
from tallyjudge.interfaces.metrics import Metric, MetricBase, MetricSpec
from tallyjudge.plugins.metrics.text.exact_match import ExactMatchMetric
from tallyjudge.plugins.metrics.text.string_presence import StringPresenceMetric
from tallyjudge.registry import PluginRegistry, PluginSpec
from tallyjudge.registry.builtins import BUILTINS

# Also included in the pyright input set: named ascore parameters must not
# break the executor-facing structural protocol.
exact: Metric = ExactMatchMetric()
presence: Metric = StringPresenceMetric()


def test_lazy_list_describe_and_construct_in_fresh_process():
    code = """
import sys
from tallyjudge.registry import registry
assert {spec.id for spec in registry.list(kind='storage')} == {'storage.local.memory', 'storage.local.jsonl'}
assert {spec.id for spec in registry.list(kind='cache')} == {'cache.local.sqlite'}
assert {spec.id for spec in registry.list(kind='event')} == {'event.local.jsonl'}
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
        registry.register(BUILTINS[0])
    registry.register(BUILTINS[0], replace=True)
    assert len(registry.list(kind="metric")) == len(
        [spec for spec in BUILTINS if spec.kind == "metric"]
    )
    assert len(registry.list(kind="model")) == 1
    with pytest.raises(PluginError, match="configuration"):
        registry.create(BUILTINS[0].id, model="unexpected")
    registry.register(PluginSpec("wrong", "metric", 1, ExactMatchMetric))
    with pytest.raises(PluginError, match="must match"):
        registry.create("wrong")
    with pytest.raises(PluginError, match="matching MetricSpec"):
        registry.describe("wrong")
    registry.register(PluginSpec("version", "metric", 99, "must_not_import:Factory"))
    with pytest.raises(PluginError, match="api_version"):
        registry.create("version")
    registry.register(PluginSpec("unloaded", "metric", 1, "must_not_import:Factory"))
    with pytest.raises(PluginError, match="configuration"):
        registry.create("unloaded", bad=True)
    with pytest.raises(PluginError, match="Cannot load"):
        registry.create("unloaded")
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
        return ConstantMetric(**config)

    factory.spec = ConstantMetric.spec
    registry = PluginRegistry(
        [PluginSpec(ConstantMetric.spec.id, "metric", 1, factory, config_type=Settings)]
    )
    assert registry.describe(ConstantMetric.spec.id) is ConstantMetric.spec
    metric = registry.create(ConstantMetric.spec.id, score=0.75)
    assert metric.config_snapshot() == {"score": 0.75}
    with pytest.raises(PluginError, match="Invalid plugin configuration"):
        registry.create(ConstantMetric.spec.id, score="not a number")


def test_load_errors_preserve_cause_and_plugin_id(monkeypatch):
    import importlib

    original = ModuleNotFoundError("missing_optional_sdk")

    def broken_import(name):
        raise original

    monkeypatch.setattr(importlib, "import_module", broken_import)
    registry = PluginRegistry(
        [PluginSpec("metric.test.missing", "metric", 1, "broken.module:Factory")]
    )
    with pytest.raises(PluginError, match="metric.test.missing") as caught:
        registry.create("metric.test.missing")
    assert caught.value.__cause__ is original
    assert not isinstance(caught.value.__cause__, PluginError)


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

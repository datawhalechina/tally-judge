import importlib
import logging
import sys
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Protocol, runtime_checkable

import pytest

from tallyjudge import Dataset, Evaluator
from tallyjudge.registry import KindSpec, PluginRegistry, PluginSpec
from tallyjudge.registry.builtins import BUILTINS
from tallyjudge.sdk import PluginError
from tallyjudge.sdk.plugin import manifest_source

EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "length-ratio-plugin"
GROUP = "tallyjudge.plugins"
BUILTIN_PLUGINS = [spec for spec in BUILTINS if isinstance(spec, PluginSpec)]
PLUGIN_ID = "metric.length_ratio.response_to_reference"


def _installed(monkeypatch, *values):
    entries = [EntryPoint(f"entry{i}", value, GROUP) for i, value in enumerate(values)]
    monkeypatch.setattr(
        importlib.import_module("tallyjudge.registry.registry"),
        "entry_points",
        lambda group: entries if group == GROUP else [],
    )
    monkeypatch.syspath_prepend(str(EXAMPLE))
    return PluginRegistry(BUILTINS, entry_point_group=GROUP)


def test_installed_plugin_is_listed_lazily_and_scores(monkeypatch):
    registry = _installed(monkeypatch, "tallyjudge_length_ratio.manifest:PLUGINS")
    assert PLUGIN_ID in {spec.id for spec in registry.list(kind="metric")}
    assert "tallyjudge_length_ratio.metric" not in sys.modules
    metric = registry.create(PLUGIN_ID, cap=1.5)
    result = Evaluator([metric]).run(
        Dataset.from_records([{"response": "abcd", "reference": "ab"}])
    )
    assert result.rows[0].metrics[PLUGIN_ID].value == 1.5
    assert result.metrics[PLUGIN_ID]["config_snapshot"] == {"cap": 1.5}


def test_broken_or_conflicting_plugins_are_skipped_with_a_warning(monkeypatch, caplog):
    registry = _installed(
        monkeypatch,
        "tallyjudge_missing_plugin.manifest:PLUGINS",
        "tallyjudge.registry.builtins:BUILTINS",
        "tallyjudge_length_ratio.manifest:PLUGINS",
    )
    with caplog.at_level(logging.WARNING, logger="tallyjudge.registry.registry"):
        listed = registry.list()
    assert len(listed) == len(BUILTIN_PLUGINS) + 1
    assert "tallyjudge_missing_plugin.manifest:PLUGINS" in caplog.text
    assert "tallyjudge.registry.builtins:BUILTINS" in caplog.text
    registry.create("metric.text.exact_match")


def test_allowlist_limits_which_installed_plugins_are_read(monkeypatch):
    monkeypatch.setenv("TALLYJUDGE_PLUGINS", "entry1")
    registry = _installed(
        monkeypatch,
        "tallyjudge_length_ratio.manifest:PLUGINS",
        "tallyjudge_length_ratio.manifest:PLUGINS",
    )
    assert PLUGIN_ID in {spec.id for spec in registry.list()}
    monkeypatch.setenv("TALLYJUDGE_PLUGINS", "")
    registry = _installed(monkeypatch, "tallyjudge_length_ratio.manifest:PLUGINS")
    assert len(registry.list()) == len(BUILTIN_PLUGINS)


@runtime_checkable
class Simulator(Protocol):
    def reply(self, message: str) -> str: ...


class EchoSimulator:
    plugin_id = "simulator.example.echo"

    def reply(self, message):
        return message


class SilentSimulator:
    plugin_id = "simulator.example.silent"


def test_a_package_defines_its_own_plugin_kind_without_touching_the_kernel():
    registry = PluginRegistry(
        [
            PluginSpec(EchoSimulator.plugin_id, "simulator", 1, EchoSimulator),
            PluginSpec(SilentSimulator.plugin_id, "simulator", 1, SilentSimulator),
            KindSpec("simulator", Simulator),
        ]
    )
    assert registry.kinds()[-1] == "simulator"
    assert [spec.id for spec in registry.list(kind="simulator")] == [
        EchoSimulator.plugin_id,
        SilentSimulator.plugin_id,
    ]
    echo = registry.create_for_kind("simulator", EchoSimulator.plugin_id)
    assert echo.reply("hello") == "hello"
    with pytest.raises(PluginError, match="simulator contract"):
        registry.create(SilentSimulator.plugin_id)
    for taken in ("simulator", "metric"):
        with pytest.raises(PluginError, match="already defined"):
            registry.register(KindSpec(taken, Simulator))
    registry.register(PluginSpec("judge.example.one", "judge", 1, EchoSimulator))
    with pytest.raises(PluginError, match="Unexpected plugin kind"):
        registry.create("judge.example.one")


def test_a_generated_manifest_carries_the_kinds_a_package_declares(
    tmp_path, monkeypatch
):
    package = tmp_path / "simulators"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "contract.py").write_text(
        "from typing import Protocol, runtime_checkable\n"
        "from tallyjudge.sdk import KindSpec\n\n"
        "@runtime_checkable\n"
        "class Simulator(Protocol):\n"
        "    def reply(self, message: str) -> str: ...\n\n"
        "SIMULATOR = KindSpec('simulator', Simulator)\n"
    )
    (package / "echo.py").write_text(
        "from tallyjudge.sdk import Plugin\n\n"
        "class Echo:\n"
        "    plugin = Plugin('simulator')\n"
        "    plugin_id = 'simulator.simulators.echo'\n\n"
        "    def reply(self, message):\n"
        "        return message\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    source = manifest_source("simulators")
    assert "KindSpec('simulator', 'simulators.contract:Simulator')" in source
    (package / "manifest.py").write_text(source)
    for name in [name for name in sys.modules if name.startswith("simulators")]:
        monkeypatch.delitem(sys.modules, name)
    registry = _installed(monkeypatch, "simulators.manifest:PLUGINS")
    assert "simulator" in registry.kinds()
    assert "simulators.echo" not in sys.modules
    assert registry.create("simulator.simulators.echo").reply("hi") == "hi"

from tallyjudge.registry.builtins import BUILTINS
from tallyjudge.registry.registry import PluginRegistry
from tallyjudge.sdk.plugin import KindSpec, PluginSpec

registry = PluginRegistry(BUILTINS, entry_point_group="tallyjudge.plugins")

__all__ = ["KindSpec", "PluginRegistry", "PluginSpec", "registry"]

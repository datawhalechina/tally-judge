from tallyjudge.registry.builtins import BUILTINS
from tallyjudge.registry.registry import PluginRegistry
from tallyjudge.registry.specs import PluginSpec

registry = PluginRegistry(BUILTINS)

__all__ = ["PluginRegistry", "PluginSpec", "registry"]

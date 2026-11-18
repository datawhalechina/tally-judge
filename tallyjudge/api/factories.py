"""Factories for implemented plugin kinds."""

from typing import Any

from tallyjudge.interfaces.metrics import Metric
from tallyjudge.registry import registry


def create_metric(plugin_id: str, **config: Any) -> Metric:
    return registry.create(plugin_id, **config)

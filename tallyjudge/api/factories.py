"""Factories for implemented plugin kinds."""

from typing import Any

from tallyjudge.interfaces.metrics import Metric
from tallyjudge.interfaces.models import EmbeddingModel, LanguageModel
from tallyjudge.registry import registry


def create_metric(plugin_id: str, **config: Any) -> Metric:
    """Build one metric from its registered ID, settings and required model objects."""
    return registry.create_for_kind("metric", plugin_id, **config)


def create_language_model(plugin_id: str, **config: Any) -> LanguageModel:
    """Build an async generation plugin using the caller's client."""
    return registry.create_for_kind("model", plugin_id, **config)


def create_embedding_model(plugin_id: str, **config: Any) -> EmbeddingModel:
    """Build an async embedding plugin using the caller's client."""
    return registry.create_for_kind("embedding", plugin_id, **config)

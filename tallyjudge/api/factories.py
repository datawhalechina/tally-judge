"""Factories for implemented plugin kinds."""

from typing import Any

from tallyjudge.interfaces.cache import ScoreCache
from tallyjudge.interfaces.events import EventSink
from tallyjudge.interfaces.metrics import Metric
from tallyjudge.interfaces.models import EmbeddingModel, LanguageModel
from tallyjudge.interfaces.optimization import PromptOptimizer
from tallyjudge.interfaces.storage import StorageBackend
from tallyjudge.interfaces.synthesis import Synthesizer
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


def create_optimizer(plugin_id: str, **config: Any) -> PromptOptimizer:
    """Build an optimizer for labeled application prompt experiments."""
    return registry.create_for_kind("optimization", plugin_id, **config)


def create_synthesizer(plugin_id: str, **config: Any) -> Synthesizer:
    """Build a dataset generator using its registered ID and supplied model."""
    return registry.create_for_kind("synthesis", plugin_id, **config)


def create_storage_backend(plugin_id: str, **config: Any) -> StorageBackend:
    """Construct a run store from its registered ID and local settings."""
    return registry.create_for_kind("storage", plugin_id, **config)


def create_cache_backend(plugin_id: str, **config: Any) -> ScoreCache:
    """Construct a score cache with a deployment namespace."""
    return registry.create_for_kind("cache", plugin_id, **config)


def create_event_sink(plugin_id: str, **config: Any) -> EventSink:
    """Construct an observer for run, step and request events."""
    return registry.create_for_kind("event", plugin_id, **config)

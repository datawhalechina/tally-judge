from typing import TYPE_CHECKING, Any

from tallyjudge.registry import registry
from tallyjudge.sdk.analysis import Analyzer
from tallyjudge.sdk.cache import ScoreCache
from tallyjudge.sdk.events import EventSink
from tallyjudge.sdk.importing import Importer
from tallyjudge.sdk.metrics import Metric
from tallyjudge.sdk.models import EmbeddingModel, LanguageModel
from tallyjudge.sdk.running import Runner
from tallyjudge.sdk.storage import StorageBackend

if TYPE_CHECKING:
    from tallyjudge.plugins.optimization.contract import PromptOptimizer
    from tallyjudge.plugins.synthesis.contract import Synthesizer


def create_metric(plugin_id: str, **config: Any) -> Metric:
    return registry.create_for_kind("metric", plugin_id, **config)


def create_language_model(plugin_id: str, **config: Any) -> LanguageModel:
    return registry.create_for_kind("model", plugin_id, **config)


def create_embedding_model(plugin_id: str, **config: Any) -> EmbeddingModel:
    return registry.create_for_kind("embedding", plugin_id, **config)


def create_optimizer(plugin_id: str, **config: Any) -> "PromptOptimizer":
    return registry.create_for_kind("optimization", plugin_id, **config)


def create_synthesizer(plugin_id: str, **config: Any) -> "Synthesizer":
    return registry.create_for_kind("synthesis", plugin_id, **config)


def create_analyzer(plugin_id: str, **config: Any) -> Analyzer:
    return registry.create_for_kind("analyzer", plugin_id, **config)


def create_importer(plugin_id: str, **config: Any) -> Importer:
    return registry.create_for_kind("importer", plugin_id, **config)


def create_runner(plugin_id: str, **config: Any) -> Runner:
    return registry.create_for_kind("runner", plugin_id, **config)


def create_storage_backend(plugin_id: str, **config: Any) -> StorageBackend:
    return registry.create_for_kind("storage", plugin_id, **config)


def create_cache_backend(plugin_id: str, **config: Any) -> ScoreCache:
    return registry.create_for_kind("cache", plugin_id, **config)


def create_event_sink(plugin_id: str, **config: Any) -> EventSink:
    return registry.create_for_kind("event", plugin_id, **config)

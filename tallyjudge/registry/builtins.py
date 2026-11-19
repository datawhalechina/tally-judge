"""Explicit built-ins; MetricSpec lives only on each metric class."""

from tallyjudge.registry.specs import PluginSpec

BUILTINS = (
    PluginSpec(
        "metric.text.exact_match",
        "metric",
        1,
        "tallyjudge.plugins.metrics.text.exact_match:ExactMatchMetric",
    ),
    PluginSpec(
        "metric.text.string_presence",
        "metric",
        1,
        "tallyjudge.plugins.metrics.text.string_presence:StringPresenceMetric",
    ),
    PluginSpec(
        "metric.rag.faithfulness",
        "metric",
        1,
        "tallyjudge.plugins.metrics.rag.faithfulness:FaithfulnessMetric",
        dependencies={"llm": "language_model"},
    ),
    PluginSpec(
        "metric.rag.answer_relevance",
        "metric",
        1,
        "tallyjudge.plugins.metrics.rag.answer_relevance:AnswerRelevanceMetric",
        config_type="tallyjudge.plugins.metrics.rag.answer_relevance:AnswerRelevanceConfig",
        dependencies={"llm": "language_model", "embeddings": "embedding_model"},
    ),
    PluginSpec(
        "model.openai_compatible.chat_completions",
        "model",
        1,
        "tallyjudge.plugins.models.openai_compatible:OpenAICompatibleLanguageModel",
        config_type="tallyjudge.plugins.models.openai_compatible:ChatConfig",
        dependencies={"client": "client"},
    ),
    PluginSpec(
        "embedding.openai_compatible.embeddings",
        "embedding",
        1,
        "tallyjudge.plugins.embeddings.openai_compatible:OpenAICompatibleEmbeddingModel",
        config_type="tallyjudge.plugins.embeddings.openai_compatible:EmbeddingConfig",
        dependencies={"client": "client"},
    ),
    PluginSpec(
        "storage.local.memory",
        "storage",
        1,
        "tallyjudge.plugins.storage.memory:MemoryStorageBackend",
    ),
    PluginSpec(
        "storage.local.jsonl",
        "storage",
        1,
        "tallyjudge.plugins.storage.jsonl:JsonlStorageBackend",
        config_type="tallyjudge.plugins._local:DirectoryConfig",
    ),
    PluginSpec(
        "event.local.jsonl",
        "event",
        1,
        "tallyjudge.plugins.tracing.jsonl:JsonlEventSink",
        config_type="tallyjudge.plugins._local:DirectoryConfig",
    ),
    PluginSpec(
        "cache.local.sqlite",
        "cache",
        1,
        "tallyjudge.plugins.cache.sqlite:SqliteScoreCache",
        config_type="tallyjudge.plugins.cache.sqlite:SqliteCacheConfig",
    ),
)

"""SemanticSimilarity from Ragas collections/_semantic_similarity.py (ec00339, Apache-2.0).

SemanticSimilarityMetric preserves negative scores and inclusive thresholds. A zero norm is undefined.
"""

from typing import Any

from tallyjudge.plugins.metrics._cosine import embedding_cosine
from tallyjudge.sdk.config import EmptyConfig
from tallyjudge.sdk.gate import gated
from tallyjudge.sdk.metrics import MetricBase, MetricSpec
from tallyjudge.sdk.models import EmbeddingModel, model_snapshot
from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.results import MetricResult


class SemanticSimilarityConfig(EmptyConfig):
    model_config = {"allow_inf_nan": False}

    threshold: float | None = None


class SemanticSimilarityMetric(MetricBase):
    plugin = Plugin("metric", dependencies={"embeddings": "embedding_model"})
    plugin_id = "metric.text.semantic_similarity"

    Config = SemanticSimilarityConfig

    @staticmethod
    def spec_from_config(*, threshold: float | None = None) -> MetricSpec:
        return MetricSpec(
            SemanticSimilarityMetric.plugin_id,
            "1",
            {"reference": str, "response": str},
            value_range=(-1.0, 1.0) if threshold is None else (0.0, 1.0),
        )

    def __init__(self, embeddings: EmbeddingModel, threshold: float | None = None):
        self.embeddings = gated(embeddings)
        self.threshold = SemanticSimilarityConfig(threshold=threshold).threshold
        self.spec = self.spec_from_config(threshold=self.threshold)

    def config_snapshot(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "embeddings": model_snapshot(self.embeddings),
        }

    async def ascore(self, reference: str, response: str) -> MetricResult:
        similarity = await embedding_cosine(
            self.embeddings, reference or " ", response or " "
        )
        return MetricResult(
            similarity
            if self.threshold is None
            else float(similarity >= self.threshold)
        )

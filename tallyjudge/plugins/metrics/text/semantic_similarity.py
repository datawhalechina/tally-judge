"""SemanticSimilarity from Ragas collections/_semantic_similarity.py (ec00339, Apache-2.0).

SemanticSimilarityMetric preserves negative scores and inclusive thresholds. A zero norm is undefined.
"""

from typing import Any

from tallyjudge.core.results import MetricResult
from tallyjudge.core.usage import run_scoped_score
from tallyjudge.interfaces.metrics import MetricBase, MetricSpec
from tallyjudge.interfaces.models import EmbeddingModel
from tallyjudge.plugins.metrics._cosine import embedding_cosine
from tallyjudge.registry.specs import EmptyConfig


class SemanticSimilarityConfig(EmptyConfig):
    model_config = {"allow_inf_nan": False}

    threshold: float | None = None


class SemanticSimilarityMetric(MetricBase):
    """Compare the reference and response by cosine or an inclusive binary threshold."""

    @staticmethod
    def spec_from_config(*, threshold: float | None = None) -> MetricSpec:
        """Describe a signed cosine or a binary score without an embedding client."""
        return MetricSpec(
            "metric.text.semantic_similarity",
            "1",
            {"reference": str, "response": str},
            value_range=(-1.0, 1.0) if threshold is None else (0.0, 1.0),
        )

    def __init__(self, embeddings: EmbeddingModel, threshold: float | None = None):
        self.embeddings = embeddings
        self.threshold = SemanticSimilarityConfig(threshold=threshold).threshold
        self.spec = self.spec_from_config(threshold=self.threshold)

    def config_snapshot(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "embeddings": {
                "plugin_id": self.embeddings.plugin_id,
                "config": self.embeddings.config_snapshot(),
            },
        }

    async def ascore(self, reference: str, response: str) -> MetricResult:
        """Embed the reference first; an empty string is sent as one ASCII space."""
        return await self.ascore_inputs({"reference": reference, "response": response})

    async def _ascore_validated(self, reference: str, response: str) -> MetricResult:
        similarity = await run_scoped_score(
            embedding_cosine(self.embeddings, reference or " ", response or " ")
        )
        return MetricResult(
            similarity
            if self.threshold is None
            else float(similarity >= self.threshold)
        )

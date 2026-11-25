"""The shared input and model contract for context judgments, migrated from Ragas ec00339 (Apache-2.0)."""

from functools import cache
from typing import ClassVar

from tallyjudge.plugins.metrics._judge import ModelJudge
from tallyjudge.plugins.metrics.rag._inputs import required_text_inputs
from tallyjudge.sdk.metrics import MetricSpec
from tallyjudge.sdk.models import LanguageModel


class ContextJudge(ModelJudge):
    plugin_id: ClassVar[str]
    answer_field: ClassVar[str]

    @classmethod
    @cache
    def spec_from_config(cls) -> MetricSpec:
        return MetricSpec(
            cls.plugin_id,
            "2",
            required_text_inputs("user_input", cls.answer_field, passages=True),
            value_range=(0.0, 1.0),
        )

    def __init__(self, llm: LanguageModel):
        super().__init__(llm)
        self.spec = self.spec_from_config()

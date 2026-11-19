"""The shared input and model contract for context judgments, migrated from Ragas ec00339 (Apache-2.0)."""

from functools import cache
from typing import Annotated, Any, ClassVar

from tallyjudge.interfaces.metrics import MetricBase, MetricSpec
from tallyjudge.interfaces.models import LanguageModel


class ContextJudge(MetricBase):
    """Judge retrieved passages against a question and a plugin-selected answer."""

    plugin_id: ClassVar[str]
    answer_field: ClassVar[str]

    @classmethod
    @cache
    def spec_from_config(cls) -> MetricSpec:
        """Describe required text and passages without constructing a judge."""
        from pydantic import Field

        text = Annotated[str, Field(min_length=1)]
        contexts = Annotated[list[str], Field(min_length=1)]
        return MetricSpec(
            cls.plugin_id,
            "1",
            {
                "user_input": text,
                cls.answer_field: text,
                "retrieved_contexts": contexts,
            },
            value_range=(0.0, 1.0),
        )

    def __init__(self, llm: LanguageModel):
        self.llm = llm
        self.spec = self.spec_from_config()

    def config_snapshot(self) -> dict[str, Any]:
        return {
            "prompt_revision": "1",
            "llm": {
                "plugin_id": self.llm.plugin_id,
                "config": self.llm.config_snapshot(),
            },
        }

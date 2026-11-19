"""Snapshot the scoring rule and model deployment used by a custom judge."""

import hashlib
from typing import Any

from tallyjudge.core.dataset import canonical_json
from tallyjudge.interfaces.models import LanguageModel


def rule_digest(rule: str | dict[str, str]) -> str:
    return hashlib.sha256(canonical_json(rule).encode("utf-8")).hexdigest()


def judge_snapshot(llm: LanguageModel, *, prompt_revision: str = "1") -> dict[str, Any]:
    return {
        "prompt_revision": prompt_revision,
        "llm": {"plugin_id": llm.plugin_id, "config": llm.config_snapshot()},
    }

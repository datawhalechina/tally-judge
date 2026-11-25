import json
from datetime import datetime
from pathlib import Path
from typing import Any


def json_document(source: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(source, dict):
        return source
    return json.loads(Path(source).read_text(encoding="utf-8"))


def seconds(timestamp: str | None) -> float | None:
    return None if timestamp is None else datetime.fromisoformat(timestamp).timestamp()


def text(content: str | list[dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    return "".join(part["text"] for part in content if part["type"] == "text")

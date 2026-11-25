import re
from pathlib import Path

from pydantic import BaseModel


class DirectoryConfig(BaseModel):
    root: Path | str


def directory(root: Path | str) -> Path:
    return Path(DirectoryConfig.model_validate({"root": root}, strict=True).root)


def run_file(root: Path, prefix: str, run_id: str) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", run_id):
        raise ValueError("Invalid run ID for a local file")
    return root / f"{prefix}-{run_id}.jsonl"

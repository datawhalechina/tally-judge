"""Disk-backed score cache. SQLite commits one complete value per key."""

import json
import sqlite3
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

from pydantic import BaseModel, Field

from tallyjudge.core.dataset import canonical_json
from tallyjudge.core.results import MetricResult


class SqliteCacheConfig(BaseModel):
    path: Path | str
    namespace: str = Field(min_length=1)


class SqliteScoreCache:
    """Store successful binding scores in one SQLite file, keyed by score identity.

    Methods block on file locks; the scheduler calls them from worker threads.
    """

    def __init__(self, *, path: str | Path, namespace: str) -> None:
        self.path = Path(path)
        self.namespace = namespace
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, score TEXT NOT NULL)"
            )

    def get(self, key: str) -> MetricResult | None:
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute(
                "SELECT score FROM scores WHERE key = ?", (key,)
            ).fetchone()
        return MetricResult(**json.loads(row[0])) if row else None

    def put(self, key: str, score: MetricResult) -> None:
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                "INSERT INTO scores(key, score) VALUES (?, ?) ON CONFLICT(key) DO NOTHING",
                (key, canonical_json(asdict(score))),
            )

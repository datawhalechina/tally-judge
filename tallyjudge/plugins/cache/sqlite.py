import json
import sqlite3
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

from pydantic import BaseModel, Field

from tallyjudge.sdk.dataset import canonical_json
from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.results import MetricResult


class SqliteCacheConfig(BaseModel):
    path: Path | str
    namespace: str = Field(min_length=1)


class SqliteScoreCache:
    plugin = Plugin("cache")
    plugin_id = "cache.local.sqlite"

    Config = SqliteCacheConfig

    def __init__(self, *, path: str | Path, namespace: str) -> None:
        config = SqliteCacheConfig.model_validate(
            {"path": path, "namespace": namespace}, strict=True
        )
        self.path = Path(config.path)
        self.namespace = config.namespace
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

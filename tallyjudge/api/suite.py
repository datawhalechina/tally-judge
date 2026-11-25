import json
import subprocess
import tomllib
from contextlib import AsyncExitStack
from importlib import import_module
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tallyjudge.api.evaluator import Evaluator
from tallyjudge.engine.config import ExecutionConfig
from tallyjudge.registry import registry
from tallyjudge.sdk.dataset import Dataset
from tallyjudge.sdk.metrics import MetricBinding
from tallyjudge.sdk.results import RunResult
from tallyjudge.sdk.storage import StorageBackend
from tallyjudge.sdk.sync import run_sync

MERGED_BY_NAME = ("models", "metrics")


class _Plugin(BaseModel):
    model_config = ConfigDict(extra="allow")
    plugin: str


class _Metric(_Plugin):
    input_map: dict[str, str] = {}


class _Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    id_field: str = "case_id"


class SuiteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    dataset: _Dataset
    models: dict[str, _Plugin] = {}
    task: _Plugin | None = None
    metrics: dict[str, _Metric] = Field(min_length=1)
    execution: dict[str, Any] = {}


def _layer(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        merged[key] = {**base.get(key, {}), **value} if key in MERGED_BY_NAME else value
    return merged


class Suite:
    def __init__(self, config: SuiteConfig, root: Path) -> None:
        self.config = config
        self.root = root

    @classmethod
    def load(cls, path: str | Path, *overlays: str | Path) -> "Suite":
        path = Path(path)
        data: dict[str, Any] = {}
        for layer in (path, *map(Path, overlays)):
            data = _layer(data, tomllib.loads(layer.read_text(encoding="utf-8")))
        return cls(SuiteConfig.model_validate(data), path.resolve().parent)

    def effective_config(self) -> dict[str, Any]:
        return self.config.model_dump(mode="json", exclude_none=True)

    def _metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "suite": {"id": self.config.id, "revision": self.config.revision}
        }
        from tallyjudge.plugins.versioning.git import git_revision

        try:
            metadata["git_commit"] = git_revision(self.root)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        return metadata

    def _dataset(self) -> Dataset:
        source = self.config.dataset
        with (self.root / source.path).open(encoding="utf-8") as stream:
            return Dataset.from_records(
                (json.loads(line) for line in stream if line.strip()),
                id_field=source.id_field,
            )

    async def arun(self, *, backend: StorageBackend | None = None) -> RunResult:
        async with AsyncExitStack() as stack:
            models: dict[str, Any] = {}

            async def settings(entry: _Plugin) -> dict[str, Any]:
                spec = registry.spec(entry.plugin)
                values = dict(entry.model_extra or {})
                for name, kind in {
                    **spec.dependencies,
                    **spec.optional_dependencies,
                }.items():
                    if name in values:
                        values[name] = (
                            await client(values[name])
                            if kind == "client"
                            else await model(values[name])
                        )
                return values

            async def client(path: str) -> Any:
                module, name = path.split(":")
                created = getattr(import_module(module), name)()
                if hasattr(created, "__aenter__"):
                    created = await stack.enter_async_context(created)
                return created

            async def model(name: str) -> Any:
                if name not in models:
                    entry = self.config.models[name]
                    models[name] = registry.create(
                        entry.plugin, **await settings(entry)
                    )
                return models[name]

            bindings = [
                MetricBinding(
                    registry.create_for_kind(
                        "metric", entry.plugin, **await settings(entry)
                    ),
                    key,
                    entry.input_map,
                )
                for key, entry in self.config.metrics.items()
            ]
            task = self.config.task
            runner = (
                None
                if task is None
                else registry.create_for_kind(
                    "runner", task.plugin, **await settings(task)
                )
            )
            evaluator = Evaluator(
                bindings,
                config=ExecutionConfig(**self.config.execution),
                backend=backend,
            )
            return await evaluator._arun(
                self._dataset(), None, runner, self._metadata()
            )

    def run(self, *, backend: StorageBackend | None = None) -> RunResult:
        return run_sync(lambda: self.arun(backend=backend), "arun")

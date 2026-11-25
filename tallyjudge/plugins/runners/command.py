import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tallyjudge.sdk.plugin import Plugin


class CommandRunnerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    command: list[str] = Field(min_length=1)
    template: str | None = None
    collect: list[str] = []

    @model_validator(mode="after")
    def collect_needs_a_workspace(self) -> Self:
        if self.collect and self.template is None:
            raise ValueError("collect needs a template directory to run in")
        return self


class CommandRunner:
    plugin = Plugin("runner")
    plugin_id = "runner.command.json"
    Config = CommandRunnerConfig

    def __init__(self, **settings: Any):
        self.config = CommandRunnerConfig(**settings)

    async def _run(self, row: dict[str, Any], directory: str | None) -> dict[str, Any]:
        process = await asyncio.create_subprocess_exec(
            *self.config.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            cwd=directory,
        )
        try:
            output, _ = await process.communicate(
                json.dumps(row, ensure_ascii=False).encode("utf-8")
            )
        except asyncio.CancelledError:
            process.kill()
            raise
        if process.returncode:
            raise RuntimeError(f"Command exited with status {process.returncode}")
        return json.loads(output)

    async def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        template = self.config.template
        if template is None:
            return await self._run(row, None)
        with tempfile.TemporaryDirectory() as directory:
            await asyncio.to_thread(
                shutil.copytree, template, directory, dirs_exist_ok=True
            )
            output = await self._run(row, directory)
            files = {}
            for name in self.config.collect:
                path = Path(directory, name)
                files[name] = (
                    path.read_text(encoding="utf-8") if path.exists() else None
                )
            output["workspace_files"] = files
            return output

import asyncio
import json
from typing import Any
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field

from tallyjudge.sdk.plugin import Plugin


class HttpRunnerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    url: str = Field(pattern=r"^https?://")
    headers: dict[str, str] = {}
    timeout: float = Field(default=60.0, gt=0)


class HttpRunner:
    plugin = Plugin("runner")

    Config = HttpRunnerConfig
    plugin_id = "runner.http.json"

    def __init__(self, **settings: Any):
        self.config = HttpRunnerConfig(**settings)

    def _post(self, row: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            self.config.url,
            data=json.dumps(row, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", **self.config.headers},
        )
        with urlopen(request, timeout=self.config.timeout) as response:
            return json.load(response)

    async def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._post, row)

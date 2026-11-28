import json
from typing import Any

from pydantic import BaseModel, ConfigDict


class JsonRunnerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    input_fields: list[str] = ["user_input"]

    def encode_input(self, row: dict[str, Any]) -> bytes:
        """Encode only the dataset columns selected for the application."""
        payload = {name: row[name] for name in self.input_fields}
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

"""Static plugin metadata; loading a manifest does not load implementations."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict


class EmptyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


@dataclass(frozen=True)
class PluginSpec:
    id: str
    kind: str
    api_version: int
    factory: str | Callable[..., Any]
    required_distributions: tuple[str, ...] = ()
    config_type: type[BaseModel] = EmptyConfig

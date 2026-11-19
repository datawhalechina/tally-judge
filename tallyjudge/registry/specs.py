"""Static plugin metadata; loading a manifest does not load implementations."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

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
    config_type: type[BaseModel] | str = EmptyConfig
    dependencies: Mapping[
        str, Literal["language_model", "embedding_model", "client"]
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "dependencies", MappingProxyType(dict(self.dependencies))
        )

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
    """A keyword factory and its schema. A factory that validates its own settings
    can set ``factory_validates_config`` to share that check with direct construction.
    """

    id: str
    kind: str
    api_version: int
    factory: str | Callable[..., Any]
    required_distributions: tuple[str, ...] = ()
    config_type: type[BaseModel] | str = EmptyConfig
    dependencies: Mapping[
        str, Literal["language_model", "embedding_model", "client"]
    ] = field(default_factory=dict)
    install_extra: str | None = None
    factory_validates_config: bool = False
    optional_dependencies: Mapping[
        str, Literal["language_model", "embedding_model", "client"]
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.optional_dependencies.keys() & self.dependencies.keys():
            raise ValueError(
                f"Dependency declared as both required and optional: {self.id}"
            )
        object.__setattr__(
            self,
            "optional_dependencies",
            MappingProxyType(dict(self.optional_dependencies)),
        )
        object.__setattr__(
            self, "dependencies", MappingProxyType(dict(self.dependencies))
        )

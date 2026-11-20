"""Explicit registration and lazy, validated plugin construction."""

import importlib
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, distribution
from typing import Any

from pydantic import BaseModel, ValidationError

from tallyjudge.core.errors import PluginError
from tallyjudge.interfaces.cache import ScoreCache
from tallyjudge.interfaces.events import EventSink
from tallyjudge.interfaces.metrics import MetricSpec
from tallyjudge.interfaces.optimization import PromptOptimizer
from tallyjudge.interfaces.storage import StorageBackend
from tallyjudge.interfaces.synthesis import Synthesizer
from tallyjudge.registry.specs import PluginSpec

_PLUGIN_CONTRACTS = {
    "synthesis": Synthesizer,
    "optimization": PromptOptimizer,
    "storage": StorageBackend,
    "cache": ScoreCache,
    "event": EventSink,
}


def _load(symbol: Any, plugin_id: str) -> Any:
    if not isinstance(symbol, str):
        return symbol
    try:
        module, name = symbol.split(":", 1)
        obj = getattr(importlib.import_module(module), name)
        if not callable(obj):
            raise TypeError
        return obj
    except Exception as exc:
        raise PluginError(
            f"Cannot load the configured plugin factory or schema: {plugin_id}"
        ) from exc


def _matches_model(value: Any, kind: str) -> bool:
    method = {"language_model": "agenerate", "embedding_model": "aembed_texts"}.get(
        kind
    )
    if method is None:
        return False
    plugin_id = getattr(value, "plugin_id", None)
    return (
        isinstance(plugin_id, str)
        and bool(plugin_id)
        and getattr(value, "usage_tracking", None) in ("per_request", "untracked")
        and callable(getattr(value, "config_snapshot", None))
        and callable(getattr(value, method, None))
    )


def _config_schema(spec: PluginSpec) -> type[BaseModel]:
    schema = _load(spec.config_type, spec.id)
    if not isinstance(schema, type) or not issubclass(schema, BaseModel):
        raise PluginError(f"Configuration schema must be a Pydantic model: {spec.id}")
    return schema


def _invalid_config(spec: PluginSpec, exc: ValidationError) -> PluginError:
    fields = ", ".join(
        str(e["loc"][0]) for e in exc.errors(include_input=False) if e["loc"]
    )
    detail = f" ({fields})" if fields else ""
    return PluginError(f"Invalid plugin configuration: {spec.id}{detail}")


def _settings(
    spec: PluginSpec,
    schema: type[BaseModel],
    raw: dict[str, Any],
    *,
    for_factory: bool = False,
) -> dict[str, Any]:
    if raw.keys() - schema.model_fields.keys():
        raise PluginError(f"Unknown plugin configuration fields: {spec.id}")
    if for_factory and spec.factory_validates_config:
        return raw
    try:
        validated = schema.model_validate(raw, strict=True)
    except ValidationError as exc:
        raise _invalid_config(spec, exc) from exc
    return {name: getattr(validated, name) for name in schema.model_fields}


class PluginRegistry:
    """Map stable plugin IDs to lazily loaded factories and their configuration."""

    def __init__(self, specs: Iterable[PluginSpec] = ()) -> None:
        self._specs: dict[str, PluginSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: PluginSpec, *, replace: bool = False) -> None:
        """Register a manifest, replacing an existing ID only when requested."""
        if not isinstance(spec.id, str) or not spec.id:
            raise PluginError("Plugin ID must be a non-empty string")
        if spec.id in self._specs and not replace:
            raise PluginError(f"Plugin already registered: {spec.id}")
        self._specs[spec.id] = spec

    def list(self, *, kind: str | None = None) -> tuple[PluginSpec, ...]:
        """List manifests without importing plugin implementations."""
        return tuple(
            spec for spec in self._specs.values() if kind is None or spec.kind == kind
        )

    def _get(self, plugin_id: str, kind: str | None = None) -> PluginSpec:
        if plugin_id not in self._specs:
            raise PluginError(f"Unknown plugin: {plugin_id}")
        spec = self._specs[plugin_id]
        if type(spec.api_version) is not int or spec.api_version != 1:
            raise PluginError(f"Unsupported plugin api_version; expected 1: {spec.id}")
        if (
            spec.kind not in ("metric", "model", "embedding", *_PLUGIN_CONTRACTS)
            or kind is not None
            and spec.kind != kind
        ):
            raise PluginError(
                f"Unexpected plugin kind {spec.kind!r} for {plugin_id} (requested {kind})"
            )
        return spec

    def describe(self, plugin_id: str, **settings: Any) -> MetricSpec:
        """Resolve a metric's scoring contract from settings, without injected clients."""
        spec = self._get(plugin_id, "metric")
        factory = _load(spec.factory, spec.id)
        build_spec = getattr(factory, "spec_from_config", None)
        if build_spec is not None or settings:
            settings = _settings(spec, _config_schema(spec), settings)
        if build_spec is None:
            metric_spec = getattr(factory, "spec", None)
        else:
            try:
                metric_spec = build_spec(**settings)
            except ValueError as exc:
                raise PluginError(f"Invalid plugin configuration: {spec.id}") from exc
        if not isinstance(metric_spec, MetricSpec) or metric_spec.id != spec.id:
            raise PluginError(
                f"Factory must expose its matching MetricSpec for describe: {spec.id}"
            )
        return metric_spec

    def create(self, plugin_id: str, **config: Any) -> Any:
        """Construct a registered plugin selected by ID."""
        return self._construct(self._get(plugin_id), config)

    def create_for_kind(self, kind: str, plugin_id: str, /, **config: Any) -> Any:
        """Construct a plugin after verifying the kind requested by its public factory."""
        return self._construct(self._get(plugin_id, kind), config)

    def _construct(self, spec: PluginSpec, config: dict[str, Any]) -> Any:
        schema = _config_schema(spec)
        declared = {**spec.dependencies, **spec.optional_dependencies}
        if declared.keys() & schema.model_fields.keys():
            raise PluginError(f"Configuration and dependency names overlap: {spec.id}")
        dependencies = {}
        for name, kind in declared.items():
            dependency = config.pop(name, None)
            if dependency is None:
                if name in spec.dependencies:
                    raise PluginError(f"Missing dependency {name!r} for {spec.id}")
                continue
            if kind != "client" and not _matches_model(dependency, kind):
                raise PluginError(f"Invalid {kind} dependency {name!r} for {spec.id}")
            dependencies[name] = dependency
        options = _settings(spec, schema, config, for_factory=True)
        for name in spec.required_distributions:
            try:
                distribution(name)
            except PackageNotFoundError as exc:
                hint = (
                    f"; install tallyjudge[{spec.install_extra}]"
                    if spec.install_extra
                    else ""
                )
                raise PluginError(
                    f"Missing optional distribution {name} for plugin {spec.id}{hint}"
                ) from exc
        factory = _load(spec.factory, spec.id)
        try:
            plugin = factory(**dependencies, **options)
        except ValidationError as exc:
            raise _invalid_config(spec, exc) from exc
        except Exception as exc:
            raise PluginError(f"Plugin construction failed: {spec.id}") from exc
        if spec.kind == "metric":
            if (
                not isinstance(getattr(plugin, "spec", None), MetricSpec)
                or plugin.spec.id != spec.id
            ):
                raise PluginError(f"PluginSpec.id must match MetricSpec.id: {spec.id}")
            if not callable(getattr(plugin, "ascore_inputs", None)) or not callable(
                getattr(plugin, "config_snapshot", None)
            ):
                raise PluginError(
                    f"Factory result does not satisfy the Metric contract: {spec.id}"
                )
        elif spec.kind in _PLUGIN_CONTRACTS:
            if not isinstance(plugin, _PLUGIN_CONTRACTS[spec.kind]):
                raise PluginError(
                    f"Factory result does not match the {spec.kind} contract: {spec.id}"
                )
            if spec.kind == "synthesis" and plugin.plugin_id != spec.id:
                raise PluginError(
                    f"PluginSpec.id must match the synthesizer ID: {spec.id}"
                )
        elif (
            not _matches_model(
                plugin, "language_model" if spec.kind == "model" else "embedding_model"
            )
            or plugin.plugin_id != spec.id
        ):
            raise PluginError(
                f"Factory result does not match the model contract: {spec.id}"
            )
        return plugin

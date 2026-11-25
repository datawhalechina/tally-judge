import importlib
import inspect
import logging
import os
from collections.abc import Callable, Iterable
from importlib.metadata import PackageNotFoundError, distribution, entry_points
from typing import Any

from pydantic import BaseModel, ValidationError

from tallyjudge.sdk.analysis import Analyzer
from tallyjudge.sdk.cache import ScoreCache
from tallyjudge.sdk.config import EmptyConfig
from tallyjudge.sdk.errors import PluginError
from tallyjudge.sdk.events import EventSink
from tallyjudge.sdk.gate import gated
from tallyjudge.sdk.importing import PRODUCTS, Importer
from tallyjudge.sdk.metrics import MetricSpec
from tallyjudge.sdk.plugin import KindSpec, PluginSpec
from tallyjudge.sdk.running import Runner
from tallyjudge.sdk.storage import StorageBackend


def _load(symbol: Any, owner: str, what: str = "plugin factory") -> Any:
    if not isinstance(symbol, str):
        return symbol
    try:
        module, name = symbol.split(":", 1)
        obj = getattr(importlib.import_module(module), name)
        if not callable(obj):
            raise TypeError
        return obj
    except Exception as exc:
        raise PluginError(f"Cannot load the configured {what}: {owner}") from exc


logger = logging.getLogger(__name__)

_MODEL_METHODS = {"language_model": "agenerate", "embedding_model": "aembed_texts"}


def _is_model(value: Any, method: str) -> bool:
    plugin_id = getattr(value, "plugin_id", None)
    return (
        isinstance(plugin_id, str)
        and bool(plugin_id)
        and getattr(value, "usage_tracking", None) in ("per_request", "untracked")
        and callable(getattr(value, "config_snapshot", None))
        and callable(getattr(value, method, None))
    )


def _is_metric(plugin: Any, plugin_id: str) -> bool:
    spec = getattr(plugin, "spec", None)
    return (
        isinstance(spec, MetricSpec)
        and spec.id == plugin_id
        and callable(getattr(plugin, "ascore_inputs", None))
        and callable(getattr(plugin, "config_snapshot", None))
    )


def _model(method: str) -> Callable[[Any, str], bool]:
    return lambda plugin, plugin_id: (
        _is_model(plugin, method) and plugin.plugin_id == plugin_id
    )


def _implements(contract: type) -> Callable[[Any, str], bool]:
    return lambda plugin, plugin_id: (
        isinstance(plugin, contract) and plugin.plugin_id == plugin_id
    )


KINDS: dict[str, Callable[[Any, str], bool]] = {
    "metric": _is_metric,
    "model": _model("agenerate"),
    "embedding": _model("aembed_texts"),
    "storage": _implements(StorageBackend),
    "cache": _implements(ScoreCache),
    "event": _implements(EventSink),
    "importer": _implements(Importer),
    "runner": _implements(Runner),
    "analyzer": _implements(Analyzer),
}


def _config_schema(factory: Any, spec: PluginSpec) -> type[BaseModel]:
    schema = getattr(factory, "Config", EmptyConfig)
    if not isinstance(schema, type) or not issubclass(schema, BaseModel):
        raise PluginError(f"Configuration schema must be a Pydantic model: {spec.id}")
    return schema


def _invalid_config(spec: PluginSpec, exc: ValidationError) -> PluginError:
    fields = ", ".join(
        str(e["loc"][0]) for e in exc.errors(include_input=False) if e["loc"]
    )
    detail = f" ({fields})" if fields else ""
    return PluginError(f"Invalid plugin configuration: {spec.id}{detail}")


def _reject_unknown(
    spec: PluginSpec, schema: type[BaseModel], raw: dict[str, Any]
) -> None:
    if raw.keys() - schema.model_fields.keys():
        raise PluginError(f"Unknown plugin configuration fields: {spec.id}")


def _validated(
    spec: PluginSpec, schema: type[BaseModel], raw: dict[str, Any]
) -> dict[str, Any]:
    _reject_unknown(spec, schema, raw)
    try:
        validated = schema.model_validate(raw, strict=True)
    except ValidationError as exc:
        raise _invalid_config(spec, exc) from exc
    return {name: getattr(validated, name) for name in schema.model_fields}


class PluginRegistry:
    """Map stable plugin IDs to lazily loaded factories and their configuration."""

    def __init__(
        self,
        specs: Iterable[PluginSpec | KindSpec] = (),
        *,
        entry_point_group: str | None = None,
    ) -> None:
        self._specs: dict[str, PluginSpec] = {}
        self._kinds: dict[str, KindSpec] = {}
        self._entry_point_group = entry_point_group
        for spec in specs:
            self.register(spec)

    def _discover(self) -> None:
        group, self._entry_point_group = self._entry_point_group, None
        if group is None:
            return
        allowed = os.environ.get("TALLYJUDGE_PLUGINS")
        for entry in entry_points(group=group):
            if allowed is not None and entry.name not in allowed.split(","):
                continue
            try:
                specs = tuple(entry.load())
            except Exception:
                logger.warning(
                    "Cannot read the plugin manifest %s", entry.value, exc_info=True
                )
                continue
            for spec in specs:
                try:
                    self.register(spec)
                except Exception:
                    logger.warning(
                        "Skipping a plugin declared in %s", entry.value, exc_info=True
                    )

    def register(self, spec: PluginSpec | KindSpec, *, replace: bool = False) -> None:
        """Register a manifest, replacing an existing ID only when requested."""
        if isinstance(spec, KindSpec):
            if spec.kind in KINDS or spec.kind in self._kinds:
                raise PluginError(f"Plugin kind already defined: {spec.kind}")
            self._kinds[spec.kind] = spec
            return
        if not isinstance(spec.id, str) or not spec.id:
            raise PluginError("Plugin ID must be a non-empty string")
        if spec.id in self._specs and not replace:
            raise PluginError(f"Plugin already registered: {spec.id}")
        self._specs[spec.id] = spec

    def kinds(self) -> tuple[str, ...]:
        """Name the built-in kinds followed by the kinds packages have defined."""
        self._discover()
        return (*KINDS, *self._kinds)

    def list(self, *, kind: str | None = None) -> tuple[PluginSpec, ...]:
        """List manifests without importing plugin implementations."""
        self._discover()
        return tuple(
            spec for spec in self._specs.values() if kind is None or spec.kind == kind
        )

    def _get(self, plugin_id: str, kind: str | None = None) -> PluginSpec:
        self._discover()
        if plugin_id not in self._specs:
            raise PluginError(f"Unknown plugin: {plugin_id}")
        spec = self._specs[plugin_id]
        if spec.api_version != 1:
            raise PluginError(f"Unsupported plugin api_version; expected 1: {spec.id}")
        if spec.kind not in self.kinds() or kind is not None and spec.kind != kind:
            raise PluginError(
                f"Unexpected plugin kind {spec.kind!r} for {plugin_id} (requested {kind})"
            )
        if spec.kind == "importer" and spec.produces not in PRODUCTS:
            raise PluginError(
                f"An importer must declare produces as one of {PRODUCTS}: {spec.id}"
            )
        return spec

    def spec(self, plugin_id: str) -> PluginSpec:
        """Return a plugin's manifest row without importing its implementation."""
        return self._get(plugin_id)

    def describe(self, plugin_id: str, **settings: Any) -> MetricSpec:
        """Resolve a metric's scoring contract from settings, without injected clients."""
        spec = self._get(plugin_id, "metric")
        factory = _load(spec.factory, spec.id)
        build_spec = getattr(factory, "spec_from_config", None)
        if build_spec is not None or settings:
            settings = _validated(spec, _config_schema(factory, spec), settings)
        if build_spec is None:
            metric_spec = getattr(factory, "spec", None)
        else:
            try:
                accepted = inspect.signature(build_spec).parameters.values()
                if not any(p.kind is p.VAR_KEYWORD for p in accepted):
                    names = {p.name for p in accepted}
                    settings = {k: v for k, v in settings.items() if k in names}
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
        schema = _config_schema(factory, spec)
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
            if kind != "client" and not _is_model(dependency, _MODEL_METHODS[kind]):
                raise PluginError(f"Invalid {kind} dependency {name!r} for {spec.id}")
            dependencies[name] = dependency if kind == "client" else gated(dependency)
        _reject_unknown(spec, schema, config)
        try:
            plugin = factory(**dependencies, **config)
        except ValidationError as exc:
            raise _invalid_config(spec, exc) from exc
        except Exception as exc:
            raise PluginError(f"Plugin construction failed: {spec.id}") from exc
        conforms = KINDS.get(spec.kind) or _implements(
            _load(self._kinds[spec.kind].contract, spec.kind, "kind contract")
        )
        if not conforms(plugin, spec.id):
            raise PluginError(
                f"Factory result must match the {spec.kind} contract and plugin ID: {spec.id}"
            )
        return plugin

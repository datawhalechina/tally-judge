"""Explicit registration and lazy, validated metric construction."""

import importlib
from collections.abc import Callable, Iterable
from importlib.metadata import PackageNotFoundError, distribution
from typing import Any, cast

from pydantic import ValidationError

from tallyjudge.core.errors import PluginError
from tallyjudge.interfaces.metrics import Metric, MetricSpec
from tallyjudge.registry.specs import PluginSpec


class PluginRegistry:
    def __init__(self, specs: Iterable[PluginSpec] = ()) -> None:
        self._specs: dict[str, PluginSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: PluginSpec, *, replace: bool = False) -> None:
        if not isinstance(spec.id, str) or not spec.id:
            raise PluginError("Plugin ID must be a non-empty string")
        if spec.id in self._specs and not replace:
            raise PluginError(f"Plugin already registered: {spec.id}")
        self._specs[spec.id] = spec

    def list(self, *, kind: str | None = None) -> tuple[PluginSpec, ...]:
        return tuple(
            spec for spec in self._specs.values() if kind is None or spec.kind == kind
        )

    def _get(self, plugin_id: str) -> PluginSpec:
        if plugin_id not in self._specs:
            raise PluginError(f"Unknown plugin: {plugin_id}")
        spec = self._specs[plugin_id]
        if type(spec.api_version) is not int or spec.api_version != 1:
            raise PluginError(f"Unsupported plugin api_version; expected 1: {spec.id}")
        if spec.kind != "metric":
            raise PluginError(f"Phase 1 supports metric plugins only: {spec.id}")
        return spec

    def _load(self, spec: PluginSpec) -> Callable[..., Any]:
        if callable(spec.factory):
            return spec.factory
        try:
            module, name = spec.factory.split(":", 1)
            factory = getattr(importlib.import_module(module), name)
            if not callable(factory):
                raise TypeError
            return factory
        except Exception as exc:
            raise PluginError(
                f"Cannot load the configured plugin factory: {spec.id}"
            ) from exc

    def describe(self, plugin_id: str) -> MetricSpec:
        spec = self._get(plugin_id)
        metric_spec = getattr(self._load(spec), "spec", None)
        if not isinstance(metric_spec, MetricSpec) or metric_spec.id != spec.id:
            raise PluginError(
                f"Factory must expose its matching MetricSpec for describe: {spec.id}"
            )
        return metric_spec

    def create(self, plugin_id: str, **config: Any) -> Metric:
        spec = self._get(plugin_id)
        # Unknown keys are rejected even if a custom schema silently ignores extras.
        if set(config) - set(spec.config_type.model_fields):
            raise PluginError(f"Unknown plugin configuration fields: {spec.id}")
        try:
            validated = spec.config_type.model_validate(config, strict=True)
        except ValidationError as exc:
            raise PluginError(f"Invalid plugin configuration: {spec.id}") from exc
        for name in spec.required_distributions:
            try:
                distribution(name)
            except PackageNotFoundError as exc:
                raise PluginError(
                    f"Missing optional distribution {name} for plugin {spec.id}"
                ) from exc
        factory = self._load(spec)
        try:
            metric = factory(
                **{
                    name: getattr(validated, name)
                    for name in type(validated).model_fields
                }
            )
        except Exception as exc:
            raise PluginError(f"Plugin construction failed: {spec.id}") from exc
        if (
            not isinstance(getattr(metric, "spec", None), MetricSpec)
            or metric.spec.id != spec.id
        ):
            raise PluginError(f"PluginSpec.id must match MetricSpec.id: {spec.id}")
        if not callable(getattr(metric, "ascore_inputs", None)) or not callable(
            getattr(metric, "config_snapshot", None)
        ):
            raise PluginError(
                f"Factory result does not satisfy the Metric contract: {spec.id}"
            )
        return cast(Metric, metric)

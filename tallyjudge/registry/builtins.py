"""Explicit built-ins; MetricSpec lives only on each metric class."""

from tallyjudge.registry.specs import PluginSpec

BUILTINS = (
    PluginSpec(
        "metric.text.exact_match",
        "metric",
        1,
        "tallyjudge.plugins.metrics.text.exact_match:ExactMatchMetric",
    ),
    PluginSpec(
        "metric.text.string_presence",
        "metric",
        1,
        "tallyjudge.plugins.metrics.text.string_presence:StringPresenceMetric",
    ),
)

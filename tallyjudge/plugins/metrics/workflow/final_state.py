from typing import Any

from tallyjudge.sdk.errors import UndefinedScore
from tallyjudge.sdk.metrics import MetricBase, MetricSpec
from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.results import MetricResult

_ABSENT = object()


def _leaves(
    value: Any, path: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict) and value:
        return [
            leaf
            for key, child in value.items()
            for leaf in _leaves(child, (*path, key))
        ]
    return [(path, value)]


def _at(state: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        if not isinstance(state, dict) or key not in state:
            return _ABSENT
        state = state[key]
    return state


class FinalStateMetric(MetricBase):
    plugin = Plugin("metric")
    spec = MetricSpec(
        "metric.workflow.final_state",
        "1",
        {"final_state": dict[str, Any], "expected_state": dict[str, Any]},
        value_range=(0.0, 1.0),
    )

    async def ascore(
        self, final_state: dict[str, Any], expected_state: dict[str, Any]
    ) -> MetricResult:
        expected = _leaves(expected_state)
        if not expected_state:
            raise UndefinedScore("no_expected_state")
        mismatched = [
            ".".join(path)
            for path, value in expected
            if _at(final_state, path) != value
        ]
        return MetricResult(
            (len(expected) - len(mismatched)) / len(expected),
            details={"mismatched": mismatched},
        )

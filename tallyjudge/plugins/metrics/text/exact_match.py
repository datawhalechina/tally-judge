"""Exact equality, adapted from Ragas collections/_string.py (ec00339).

Original algorithm distributed under Apache-2.0; see repository LICENSE.
Revision 1 preserves case and whitespace and strictly requires strings.
"""

from tallyjudge.sdk.metrics import MetricBase, MetricSpec
from tallyjudge.sdk.plugin import Plugin
from tallyjudge.sdk.results import MetricResult


class ExactMatchMetric(MetricBase):
    plugin = Plugin("metric")

    spec = MetricSpec(
        "metric.text.exact_match",
        "1",
        {"reference": str, "response": str},
        value_range=(0.0, 1.0),
    )

    async def ascore(self, reference: str, response: str) -> MetricResult:
        return MetricResult(float(reference == response))

"""Substring membership, adapted from Ragas collections/_string.py (ec00339).

Original algorithm distributed under Apache-2.0; see repository LICENSE.
Revision 1 is case-sensitive. An empty reference always scores 1.0.
"""

from tallyjudge.core.results import MetricResult
from tallyjudge.interfaces.metrics import MetricBase, MetricSpec


class StringPresenceMetric(MetricBase):
    spec = MetricSpec(
        "metric.text.string_presence",
        "1",
        {"reference": str, "response": str},
        value_range=(0.0, 1.0),
    )

    async def ascore(self, reference: str, response: str) -> MetricResult:
        return MetricResult(float(reference in response))

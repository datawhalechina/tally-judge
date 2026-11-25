import asyncio
import json
import logging

import pytest

from tallyjudge import (
    Dataset,
    Evaluator,
    ExecutionConfig,
    MetricBinding,
    MetricResult,
    RunCollector,
    create_metric,
)
from tallyjudge.sdk.metrics import MetricBase, MetricSpec


@pytest.mark.asyncio
async def test_live_snapshot_and_external_cancel_distinguish_unstarted_rows():
    entered, release = asyncio.Event(), asyncio.Event()

    class WaitingMetric(MetricBase):
        spec = MetricSpec("metric.test.wait", "1", {})

        async def ascore(self):
            entered.set()
            await release.wait()
            return MetricResult(1.0)

    collector = RunCollector()
    task = asyncio.create_task(
        Evaluator([WaitingMetric()], config=ExecutionConfig(max_concurrency=1)).arun(
            Dataset.from_records([{}, {}]), collector=collector
        )
    )
    await entered.wait()
    live = collector.snapshot()
    assert live.run_status == "running"
    assert live.stop_reason is None
    assert [row.metrics["metric.test.wait"].status for row in live.rows] == [
        "running",
        "not_started",
    ]
    assert live.summary["metric.test.wait"]["status_counts"]["not_started"] == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    final = collector.snapshot()
    assert (final.run_status, final.stop_reason) == ("cancelled", "external_cancel")
    assert [row.metrics["metric.test.wait"].status for row in final.rows] == [
        "cancelled",
        "not_started",
    ]
    assert live.run_status == "running"
    for include_inputs in (True, False):
        assert json.loads(
            final.to_json(include_inputs=include_inputs)
        ) == final.to_dict(include_inputs=include_inputs)


def test_validation_is_one_warning_per_row_and_result_has_safe_fields(caplog):
    metric = create_metric("metric.text.exact_match")
    with caplog.at_level(logging.WARNING, logger="tallyjudge.engine"):
        result = Evaluator(
            [MetricBinding(metric, key="mapped", input_map={"response": "answer"})]
        ).run(
            Dataset.from_records(
                [
                    {"answer": 987654321, "reference": "private_reference"}
                    for _ in range(50)
                ]
            )
        )
    assert (result.run_status, result.stop_reason) == ("completed", None)
    assert len(caplog.records) == 50
    assert all(
        record.levelno == logging.WARNING and record.exc_info is None
        for record in caplog.records
    )
    assert "987654321" not in caplog.text
    assert "private_reference" not in caplog.text
    assert "mapped" in caplog.text
    error = result.rows[0].metrics["mapped"].error
    assert (error.category, error.field, error.expected, error.exception_type) == (
        "validation_error",
        "response",
        "str",
        None,
    )
    assert "987654321" not in result.to_json(include_inputs=False)
    assert result.summary["mapped"]["coverage"] == 0.0


def test_missing_input_reports_expected_parameter():
    metric = create_metric("metric.text.exact_match")
    result = Evaluator([metric]).run(Dataset.from_records([{"reference": "a"}]))
    error = result.rows[0].metrics[metric.spec.id].error
    assert error.field == "response"
    assert error.expected == "str"
    assert error.exception_type is None


def test_empty_dataset_is_completed_with_undefined_coverage():
    metric = create_metric("metric.text.exact_match")
    result = Evaluator([metric]).run(Dataset.from_records([]))
    assert (result.run_status, result.stop_reason) == ("completed", None)
    assert result.rows == []
    assert result.summary[metric.spec.id]["coverage"] is None
    assert result.summary[metric.spec.id]["mean"] is None


@pytest.mark.asyncio
async def test_generic_validation_reports_spec_type_and_debug_traceback(caplog):
    class ContextMetric(MetricBase):
        spec = MetricSpec("metric.test.contexts", "1", {"contexts": list[str]})

        async def ascore(self, contexts):
            return MetricResult(1.0)

    with caplog.at_level(logging.DEBUG, logger="tallyjudge.engine"):
        result = await Evaluator([ContextMetric()]).arun(
            Dataset.from_records([{"contexts": [1]}])
        )
    error = result.rows[0].metrics["metric.test.contexts"].error
    assert (error.field, error.expected) == ("contexts", "list[str]")
    assert sum(record.levelno == logging.WARNING for record in caplog.records) == 1
    assert any(
        record.levelno == logging.DEBUG and record.exc_info for record in caplog.records
    )

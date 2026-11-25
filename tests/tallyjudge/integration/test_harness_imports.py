import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tallyjudge import Dataset, Evaluator, create_importer, create_metric

DATA = Path(__file__).with_name("data")
INSPECT_LOG = DATA / "inspect_mockllm_log.json"
ATIF = DATA / "atif_trajectory.json"


def test_an_inspect_log_becomes_records_that_existing_metrics_score():
    [record] = create_importer("importer.inspect.log", include_content=True).convert(
        INSPECT_LOG
    )
    assert {key: value for key, value in record.items() if key != "trace"} == {
        "case_id": "refund-1",
        "user_input": "Can order 1042 be refunded?",
        "reference": "refundable",
        "response": "Order 1042 was delivered and is refundable.",
        "final_state": {"looked_up": "1042"},
        "inspect_scores": {"includes": "C"},
    }
    trace = record["trace"]
    assert trace["output"] == record["response"]
    steps = trace["steps"]
    assert [(step["kind"], step["name"], step["status"]) for step in steps] == [
        ("model", "mockllm/model", "success"),
        ("tool", "lookup_order", "success"),
        ("model", "mockllm/model", "success"),
        ("tool", "lookup_order", "error"),
        ("model", "mockllm/model", "success"),
    ]
    assert (steps[0]["input_tokens"], steps[0]["output_tokens"]) == (85, 31)
    assert steps[1]["arguments"] == {"order_id": "1042"}
    assert json.loads(steps[1]["result"]) == {"status": "delivered", "refundable": True}
    assert steps[1]["started_at"] < steps[1]["ended_at"] < steps[2]["started_at"]
    assert steps[3]["attributes"] == {"inspect.error_type": "file_not_found"}

    metrics = [
        create_metric("metric.workflow." + name)
        for name in ("tool_recall", "step_efficiency", "forbidden_tools", "final_state")
    ]
    row = {
        **record,
        "expected_tools": ["lookup_order"],
        "forbidden_tools": ["issue_refund"],
        "expected_state": {"looked_up": "1042"},
    }
    scores = Evaluator(metrics).run(Dataset.from_records([row])).rows[0].metrics
    assert [scores[metric.spec.id].value for metric in metrics] == [
        1.0,
        0.5,
        "compliant",
        1.0,
    ]


def test_an_inspect_log_is_imported_without_content_by_default():
    document = json.loads(INSPECT_LOG.read_text(encoding="utf-8"))
    [record] = create_importer("importer.inspect.log").convert(document)
    trace = record["trace"]
    assert trace["output"] is None
    assert all(
        step["arguments"] is None and step["result"] is None for step in trace["steps"]
    )


def test_an_atif_trajectory_becomes_a_trace_of_this_run_only():
    trace = create_importer("importer.atif.trajectory", include_content=True).convert(
        ATIF
    )
    assert (trace.run_id, trace.output) == ("session-123", "hello.txt contains Hello.")
    assert trace.attributes == {"atif.schema_version": "ATIF-v1.8"}
    assert [
        (step.step_id, step.parent_id, step.kind, step.name) for step in trace.steps
    ] == [
        ("2", None, "model", "openai/gpt-5.6-sol"),
        ("2.call-1", "2", "tool", "write_file"),
        ("3.call-2", None, "tool", "read_file"),
        ("5", None, "model", "openai/gpt-5.6-mini"),
    ]
    model, write = trace.steps[:2]
    assert (model.input_tokens, model.output_tokens) == (120, 18)
    assert model.started_at == datetime(2026, 11, 25, 3, tzinfo=UTC).timestamp()
    assert (write.call_id, write.result) == ("call-1", "File created")
    assert write.arguments == {"path": "hello.txt", "content": "Hello"}

    metric = create_metric("metric.workflow.tool_sequence")
    row = {"trace": trace.model_dump(), "expected_tools": ["write_file", "read_file"]}
    run = Evaluator([metric]).run(Dataset.from_records([row]))
    assert run.rows[0].metrics[metric.spec.id].value == 1.0


def test_an_atif_trajectory_is_imported_without_content_by_default():
    document = json.loads(ATIF.read_text(encoding="utf-8"))
    trace = create_importer("importer.atif.trajectory", run_id="trial-7").convert(
        document
    )
    assert (trace.run_id, trace.output) == ("trial-7", None)
    assert all(step.arguments is None and step.result is None for step in trace.steps)
    del document["session_id"]
    with pytest.raises(ValueError, match="pass run_id"):
        create_importer("importer.atif.trajectory").convert(document)

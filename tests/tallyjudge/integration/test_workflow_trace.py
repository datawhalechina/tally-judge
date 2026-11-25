import pytest

from tallyjudge import Dataset, Evaluator, create_importer, create_metric
from tallyjudge.sdk.workflow import WorkflowStep, WorkflowTrace


def langchain_trace(events, *, run_id, include_content=False):
    return create_importer(
        "importer.langchain.trace", run_id=run_id, include_content=include_content
    ).convert(events)


def evt(kind, key, name, parents=(), **data):
    return {
        "event": "on_" + kind,
        "run_id": key,
        "parent_ids": list(parents),
        "name": name,
        "data": data,
    }


@pytest.mark.parametrize(
    "route,expected,score",
    [
        ("ok", ["lookup"], 1.0),
        ("skip", [], 1.0),
        ("error", ["lookup"], 0.0),
        ("retry", ["lookup"], 1.0),
    ],
)
def test_hand_recorded_and_streamed_branch(route, expected, score):
    root = WorkflowStep(step_id="flow", name="route", kind="workflow", status="success")
    events = [evt("chain_start", "flow", "route")]
    attempts = []
    if route != "skip":
        statuses = (
            ["error", "success"]
            if route == "retry"
            else ["error" if route == "error" else "success"]
        )
        for i, status in enumerate(statuses):
            key = str(i)
            events.extend(
                [
                    evt(
                        "tool_start",
                        key,
                        "lookup",
                        ["flow"],
                        input={
                            "type": "tool_call",
                            "id": "c",
                            "args": {"q": "private"},
                        },
                    ),
                    evt(
                        "tool_error" if status == "error" else "tool_end",
                        key,
                        "lookup",
                        ["flow"],
                        output={
                            "type": "tool",
                            "content": "private answer",
                            "tool_call_id": "c",
                        },
                    ),
                ]
            )
            attempts.append(
                WorkflowStep(
                    step_id=key,
                    parent_id="flow",
                    name="lookup",
                    kind="tool",
                    call_id="c",
                    status=status,
                )
            )
    events.append(evt("chain_end", "flow", "route"))
    trace = langchain_trace(events, run_id="flow")
    assert trace == WorkflowTrace(run_id="flow", steps=[root, *attempts])
    assert "private" not in trace.model_dump_json()
    metric = create_metric("metric.workflow.tool_sequence")
    row = {"case_id": "sample", "trace": trace.model_dump(), "expected_tools": expected}
    run = Evaluator([metric]).run(Dataset.from_records([row]))
    assert run.row_for("sample").metrics[metric.spec.id].value == score
    assert '"steps"' not in run.to_json(include_inputs=False)


def test_mismatched_tool_result_and_incomplete_trace():
    e = evt(
        "tool_start",
        "t",
        "lookup",
        input={"type": "tool_call", "id": "c", "args": {"q": "secret"}},
    )
    trace = langchain_trace([e], run_id="run", include_content=True)
    assert trace.steps[0].arguments == {"q": "secret"}
    metric = create_metric("metric.workflow.tool_sequence")
    result = Evaluator([metric]).run(
        Dataset.from_records([{"trace": trace.model_dump(), "expected_tools": []}])
    )
    assert result.rows[0].metrics[metric.spec.id].status == "undefined"
    with pytest.raises(ValueError, match="different call"):
        langchain_trace(
            [
                e,
                evt(
                    "tool_end",
                    "t",
                    "lookup",
                    output={
                        "type": "tool",
                        "content": "answer",
                        "tool_call_id": "different",
                    },
                ),
            ],
            run_id="run",
        )


@pytest.mark.asyncio
async def test_live_langchain_branch_with_local_tool():
    tools = pytest.importorskip("langchain_core.tools")
    runnable = pytest.importorskip("langchain_core.runnables")

    @tools.tool(description="Look up a place.")
    def lookup(query: str) -> str:
        return "London"

    flow = runnable.RunnableBranch(
        (lambda row: row["query"] == "lookup", lookup),
        runnable.RunnableLambda(lambda row: "skipped"),
    )
    for query, expected in (("lookup", ["lookup"]), ("skip", [])):
        events = [e async for e in flow.astream_events({"query": query}, version="v2")]
        trace = langchain_trace(
            events, run_id=events[0]["run_id"], include_content=True
        )
        score = await create_metric("metric.workflow.tool_sequence").ascore_inputs(
            {"trace": trace.model_dump(), "expected_tools": expected}
        )
        assert score.value == 1
        if expected:
            (tool,) = (step for step in trace.steps if step.kind == "tool")
            assert tool.result == "London" and tool.parent_id == events[0]["run_id"]


def test_trace_rejects_unrelated_runs_and_invalid_parent_links():
    with pytest.raises(ValueError, match="different root"):
        langchain_trace(
            [evt("chain_start", "a", "A"), evt("chain_start", "b", "B")], run_id="mix"
        )
    malformed = {
        "run_id": "one",
        "steps": [
            {
                "step_id": "child",
                "parent_id": "missing",
                "status": "success",
                "name": "lookup",
                "kind": "tool",
            }
        ],
    }
    metric = create_metric("metric.workflow.tool_sequence")
    result = Evaluator([metric]).run(
        Dataset.from_records([{"trace": malformed, "expected_tools": ["lookup"]}])
    )
    failure = result.rows[0].metrics[metric.spec.id]
    assert (
        failure.error.category == "validation_error" and failure.error.field == "trace"
    )


@pytest.mark.asyncio
async def test_successful_tool_can_return_a_business_error_status():
    tools = pytest.importorskip("langchain_core.tools")
    payload = {
        "status": "error",
        "tool_call_id": "earlier-job",
        "content": "Historical status",
        "code": 42,
    }

    @tools.tool(description="Inspect a job.")
    def inspect_job(job_id: str) -> dict:
        return payload

    events = [
        e async for e in inspect_job.astream_events({"job_id": "a"}, version="v2")
    ]
    trace = langchain_trace(events, run_id=events[0]["run_id"], include_content=True)
    (step,) = trace.steps
    assert step.status == "success" and step.call_id is None and step.result == payload
    score = await create_metric("metric.workflow.tool_sequence").ascore_inputs(
        {"trace": trace.model_dump(), "expected_tools": ["inspect_job"]}
    )
    assert score.value == 1.0

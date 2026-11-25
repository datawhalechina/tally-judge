import json

import pytest

from tallyjudge import (
    Dataset,
    Evaluator,
    create_importer,
    create_language_model,
    create_metric,
)
from tallyjudge.sdk.workflow import WorkflowTrace


def step(key, name, kind="tool", status="success", **fields):
    return {"step_id": key, "name": name, "kind": kind, "status": status, **fields}


TRACE = {
    "run_id": "refund",
    "output": "订单 1042 已退款。",
    "steps": [
        step("0", "agent", kind="workflow"),
        step(
            "1",
            "lookup_order",
            parent_id="0",
            arguments={"order_id": "1042"},
            result={"refundable": True},
        ),
        step(
            "2",
            "create_refund",
            parent_id="0",
            status="error",
            arguments={"order_id": "1042"},
        ),
        step(
            "3",
            "create_refund",
            parent_id="0",
            arguments={"order_id": "1042"},
            result={"refund_id": "r-7"},
        ),
        step("4", "send_email", parent_id="0", arguments={"to": "user"}, result="sent"),
    ],
}
ROW = {
    "user_input": "给订单 1042 退款",
    "trace": TRACE,
    "expected_tools": ["lookup_order", "create_refund"],
    "expected_calls": [
        {"name": "create_refund", "arguments": {"order_id": "1042"}},
        {"name": "create_refund", "arguments": {"order_id": "9999"}},
    ],
    "forbidden_tools": ["delete_order", "send_email"],
}
DETERMINISTIC = {
    "metric.workflow.tool_recall": (1.0, {"missing": []}),
    "metric.workflow.tool_precision": (2 / 3, {"unexpected": ["send_email"]}),
    "metric.workflow.tool_arguments": (0.5, {"unmatched": ["create_refund"]}),
    "metric.workflow.step_efficiency": (0.5, {"attempts": 4}),
    "metric.workflow.forbidden_tools": ("violated", {"attempted": ["send_email"]}),
    "metric.workflow.tool_sequence": (0.0, None),
}


def test_trace_metrics_score_a_recorded_run_and_say_why():
    metrics = [create_metric(key) for key in DETERMINISTIC]
    scores = Evaluator(metrics).run(Dataset.from_records([ROW])).rows[0].metrics
    assert {
        key: (score.value, score.details) for key, score in scores.items()
    } == DETERMINISTIC


def test_unfinished_or_unrecorded_traces_are_undefined_not_guessed():
    open_trace = dict(
        TRACE, steps=[*TRACE["steps"], step("5", "x", status="incomplete")]
    )
    bare = dict(TRACE, steps=[step("1", "create_refund")])
    metrics = [
        create_metric("metric.workflow.tool_recall"),
        create_metric("metric.workflow.tool_arguments"),
    ]
    result = Evaluator(metrics).run(
        Dataset.from_records([dict(ROW, trace=open_trace), dict(ROW, trace=bare)])
    )
    codes = [
        [score.error.code if score.error else None for score in row.metrics.values()]
        for row in result.rows
    ]
    assert codes == [
        ["incomplete_trace", "incomplete_trace"],
        [None, "arguments_not_recorded"],
    ]


@pytest.mark.asyncio
async def test_goal_completion_judges_tool_results_not_the_final_claim(client):
    client.chat_reply(
        json.dumps({"reason": "create_refund 返回了退款号", "verdict": 1})
    )
    judge = create_language_model(
        "model.openai_compatible.chat_completions", client=client, model="judge"
    )
    metric = create_metric("metric.workflow.goal_completion", llm=judge)
    claimed_only = dict(TRACE, steps=[step("0", "agent", kind="workflow")])
    result = await Evaluator([metric]).arun(
        Dataset.from_records([ROW, dict(ROW, trace=claimed_only)])
    )
    done, unverifiable = (row.metrics[metric.spec.id] for row in result.rows)
    assert (done.value, done.reason) == (1.0, "create_refund 返回了退款号")
    assert (unverifiable.status, unverifiable.error.code) == (
        "undefined",
        "results_not_recorded",
    )
    prompt = client.calls[0][1]["messages"][0]["content"]
    assert '\\"refund_id\\": \\"r-7\\"' in prompt and "订单 1042 已退款。" in prompt
    assert len(client.calls) == 1


def attribute(key, value):
    kind = "intValue" if isinstance(value, int) else "stringValue"
    return {"key": key, "value": {kind: str(value) if kind == "intValue" else value}}


def span(key, name, kind, start, parent=None, error=False, **attributes):
    return {
        "traceId": "t1",
        "spanId": key,
        **({"parentSpanId": parent} if parent else {}),
        "name": name,
        "startTimeUnixNano": str(start * 10**9),
        "endTimeUnixNano": str((start + 1) * 10**9),
        "status": {"code": 2} if error else {},
        "attributes": [
            attribute("openinference.span.kind", kind),
            *(attribute(k.replace("__", "."), v) for k, v in attributes.items()),
        ],
    }


def test_openinference_export_becomes_a_parent_first_trace():
    export = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            span(
                                "b",
                                "ToolRun",
                                "TOOL",
                                2,
                                parent="a",
                                tool__name="create_refund",
                                input__value='{"order_id": "1042"}',
                                output__value='{"refund_id": "r-7"}',
                            ),
                            span(
                                "c",
                                "ChatModel",
                                "LLM",
                                3,
                                parent="a",
                                error=True,
                                llm__token_count__prompt=12,
                                llm__token_count__completion=3,
                            ),
                            span(
                                "a",
                                "support_agent",
                                "AGENT",
                                1,
                                agent__name="support",
                                output__value="已退款",
                            ),
                        ]
                    }
                ]
            }
        ]
    }
    importer = create_importer("importer.otlp.openinference", include_content=True)
    trace = importer.convert(export)
    assert (trace.run_id, trace.output) == ("t1", "已退款")
    assert [(s.step_id, s.parent_id, s.kind, s.status) for s in trace.steps] == [
        ("a", None, "workflow", "success"),
        ("b", "a", "tool", "success"),
        ("c", "a", "model", "error"),
    ]
    agent, tool, model = trace.steps
    assert (agent.agent_id, agent.started_at, agent.ended_at) == ("support", 1.0, 2.0)
    assert (tool.name, tool.arguments) == ("create_refund", {"order_id": "1042"})
    assert (model.input_tokens, model.output_tokens) == (12, 3)
    assert tool.attributes == {"openinference.span.kind": "TOOL"}
    redacted = create_importer("importer.otlp.openinference").convert(export)
    assert redacted.output is None and redacted.steps[1].arguments is None


def test_jsonl_steps_become_a_trace_named_after_the_file(tmp_path):
    path = tmp_path / "refund-run.jsonl"
    path.write_text(
        "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in TRACE["steps"]),
        encoding="utf-8",
    )
    trace = create_importer("importer.jsonl.trace").convert(path)
    assert trace.run_id == "refund-run" and len(trace.steps) == 5


def test_trace_attributes_must_be_namespaced():
    kept = WorkflowTrace.model_validate(dict(TRACE, attributes={"acme.session": "s1"}))
    assert kept.attributes == {"acme.session": "s1"}
    with pytest.raises(ValueError):
        WorkflowTrace.model_validate(dict(TRACE, attributes={"session": "s1"}))


def test_final_state_checks_the_world_the_application_left_behind():
    metric = create_metric("metric.workflow.final_state")
    rows = [
        {
            "response": "订单 1042 已退款。",
            "final_state": {"orders": {"1042": {"refund": "completed", "amount": 30}}},
            "expected_state": {
                "orders": {"1042": {"refund": "completed", "amount": 30}}
            },
        },
        {
            "response": "订单 1042 已退款。",
            "final_state": {"orders": {"1042": {"refund": "none", "amount": 30}}},
            "expected_state": {
                "orders": {"1042": {"refund": "completed", "amount": 30}}
            },
        },
    ]
    scores = [
        row.metrics[metric.spec.id]
        for row in Evaluator([metric]).run(Dataset.from_records(rows)).rows
    ]
    assert [(s.value, s.details) for s in scores] == [
        (1.0, {"mismatched": []}),
        (0.5, {"mismatched": ["orders.1042.refund"]}),
    ]


@pytest.mark.asyncio
async def test_a_real_langgraph_run_is_imported_and_scored():
    pytest.importorskip("langgraph")
    from typing import TypedDict

    from langchain_core.tools import tool
    from langgraph.graph import END, START, StateGraph

    @tool(description="Look up an order.")
    def lookup_order(order_id: str) -> str:
        return json.dumps({"order_id": order_id, "refundable": True})

    @tool(description="Create a refund.")
    def create_refund(order_id: str) -> str:
        return json.dumps({"refund_id": "r-7"})

    class State(TypedDict, total=False):
        order_id: str
        order: str
        refund: str

    async def check(state: State):
        return {"order": await lookup_order.ainvoke({"order_id": state["order_id"]})}

    async def refund(state: State):
        return {"refund": await create_refund.ainvoke({"order_id": state["order_id"]})}

    graph = StateGraph(State)
    graph.add_node("check", check)
    graph.add_node("refund", refund)
    graph.add_edge(START, "check")
    graph.add_edge("check", "refund")
    graph.add_edge("refund", END)
    events = [
        event
        async for event in graph.compile().astream_events(
            {"order_id": "1042"}, version="v2"
        )
    ]
    trace = create_importer("importer.langchain.trace", include_content=True).convert(
        events
    )
    tools = [step for step in trace.steps if step.kind == "tool"]
    assert [(t.name, t.arguments, t.attributes) for t in tools] == [
        ("lookup_order", {"order_id": "1042"}, {"langgraph.node": "check"}),
        ("create_refund", {"order_id": "1042"}, {"langgraph.node": "refund"}),
    ]
    row = {
        "trace": trace.model_dump(),
        "expected_tools": ["lookup_order", "create_refund"],
    }
    metrics = [
        create_metric("metric.workflow.tool_sequence"),
        create_metric("metric.workflow.step_efficiency"),
    ]
    result = await Evaluator(metrics).arun(Dataset.from_records([row]))
    assert [score.value for score in result.rows[0].metrics.values()] == [1.0, 1.0]

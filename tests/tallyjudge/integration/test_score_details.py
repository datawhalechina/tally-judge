import json

import pytest

from tallyjudge import (
    Dataset,
    Evaluator,
    RunResult,
    create_language_model,
    create_metric,
    create_storage_backend,
    numeric_metric,
)
from tallyjudge.sdk import MetricBase, MetricResult, MetricSpec

FAITH = "metric.rag.faithfulness"
ROW = {
    "user_input": "图书馆什么时候开放？",
    "response": "工作日上午九点开放，周末也开放。",
    "retrieved_contexts": ["图书馆公告：工作日上午九点开放。"],
}


def _judge(client, verdicts):
    client.chat_reply(json.dumps({"statements": ["工作日九点开放", "周末开放"]}))
    client.chat_reply(
        json.dumps(
            {
                "statements": [
                    {"statement": str(i), "reason": f"理由{i}", "verdict": v}
                    for i, v in enumerate(verdicts)
                ]
            }
        )
    )
    return create_language_model(
        "model.openai_compatible.chat_completions", client=client, model="judge"
    )


@pytest.mark.asyncio
async def test_faithfulness_names_the_unsupported_statement_and_survives_storage(
    client, tmp_path
):
    store = create_storage_backend("storage.local.jsonl", root=tmp_path)
    metric = create_metric(FAITH, llm=_judge(client, [1, 0]))
    result = await Evaluator([metric], backend=store).arun(Dataset.from_records([ROW]))
    expected = [
        {"statement": "工作日九点开放", "supported": True, "reason": "理由0"},
        {"statement": "周末开放", "supported": False, "reason": "理由1"},
    ]
    score = store.load_run(result.run_id).rows[0].metrics[FAITH]
    assert score.value == 0.5 and score.details == {"statements": expected}
    assert result.metrics[FAITH]["revision"] == "2"


@pytest.mark.asyncio
async def test_faithfulness_rejects_a_verdict_count_that_differs_from_the_statements(
    client,
):
    metric = create_metric(FAITH, llm=_judge(client, [1]))
    result = await Evaluator([metric]).arun(Dataset.from_records([ROW]))
    score = result.rows[0].metrics[FAITH]
    assert (score.status, score.error.code) == ("error", "verdict_count")


def test_details_must_be_a_json_object():
    class Evidence(MetricBase):
        spec = MetricSpec("metric.test.evidence", "1", {"response": str})

        async def ascore(self, response):
            return MetricResult(1.0, details={"seen": {response}})

    result = Evaluator([Evidence()]).run(Dataset.from_records([{"response": "a"}]))
    assert result.rows[0].metrics["metric.test.evidence"].status == "invalid_output"


def test_scores_without_details_keep_the_export_compact_and_version_3_still_loads():
    @numeric_metric("metric.test.plain", revision="1")
    def plain(response: str) -> float:
        return 1.0

    data = Evaluator([plain()]).run(Dataset.from_records([{"response": "a"}])).to_dict()
    assert "details" not in data["rows"][0]["metrics"]["metric.test.plain"]
    data["schema_version"] = 3
    restored = RunResult.from_dict(data)
    assert restored.schema_version == 4
    assert restored.rows[0].metrics["metric.test.plain"].details is None


@pytest.mark.asyncio
async def test_conversation_judge_reads_every_turn_of_a_multi_turn_dialogue(client):
    client.chat_reply(json.dumps({"reason": "第二轮直接修改了地址", "verdict": 0}))
    judge = create_language_model(
        "model.openai_compatible.chat_completions", client=client, model="judge"
    )
    metric = create_metric(
        "metric.conversation.criteria",
        llm=judge,
        criteria="助手在修改订单之前先确认订单号。",
    )
    messages = [
        {"role": "user", "content": "帮我改收货地址"},
        {"role": "assistant", "content": "已改好"},
        {"role": "user", "content": "谢谢"},
    ]
    result = await Evaluator([metric]).arun(
        Dataset.from_records([{"messages": messages}, {"messages": []}])
    )
    judged, empty = (row.metrics[metric.spec.id] for row in result.rows)
    assert (judged.value, judged.reason) == (0.0, "第二轮直接修改了地址")
    assert judged.inputs_sha256 is not None
    assert (empty.status, empty.error.field) == ("error", "messages")
    prompt = client.calls[0][1]["messages"][0]["content"]
    assert "帮我改收货地址" in prompt and "谢谢" in prompt and len(client.calls) == 1

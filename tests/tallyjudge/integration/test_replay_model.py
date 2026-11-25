import json

import pytest

from tallyjudge import Dataset, Evaluator, create_language_model, create_metric

FAITH = "metric.rag.faithfulness"
REPLAY = "model.replay.jsonl"
ROW = {
    "user_input": "图书馆什么时候开放？",
    "response": "工作日上午九点开放，周末也开放。",
    "retrieved_contexts": ["图书馆公告：工作日上午九点开放。"],
}


def _score(model):
    metric = create_metric(FAITH, llm=model)
    return Evaluator([metric]).arun(Dataset.from_records([ROW]))


@pytest.mark.asyncio
async def test_recorded_replies_reproduce_the_score_without_requests(client, tmp_path):
    recording = tmp_path / "judge.jsonl"
    client.chat_reply(json.dumps({"statements": ["工作日九点开放", "周末开放"]}))
    client.chat_reply(
        json.dumps(
            {
                "statements": [
                    {"statement": "工作日九点开放", "reason": "公告写明", "verdict": 1},
                    {"statement": "周末开放", "reason": "公告未提", "verdict": 0},
                ]
            }
        )
    )
    live = create_language_model(
        "model.openai_compatible.chat_completions", client=client, model="judge"
    )
    recorded = await _score(create_language_model(REPLAY, path=recording, llm=live))
    assert recorded.rows[0].metrics[FAITH].value == 0.5
    assert len(client.calls) == 2

    replayed = await _score(create_language_model(REPLAY, path=recording))
    score = replayed.rows[0].metrics[FAITH]
    assert score.value == 0.5 and len(client.calls) == 2
    assert score.usage.untracked_sources == [REPLAY]


@pytest.mark.asyncio
async def test_prompt_without_a_recording_is_a_model_error(tmp_path):
    result = await _score(create_language_model(REPLAY, path=tmp_path / "empty.jsonl"))
    score = result.rows[0].metrics[FAITH]
    assert (score.status, score.error.category) == ("error", "model_error")

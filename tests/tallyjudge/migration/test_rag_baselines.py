"""Scoring and transport regressions observed in the isolated Ragas ec00339 run."""

import hashlib
import json

import pydantic
import pytest

from tallyjudge import (
    Dataset,
    Evaluator,
    create_embedding_model,
    create_language_model,
    create_metric,
)

QUESTION = "法国的首都是哪里？"
ROW = {
    "user_input": QUESTION,
    "response": "巴黎。",
    "retrieved_contexts": ["法国位于欧洲。", "首都是巴黎。"],
}
FAITH = "metric.rag.faithfulness"
RELEVANCE = "metric.rag.answer_relevance"

# Upstream SHA-256 pairs: exact prompt bytes on Pydantic 2.14.3, then the same
# prompt with only the JSON Schema's key order and insignificant whitespace normalized.
PROMPTS = {
    "ap_reference_first": (
        "96840e20733611f9525d2b6189aadf433cfd36cf25c4d4e2f80ecfa27721a6af",
        "0ee87fa8290beca14ce31018cbe648f13bbd2df16d9ece7fd394496e16529c23",
    ),
    "ap_reference_second": (
        "197708eeec8e0699e8803bfc1499ce87bb512e369080b4345e65ba69f1a359d3",
        "651f4f58681e335b7d212b6052d9a69a2ce93755fafb60f6d3e0f637033c65bd",
    ),
    "ap_reference_blank": (
        "9657dea68f8939f97724f8d164d5dbfa97afbd0a169e9ad46f79311a6014ae91",
        "bb870973afbf3466a0c4676c7f086bfa5b978e260750c1e3262e1493180cdc6a",
    ),
    "ap_response_first": (
        "99a0441df112183cd658d67b5b1b28b1a0021c06687d2bc04498c8deb8bd5d4f",
        "a53ebad2175ae8890a65e8af02a4b5118f267e4d382af4eac42ea25a4aef741c",
    ),
    "ap_response_second": (
        "da5cb80e9bb9cb5e25301cc41270bd42b8ba03f7eb1f595c9c653a9552d1320e",
        "24a44659510a77889404cc453b55e61d4895426e675c302bdd1d750a76fd5dae",
    ),
    "ap_response_blank": (
        "378d973342d636bc1c7d8099c10f1ff272ae4f220f6c86c2fab01c3a55b2c444",
        "4d1f6c0e5d3f35c787b58918131a55cb00e8b5b88f3d252dd4b58773e8a0b062",
    ),
    "statements": (
        "cb84d943e48a73f3628d802fab202a3bcf7fb5349f6648566ef54f3ead5ed6e3",
        "16f49260a0982c3d5179f7d25edbcfab9578ef075bce8ae48d1034e919937cb3",
    ),
    "verdicts": (
        "1dcab1efff4c5f8c7617b649562b225ff335f00c291238de6bd7390c3b70baba",
        "cc2945bb80833857d9317a83d1f7f2cd80f46aa0e4a638471cfa6b92b59e5498",
    ),
    "questions": (
        "cdad171309b2939d766872501f73b11ae068d7de1c5089b0179480e38e63e98d",
        "9fc5231b8369a4f6255291d9c4b1b5a98b808712f9b2887adf357abc46106f3d",
    ),
}


@pytest.fixture
def judge(client):
    return create_language_model(
        "model.openai_compatible.chat_completions",
        client=client,
        model="test",
        structured_retries=0,
    )


async def evaluate(metric, record):
    result = await Evaluator([metric]).arun(Dataset.from_records([record]))
    return result.rows[0].metrics[metric.spec.id]


@pytest.mark.parametrize("field", ["reference", "response"])
@pytest.mark.parametrize(
    ("passages", "verdicts", "status", "value", "code"),
    [
        (["first", "second", "first"], [1, 0, 1], "success", 5 / 6, None),
        (["second", "first", "first"], [0, 1, 1], "success", 7 / 12, None),
        (["first", "second"], [0, 0], "undefined", None, "no_relevant_contexts"),
        ([], [], "error", None, None),
        ([""], [1], "success", 1.0, None),
    ],
)
@pytest.mark.asyncio
async def test_archived_context_precision(
    field, passages, verdicts, status, value, code, judge, client
):
    suffix = "with_reference" if field == "reference" else "without_reference"
    metric = create_metric("metric.rag.context_precision_" + suffix, llm=judge)
    for verdict in verdicts:
        client.chat_reply(json.dumps({"reason": "fixture", "verdict": verdict}))
    row = {
        "user_input": "question",
        "reference": "gold",
        "response": "observed",
        "retrieved_contexts": passages,
    }
    result = await evaluate(metric, row)
    assert result.status == status
    if status == "success":
        assert result.value == pytest.approx(value)
    elif code:
        assert result.value is None and result.error.code == code
    else:
        assert (result.error.category, result.error.field) == (
            "validation_error",
            "retrieved_contexts",
        )
    assert_calls(client, [f"ap_{field}_{passage or 'blank'}" for passage in passages])
    if verdicts:
        assert result.usage.generation.requests == len(verdicts)
        assert result.usage.generation.requests_without_usage == len(verdicts)


def assert_prompt(text, name):
    raw, normalized = PROMPTS[name]
    if pydantic.__version__ == "2.14.3":
        assert hashlib.sha256(text.encode()).hexdigest() == raw
    prefix, rest = text.split("as specified in JSON Schema:\n", 1)
    schema, suffix = rest.split("Do not use single quotes", 1)
    canonical = json.dumps(
        [prefix, json.loads(schema), suffix],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert hashlib.sha256(canonical.encode()).hexdigest() == normalized


def assert_calls(client, prompts, embedding_batches=()):
    assert [kind for kind, _ in client.calls] == (
        ["generation"] * len(prompts) + ["embedding"] * len(embedding_batches)
    )
    for name, (_, request) in zip(prompts, client.calls[: len(prompts)], strict=True):
        [message] = request["messages"]
        assert message["role"] == "user"
        assert_prompt(message["content"], name)
    assert [req["input"] for _, req in client.calls[len(prompts) :]] == list(
        embedding_batches
    )
    assert not client.responses


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verdicts,expected,status,reason",
    [
        pytest.param([1, 1, 1], 1.0, "success", None, id="supported"),
        pytest.param([0, 1, 0], 1 / 3, "success", None, id="mixed"),
        pytest.param([1, 0], 0.5, "success", None, id="fewer-verdicts-than-statements"),
        pytest.param([], None, "undefined", "no_verdicts", id="empty-verdicts"),
        pytest.param(
            ["1", True, 1.0], 1.0, "success", None, id="coerced-binary-verdicts"
        ),
        pytest.param([2], None, "error", "schema_validation", id="invalid-verdict"),
    ],
)
async def test_faithfulness_verdicts(client, judge, verdicts, expected, status, reason):
    client.chat_reply(json.dumps({"statements": ["甲", "乙", "丙"]}))
    client.chat_reply(
        json.dumps(
            {
                "statements": [
                    {"statement": str(i), "reason": "fixture", "verdict": v}
                    for i, v in enumerate(verdicts)
                ]
            }
        )
    )
    score = await evaluate(create_metric(FAITH, llm=judge), ROW)
    assert (score.status, score.value) == (status, expected)
    assert (score.error.code if score.error else None) == reason
    assert_calls(client, ["statements", "verdicts"])


@pytest.mark.asyncio
async def test_faithfulness_without_statements(client, judge):
    client.chat_reply('{"statements":[]}')
    score = await evaluate(create_metric(FAITH, llm=judge), ROW)
    assert (score.status, score.value, score.error.code) == (
        "undefined",
        None,
        "no_statements",
    )
    assert_calls(client, ["statements"])


@pytest.mark.asyncio
async def test_faithfulness_rejects_string_context_before_model_call(client, judge):
    score = await evaluate(
        create_metric(FAITH, llm=judge), dict(ROW, retrieved_contexts="巴黎。")
    )
    assert (score.status, score.error.category, score.error.field) == (
        "error",
        "validation_error",
        "retrieved_contexts",
    )
    assert_calls(client, [])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "questions,embedded,expected",
    [
        pytest.param([("B", 0)], ["B"], -1.0, id="negative-cosine"),
        pytest.param(
            [("A", 1), ("", 0), ("C", 0)], ["A", "C"], 0.5, id="partial-questions"
        ),
        pytest.param(
            [("A", 2), ("B", -1), ("C", 1)], ["A", "B", "C"], 0.0, id="all-evasive"
        ),
        pytest.param([("", 1)], [], 0.0, id="empty-questions"),
        pytest.param([("Z", 1)], ["Z"], None, id="zero-vector"),
    ],
)
async def test_answer_relevance(client, judge, questions, embedded, expected):
    vectors = {
        QUESTION: [1.0, 0.0],
        "A": [1.0, 0.0],
        "B": [-1.0, 0.0],
        "C": [0.0, 1.0],
        "Z": [0.0, 0.0],
    }
    for question, flag in questions:
        client.chat_reply(json.dumps({"question": question, "noncommittal": flag}))
    batches = [[QUESTION], embedded] if embedded else []
    for batch in batches:
        client.embedding_reply(enumerate(vectors[q] for q in batch))
    embeddings = create_embedding_model(
        "embedding.openai_compatible.embeddings", client=client, model="test"
    )
    metric = create_metric(
        RELEVANCE, llm=judge, embeddings=embeddings, strictness=len(questions)
    )
    score = await evaluate(metric, ROW)
    if expected is None:
        assert (score.status, score.value, score.error.code) == (
            "undefined",
            None,
            "zero_norm_embedding",
        )
    else:
        assert score.status == "success"
        assert score.value == pytest.approx(expected, abs=1e-12, rel=0.0)
    assert_calls(client, ["questions"] * len(questions), batches)

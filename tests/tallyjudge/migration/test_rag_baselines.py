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
    "ac_answer": (
        "9cb8e616cefd81009b5f7dbe12a1b55547a610ccf281c2bd4176bba3eac36332",
        "cbebcbdf146a10df029c8d35dec3cc0060312a36e0d671fd93febb46d431d974",
    ),
    "ac_reference": (
        "47ea6f39a6736ae00bf904d85df8d5cb9af7d8778b8d030b0382752d0fa74788",
        "c8e14ac93f910bd7d71fb358dfa80a8e8635ae5fdf1442982ed36228c1f208a7",
    ),
    "ac_classify_weighted": (
        "a0aa6cac4a30e4d7aa65725809630ea385797c5dc5baa614464d01c2fea8aa1d",
        "521e5147d86b5a0e6f01d984cf1345b5235da37302449ed19e54578f247be929",
    ),
    "ac_classify_answer_only": (
        "1367488fbeaec70d90b8cdb6e3d4f5ebc92522e28f23ebc54fe7ea95925bbff8",
        "90774a54e438569f6d87cb06e4986e6f8c2c0b7490c0646a306f0325eab40220",
    ),
    "ac_classify_reference_only": (
        "f6a5d2b7405d0939bb0b7d8c1dcbf9d0cacf8af70610f9b7c1b163851dff93be",
        "d2dca023647ba1b525b8809b8190dba4d1b81c2711c16b93742a35058935dc99",
    ),
    "ac_classify_no_classified_statements": (
        "de9de65edc9ca6d71183112c9fe2d21efabbfc6cb5a8fe10472594b39165d701",
        "a06cf0ee8174992a1063b3d4157b3539db4360ad789fccd3c0631355a2f9102f",
    ),
    "ac_empty_answer": (
        "09e49f9ae48d49227cde30d1f62672b07348712e6e8d91bc81460ce355aa5e57",
        "6303c6e98222a6587bb38b30b36752e066251a1b81b2ed8a5506c80a5dbb3a66",
    ),
    "recall_ranked": (
        "a854e51d2f589314f20b8589094d7029dfb54bab612b782da87b8ced68a28959",
        "53043749ce08fe56c5ffa15c31ca75caf238da1ed75554e59ab85cc1fe8e097e",
    ),
    "recall_blank": (
        "eb579ed124ee4310c404c0f114a53c6fabf1459bf9f16c340c2600e46ce87aa7",
        "1eb6a564f33bbc1e8286cb10eb0702a672348c331be916c25ec799d1a60c91ff",
    ),
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


@pytest.mark.parametrize(
    ("passages", "bits", "expected"),
    [
        (["first", "second", "first"], [1, 0, 1], 2 / 3),
        (["first", "second", "first"], [0, 0], 0.0),
        (["first", "second", "first"], [], "no_classifications"),
        ([], [], "validation_error"),
        ([""], [1], 1.0),
    ],
)
@pytest.mark.asyncio
async def test_archived_context_recall(passages, bits, expected, judge, client):
    metric = create_metric("metric.rag.context_recall", llm=judge)
    if passages:
        items = [
            {"statement": "same statement", "reason": "fixture", "attributed": flag}
            for flag in bits
        ]
        client.chat_reply(json.dumps({"classifications": items}))
    score = await evaluate(
        metric,
        {"user_input": "question", "reference": "gold", "retrieved_contexts": passages},
    )
    if expected == "validation_error":
        assert (score.status, score.error.category, score.error.field) == (
            "error",
            expected,
            "retrieved_contexts",
        )
        assert_calls(client, [])
        return
    if expected == "no_classifications":
        assert (score.status, score.value, score.error.code) == (
            "undefined",
            None,
            expected,
        )
    else:
        assert (score.status, score.value) == ("success", pytest.approx(expected))
    assert (
        score.usage.generation.requests
        == score.usage.generation.requests_without_usage
        == 1
    )
    assert_calls(client, ["recall_ranked" if passages[0] else "recall_blank"])


@pytest.mark.parametrize(
    ("response", "threshold", "expected"),
    [
        ("X", None, 1.0),
        ("Y", None, 0.0),
        ("-X", None, -1.0),
        ("Y", 0.0, 1.0),
        ("-X", 0.0, 0.0),
        ("X", 1.0, 1.0),
        ("Z", None, "zero_norm_embedding"),
        ("Z", 0.0, "zero_norm_embedding"),
        ("", None, 1.0),
        ("bad", None, "invalid_embedding"),
    ],
)
@pytest.mark.asyncio
async def test_archived_semantic_similarity(response, threshold, expected, client):
    vectors = {
        "X": [1.0, 0.0],
        "Y": [0.0, 1.0],
        "-X": [-1.0, 0.0],
        "Z": [0.0, 0.0],
        " ": [1.0, 0.0],
        "bad": [1.0],
    }
    embeddings = create_embedding_model(
        "embedding.openai_compatible.embeddings", client=client, model="fixture"
    )
    metric = create_metric(
        "metric.text.semantic_similarity", embeddings=embeddings, threshold=threshold
    )
    client.embedding_reply([(0, vectors["X"])])
    client.embedding_reply([(0, vectors[response or " "])])
    score = await evaluate(metric, {"reference": "X", "response": response})
    if isinstance(expected, str):
        status = "undefined" if expected == "zero_norm_embedding" else "error"
        assert (score.status, score.value, score.error.code) == (status, None, expected)
    else:
        assert (score.status, score.value) == ("success", expected)
    assert (
        score.usage.embedding.requests
        == score.usage.embedding.requests_without_usage
        == 2
    )
    assert_calls(client, [], [["X"], [response or " "]])


# The classifier is called when either extracted side has statements.
MIXED = (["A", "B"], ["A", "C", "D"], (1, 1, 2), "ac_classify_weighted")
ANSWER_ONLY = (["B"], [], (0, 1, 0), "ac_classify_answer_only")
REFERENCE_ONLY = ([], ["C"], (0, 0, 1), "ac_classify_reference_only")
EMPTY = ([], [], None, None)
UNCLASSIFIED = (["A"], ["B"], (0, 0, 0), "ac_classify_no_classified_statements")


@pytest.mark.parametrize(
    "statements,weights,beta,expected,response,zero_vector",
    [
        pytest.param(MIXED, (0.75, 0.25), 1.0, 0.05, "observed", False, id="weighted"),
        pytest.param(MIXED, (2, 0), 2.0, 5 / 14, "observed", False, id="beta-two"),
        pytest.param(MIXED, (3, 1), 1.0, 0.05, "observed", False, id="scaled-weights"),
        pytest.param(
            ANSWER_ONLY, (1, 0), 1.0, 0.0, "observed", False, id="answer-only"
        ),
        pytest.param(
            REFERENCE_ONLY, (1, 0), 1.0, 0.0, "observed", False, id="reference-only"
        ),
        pytest.param(
            ANSWER_ONLY,
            (0.75, 0.25),
            1.0,
            -0.25,
            "observed",
            False,
            id="negative-total",
        ),
        pytest.param(EMPTY, (1, 0), 1.0, 1.0, "observed", False, id="empty-factual"),
        pytest.param(
            EMPTY, (0.75, 0.25), 1.0, 0.5, "observed", False, id="empty-weighted"
        ),
        pytest.param(
            UNCLASSIFIED,
            (0.75, 0.25),
            1.0,
            "no_classified_statements",
            "observed",
            False,
            id="empty-classification",
        ),
        pytest.param(
            REFERENCE_ONLY,
            (1, 0),
            0.0,
            "no_predicted_statements",
            "observed",
            False,
            id="no-predictions",
        ),
        pytest.param(ANSWER_ONLY, (1, 0), 0.0, 0.0, "observed", False, id="beta-zero"),
        pytest.param(
            MIXED,
            (0.75, 0.25),
            1.0,
            "zero_norm_embedding",
            "observed",
            True,
            id="zero-vector",
        ),
        pytest.param(MIXED, (0, 1), 1.0, -1.0, "observed", False, id="only-similarity"),
        pytest.param(
            REFERENCE_ONLY, (0.75, 0.25), 1.0, -0.25, "", False, id="raw-empty-response"
        ),
    ],
)
@pytest.mark.asyncio
async def test_archived_answer_correctness(
    statements, weights, beta, expected, response, zero_vector, client, judge
):
    answer, truth, counts, classifier = statements
    prompts = ["ac_answer" if response else "ac_empty_answer", "ac_reference"]
    for extracted in (answer, truth):
        client.chat_reply(json.dumps({"statements": extracted}))
    if counts is not None:
        item = {"statement": "item", "reason": "fixture"}
        client.chat_reply(
            json.dumps(
                {
                    name: [item] * count
                    for name, count in zip(("TP", "FP", "FN"), counts, strict=True)
                }
            )
        )
        prompts.append(classifier)
    embedding_batches = []
    dependencies = {"llm": judge}
    if weights[1]:
        dependencies["embeddings"] = create_embedding_model(
            "embedding.openai_compatible.embeddings", client=client, model="fixture"
        )
        if expected != "no_classified_statements":
            embedding_batches = [[response], ["gold"]]
            client.embedding_reply([(0, [0.0, 0.0] if zero_vector else [-1.0, 0.0])])
            client.embedding_reply([(0, [1.0, 0.0])])
    metric = create_metric(
        "metric.rag.answer_correctness", weights=weights, beta=beta, **dependencies
    )
    score = await evaluate(
        metric, {"user_input": "question", "response": response, "reference": "gold"}
    )
    if isinstance(expected, str):
        assert (score.status, score.value, score.error.code) == (
            "undefined",
            None,
            expected,
        )
    else:
        assert score.status == "success" and score.value == pytest.approx(
            expected, rel=0.0, abs=1e-12
        )
    assert score.usage.generation.requests == len(prompts)
    assert (score.usage.embedding.requests if score.usage.embedding else 0) == len(
        embedding_batches
    )
    assert_calls(client, prompts, embedding_batches)


@pytest.mark.asyncio
async def test_correctness_reports_second_extraction_failure_and_usage(client, judge):
    client.chat_reply('{"statements":["A"]}')
    client.chat_reply("broken JSON")
    metric = create_metric(
        "metric.rag.answer_correctness", llm=judge, weights=[1.0, 0.0]
    )
    score = await evaluate(
        metric, {"user_input": "question", "response": "observed", "reference": "gold"}
    )
    assert (score.status, score.error.category) == ("error", "model_output_error")
    assert score.usage.generation.requests == 2
    assert_calls(client, ["ac_answer", "ac_reference"])


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

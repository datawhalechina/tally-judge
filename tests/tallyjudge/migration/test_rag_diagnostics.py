"""Offline score and prompt regressions from Ragas ec00339 (Apache-2.0)."""

import hashlib
import json
from types import SimpleNamespace

import pydantic
import pytest

from tallyjudge import Dataset, Evaluator, create_language_model, create_metric

PASSAGES = ["first passage", "second passage"]
# Exact UTF-8 on the pinned Pydantic, plus a digest with only schema key order normalized.
PROMPTS = {
    "entity_reference": (
        "e2de7ef6da39a5e2e93a47824237e008a04570e954dc4a73716de8a0e55ea2eb",
        "8edfcbc1185345fa6ba908b2a6b15adf90279939595162133d7904bef8d3d23a",
    ),
    "entity_passages": (
        "773972e16efa2127a8c27cfd0686f12d24339d42e9d5810c9d1ffbe319e5622f",
        "d910bcb1e7884f1ac437f8687066541661e7185e90f132d8124719771976e20f",
    ),
    "entity_empty": (
        "8d5cb2bc8b4e5f0e4e30274042d90b47ff90f433f60678e8f19d70605ea0e9b1",
        "0514d11e2b70ddd0f6ea5bd04cd0955aec04482a563f09a1fb7f8fbdfd428f3a",
    ),
    "answer_claims": (
        "d1cdcdd4361e135760f14098ab5c7b2073b27a17a6207d3af9e619722ba741a4",
        "dd04c8562a371da1c7ae6c8ebabdf6ebd32e6ee9284ca5f836041c58d027f189",
    ),
    "reference_claims": (
        "99a79f148fa9ac9c8ffc75c1f35dfb5aa6989d0449f0623d6999bcec3afb044b",
        "828ec001728b6e2180c4facf2dd6b8d5ea4c1005dab30035dbf663455d8748e0",
    ),
    "verify_answer": (
        "3df3f7c67a27ed982b1cb31e073e0f441d7d7ff15d34fb00d31d958128938917",
        "99d2ca427feb9bf60786f925ff02273e9fe75a764671dac8691505823788f9e5",
    ),
    "verify_reference": (
        "657845061c89d70d594d18713440f57fe7777f8f59f5bf73bf2f8a4e3d4e565f",
        "71ca47ce4d794cee8a3d075b04b80a67014f65ac0fb5885dfe7c141d290c0aab",
    ),
    "fine_answer": (
        "a276229bd242a587e457e120aca49c1f7c99ad1ca8890bc75bc29ea778b8fa10",
        "17b4243e970c57c8f3aaf7e8710c979c8578c2feddf2eba0324e52dff846ff54",
    ),
    "fine_reference": (
        "9d726886a3a3e3a2fa13c4c89a60e35bcd3752e88e9d70c5f4a131a761829391",
        "d4fba15275fd7aac6b9ca90de4ab9eaff60485a3e5fc91d2caa8408a653125d8",
    ),
    "verify_one_answer": (
        "bbcf8e41adf0b625e0659ba74a3fce3d690bf39f91a0d0fd8998211056ec5e89",
        "dfa21cd5b0b6fb22524478847ba90f6907a0e836b3454d9edd5410fac78b8a63",
    ),
    "verify_one_reference": (
        "336f701ad37c7ba2b36652c0291757093604ce6891a6faf30eaa39a6deed88d6",
        "6b152fd315a5a49782bc3173681a7bfc4552d8a8fbb97b879d70a8b081ddd65c",
    ),
    "noise_gold": (
        "bb478efde451095c0091ba2aad2e11fd49043247aaf31edbf1f52926c12371ff",
        "c8f17825987a12877aa2cb2dc735f560bac94ed9701b75e05f28556bcaf93849",
    ),
    "noise_answer": (
        "b304a6e4f7964f23df45daade3870a82b8486ecb2c2f28c3212f5fec21a2ba05",
        "79ad9b8d30f57ecb656a02f7415baf37979e2e6d5f9dde7bbad26c26b029f3ce",
    ),
    "first_gold": (
        "bcfcf05a62cc0743e18b225d8347372293f5ca335c5205f7dcb59402c28b40b7",
        "be25ced304c3d574679be752a154e23b93f2a063ab19333436088aa57b3bd161",
    ),
    "first_answer": (
        "7d20721a4e8c3f8609fef225aa56b125aef92b25a7706f8439f1f8068b122dc8",
        "362f08e23437fa63b5cf5cfd23344cedf026c310c61cf73de0be5bd380a1e5b0",
    ),
    "second_gold": (
        "055d91277b309a9e4f1785ee452222a9666a92764c2ed39dceb2bd91b30c464d",
        "31a9c821299d8248dcdafc02cbf445c3570898119ba9d0935ef6f7f7c52ed940",
    ),
    "second_answer": (
        "67a63ba7dde2a0e1cfa165600040123b59d6d0b01bc204666debf5ec43227294",
        "5f679332eb59ddbfd9bcd5dac7606681d5ae9ce5a630d9d2dab7e8ebfc8462ef",
    ),
    "gold_answer": (
        "9ca747b43dbea7ec7685786a096cb2c357cbf5f14d75068d0b9bdacccbf367ce",
        "eb0a94044ed27b45cd95f2bd97e90ea67fa4b15b4bc1447a7d188bb7e9e2e633",
    ),
    "accuracy_first": (
        "584d4adc945c28daad881083b85d373a02c2d0ef8b55c9e960ede02e995d8e70",
        "63551af2a4cb72c56a4e8d78ffb91282c563009cbc0ec536d239dc8482e71134",
    ),
    "accuracy_swapped": (
        "a22413464c2b61f1f7e4c0ae3013730f94643ab5386fdb573608f99fc7b5543d",
        "8b2829afe08abaa8e13d3e8fdcd285d38266fde213dc657d330e8d11f660632d",
    ),
    "accuracy_blank_first": (
        "241dc12164f87e1d5ed621e4110bb809ad56b93834e255efc008af9fe21e6737",
        "c2c35a5df92de8080c123fd5438e936d00f3ad3bb6a027f16842c1b39a2924ee",
    ),
    "accuracy_blank_swapped": (
        "be34756a3c25521b25ebefe23ad90836cc1fe61a73bc211d2ae2343992275bc0",
        "cf903edef10ca1b8f9a20b365040a6163d85596f53a0f98dbf633e4d13fed63d",
    ),
    "relevance_first": (
        "d1660485b317707df9be6fce42f37b8392b1a47b111c958b52d1fa4c098f028a",
        "73bf7c16c184eec2778ab01d16819c5609335a146eb0aeb23599dbeaf506231c",
    ),
    "relevance_second": (
        "a4f7be2bd58f337b060357e6cb9493af63a78c8abed97e3ad7a4bacb0af7fb14",
        "de3f296a30406694754b58c2f3b5b5248e1c7a3a38ea0c18be49cac8249226c4",
    ),
    "groundedness_first": (
        "91a630d91d3fd5349ff932db28065380bd98e7f994c210935cddcd3fb16d7cbf",
        "89f6d6b877ae2b49c35107f111f744f3ba15bf2a593f1790146c1d506c0af679",
    ),
    "groundedness_second": (
        "d36ef81e66dd5c63eac4165cf5059dfd030ccbb4e624f21a48ce9c673d672143",
        "b5e8dc7dd30d65892a80dcd279af33691cf1da29b03ebcab0175d6558d3616ca",
    ),
    "low_high": (
        "18c9f0311cea2d6483d30a601185de46183d476d00c659bad4d7bfeb7bb047cb",
        "0fab74574b3a3bcd9672606e7236c3e3c84ded6d22aa2535314ed046b2fb1c31",
    ),
    "high_low": (
        "3d90f89588f18452d61c6194b32971fea9a6a0dc2bca7e717614c94186261cdf",
        "4bfa4bc5aa5622770ac3b7bf47e8ac2f502eab3a750f9658d990428351b63c0d",
    ),
}


def assert_archived_prompt(text, name):
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


async def replay(client, metric, inputs, replies, prompts, *, retries=0, **settings):
    llm = create_language_model(
        "model.openai_compatible.chat_completions",
        client=client,
        model="offline",
        temperature=0.0,
        structured_retries=retries,
    )
    plugin = create_metric("metric.rag." + metric, llm=llm, **settings)
    for reply in replies:
        client.chat_reply(
            json.dumps(reply),
            tokens=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
        )
    run = await Evaluator([plugin]).arun(Dataset.from_records([inputs]))
    score = run.rows[0].metrics[plugin.spec.id]
    for (kind, request), name in zip(client.calls, prompts, strict=True):
        message = request["messages"][0]
        assert (kind, message["role"]) == ("generation", "user")
        assert_archived_prompt(message["content"], name)
    assert len(client.responses) == len(replies) - len(prompts)
    if prompts:
        usage = score.usage.generation
        assert usage.requests == len(prompts)
        assert usage.known_tokens == {
            "input_tokens": 3 * len(prompts),
            "output_tokens": len(prompts),
        }
    else:
        assert score.usage is None
    return score


@pytest.mark.parametrize(
    ("passages", "entity_lists", "order", "expected"),
    [
        (
            PASSAGES,
            [["巴黎", "塞纳河", "巴黎"], ["巴黎", "法国"]],
            ["entity_reference", "entity_passages"],
            1 / (2 + 1e-8),
        ),
        (
            PASSAGES,
            [["Paris"], ["paris"]],
            ["entity_reference", "entity_passages"],
            0.0,
        ),
        (PASSAGES, [[], ["Paris"]], ["entity_reference"], "no_reference_entities"),
        ([], [["Paris"], []], ["entity_reference", "entity_empty"], 0.0),
        ("text", [], [], "validation_error"),
    ],
)
@pytest.mark.asyncio
async def test_entity_recall(client, passages, entity_lists, order, expected):
    row = {"reference": "Paris is on the Seine.", "retrieved_contexts": passages}
    score = await replay(
        client,
        "context_entity_recall",
        row,
        [{"entities": xs} for xs in entity_lists],
        order,
    )
    if expected == "validation_error":
        assert (score.status, score.error.category, score.error.field) == (
            "error",
            expected,
            "retrieved_contexts",
        )
    elif expected == "no_reference_entities":
        assert (score.status, score.error.code) == ("undefined", expected)
    else:
        assert score.status == "success" and score.value == pytest.approx(
            expected, abs=1e-12, rel=0
        )


def judgments(flags):
    return {
        "statements": [
            {"statement": f"claim {index}", "reason": "fixture", "verdict": flag}
            for index, flag in enumerate(flags)
        ]
    }


ANSWER_CLAIMS = (["A", "B", "C"], [1, 1, 0])
REFERENCE_CLAIMS = (["D", "E"], [1, 0])
FACTS = {"response": "observed answer", "reference": "reference answer"}


@pytest.mark.parametrize(
    ("settings", "answer", "reference", "expected"),
    [
        ({"mode": "precision"}, ANSWER_CLAIMS, None, 0.67),
        ({"mode": "recall"}, None, REFERENCE_CLAIMS, 0.50),
        ({}, ANSWER_CLAIMS, REFERENCE_CLAIMS, 0.57),
        ({"beta": 2}, ANSWER_CLAIMS, REFERENCE_CLAIMS, 0.53),
        (
            {"atomicity": "high", "coverage": "high"},
            ANSWER_CLAIMS,
            REFERENCE_CLAIMS,
            0.57,
        ),
        ({}, ([], []), None, "no_response_statements"),
        ({}, ANSWER_CLAIMS, ([], []), "no_reference_statements"),
        ({}, (["A"], []), None, "no_verdicts"),
        ({}, (["A"], [0]), (["D"], [0]), 0.0),
        ({"beta": 0}, ANSWER_CLAIMS, (["D"], [0]), "zero_fbeta_denominator"),
        ({"beta": 4.92442763184274}, ANSWER_CLAIMS, REFERENCE_CLAIMS, 0.50),
        ({"beta": 1e200}, ANSWER_CLAIMS, REFERENCE_CLAIMS, 0.50),
    ],
)
@pytest.mark.asyncio
async def test_factual_correctness(client, settings, answer, reference, expected):
    replies, prompts = [], []
    for side, direction in (("answer", answer), ("reference", reference)):
        if direction is None:
            continue
        claims, flags = direction
        replies.append({"claims": claims})
        prompts.append(
            "fine_" + side if settings.get("atomicity") == "high" else side + "_claims"
        )
        if claims:
            replies.append(judgments(flags))
            prompts.append(("verify_one_" if len(claims) == 1 else "verify_") + side)
    score = await replay(
        client, "factual_correctness", FACTS, replies, prompts, **settings
    )
    if isinstance(expected, str):
        assert (score.status, score.error.code) == ("undefined", expected)
    else:
        assert (score.status, score.value) == ("success", expected)


NOISE_INPUT = dict(FACTS, user_input="question", retrieved_contexts=PASSAGES)
NOISE_ORDER = [
    "noise_gold",
    "noise_answer",
    "first_gold",
    "first_answer",
    "second_gold",
    "second_answer",
    "gold_answer",
]
NOISE_REPLIES = [
    {"statements": ["gold"]},
    {"statements": ["A", "B", "C", "D"]},
    judgments([1]),
    judgments([1, 1, 1, 0]),
    judgments([0]),
    judgments([0, 0, 1, 1]),
    judgments([1, 0, 0, 0]),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "value"), [("relevant", 2 / 4), ("irrelevant", 1 / 4)]
)
async def test_noise_sensitivity_counts_incorrect_statements_once(client, mode, value):
    result = await replay(
        client, "noise_sensitivity", NOISE_INPUT, NOISE_REPLIES, NOISE_ORDER, mode=mode
    )
    assert (result.status, result.value) == ("success", value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("passages", "replies", "category", "code"),
    [
        ([], [], "validation_error", None),
        (PASSAGES, [{"statements": []}], "undefined_score", "no_reference_statements"),
        (
            PASSAGES,
            [NOISE_REPLIES[0], {"statements": []}],
            "undefined_score",
            "no_response_statements",
        ),
        (
            PASSAGES,
            [*NOISE_REPLIES[:5], judgments([0, 0])],
            "model_output_error",
            "verdict_count_mismatch",
        ),
    ],
)
async def test_noise_sensitivity_stops_at_first_unusable_response(
    client, passages, replies, category, code
):
    row = dict(NOISE_INPUT, retrieved_contexts=passages)
    score = await replay(
        client, "noise_sensitivity", row, replies, NOISE_ORDER[: len(replies)]
    )
    assert (score.error.category, score.error.code) == (category, code)
    assert score.status == ("undefined" if category == "undefined_score" else "error")


DUAL_JUDGES = [
    (
        "answer_accuracy",
        dict(FACTS, user_input="question"),
        (4, 2),
        ["accuracy_first", "accuracy_swapped"],
    ),
    (
        "context_relevance",
        {"user_input": "a long question", "retrieved_contexts": PASSAGES},
        (2, 1),
        ["relevance_first", "relevance_second"],
    ),
    (
        "response_groundedness",
        {"response": "observed answer", "retrieved_contexts": PASSAGES},
        (2, 0),
        ["groundedness_first", "groundedness_second"],
    ),
]


@pytest.mark.parametrize(("metric", "row", "grades", "order"), DUAL_JUDGES)
@pytest.mark.parametrize(
    "case",
    [
        "two_ratings",
        "retry_then_valid",
        "only_first_valid",
        "only_second_valid",
        "both_invalid",
    ],
)
@pytest.mark.asyncio
async def test_dual_judges_require_two_in_scale_ratings(
    client, metric, row, grades, order, case
):
    high, other = grades
    # Feed the whole archived script, including valid grades after a failed round.
    ratings = {
        "two_ratings": [high, other],
        "retry_then_valid": [9, high, other],
        "only_first_valid": [high, 9, 9, 9, 9, 9],
        "only_second_valid": [9, 9, 9, 9, 9, high],
        "both_invalid": [9] * 10,
    }[case]
    consumed = order if ratings[0] == high else order[:1]
    score = await replay(
        client, metric, row, [{"rating": r} for r in ratings], consumed
    )
    if case == "two_ratings":
        assert (score.status, score.value) == ("success", (high + other) / (2 * high))
    else:
        assert (score.status, score.error.category, score.error.code) == (
            "error",
            "model_output_error",
            "invalid_rating",
        )


@pytest.mark.asyncio
async def test_accuracy_structured_retry_uses_the_model_budget(client):
    metric, row, _, order = DUAL_JUDGES[0]
    score = await replay(
        client,
        metric,
        row,
        [{}, {"rating": 4}, {"rating": 2}],
        [order[0], *order],
        retries=1,
    )
    retry = client.calls[1][1]["messages"]
    assert [m["role"] for m in retry] == ["user", "assistant", "user"]
    assert (score.status, score.value) == ("success", 0.75)


@pytest.mark.parametrize(
    ("metric", "row", "prompts", "invalid_field"),
    [
        (
            "answer_accuracy",
            dict(FACTS, user_input="question", response=""),
            [],
            "response",
        ),
        (
            "answer_accuracy",
            dict(FACTS, user_input="question", response=" "),
            ["accuracy_blank_first", "accuracy_blank_swapped"],
            None,
        ),
        (
            "context_relevance",
            {"user_input": "a long question", "retrieved_contexts": ["long"]},
            [],
            None,
        ),
        (
            "context_relevance",
            {
                "user_input": "a long question",
                "retrieved_contexts": [" a long question "],
            },
            [],
            None,
        ),
        (
            "context_relevance",
            {"user_input": "a long question", "retrieved_contexts": [" "]},
            [],
            None,
        ),
        (
            "context_relevance",
            {"user_input": "a long question", "retrieved_contexts": []},
            [],
            "retrieved_contexts",
        ),
        (
            "response_groundedness",
            {"response": " ", "retrieved_contexts": PASSAGES},
            [],
            None,
        ),
        (
            "response_groundedness",
            {"response": "observed answer", "retrieved_contexts": [" "]},
            [],
            None,
        ),
        (
            "response_groundedness",
            {"response": "observed answer", "retrieved_contexts": []},
            [],
            "retrieved_contexts",
        ),
    ],
)
@pytest.mark.asyncio
async def test_dual_judges_preserve_blank_and_substring_branches(
    client, metric, row, prompts, invalid_field
):
    score = await replay(client, metric, row, [{"rating": 0} for _ in prompts], prompts)
    if invalid_field:
        assert (score.status, score.error.category, score.error.field) == (
            "error",
            "validation_error",
            invalid_field,
        )
    else:
        assert (score.status, score.value) == ("success", 0.0)


@pytest.mark.asyncio
async def test_granularity_examples_are_bound_to_each_metric_instance(client):
    llm = create_language_model(
        "model.openai_compatible.chat_completions", client=client, model="offline"
    )
    variants = [
        ("low", "low", "answer_claims"),
        ("low", "high", "low_high"),
        ("high", "low", "high_low"),
        ("high", "high", "fine_answer"),
    ]
    metrics = [
        create_metric(
            "metric.rag.factual_correctness",
            llm=llm,
            mode="precision",
            atomicity=atomicity,
            coverage=coverage,
        )
        for atomicity, coverage, _ in variants
    ]
    for metric, (atomicity, coverage, name) in zip(metrics, variants, strict=True):
        client.chat_reply(json.dumps({"claims": ANSWER_CLAIMS[0]}))
        client.chat_reply(json.dumps(judgments(ANSWER_CLAIMS[1])))
        value = await metric.ascore(**FACTS)
        assert value.value == 0.67
        assert_archived_prompt(client.calls[-2][1]["messages"][0]["content"], name)
        snapshot = metric.config_snapshot()
        assert (snapshot["atomicity"], snapshot["coverage"]) == (atomicity, coverage)

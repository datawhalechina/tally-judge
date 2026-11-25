"""Model-boundary replay of Ragas ec00339 using generate_prompt_baselines.py."""

import hashlib
import json
import sys
from types import SimpleNamespace
from uuid import UUID

import pydantic
import pytest

from tallyjudge import Dataset, Evaluator, create_metric
from tallyjudge.plugins.synthesis.graph import (
    EvidenceGraph,
    build_entity_graph,
    build_graph,
)
from tallyjudge.plugins.synthesis.scenarios import (
    MultiHopScenario,
    SingleHopScenario,
    generate_sample,
)
from tallyjudge.sdk.models import GenerationResult
from tallyjudge.sdk.usage import UsageSummary

ARCHIVED = {
    "single": [
        (
            "1dadaa521afb0b97713d85b12d15887581e6ebd89744f3655435d8322336448d",
            "6ec0bea5e8d80066a04f4eb250c8f63c7a684ae4c87b2fde05a312b87ead2ad4",
            "GeneratedQueryAnswer",
        )
    ],
    "multi": [
        (
            "d24198902f5fd5efff398190835bff329573893dab41d75d71ee4593fddd6a88",
            "a39a65209487da8d80ac2005ecdd8af14c71245ff9f5a7d7546264c29b4729e5",
            "GeneratedQueryAnswer",
        )
    ],
    "multi_order": [
        (
            "1b2e517b64817cb89f7ba3b53060162c869ddf3a8e37be9977deb27226ba482a",
            "0939e49199da479c4a3766e20ad6f6cdd91d9e1596d480dfa420ee00ecc22468",
            "GeneratedQueryAnswer",
        )
    ],
    "ner": [
        (
            "a1d66bf94f045afc6cba9f8bcaf32b87885352e3543f70925d8d19d5fe817d05",
            "fb803324a26ccd1166ae4105a44bfd28c1a950445bc271c9bf7782048c3edb23",
            "NEROutput",
        )
    ],
    "ner_windows": [
        (
            "dc609a0d9e8cca551813fe6ad48c7acba61ee64bd77f18380370b2984d696f8b",
            "8bc22fcf20aea0a56fb1d7809336fe4c866973240cf1dc723fe0f8c656a299d3",
            "NEROutput",
        ),
        (
            "b77349aa9c85c38824ddcac8ecd698d571ff585e8d82571eed0d5e19c3487df7",
            "9f73393d3527c4aa8fc9095ff37933bae301ea20910064e922098de09bc6e4c5",
            "NEROutput",
        ),
        (
            "58c52f43eadec78b65ee20b851fe4f04377747b87c44e34300022633acb22847",
            "f3e862d199ebe865e5e0a7d56328936cc6f2535989beaf1ec7cb020a8aa5a6cb",
            "NEROutput",
        ),
    ],
    "rubric_free": [
        (
            "526054f08295a5d0f7abc1910adf8141f9f4bb849920b4085c01d475bf2e83af",
            "e8d8dde4c5ae5bc7590a5cfa8005a4590ebca05560037762b21f91c306e089f1",
            "RubricScoreOutput",
        )
    ],
    "rubric_reference": [
        (
            "98fe7a891c06fdee927fba89f7c9e96763bd3e188c7ac10ea96d1231505cdd78",
            "b086a46c317c78f75b4dd07167d327fe47d4b0a36ccc276004c5f025fd29b6be",
            "RubricScoreOutput",
        )
    ],
    "rubric_custom": [
        (
            "9123cadc821c2d82468f1ee4d304dd23f6102f461eba85b1561431d5d7351299",
            "c38d897b57eb6fa982f809b2eca965043d292033c83c73d02721907a6913cb2a",
            "RubricScoreOutput",
        )
    ],
    "rubric_instance": [
        (
            "222cd9622ce427748abd4d9f37cbf9efae2de487b58aaa71c10cad4e8259fbee",
            "bc85c0b17bb0d07fe5d5df0e2bf02c7e11b31d6dee79ae28ebbcaa7d15ed196d",
            "InstanceRubricScoreOutput",
        )
    ],
}


class ScriptedModel:
    plugin_id = "model.fixture.archive"
    usage_tracking = "untracked"

    def __init__(self, case, responses):
        self.expected = iter(ARCHIVED[case])
        self.responses = iter(responses)

    def config_snapshot(self):
        return {}

    async def agenerate(self, prompt, response_model):
        raw, normalized, schema_name = next(self.expected)
        [message] = prompt.messages
        assert message.role == "user" and response_model.__name__ == schema_name
        text = message.content
        if pydantic.__version__ == "2.14.3":
            assert hashlib.sha256(text.encode()).hexdigest() == raw
        prefix, rest = text.split("as specified in JSON Schema:\n", 1)
        schema, suffix = rest.split("Do not use single quotes", 1)
        content = json.dumps(
            [prefix, json.loads(schema), suffix],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        assert hashlib.sha256(content.encode()).hexdigest() == normalized
        data = response_model.model_validate_json(json.dumps(next(self.responses)))
        return GenerationResult(data, UsageSummary())

    def assert_consumed(self):
        assert list(self.expected) == [] and list(self.responses) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case,indices,style,length,guidance",
    [
        ("single", [1], "Perfect grammar", "medium", None),
        ("multi", [1, 2], "Perfect grammar", "medium", None),
        ("multi_order", [3, 2, 1], "Web search like queries", "short", "Compare"),
    ],
)
async def test_scenario_replays_archived_prompt_and_sample(
    case, indices, style, length, guidance
):
    texts = ["Ada designed Kit.", "Kit is in Rome.", "Rye costs five coins."]
    nodes = [
        {
            "id": str(UUID(int=i)),
            "type": "chunk",
            "properties": {
                "page_content": text,
                "document_metadata": {"source": f"file-{i}"},
            },
        }
        for i, text in enumerate(texts, 1)
    ]
    graph = EvidenceGraph.from_ragas({"nodes": nodes, "relationships": []})
    scene = {
        "nodes": [nodes[i - 1] for i in indices],
        "persona": {"name": "Librarian", "role_description": "Checks provenance"},
        "style": style,
        "length": length,
    }
    if case == "single":
        scenario = SingleHopScenario.from_ragas({**scene, "term": "Kit"}, graph)
    else:
        scenario = MultiHopScenario.from_ragas(
            {**scene, "combinations": ["Rome", "Kit"] if guidance else ["Kit"]}, graph
        )
    model = ScriptedModel(case, [{"query": "Who designed Kit?", "answer": "Ada"}])
    sample = await generate_sample(scenario, llm=model, llm_context=guidance)
    contexts = [texts[i - 1] for i in indices]
    if case != "single":
        contexts = [f"<{i}-hop>\n\n{text}" for i, text in enumerate(contexts, 1)]
    assert sample == {
        "user_input": "Who designed Kit?",
        "reference": "Ada",
        "reference_contexts": contexts,
        "persona_name": "Librarian",
        "query_style": "WEB_SEARCH_LIKE" if guidance else "PERFECT_GRAMMAR",
        "query_length": length.upper(),
        "reference_context_ids": [node["id"] for node in scene["nodes"]],
        "reference_document_ids": [f"file-{i}" for i in indices],
    }
    model.assert_consumed()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case,text,limit,responses",
    [
        ("ner", "Ada designed Kit.", 32000, [["Ada", "Kit", "Alias"]]),
        ("ner_windows", "1234567", 3, [["x"], ["x", "y"], ["z"]]),
    ],
)
async def test_graph_ner_replays_windows_and_preserves_labels(
    case, text, limit, responses, monkeypatch
):
    class ByteEncoding:
        def encode(self, value, *, disallowed_special):
            assert disallowed_special == ()
            return list(value.encode("ascii"))

        def decode(self, tokens):
            return bytes(tokens).decode("ascii")

    def get_encoding(name):
        assert name == "o200k_base"
        return ByteEncoding()

    monkeypatch.setitem(
        sys.modules, "tiktoken", SimpleNamespace(get_encoding=get_encoding)
    )
    model = ScriptedModel(case, [{"entities": batch} for batch in responses])
    graph = await build_entity_graph(
        [{"chunk_id": "a", "document_id": "source", "text": text}],
        llm=model,
        max_token_limit=limit,
    )
    assert graph.chunks[0]["entities"] == [
        label for batch in responses for label in batch
    ]
    model.assert_consumed()


def test_archived_overlap_removes_frequent_label_and_keeps_fuzzy_match():
    pytest.importorskip("rapidfuzz")
    entities = [["Guide", "Martha"], ["Guide", "Marhta"], ["Guide", "Rye"]]
    chunks = [
        {
            "chunk_id": str(i),
            "document_id": "guide",
            "text": "source",
            "entities": labels,
        }
        for i, labels in enumerate(entities, 1)
    ]
    assert build_graph(chunks).relations == (("1", "2"),)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case,kind,settings,inputs",
    [
        (
            "rubric_free",
            "rubric_without_reference",
            {},
            {"user_input": "Who?", "response": "Ada"},
        ),
        (
            "rubric_reference",
            "rubric_with_reference",
            {},
            {"user_input": "Who?", "response": "Ada", "reference": "Ada"},
        ),
        (
            "rubric_custom",
            "rubric_without_reference",
            {"rubrics": {"second": "B", "first": "A"}},
            {"response": "Ada"},
        ),
        (
            "rubric_instance",
            "instance_rubric",
            {},
            {"response": "Ada", "rubrics": {"second": "B", "first": "A"}},
        ),
    ],
)
async def test_rubric_replays_archived_prompt_score_and_feedback(
    case, kind, settings, inputs
):
    model = ScriptedModel(case, [{"feedback": "Archived feedback", "score": "4"}])
    metric = create_metric("metric.custom." + kind, llm=model, **settings)
    run = await Evaluator([metric]).arun(Dataset.from_records([inputs]))
    result = run.rows[0].metrics[metric.spec.id]
    assert (result.status, result.value, result.reason) == (
        "success",
        4.0,
        "Archived feedback",
    )
    model.assert_consumed()

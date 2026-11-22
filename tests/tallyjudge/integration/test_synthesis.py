"""Generate test sets through the public API and score the application against them."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from tallyjudge import (
    Dataset,
    Evaluator,
    MetricBinding,
    SynthesisCollector,
    create_language_model,
    create_metric,
    create_synthesizer,
    experiment,
)
from tallyjudge.core.errors import ClientLoopError, PluginError
from tallyjudge.core.prompts import PromptInput

KEY = "synthesis.testset.single_hop"
HOURS = {"document_id": "library", "chunk_id": "hours", "text": "图书馆周一闭馆。"}
LOANS = {"document_id": "service", "chunk_id": "loans", "text": "图书借期为14天。"}
QA_HOURS = {"query": "图书馆哪天闭馆？", "answer": "周一"}
QA_LOANS = {"query": "图书借期多长？", "answer": "14天"}
TOKENS = SimpleNamespace(prompt_tokens=7, completion_tokens=3)


def synthesizer(client, *, retries=0, seed=7):
    judge = create_language_model(
        "model.openai_compatible.chat_completions",
        client=client,
        model="fixture",
        temperature=0.0,
        structured_retries=retries,
    )
    return create_synthesizer(KEY, llm=judge, seed=seed)


def reply(client, pair, tokens=TOKENS):
    client.chat_reply(json.dumps(pair, ensure_ascii=False), tokens=tokens)


@pytest.mark.asyncio
async def test_review_a_synthetic_reference_against_its_own_evidence(client):
    llm = create_language_model(
        "model.openai_compatible.chat_completions",
        client=client,
        model="fixture",
        temperature=0.0,
    )
    client.chat_reply(json.dumps({"query": "Where is Kit?", "answer": "Paris"}))
    generated = await create_synthesizer(KEY, llm=llm).arun(
        [{"document_id": "guide", "chunk_id": "place", "text": "Kit is kept in Rome."}],
        testset_size=1,
    )
    assert generated.generated_count == generated.usage.generation.requests == 1

    binding = MetricBinding(
        create_metric("metric.rag.faithfulness", llm=llm),
        input_map={"response": "reference", "retrieved_contexts": "reference_contexts"},
    )
    client.chat_reply('{"statements":["Kit is kept in Paris."]}')
    client.chat_reply(
        json.dumps(
            {
                "statements": [
                    {
                        "statement": "Kit is kept in Paris.",
                        "reason": "The passage says Rome.",
                        "verdict": 0,
                    }
                ]
            }
        )
    )
    review = await Evaluator([binding]).arun(generated.to_dataset())
    score = review.row_for(generated.candidates[0].case_id).metrics[binding.key]
    assert score.value == 0.0 and score.usage.generation.requests == 2
    assert len(client.calls) == 3


@pytest.mark.asyncio
async def test_generated_jsonl_feeds_two_experiments(client, tmp_path):
    for pair in (QA_HOURS, QA_LOANS):
        reply(client, pair)
    result = await synthesizer(client).arun([HOURS, LOANS], testset_size=3)
    assert (result.testset_size, result.attempted_count, result.generated_count) == (
        3,
        2,
        2,
    )
    assert [c.status for c in result.candidates] == ["success", "success"]
    assert [c.chunk_id for c in result.candidates] == ["hours", "loans"]
    assert result.usage.generation.requests == len(client.calls) == 2
    assert result.usage.generation.known_tokens == {
        "input_tokens": 14,
        "output_tokens": 6,
    }
    assert result.usage.generation.complete and result.usage.embedding.requests == 0
    assert [
        json.loads(
            call[1]["messages"][0]["content"]
            .rsplit("input: ", 1)[1]
            .removesuffix("\nOutput: ")
        )["context"]
        for call in client.calls
    ] == [
        HOURS["text"],
        LOANS["text"],
    ]
    dataset = result.to_dataset()
    path = tmp_path / "samples.jsonl"
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in result.to_records()
        ),
        encoding="utf-8",
    )
    restored = Dataset.from_records(
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    )
    assert restored.manifest == dataset.manifest
    assert len({row["case_id"] for row in restored.records}) == 2
    assert [
        (row["reference_context_ids"], row["reference_document_ids"])
        for row in restored.records
    ] == [(["hours"], ["library"]), (["loans"], ["service"])]
    detached = result.to_records()
    detached[0]["reference_contexts"].clear()
    assert result.to_dataset().records == restored.records

    @experiment
    def original(row):
        return {
            "response": {"hours": "周一", "loans": "7天"}[
                row["reference_context_ids"][0]
            ]
        }

    @experiment
    def revised(row):
        return {
            "response": {"hours": "周一", "loans": "14天"}[
                row["reference_context_ids"][0]
            ]
        }

    metric = create_metric("metric.text.exact_match")
    first = await original.arun(restored, metrics=[metric])
    second = await revised.arun(restored, metrics=[metric])
    assert first.summary[metric.spec.id]["mean"] == 0.5
    assert second.summary[metric.spec.id]["mean"] == 1.0
    assert (
        first.summary[metric.spec.id]["coverage"]
        == second.summary[metric.spec.id]["coverage"]
        == 1.0
    )
    assert len(client.calls) == 2


@pytest.mark.parametrize(
    "bad,code",
    [
        ("not JSON", "invalid_json"),
        (
            json.dumps({"query": "missing answer"}),
            "schema_validation",
        ),
        (json.dumps(dict(QA_LOANS, answer=42)), "schema_validation"),
    ],
)
def test_invalid_second_output_records_source_without_replacement(client, bad, code):
    reply(client, QA_HOURS)
    client.chat_reply(bad)
    result = synthesizer(client).run([HOURS, LOANS], testset_size=3)
    assert (result.testset_size, result.attempted_count, result.generated_count) == (
        3,
        2,
        1,
    )
    failed = result.candidates[1]
    assert (failed.chunk_id, failed.document_id, failed.status) == (
        "loans",
        "service",
        "error",
    )
    assert (failed.error.category, failed.error.code) == ("model_output_error", code)
    assert failed.usage.generation.requests_without_usage == 1
    assert (
        result.usage.generation.requests == 2 and not result.usage.generation.complete
    )
    assert result.to_dataset().records[0]["reference"] == QA_HOURS["answer"]
    assert len(result.to_dataset()) == 1


def test_transport_failure_then_structured_retry_has_separate_usage(client):
    client.responses.append(OSError("private transport details"))
    client.chat_reply("invalid", tokens=TOKENS)
    reply(client, QA_LOANS)
    result = synthesizer(client, retries=1).run([HOURS, LOANS], testset_size=2)
    failed, success = result.candidates
    assert (failed.status, failed.error.category, failed.error.exception_type) == (
        "error",
        "model_error",
        "OSError",
    )
    assert failed.usage.generation.requests == 1
    assert success.status == "success" and success.usage.generation.requests == 2
    assert len(client.calls) == result.usage.generation.requests == 3
    assert result.usage.generation.known_tokens == {
        "input_tokens": 14,
        "output_tokens": 6,
    }
    assert "private transport details" not in repr(failed)


@pytest.mark.asyncio
async def test_cancel_while_second_request_is_in_flight_preserves_partial_result(
    client,
):
    started, release, late = asyncio.Event(), asyncio.Event(), asyncio.Event()
    reply(client, QA_HOURS)
    client.chat_reply("invalid after cancellation")
    late_response = client.responses.pop()

    async def delayed():
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        late.set()
        return late_response

    client.responses.append(delayed)
    collector = SynthesisCollector()
    generator = synthesizer(client, retries=1, seed=5)
    third = dict(LOANS, chunk_id="third")
    task = asyncio.create_task(
        generator.arun([HOURS, LOANS, third], testset_size=3, collector=collector)
    )
    try:
        async with asyncio.timeout(4):
            await started.wait()
            live = collector.snapshot()
            assert [(c.chunk_id, c.status) for c in live.candidates] == [
                ("hours", "success"),
                ("loans", "running"),
            ]
            assert live.usage.generation.in_flight_requests == 1
            live.to_records()[0].clear()
            with pytest.raises(ValueError, match="already collecting"):
                await generator.arun([], testset_size=0, collector=collector)
            task.cancel("operator")
            with pytest.raises(asyncio.CancelledError) as cancelled:
                await task
            assert cancelled.value.args == ("operator",)
            saved = collector.snapshot()
            assert [c.status for c in saved.candidates] == ["success", "cancelled"]
            assert (
                saved.attempted_count == 2
                and saved.generated_count == len(saved.to_records()) == 1
            )
            assert (
                saved.usage.generation.requests == 2
                and saved.usage.generation.in_flight_requests == 1
            )
            assert saved.usage.generation.known_tokens == {
                "input_tokens": 7,
                "output_tokens": 3,
            }
            release.set()
            await late.wait()
            await asyncio.sleep(0)
            assert collector.snapshot() == saved and len(client.calls) == 2
            clean = await generator.arun([], testset_size=2, collector=collector)
            assert clean == collector.snapshot() and clean.attempted_count == 0
    finally:
        release.set()
        if not task.done():
            task.cancel()


@pytest.mark.parametrize(
    "chunks,size,match",
    [
        ([HOURS, dict(LOANS, chunk_id="hours")], 1, "Duplicate chunk_id"),
        ([HOURS, dict(LOANS, text="  ")], 2, "text"),
        ([dict(HOURS, document_id="")], 1, "document_id"),
        ([dict(HOURS, chunk_id=False)], 1, "chunk_id"),
        ([{"chunk_id": "one", "text": "excerpt"}], 1, "document_id"),
        ([], -1, "testset_size"),
        ([], True, "testset_size"),
        ([], 2.0, "testset_size"),
    ],
)
def test_invalid_input_fails_before_a_request(client, chunks, size, match):
    with pytest.raises(ValueError, match=match):
        synthesizer(client).run(chunks, testset_size=size)
    assert not client.calls


def test_no_candidates_and_registry_errors(client):
    def unused():
        raise AssertionError("The generator consumed chunks when zero were requested")
        yield

    generator = synthesizer(client)
    for chunks, size in ((unused(), 0), ([], 4)):
        result = generator.run(chunks, testset_size=size)
        assert (
            result.attempted_count
            == result.generated_count
            == result.usage.generation.requests
            == 0
        )
        assert result.to_dataset().records == []
    with pytest.raises(PluginError, match="Missing dependency"):
        create_synthesizer(KEY)
    with pytest.raises(PluginError, match="Unexpected plugin kind"):
        create_synthesizer("metric.text.exact_match")
    with pytest.raises(PluginError, match="seed"):
        synthesizer(client, seed=True)
    assert not client.calls


def test_client_loop_error_aborts_once_and_remains_visible(client):
    generator = synthesizer(client)
    reply(client, QA_HOURS)
    generator.run([HOURS], testset_size=1)
    collector = SynthesisCollector()
    with pytest.raises(ClientLoopError):
        generator.run([HOURS, LOANS], testset_size=2, collector=collector)
    snapshot = collector.snapshot()
    assert snapshot.attempted_count == len(client.calls) == 1
    assert snapshot.candidates[0].error.exception_type == "ClientLoopError"


@pytest.mark.asyncio
async def test_generation_application_and_faithfulness_account_for_their_own_requests(
    client,
):
    class AppReply(BaseModel):
        answer: str

    llm = create_language_model(
        "model.openai_compatible.chat_completions",
        client=client,
        model="fixture",
        temperature=0.0,
    )
    reply(client, QA_HOURS)
    generated = await create_synthesizer(KEY, llm=llm).arun([HOURS], testset_size=1)

    @experiment
    async def library(row):
        passage = "图书馆周一闭馆。"
        prompt = PromptInput.from_text(row["user_input"] + "\n" + passage)
        response = await llm.agenerate(prompt, AppReply)
        return {"response": response.data.answer, "retrieved_contexts": [passage]}

    reply(client, {"answer": "图书馆周一闭馆。"})
    reply(client, {"statements": ["图书馆周一闭馆。"]})
    reply(
        client,
        {
            "statements": [
                {"statement": "图书馆周一闭馆。", "reason": "见公告", "verdict": 1}
            ]
        },
    )
    faithfulness = create_metric("metric.rag.faithfulness", llm=llm)
    run = await library.arun(generated.to_dataset(), metrics=[faithfulness])
    row = run.row_for(generated.candidates[0].case_id)
    score = row.metrics[faithfulness.spec.id]
    assert score.value == 1.0 and run.summary[faithfulness.spec.id]["coverage"] == 1.0
    assert (
        generated.usage.generation.requests,
        row.task.usage.generation.requests,
        score.usage.generation.requests,
    ) == (1, 1, 2)
    assert len(client.calls) == 4


@pytest.mark.asyncio
async def test_broken_model_aborts_and_snapshot_has_a_terminal_state(
    client, monkeypatch
):
    async def broken(prompt, response_model):
        return SimpleNamespace(data=None)

    llm = create_language_model(
        "model.openai_compatible.chat_completions",
        client=client,
        model="fixture",
        temperature=0.0,
    )
    monkeypatch.setattr(llm, "agenerate", broken)

    collector = SynthesisCollector()
    generator = create_synthesizer(KEY, llm=llm)
    with pytest.raises(AttributeError):
        await generator.arun(
            [{"chunk_id": "hours", "document_id": "library", "text": "x"}],
            testset_size=1,
            collector=collector,
        )
    result = collector.snapshot()
    [failed] = result.candidates
    assert (failed.status, failed.error.category, failed.error.exception_type) == (
        "error",
        "internal_error",
        "AttributeError",
    )
    assert result.generated_count == 0 and result.to_records() == []


@pytest.mark.asyncio
async def test_sync_entry_inside_loop_points_to_arun(client):
    with pytest.raises(RuntimeError, match="arun"):
        synthesizer(client).run([], testset_size=0)


@pytest.mark.asyncio
async def test_sample_id_tracks_the_source_and_the_term_in_the_prompt(client):
    model = create_language_model(
        "model.openai_compatible.chat_completions", client=client, model="fixture"
    )
    generator = create_synthesizer("synthesis.testset.single_hop", llm=model)
    ids = []
    for text, labels in [
        ("Ada built Kit.", ["Ada"]),
        ("Ada built Kit.", ["Ada", "Kit"]),
        ("Ada built Kit.", ["Kit"]),
        ("Ada sold Kit.", ["Kit"]),
    ]:
        client.chat_reply('{"query":"Who?","answer":"Ada"}')
        result = await generator.arun(
            [
                {
                    "chunk_id": "a",
                    "document_id": "guide",
                    "text": text,
                    "entities": labels,
                }
            ],
            testset_size=1,
        )
        ids.append(result.to_records()[0]["case_id"])
    assert ids[0] == ids[1]
    assert len(set(ids)) == 3

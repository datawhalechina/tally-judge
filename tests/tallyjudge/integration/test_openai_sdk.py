"""Exercise the real SDK against an in-process HTTP transport. No endpoint or key is used."""

import asyncio
import json

import pytest

from tallyjudge import (
    Dataset,
    Evaluator,
    RunCollector,
    create_embedding_model,
    create_language_model,
    create_metric,
)
from tallyjudge.core.errors import ClientLoopError, ModelError
from tallyjudge.core.prompts import PromptInput

httpx = pytest.importorskip("httpx")
openai = pytest.importorskip("openai")

CHAT = "model.openai_compatible.chat_completions"
EMBED = "embedding.openai_compatible.embeddings"
FAITH = "metric.rag.faithfulness"
ROW = {"user_input": "q", "response": "a", "retrieved_contexts": ["evidence"]}


def sdk(handler):
    return openai.AsyncOpenAI(
        api_key="test-secret",
        base_url="https://test.invalid/v1",
        max_retries=3,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def completion(content):
    return httpx.Response(
        200,
        json={
            "id": "id",
            "object": "chat.completion",
            "created": 1,
            "model": "served",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        },
    )


@pytest.mark.asyncio
async def test_sdk_request_bodies_order_and_client_ownership():
    sent = []
    outputs = iter(
        [
            "bad",
            '{"statements":["one"]}',
            '{"statements":[{"statement":"one","reason":"ok","verdict":1}]}',
        ]
    )

    def handler(request):
        body = json.loads(request.content)
        sent.append((request.url.path, body))
        if request.url.path.endswith("embeddings"):
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": "served",
                    "data": [
                        {
                            "object": "embedding",
                            "index": i,
                            "embedding": [float(i), 1.0],
                        }
                        for i in reversed(range(len(body["input"])))
                    ],
                    "usage": {"prompt_tokens": 2, "total_tokens": 2},
                },
            )
        return completion(next(outputs))

    async with sdk(handler) as client:
        judge = create_language_model(
            CHAT, client=client, model="judge", temperature=0.01, request_timeout=3.0
        )
        emb = create_embedding_model(EMBED, client=client, model="vec", dimensions=2)
        m = create_metric(FAITH, llm=judge)
        r = await Evaluator([m]).arun(Dataset.from_records([ROW]))
        score = r.rows[0].metrics[FAITH]
        assert score.value == 1.0 and score.usage.generation.requests == 3
        assert score.usage.generation.known_tokens == {
            "input_tokens": 9,
            "output_tokens": 3,
        }
        assert sent[0][0] == "/v1/chat/completions" and sent[0][1][
            "response_format"
        ] == {"type": "json_object"}
        assert sent[0][1]["temperature"] == 0.01 and len(sent[1][1]["messages"]) == 3
        for _ in range(2):
            out = await emb.aembed_texts(["A", "B"])
            assert out.vectors == [[0.0, 1.0], [1.0, 1.0]]
            assert out.usage.known_tokens["input_tokens"] == 2 and out.model == "served"
        assert sent[-1] == (
            "/v1/embeddings",
            {
                "input": ["A", "B"],
                "model": "vec",
                "dimensions": 2,
                "encoding_format": "float",
            },
        )
        assert not client.is_closed()
        assert "test-secret" not in r.to_json()
        with pytest.raises(ClientLoopError, match="another event loop"):
            await asyncio.to_thread(asyncio.run, emb.aembed_texts(["other"]))
        assert len(sent) == 5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,original", [(429, "RateLimitError"), (500, "InternalServerError")]
)
@pytest.mark.parametrize("kind", ["chat", "embedding"])
async def test_sdk_transport_errors_have_one_attempt_and_cause(status, original, kind):
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(
            status, json={"error": {"message": "private service message"}}
        )

    async with sdk(handler) as client:
        if kind == "chat":
            from tallyjudge.plugins.metrics.rag.faithfulness import (
                StatementGeneratorOutput,
            )

            model = create_language_model(CHAT, client=client, model="test")
            call = model.agenerate(PromptInput.from_text("q"), StatementGeneratorOutput)
        else:
            call = create_embedding_model(
                EMBED, client=client, model="test"
            ).aembed_texts(["q"])
        with pytest.raises(ModelError) as caught:
            await call
        assert type(caught.value.__cause__).__name__ == original
        assert len(attempts) == 1


@pytest.mark.asyncio
async def test_sdk_cancel_then_reuse_on_same_loop():
    entered = asyncio.Event()
    count = 0

    async def handler(request):
        nonlocal count
        count += 1
        if count == 1:
            entered.set()
            await asyncio.Future()
        return completion('{"statements":[]}')

    async with sdk(handler) as client:
        metric = create_metric(
            FAITH, llm=create_language_model(CHAT, client=client, model="test")
        )
        evaluator = Evaluator([metric])
        dataset = Dataset.from_records([ROW])
        collector = RunCollector()
        task = asyncio.create_task(evaluator.arun(dataset, collector=collector))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        score = collector.snapshot().rows[0].metrics[FAITH]
        assert (
            score.status == "cancelled"
            and score.usage.generation.in_flight_requests == 1
        )
        next_run = await evaluator.arun(dataset)
        assert next_run.rows[0].metrics[FAITH].error.code == "no_statements"
        assert count == 2 and not client.is_closed()


@pytest.mark.asyncio
async def test_multi_hop_request_accepts_plain_evidence_in_json_mode():
    from tallyjudge import create_synthesizer
    from tallyjudge.plugins.synthesis.graph import EvidenceGraph

    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        assert body["response_format"] == {"type": "json_object"}
        assert "evidence_quotes" not in body["messages"][0]["content"]
        if not any("json" in m["content"].lower() for m in body["messages"]):
            return httpx.Response(
                400, json={"error": {"message": "JSON instruction required"}}
            )
        return completion(
            json.dumps(
                {
                    "query": "Where does Ada's creation live?",
                    "answer": "Rome",
                }
            )
        )

    graph = EvidenceGraph(
        (
            {"chunk_id": "a", "document_id": "bio", "text": "Ada built Kit."},
            {"chunk_id": "b", "document_id": "place", "text": "Kit lives in Rome."},
        ),
        (("a", "b"),),
    )
    async with sdk(handler) as client:
        llm = create_language_model(CHAT, client=client, model="judge")
        result = await create_synthesizer("synthesis.testset.multi_hop", llm=llm).arun(
            graph, testset_size=1
        )
    assert result.generated_count == result.usage.generation.requests == len(calls) == 1
    assert result.to_records()[0]["reference_context_ids"] == ["a", "b"]

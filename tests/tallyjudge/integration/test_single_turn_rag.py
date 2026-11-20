"""End-to-end contracts for the fifteen single-turn RAG metrics."""

import json

import pytest

from tallyjudge import Dataset, Evaluator, create_metric, create_storage_backend
from tallyjudge.core.errors import PluginError
from tallyjudge.core.usage import Usage, UsageSummary, record_request
from tallyjudge.interfaces.models import EmbeddingResult, GenerationResult
from tallyjudge.plugins.metrics.rag.answer_correctness import AnswerCorrectnessMetric
from tallyjudge.registry import PluginSpec, registry

AC = "metric.rag.answer_correctness"


class Judge:
    plugin_id = "model.test.rag"
    usage_tracking = "per_request"

    def __init__(self):
        self.calls = []

    def config_snapshot(self):
        return {}

    async def agenerate(self, prompt, response_model):
        records = []
        with record_request("generation", records) as request:
            self.calls.append(response_model.__name__)
            request.usage = Usage(3, 1)
            message = prompt.messages[0].content
            data = json.loads(
                message.rsplit("\ninput: ", 1)[1].removesuffix("\nOutput: ")
            )
            output = self._reply(response_model.__name__, data)
        return GenerationResult(
            response_model.model_validate(output),
            UsageSummary.from_records(records),
            self.plugin_id,
        )

    @staticmethod
    def _reply(schema, data):
        match schema:
            case "StatementGeneratorOutput":
                claims = (
                    ["gold fact"]
                    if data["answer"] == "gold"
                    else ["gold fact", "from hit", "both", "from miss"]
                )
                return {"statements": claims}
            case "NLIStatementOutput":
                return {
                    "statements": [
                        {"statement": s, "reason": "test", "verdict": 1}
                        for s in data["statements"]
                    ]
                }
            case "AnswerRelevanceOutput":
                return {"question": "q", "noncommittal": 0}
            case "ContextPrecisionOutput":
                return {"reason": "test", "verdict": int(data["context"] == "hit")}
            case "ContextRecallOutput":
                return {
                    "classifications": [
                        {
                            "statement": "claim",
                            "reason": "test",
                            "attributed": int(x == "hit"),
                        }
                        for x in data["context"].split("\n")
                    ]
                }
            case "ClassificationWithReason":
                return {
                    "TP": [{"statement": "claim", "reason": "test"}],
                    "FP": [],
                    "FN": [],
                }
            case "EntitiesList":
                if data["text"] == "gold":
                    return {"entities": ["Paris", "Seine"]}
                return {
                    "entities": ["Paris"] if "hit" in data["text"].split("\n") else []
                }
            case "ClaimDecompositionOutput":
                return {"claims": ["claim"]}
            case "StatementFaithfulnessOutput":
                support = {
                    "hit": {"gold fact", "from hit", "both"},
                    "miss": {"from miss", "both"},
                    "gold": {"gold fact"},
                }
                return {
                    "statements": [
                        {
                            "statement": s,
                            "reason": "test",
                            "verdict": int(s in support[data["context"]]),
                        }
                        for s in data["statements"]
                    ]
                }
            case "AnswerAccuracyOutput":
                return {"rating": 4 if data["user_answer"] == "answer" else 2}
            case "ContextRelevanceOutput" | "ResponseGroundednessOutput":
                return {"rating": 2}
            case other:
                raise AssertionError(other)


class Vectors:
    plugin_id = "embedding.test.rag"
    usage_tracking = "per_request"

    def config_snapshot(self):
        return {}

    async def aembed_texts(self, texts):
        records = []
        with record_request("embedding", records) as request:
            request.usage = Usage(2, 0)
            vectors = [[1.0, 0.0] for _ in texts]
        return EmbeddingResult(
            vectors, UsageSummary.from_records(records), self.plugin_id
        )


def test_optional_embeddings_follow_correctness_weights_at_construction():
    judge = Judge()
    spec = next(s for s in registry.list(kind="metric") if s.id == AC)
    assert spec.dependencies == {"llm": "language_model"}
    assert spec.optional_dependencies == {"embeddings": "embedding_model"}
    assert registry.describe(AC).value_range == (-0.25, 1.0)
    assert registry.describe(AC, weights=[1.0, 0.0]).value_range == (0.0, 1.0)
    pure = create_metric(AC, llm=judge, weights=[1.0, 0.0])
    assert (
        pure.config_snapshot()
        == AnswerCorrectnessMetric(llm=judge, weights=[1.0, 0.0]).config_snapshot()
    )
    assert (
        pure.config_snapshot()
        == create_metric(
            AC, llm=judge, weights=[1.0, 0.0], embeddings=Vectors()
        ).config_snapshot()
    )
    with pytest.raises(PluginError, match=AC) as missing:
        create_metric(AC, llm=judge)
    assert "embeddings" in str(missing.value.__cause__)
    with pytest.raises(PluginError, match="Invalid embedding_model"):
        create_metric(AC, llm=judge, embeddings=object())
    with pytest.raises(PluginError, match="Missing dependency"):
        create_metric(AC, weights=[0.0, 1.0], embeddings=Vectors())
    with pytest.raises(ValueError, match="both required and optional"):
        PluginSpec(
            AC,
            "metric",
            1,
            AnswerCorrectnessMetric,
            dependencies={"llm": "language_model"},
            optional_dependencies={"llm": "language_model"},
        )


@pytest.mark.parametrize(
    "settings",
    [
        {"weights": []},
        {"weights": [1.0]},
        {"weights": [0.0, 0.0]},
        {"weights": [-0.1, 1.0]},
        {"weights": [1.0, float("inf")]},
        {"beta": 1},
        {"beta": float("nan")},
    ],
)
def test_invalid_correctness_settings_rejected_in_both_public_paths(settings):
    with pytest.raises(PluginError):
        registry.describe(AC, **settings)
    with pytest.raises(PluginError):
        create_metric(AC, llm=Judge(), embeddings=Vectors(), **settings)


@pytest.mark.asyncio
async def test_fifteen_metrics_on_one_dataset_export_scores_failures_and_usage(
    tmp_path,
):
    judge, embeddings = Judge(), Vectors()
    base = {
        "user_input": "q",
        "response": "answer",
        "reference": "gold",
        "reference_context_ids": ["A", "C", "D"],
    }
    rows = [
        dict(
            base,
            case_id="ranked",
            retrieved_contexts=["hit", "miss", "hit"],
            retrieved_context_ids=["B", "A", "A"],
        ),
        dict(
            base,
            case_id="miss",
            retrieved_contexts=["miss"],
            retrieved_context_ids=["absent"],
        ),
        dict(
            base,
            case_id="invalid",
            response=3,
            reference="",
            retrieved_contexts=[],
            retrieved_context_ids=[],
            reference_context_ids=[],
        ),
    ]
    id_precision, id_recall = (
        "metric.retrieval.id_context_precision",
        "metric.retrieval.id_context_recall",
    )
    with_ref, without = (
        "metric.rag.context_precision_with_reference",
        "metric.rag.context_precision_without_reference",
    )
    recall = "metric.rag.context_recall"
    diagnostics = {
        "context_entity_recall": (1 / (2 + 1e-8), 2),
        "factual_correctness": (1.0, 4),
        "noise_sensitivity": (0.5, 9),
        "answer_accuracy": (0.75, 2),
        "context_relevance": (1.0, 2),
        "response_groundedness": (1.0, 2),
    }
    metrics = [
        create_metric("metric.rag.faithfulness", llm=judge),
        create_metric(
            "metric.rag.answer_relevance",
            llm=judge,
            embeddings=embeddings,
            strictness=1,
        ),
        create_metric(id_precision),
        create_metric(id_recall),
        create_metric(with_ref, llm=judge),
        create_metric(without, llm=judge),
        create_metric(recall, llm=judge),
        create_metric("metric.text.semantic_similarity", embeddings=embeddings),
        create_metric(AC, llm=judge, embeddings=embeddings),
        *(create_metric("metric.rag." + name, llm=judge) for name in diagnostics),
    ]
    store = create_storage_backend("storage.local.jsonl", root=tmp_path)
    run = await Evaluator(metrics, backend=store).arun(Dataset.from_records(rows))
    saved = store.load_run(run.run_id)
    assert (
        saved.rows == run.rows
        and saved.summary == run.summary
        and saved.metrics == run.metrics
        and len(saved.metrics) == 15
    )
    first = saved.row_for("ranked").metrics
    assert all(score.status == "success" for score in first.values())
    for key, value in (
        (id_precision, 1 / 2),
        (id_recall, 1 / 3),
        (with_ref, 5 / 6),
        (without, 5 / 6),
        (recall, 2 / 3),
        (AC, 1.0),
    ):
        assert first[key].value == pytest.approx(value)
    for name, (value, requests) in diagnostics.items():
        key = "metric.rag." + name
        assert first[key].value == pytest.approx(value, abs=1e-12, rel=0)
        assert first[key].usage.generation.requests == requests
        assert first[key].usage.generation.known_tokens == {
            "input_tokens": 3 * requests,
            "output_tokens": requests,
        }
        assert saved.summary[key]["coverage"] == pytest.approx(2 / 3)
        assert (
            saved.metrics[key]["revision"]
            == saved.metrics[key]["config_snapshot"]["prompt_revision"]
            == "1"
        )
    assert saved.usage.generation.requests == len(judge.calls)
    assert saved.usage.embedding.requests > 0
    miss = saved.row_for("miss").metrics
    assert (
        miss[id_precision].value,
        miss[recall].value,
        miss[with_ref].error.code,
    ) == (0.0, 0.0, "no_relevant_contexts")
    invalid = saved.row_for("invalid").metrics
    assert (
        invalid[AC].error.field == "response"
        and invalid[id_recall].error.code == "no_reference_ids"
        and invalid["metric.rag.context_entity_recall"].error.code
        == "no_reference_entities"
        and invalid["metric.rag.context_relevance"].error.field == "retrieved_contexts"
    )
    assert saved.summary[with_ref]["coverage"] == pytest.approx(1 / 3)
    assert saved.summary[recall]["mean"] == pytest.approx(1 / 3)
    assert saved.summary[id_precision]["coverage"] == pytest.approx(2 / 3)

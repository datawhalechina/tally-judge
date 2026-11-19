"""End-to-end contracts for the nine single-turn RAG metrics."""

import json
from types import SimpleNamespace

import pytest

from tallyjudge import Dataset, Evaluator, create_metric, create_storage_backend
from tallyjudge.core.errors import PluginError
from tallyjudge.plugins.metrics.rag.answer_correctness import AnswerCorrectnessMetric
from tallyjudge.registry import PluginSpec, registry

AC = "metric.rag.answer_correctness"


class Judge:
    plugin_id = "model.test.rag"
    usage_tracking = "untracked"

    def __init__(self):
        self.calls = []

    def config_snapshot(self):
        return {}

    async def agenerate(self, prompt, response_model):
        self.calls.append(response_model.__name__)
        message = prompt.messages[0].content
        data = json.loads(message.rsplit("\ninput: ", 1)[1].removesuffix("\nOutput: "))
        verdict = {"statement": "claim", "reason": "test", "verdict": 1}
        match response_model.__name__:
            case "StatementGeneratorOutput":
                output = {"statements": ["claim"]}
            case "NLIStatementOutput":
                output = {"statements": [verdict]}
            case "AnswerRelevanceOutput":
                output = {"question": "q", "noncommittal": 0}
            case "ContextPrecisionOutput":
                output = {"reason": "test", "verdict": int(data["context"] == "hit")}
            case "ContextRecallOutput":
                output = {
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
                output = {
                    "TP": [{"statement": "claim", "reason": "test"}],
                    "FP": [],
                    "FN": [],
                }
            case other:
                raise AssertionError(other)
        return SimpleNamespace(data=response_model.model_validate(output))


class Vectors:
    plugin_id = "embedding.test.rag"
    usage_tracking = "untracked"

    def config_snapshot(self):
        return {}

    async def aembed_texts(self, texts):
        return SimpleNamespace(vectors=[[1.0, 0.0] for _ in texts])


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
async def test_nine_metrics_on_one_dataset_export_failures_and_coverage(tmp_path):
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
    ]
    store = create_storage_backend("storage.local.jsonl", root=tmp_path)
    run = await Evaluator(metrics, backend=store).arun(Dataset.from_records(rows))
    saved = store.load_run(run.run_id)
    assert (
        saved.rows == run.rows
        and saved.summary == run.summary
        and len(saved.metrics) == 9
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
    )
    assert saved.summary[with_ref]["coverage"] == pytest.approx(1 / 3)
    assert saved.summary[recall]["mean"] == pytest.approx(1 / 3)
    assert saved.summary[id_precision]["coverage"] == pytest.approx(2 / 3)

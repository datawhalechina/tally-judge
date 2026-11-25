import json

import pytest

from tallyjudge import Dataset, create_metric, create_optimizer, numeric_metric
from tallyjudge.sdk.gate import record_request
from tallyjudge.sdk.models import LanguageModelBase
from tallyjudge.sdk.usage import Usage


class ScriptModel(LanguageModelBase):
    plugin_id = "model.test.script"
    structured_retries = 1

    def __init__(self, *, invalid=False):
        self.calls = []
        self.invalid = invalid

    def config_snapshot(self):
        return {"model": "offline"}

    async def _acomplete(self, messages, records):
        with record_request("generation", records) as request:
            request.usage = Usage(2, 1)
            system, question = messages[0].content, messages[-1].content
            self.calls.append((system, question))
            if self.invalid:
                return "malformed", "fixture"
            if system.startswith("Infer"):
                reply = {"instruction": "improved"}
            elif system.startswith("Give up"):
                reply = {"feedbacks": ["answer precisely"]}
            elif system.startswith("Rewrite"):
                reply = {"instruction": "fixed"}
            elif system.startswith("Cross over"):
                reply = {"instruction": "crossed"}
            elif "proposed_instruction" in system:
                reply = {"proposed_instruction": "improved"}
            else:
                reply = {
                    "response": "right"
                    if "improved" in system or "fixed" in system or "crossed" in system
                    else "wrong"
                }
            return json.dumps(reply), "fixture"


def labeled(*ids):
    return Dataset.from_records(
        [
            {"case_id": i, "user_input": "question " + i, "reference": "right"}
            for i in ids
        ]
    )


def optimize(kind, model, candidates=6, requests=100, **partitions):
    plugin = create_optimizer(
        "optimization.prompt." + kind,
        llm=model,
        max_candidates=candidates,
        max_requests=requests,
        seed=4,
    )
    return plugin.run(
        "original",
        train=labeled("train-a", "train-b"),
        holdout=partitions.get("holdout", labeled("secret")),
        metric=create_metric("metric.text.exact_match"),
    )


def test_genetic_four_stages_then_holdout():
    model = ScriptModel()
    report = optimize("genetic", model)
    assert {"original", "improved", "fixed", "crossed"} == set(report.training_scores)
    assert (
        report.instruction == "improved" and report.requests == len(model.calls) <= 100
    )
    assert all("secret" not in system + user for system, user in model.calls[:-2])
    assert all(user == "question secret" for _, user in model.calls[-2:])
    summary = report.holdout.metrics["metric.text.exact_match"]
    assert (summary.paired, summary.wins, summary.ties, summary.losses) == (1, 1, 0, 0)
    assert (
        report.holdout.left_usage.generation.requests
        == report.holdout.right_usage.generation.requests
        == 1
    )
    assert report.config_snapshot["objective"]["id"] == "metric.text.exact_match"


def test_tie_overlap_and_budget_on_real_transport_retries():
    ok = ScriptModel()
    tied = optimize("genetic", ok, candidates=1)
    assert (
        tied.instruction == "original"
        and tied.holdout.metrics["metric.text.exact_match"].ties == 1
    )
    with pytest.raises(ValueError, match="overlap"):
        optimize("genetic", ScriptModel(), holdout=labeled("train-a"))
    model = ScriptModel(invalid=True)
    report = optimize("genetic", model, requests=3)
    assert report.budget_exhausted and report.requests == len(model.calls) == 3
    assert report.holdout.metrics["metric.text.exact_match"].paired == 0


def test_real_dspy_mipro_compile_offline():
    pytest.importorskip("dspy")
    pytest.importorskip("optuna")
    model = ScriptModel()
    report = optimize("dspy_mipro", model, candidates=3, requests=90)
    assert not report.budget_exhausted
    assert any("proposed_instruction" in s for s, _ in model.calls)
    assert all("secret" not in s + q for s, q in model.calls[:-2])
    assert (
        report.requests == len(model.calls)
        and report.holdout.metrics["metric.text.exact_match"].paired == 1
    )
    assert report.instruction in {"original", "improved"}


def test_real_dspy_cannot_exceed_request_cap():
    pytest.importorskip("dspy")
    pytest.importorskip("optuna")
    model = ScriptModel()
    result = optimize("dspy_mipro", model, candidates=3, requests=5)
    assert result.requests == len(model.calls) == 5
    assert result.budget_exhausted and result.instruction == "original"
    assert result.holdout.metrics["metric.text.exact_match"].paired == 1


def test_nested_optimizers_share_the_outer_transport_budget():
    model = ScriptModel()

    def optimizer():
        return create_optimizer(
            "optimization.prompt.genetic", llm=model, max_candidates=1, max_requests=3
        )

    @numeric_metric("metric.test.nested", revision="1")
    async def nested(response: str) -> float:
        await optimizer().arun(
            "inner",
            train=labeled("inner-train"),
            holdout=labeled("inner-holdout"),
            metric=create_metric("metric.text.exact_match"),
        )
        return 1.0

    result = optimizer().run(
        "outer",
        train=labeled("outer-train"),
        holdout=labeled("outer-holdout"),
        metric=nested(),
    )
    assert result.requests == len(model.calls) == 3
    assert result.budget_exhausted


def test_genetic_feedback_uses_the_bound_objective_and_only_labeled_fields():
    @numeric_metric("metric.test.acceptable", revision="1", value_range=(0, 1))
    def acceptable(response: str) -> float:
        return float(response in {"right", "wrong"})

    model = ScriptModel()
    train = Dataset.from_records(
        [
            {
                "case_id": "A",
                "user_input": "question",
                "reference": "right",
                "internal_note": "staff-only-marker",
            }
        ]
    )
    optimizer = create_optimizer(
        "optimization.prompt.genetic", llm=model, max_candidates=5, max_requests=50
    )
    report = optimizer.run(
        "original", train=train, holdout=labeled("B"), metric=acceptable()
    )
    assert report.training_scores["original"] == 1.0
    assert all(not system.startswith("Give up") for system, _ in model.calls)
    assert all("staff-only-marker" not in user for _, user in model.calls)
    answer = optimizer.run(
        "original",
        train=train,
        holdout=labeled("B"),
        metric=create_metric("metric.text.exact_match"),
    )
    assert answer.instruction != "original"
    feedback = [
        json.loads(user)["errors"]
        for system, user in model.calls
        if system.startswith("Give up")
    ]
    assert feedback and all(
        entry["prediction"] == "wrong" and entry["score"] == 0
        for group in feedback
        for entry in group
    )
    assert all("staff-only-marker" not in user for _, user in model.calls)


def test_dspy_trial_rejects_score_outside_the_objective_range(monkeypatch):
    dspy = pytest.importorskip("dspy")
    from types import SimpleNamespace

    @numeric_metric("metric.test.range", revision="1", value_range=(0, 1))
    def out_of_range(response: str) -> float:
        return 2.0

    seen = []

    class Trial:
        def __init__(self, metric, **kwargs):
            self.metric = metric

        def compile(self, program, *, trainset, **kwargs):
            with pytest.raises(ValueError, match="invalid_output") as error:
                self.metric(trainset[0], SimpleNamespace(response="score me"))
            seen.append(str(error.value))
            return program

    monkeypatch.setattr(dspy, "MIPROv2", Trial)
    optimizer = create_optimizer(
        "optimization.prompt.dspy_mipro", llm=ScriptModel(), max_candidates=2
    )
    report = optimizer.run(
        "original", train=labeled("A"), holdout=labeled("B"), metric=out_of_range()
    )
    assert len(seen) == 1
    assert (
        report.holdout.rows[0].metrics["metric.test.range"].left.status
        == "invalid_output"
    )


def test_optimization_report_exposes_resolvable_types():
    from typing import get_type_hints

    from tallyjudge.plugins.optimization.contract import OptimizationResult

    assert (
        get_type_hints(OptimizationResult)["holdout"].__name__ == "ExperimentComparison"
    )

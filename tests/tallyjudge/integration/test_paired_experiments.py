import asyncio

import pytest

from tallyjudge import (
    Dataset,
    Evaluator,
    MetricBinding,
    RunCollector,
    RunResult,
    compare_experiments,
    compare_runs,
    create_metric,
    create_storage_backend,
    experiment,
    numeric_metric,
)
from tallyjudge.sdk.dataset import arguments_sha256

ROWS = [
    {"case_id": "a", "user_input": "days?", "reference": "14"},
    {"case_id": "b", "user_input": "when?", "reference": "Monday"},
]


def test_pairing_uses_source_and_judge_but_allows_business_answer_changes(tmp_path):
    dataset = Dataset.from_records(ROWS)
    metric = create_metric("metric.text.exact_match")
    settings = {"prompt_version": "v1", "model": {"name": "local"}}

    @experiment(metadata=settings)
    def first(row):
        return {"response": "7" if row["case_id"] == "a" else "Monday"}

    settings["model"]["name"] = "changed"

    @experiment(metadata={"prompt_version": "v2"})
    def second(row):
        return {"response": row["reference"]}

    backend = create_storage_backend("storage.local.jsonl", root=tmp_path)
    left = first.run(dataset, metrics=[metric], backend=backend, include_inputs=False)
    right = second.run(Dataset.from_records(reversed(ROWS)), metrics=[metric])
    restored = backend.load_run(left.run_id)
    assert restored.schema_version == 4 and restored.inputs is None
    assert restored.metadata == {"prompt_version": "v1", "model": {"name": "local"}}
    paired = compare_experiments(restored, right)
    values = paired.metrics[metric.spec.id]
    assert (
        values.left_valid,
        values.right_valid,
        values.paired,
        values.wins,
        values.ties,
        values.losses,
        values.mean_delta,
    ) == (2, 2, 2, 1, 1, 0, 0.5)
    assert paired.rows[0].left_task.output["response"] == "7"
    assert paired.rows[0].right_task.output["response"] == "14"
    assert compare_runs(left, right).rows[0].metrics[metric.spec.id].delta is None
    for version in (1, 2):
        archive = left.to_dict()
        archive.pop("metadata")
        archive["schema_version"] = version
        old = RunResult.from_dict(archive)
        assert old.schema_version == 4 and old.metadata is None


def test_experiment_keeps_dataset_reference_and_reports_missing_and_skipped_rows():
    metric = create_metric("metric.text.exact_match")
    baseline = experiment(lambda row: {"response": row["reference"]})
    changed = experiment(lambda row: {"reference": "30", "response": "30"})
    a = baseline.run(Dataset.from_records(ROWS), metrics=[metric])
    b = changed.run(Dataset.from_records(ROWS[:1]), metrics=[metric])
    assert b.rows[0].metrics[metric.spec.id].value == 0.0
    assert b.input_for("a")["reference"] == "14"
    assert b.rows[0].task.output["reference"] == "30"
    result = compare_experiments(a, b)
    assert result.rows[0].metrics[metric.spec.id].delta == -1.0
    assert result.rows[1].changes == ("row_missing_right",)
    assert result.metrics[metric.spec.id].paired == 1
    failure = experiment(lambda row: None).run(
        Dataset.from_records(ROWS), metrics=[metric]
    )
    assert compare_experiments(a, failure).metrics[metric.spec.id].right_coverage == 0
    generated = baseline.run(
        Dataset.from_records([{"reference": "yes"}]), metrics=[metric]
    )
    with pytest.raises(ValueError, match="provided"):
        compare_experiments(generated, generated)


@pytest.mark.asyncio
async def test_cancelled_experiment_saves_metadata():
    started = asyncio.Event()

    @experiment(metadata={"prompt_version": "cancel"})
    async def blocked(row):
        started.set()
        await asyncio.Event().wait()

    backend = create_storage_backend("storage.local.memory")
    collector = RunCollector()
    task = asyncio.create_task(
        blocked.arun(Dataset.from_records(ROWS), backend=backend, collector=collector)
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    stored = backend.load_run(collector.snapshot().run_id)
    assert (
        stored.metadata == {"prompt_version": "cancel"}
        and stored.run_status == "cancelled"
    )


def test_lower_score_counts_as_improvement():
    from tallyjudge.sdk.metrics import MetricBase, MetricSpec
    from tallyjudge.sdk.results import MetricResult

    class Loss(MetricBase):
        spec = MetricSpec(
            "metric.test.loss",
            "1",
            {"reference": str, "response": str},
            direction="lower_is_better",
        )

        async def ascore(self, reference, response):
            return MetricResult(float(reference != response))

    loss = Loss()
    data = Dataset.from_records([{"case_id": "fixed", "response": "truth"}])
    binding = MetricBinding(
        loss, input_map={"reference": "response", "response": "answer"}
    )
    left = experiment(lambda row: {"answer": "wrong"}).run(data, metrics=[binding])
    right = experiment(lambda row: {"answer": "truth"}).run(data, metrics=[binding])
    report = compare_experiments(left, right, output_fields=["answer", "response"])
    assert (
        report.metrics[binding.key].wins == 1
        and report.metrics[binding.key].mean_delta == -1.0
    )
    changed = experiment(lambda row: {"answer": "new", "response": "new"}).run(
        data, metrics=[binding]
    )
    result = compare_experiments(left, changed, output_fields=["answer", "response"])
    assert changed.rows[0].metrics[binding.key].value == 1.0
    assert result.rows[0].metrics[binding.key].delta == 0.0


@pytest.mark.parametrize(
    ("column", "gold", "answer"),
    [
        ("user_input", "loan?", "hours?"),
        ("reference_contexts", ["14 days"], ["30 days"]),
        ("reference_context_ids", ["A"], ["B"]),
        ("expected_tools", ["lookup"], ["refund"]),
        ("forbidden_tools", ["refund"], []),
    ],
)
def test_question_and_reference_columns_stay_fixed_in_custom_bindings(
    column, gold, answer
):
    @numeric_metric("metric.test.agreement", revision="1")
    def agreement(label: str | list[str], prediction: str | list[str]) -> float:
        return float(label == prediction)

    binding = MetricBinding(agreement(), input_map={"label": column})
    data = Dataset.from_records([{column: gold, "prediction": gold}])
    run = experiment(lambda _: {column: answer, "prediction": answer}).run(
        data, metrics=[binding]
    )
    assert run.rows[0].metrics[binding.key].value == 0.0


def test_experiment_output_cannot_supply_a_missing_required_reference():
    metric = create_metric("metric.text.exact_match")
    data = Dataset.from_records([{"case_id": "loan", "user_input": "how long?"}])
    run = experiment(lambda _: {"response": "30", "reference": "30"}).run(
        data, metrics=[metric]
    )
    score = run.rows[0].metrics[metric.spec.id]
    assert score.status == "error"
    assert (score.error.category, score.error.field) == (
        "validation_error",
        "reference",
    )


def test_historical_score_using_the_returned_reference_remains_incomparable():
    metric = create_metric("metric.text.exact_match")
    output = {"response": "30", "reference": "30"}
    run = experiment(lambda _: output).run(
        Dataset.from_records(ROWS[:1]), metrics=[metric]
    )
    # A v2 run scored both arguments from the returned output and stored their joint hash.
    saved = run.to_dict()
    saved["schema_version"] = 2
    old_score = saved["rows"][0]["metrics"][metric.spec.id]
    old_score.update(value=1.0, inputs_sha256=arguments_sha256(output))
    del old_score["input_sha256_by_name"]
    legacy = RunResult.from_dict(saved)
    pair = compare_experiments(legacy, run).rows[0].metrics[metric.spec.id]
    assert (pair.left.value, pair.right.value) == (1.0, 0.0)
    assert pair.changes == ("control_input_changed",) and pair.delta is None


def test_label_and_ranking_coverage_survives_pairing():
    from tallyjudge import categorical_metric, ranking_metric

    @categorical_metric("metric.test.label", revision="1", labels=["ok"])
    def label(response: str) -> str:
        return "ok"

    @ranking_metric("metric.test.rank", revision="1", length=1)
    def rank(response: str) -> list[str]:
        return [response]

    data = Dataset.from_records(ROWS)
    run = experiment(lambda row: {"response": row["reference"]}).run(
        data, metrics=[label(), rank()]
    )
    for summary in compare_experiments(run, run).metrics.values():
        assert (summary.left_valid, summary.right_valid, summary.paired) == (2, 2, 2)
        assert summary.left_coverage == summary.right_coverage == 1.0
        assert summary.mean_delta is None
    other = experiment(lambda row: {"response": "different", "reference": "new"}).run(
        data, metrics=[label(), rank()]
    )
    one = experiment(lambda row: {}).run(Dataset.from_records(ROWS[:1]), metrics=[])
    assert all(s.paired == 0 for s in compare_experiments(other, one).metrics.values())


def test_equal_argument_digests_prove_controls_in_redacted_archives():
    data = Dataset.from_records(ROWS)
    metric = create_metric("metric.text.exact_match")

    def archived(func):
        run = experiment(func).run(data, metrics=[metric])
        return _archived(run, keep_inputs=False, legacy=True)

    left = archived(lambda row: {"response": row["reference"]})
    right = archived(
        lambda row: {"response": row["reference"], "reference": row["reference"]}
    )
    comparison = compare_experiments(left, right)
    assert (
        comparison.metrics[metric.spec.id].ties
        == comparison.metrics[metric.spec.id].paired
        == 2
    )
    different = archived(lambda row: {"response": "other", "reference": "other"})
    assert compare_experiments(left, different).rows[0].metrics[
        metric.spec.id
    ].changes == ("control_input_unverified",)


def _archived(run, *, keep_inputs=False, legacy=False):
    data = run.to_dict(include_inputs=keep_inputs)
    if legacy:
        for row in data["rows"]:
            for score in row["metrics"].values():
                score.pop("input_sha256_by_name", None)
    return RunResult.from_dict(data)


@pytest.mark.parametrize("source", ["dataset", "experiment"])
@pytest.mark.parametrize("keep_inputs", [True, False])
def test_pairing_protects_ordered_rules_in_the_effective_inputs(source, keep_inputs):
    @numeric_metric("metric.test.ordered_rules", revision="1")
    def grade(response: str, rules: dict[str, int]) -> float:
        return float(next(iter(rules.values())))

    binding = MetricBinding(grade(), input_map={"rules": "grading_rules"})

    def trial(rules):
        row = {"case_id": "a"}
        output = {"response": "fixed"}
        (row if source == "dataset" else output)["grading_rules"] = rules
        run = experiment(lambda _: output).run(
            Dataset.from_records([row]), metrics=[binding]
        )
        if source == "dataset" and not keep_inputs:
            assert "private rubric" not in run.to_json(include_inputs=False)
        return _archived(run, keep_inputs=keep_inputs)

    left = trial({"private rubric": 1, "other": 5})
    right = trial({"other": 5, "private rubric": 1})
    pair = compare_experiments(left, right)
    assert pair.rows[0].metrics[binding.key].changes == ("control_input_changed",)
    assert pair.metrics[binding.key].paired == 0
    assert pair.metrics[binding.key].mean_delta is None


def test_pairing_checks_scored_defaults_and_allows_an_explicit_override():
    def scaler(default):
        @numeric_metric("metric.test.paired_scale", revision="1")
        def grade(response: str, scale: float = default) -> float:
            return len(response) * scale

        return grade()

    key = "metric.test.paired_scale"
    dataset = Dataset.from_records(
        [{"case_id": "default"}, {"case_id": "override", "scale": 3.0}]
    )
    left, right = [
        _archived(
            experiment(lambda _: {"response": reply}).run(
                dataset, metrics=[scaler(scale)]
            )
        )
        for scale, reply in [(1.0, "a"), (2.0, "ab")]
    ]
    pair = compare_experiments(left, right)
    assert pair.rows[0].metrics[key].changes == ("control_input_changed",)
    assert pair.rows[1].metrics[key].delta == 3.0
    assert (pair.metrics[key].paired, pair.metrics[key].wins) == (1, 1)


def test_mixed_archive_only_pairs_recoverable_legacy_arguments():
    from typing import Annotated

    from pydantic import AfterValidator

    @numeric_metric("metric.test.normalized", revision="1")
    def grade(response: str, rules: Annotated[str, AfterValidator(str.strip)]) -> float:
        return float(response == rules)

    metric = grade()
    rows = Dataset.from_records(
        [{"case_id": "exact", "rules": "A"}, {"case_id": "normalized", "rules": " A "}]
    )
    baseline = experiment(lambda _: {"response": "B"}).run(rows, metrics=[metric])
    candidate = _archived(
        experiment(lambda _: {"response": "A"}).run(rows, metrics=[metric])
    )
    pair = compare_experiments(
        _archived(baseline, keep_inputs=True, legacy=True), candidate
    )
    assert pair.rows[0].metrics[metric.spec.id].delta == 1.0
    assert pair.rows[1].metrics[metric.spec.id].changes == ("control_input_unverified",)
    assert pair.metrics[metric.spec.id].paired == 1

    neither_original_nor_fields = _archived(baseline, legacy=True)
    assert (
        compare_experiments(neither_original_nor_fields, candidate)
        .metrics[metric.spec.id]
        .paired
        == 0
    )


@pytest.mark.parametrize("reverse", [True, False])
def test_one_retained_original_can_only_reconstruct_matching_old_controls(reverse):
    @numeric_metric("metric.test.legacy_rules", revision="1")
    def grade(response: str, rules: dict[str, int]) -> float:
        return float(next(iter(rules.values())) + len(response))

    metric = grade()

    def trial(rules, response, *, keep_inputs):
        data = Dataset.from_records([{"case_id": "a", "rules": rules}])
        run = experiment(lambda _: {"response": response}).run(data, metrics=[metric])
        return _archived(run, keep_inputs=keep_inputs, legacy=True)

    rules = {"a": 1, "b": 5}
    left = trial(rules, "a", keep_inputs=False)
    right_rules = dict(reversed(rules.items())) if reverse else rules
    right = trial(right_rules, "ab", keep_inputs=True)
    pair = compare_experiments(left, right)
    expected = ("control_input_unverified",) if reverse else ()
    assert pair.rows[0].metrics[metric.spec.id].changes == expected
    assert pair.metrics[metric.spec.id].mean_delta == (None if reverse else 1.0)


def test_experiment_hashes_stay_out_of_ordinary_exports():
    metric = create_metric("metric.text.exact_match")
    dataset = Dataset.from_records([{"case_id": 1, "response": "a", "reference": "a"}])
    plain = Evaluator([metric]).run(dataset)
    run = experiment(lambda _: {}).run(dataset, metrics=[metric])
    key = metric.spec.id
    assert "input_sha256_by_name" not in plain.to_dict()["rows"][0]["metrics"][key]
    data = run.to_dict()
    assert set(data["rows"][0]["metrics"][key]["input_sha256_by_name"]) == {
        "response",
        "reference",
    }


def test_pair_mean_preserves_large_finite_deltas():
    @numeric_metric("metric.test.large", revision="1")
    def magnitude(response: float) -> float:
        return response

    dataset = Dataset.from_records([{"case_id": "a"}, {"case_id": "b"}])
    left, right = [
        experiment(lambda _: {"response": value}).run(dataset, metrics=[magnitude()])
        for value in (0.0, 1e308)
    ]
    paired = compare_experiments(left, right).metrics[magnitude.spec.id]
    assert paired.wins == 2 and paired.mean_delta == 1e308

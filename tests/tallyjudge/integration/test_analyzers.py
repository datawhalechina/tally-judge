import json

import pytest

from tallyjudge import (
    Dataset,
    Evaluator,
    create_analyzer,
    create_metric,
    create_storage_backend,
    numeric_metric,
)
from tallyjudge.cli import main

EXACT = "metric.text.exact_match"
ROWS = [
    {"case_id": "a", "topic": "hours", "response": "周一", "reference": "周一"},
    {"case_id": "b", "topic": "hours", "response": "周二", "reference": "周一"},
    {"case_id": "c", "topic": "loans", "response": "14天", "reference": "14天"},
]


def test_summary_by_field_reports_each_group_with_the_builtin_summary():
    run = Evaluator([create_metric(EXACT)]).run(Dataset.from_records(ROWS))
    report = create_analyzer("analyzer.summary.by_field", field="topic").analyze([run])
    groups = report["runs"][0]["groups"]
    assert {name: g[EXACT]["mean"] for name, g in groups.items()} == {
        "hours": 0.5,
        "loans": 1.0,
    }
    assert groups["hours"][EXACT]["total"] == 2


def test_summary_by_field_needs_the_saved_inputs(tmp_path):
    store = create_storage_backend("storage.local.jsonl", root=tmp_path)
    run = Evaluator([create_metric(EXACT)], backend=store, include_inputs=False).run(
        Dataset.from_records(ROWS)
    )
    analyzer = create_analyzer("analyzer.summary.by_field", field="topic")
    with pytest.raises(ValueError, match="without its inputs"):
        analyzer.analyze([store.load_run(run.run_id)])


def test_stability_reports_which_cases_moved_between_runs():
    def run(offset):
        @numeric_metric("metric.test.drift", revision="1")
        def drift(response: str) -> float:
            return 1.0 + (offset if response == "周二" else 0.0)

        return Evaluator([drift()]).run(Dataset.from_records(ROWS))

    report = create_analyzer("analyzer.stability.repeats").analyze(
        [run(0.0), run(0.5), run(0.25)]
    )
    assert report["metrics"]["metric.test.drift"] == {
        "compared": 3,
        "identical": 2,
        "mean_range": 0.5 / 3,
        "max_range": 0.5,
        "unstable_cases": ["b"],
    }
    with pytest.raises(ValueError, match="at least two"):
        create_analyzer("analyzer.stability.repeats").analyze([run(0.0)])


def test_command_line_analyzes_stored_runs(tmp_path, capsys):
    samples = tmp_path / "samples.jsonl"
    samples.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ROWS),
        encoding="utf-8",
    )
    store = str(tmp_path / "runs")
    assert (
        main(["run", "--input", str(samples), "--metric", EXACT, "--store", store]) == 0
    )
    run_id = json.loads(capsys.readouterr().out)["run_id"]
    command = ["runs", "analyze", run_id, "--store", store]
    command += ["--analyzer", "analyzer.summary.by_field", "--option", "field=topic"]
    assert main(command) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["runs"][0]["groups"]["loans"][EXACT]["mean"] == 1.0


def test_agreement_compares_scores_with_human_labels_within_a_tolerance():
    @numeric_metric("metric.test.judge", revision="1")
    def judge(response: str) -> float:
        return {"周一": 1.0, "周二": 0.4, "14天": 0.0}[response]

    rows = [dict(row, human=label) for row, label in zip(ROWS, [1, 0.5, "n/a"])]
    run = Evaluator([judge()]).run(Dataset.from_records(rows))
    settings = {"metric": "metric.test.judge", "field": "human"}
    strict = create_analyzer("analyzer.agreement.with_labels", **settings)
    loose = create_analyzer("analyzer.agreement.with_labels", tolerance=0.1, **settings)
    report = strict.analyze([run])["runs"][0]
    assert (report["compared"], report["agreed"], report["disagreements"]) == (
        2,
        1,
        ["b"],
    )
    assert report["mean_absolute_error"] == pytest.approx(0.05)
    assert loose.analyze([run])["runs"][0]["agreement"] == 1.0


def test_calibration_set_is_consistent_and_its_suite_loads():
    from pathlib import Path

    from tallyjudge import Suite
    from tallyjudge.registry import registry
    from tallyjudge.sdk.metrics import validate_inputs

    root = Path(__file__).resolve().parents[3] / "calibration"
    if not root.exists():
        pytest.skip("the calibration set is kept outside the repository")
    spec = registry.describe("metric.rag.faithfulness")
    rows = [
        json.loads(line)
        for line in (root / "faithfulness.zh.jsonl").read_text("utf-8").splitlines()
    ]
    assert len({row["case_id"] for row in rows}) == len(rows) == 12
    for row in rows:
        validate_inputs(spec, {name: row[name] for name in spec.inputs})
        labels = [item["supported"] for item in row["human_statements"]]
        assert row["human_faithfulness"] == sum(labels) / len(labels)
    suite = Suite.load(root / "faithfulness.zh.toml")
    assert suite.effective_config()["metrics"]["faithfulness"]["llm"] == "judge"

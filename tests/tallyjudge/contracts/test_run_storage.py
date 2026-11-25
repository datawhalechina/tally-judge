import copy
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from tallyjudge import (
    Dataset,
    Evaluator,
    RunResult,
    compare_runs,
    create_metric,
    create_storage_backend,
)
from tallyjudge.sdk.results import CleanupWarning
from tallyjudge.sdk.usage import ScoreUsage, UsageSummary

EXACT = "metric.text.exact_match"


def evaluate(records):
    return Evaluator([create_metric(EXACT)]).run(
        Dataset.from_records(records, id_field="ticket")
    )


SAMPLES = [
    {"ticket": 42, "response": "甲", "reference": "甲"},
    {"ticket": 8, "response": "私有", "reference": "乙"},
    {"ticket": 9, "response": 7, "reference": "丙"},
]


@pytest.fixture(params=["memory", "jsonl"])
def backend(request, tmp_path):
    options = {"root": tmp_path} if request.param == "jsonl" else {}
    return create_storage_backend("storage.local." + request.param, **options)


@pytest.mark.parametrize("keep", [True, False])
def test_terminal_roundtrip_by_custom_id(backend, keep):
    result = evaluate(SAMPLES)
    result.save(backend=backend, include_inputs=keep)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(
            pool.map(
                lambda _: result.save(backend=backend, include_inputs=keep), range(3)
            )
        )
    restored = backend.load_run(result.run_id)
    assert [row.metrics[EXACT].status for row in restored.rows] == [
        "success",
        "success",
        "error",
    ]
    assert restored.summary[EXACT]["mean"] == 0.5
    assert restored.summary[EXACT]["coverage"] == 2 / 3
    assert restored.row_for(9).metrics[EXACT].error.field == "response"
    assert restored.to_dict() == result.to_dict(include_inputs=keep)
    if keep:
        assert restored.input_for("8") == SAMPLES[1]
        restored.input_for("8").clear()
        assert restored.input_for("8") == SAMPLES[1]
    else:
        assert restored.inputs is None and "inputs" not in restored.to_dict()
        with pytest.raises(LookupError, match="not saved"):
            restored.input_for(8)
    with pytest.raises(KeyError):
        restored.row_for("missing")
    restored.rows.clear()
    assert len(backend.load_run(result.run_id).rows) == 3
    with pytest.raises(FileNotFoundError):
        backend.load_run("missing")


@pytest.mark.parametrize("status", ["completed", "stopped", "aborted", "cancelled"])
def test_terminal_states_empty_inputs_and_conflict(backend, status):
    result = evaluate([])
    result.run_status = "running"
    with pytest.raises(ValueError, match="terminal"):
        backend.save_run(result)
    result.run_status = status
    result.save(backend=backend)
    restored = backend.load_run(result.run_id)
    assert (restored.run_status, restored.inputs, restored.summary[EXACT]["mean"]) == (
        status,
        [],
        None,
    )
    with pytest.raises(FileExistsError):
        result.save(backend=backend, include_inputs=False)
    assert backend.load_run(result.run_id).inputs == []


def test_parallel_competing_writers_publish_one_complete_result(backend):
    result = evaluate(SAMPLES)
    other = evaluate(list(reversed(SAMPLES)))
    other.run_id = result.run_id

    def save(run):
        try:
            run.save(backend=backend)
            return run
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        winners = [winner for winner in pool.map(save, [result, other]) if winner]
    assert len(winners) == 1
    assert backend.load_run(result.run_id).to_dict() == winners[0].to_dict()


def test_previous_v1_without_code_or_usage_and_partial_token_snapshot(backend):
    from dataclasses import asdict

    data = evaluate(SAMPLES).to_dict()
    data["schema_version"] = 1
    data.pop("events_complete")
    data.pop("usage")
    data["metrics"][EXACT].pop("output_kind")
    for row in data["rows"]:
        row.pop("task")
        score = row["metrics"][EXACT]
        score.pop("cache_hit")
        score.pop("usage")
        if score["error"]:
            score["error"].pop("code")
    data["summary"] = {"ignore_this_stale_summary": True}
    old = RunResult.from_dict(data)
    assert old.schema_version == 4 and old.usage is None
    assert old.row_for(9).metrics[EXACT].error.code is None
    assert old.summary[EXACT]["mean"] == 0.5
    assert all(
        row.metrics[EXACT].delta == 0.0
        for row in compare_runs(old, evaluate(SAMPLES)).rows[:2]
    )
    partial = ScoreUsage(
        UsageSummary(
            requests=2,
            in_flight_requests=1,
            requests_without_usage=1,
            complete=False,
            known_tokens={"input_tokens": 11, "output_tokens": None},
            known_counts={"input_tokens": 1, "output_tokens": 0},
        ),
        UsageSummary(),
        ["custom"],
    )
    data.update(
        run_status="cancelled",
        stop_reason="external_cancel",
        cleanup_warnings=[asdict(CleanupWarning("42", 0, EXACT))],
    )
    data["rows"][0]["metrics"][EXACT].update(
        status="cancelled", value=None, usage=asdict(partial)
    )
    old = RunResult.from_dict(data)
    old.save(backend=backend)
    restored = backend.load_run(old.run_id)
    assert restored.usage == partial
    assert restored.cleanup_warnings == old.cleanup_warnings
    assert restored.to_dict() == old.to_dict()


@pytest.mark.parametrize(
    "damage", ["truncate", "duplicate", "version", "input", "score", "json"]
)
def test_jsonl_reports_corrupt_file_and_reason(tmp_path, damage):
    store = create_storage_backend("storage.local.jsonl", root=tmp_path)
    result = evaluate(SAMPLES)
    result.save(backend=store)
    (path,) = tmp_path.iterdir()
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if damage == "truncate":
        lines.pop()
    elif damage == "duplicate":
        lines[2] = lines[1]
    elif damage == "version":
        lines[0]["schema_version"] = 99
    elif damage == "score":
        lines[1]["row"]["metrics"][EXACT]["value"] = "unexpected label"
    elif damage == "input":
        lines[2]["input"]["response"] = "changed"
    text = "\n".join(json.dumps(line) for line in lines)
    path.write_text(text if damage != "json" else text[:-4], encoding="utf-8")
    with pytest.raises(ValueError, match=path.name):
        store.load_run(result.run_id)


def test_readback_preserves_binding_order_and_idempotent_resave(backend):
    keys = ["metric.text.string_presence", EXACT]
    result = Evaluator([create_metric(key) for key in keys]).run(
        Dataset.from_records(SAMPLES)
    )
    result.save(backend=backend)
    restored = backend.load_run(result.run_id)
    assert list(restored.metrics) == list(restored.rows[0].metrics) == keys
    restored.save(backend=backend)


def test_bad_write_preserves_complete_file_and_confines_ids(tmp_path):
    store = create_storage_backend("storage.local.jsonl", root=tmp_path)
    result = evaluate(SAMPLES)
    result.save(backend=store)
    original = result.to_dict()
    result.inputs[1]["bad"] = object()
    with pytest.raises((TypeError, ValueError)):
        result.save(backend=store)
    assert store.load_run(result.run_id).to_dict() == original
    assert len(list(tmp_path.iterdir())) == 1
    with pytest.raises(ValueError, match="run ID"):
        store.load_run("../outside")


def test_compare_aligns_by_identity_and_reports_changed_settings():
    left = evaluate(SAMPLES)
    right = evaluate(
        [SAMPLES[1], dict(SAMPLES[0], response="乙"), dict(SAMPLES[0], ticket=10)]
    )
    comparison = compare_runs(left, right)
    rows = {row.case_id: row for row in comparison.rows}
    assert comparison.warnings == ()
    assert rows["42"].changes == ("input_changed",)
    assert rows["42"].metrics[EXACT].delta is None
    assert rows["8"].metrics[EXACT].delta == 0.0
    assert rows["9"].changes == ("row_missing_right",)
    assert rows["10"].changes == ("row_missing_left",)
    right.metrics[EXACT].update(
        id="other", revision="99", input_map={}, config_snapshot={"option": True}
    )
    score = compare_runs(left, right).rows[1].metrics[EXACT]
    assert set(score.changes) == {"id", "revision", "input_map", "config_snapshot"}
    assert score.delta is None
    renamed = copy.deepcopy(right)
    renamed.metrics["other"] = renamed.metrics.pop(EXACT)
    for row in renamed.rows:
        row.metrics["other"] = row.metrics.pop(EXACT)
    diffs = compare_runs(left, renamed).rows[0].metrics
    assert diffs[EXACT].changes == ("binding_missing_right",)
    assert diffs["other"].changes == ("binding_missing_left",)


def test_generated_ids_require_the_same_ordered_manifest():
    one = [{"response": "a"}, {"response": "b"}]
    result = evaluate(one)
    assert compare_runs(result, evaluate(one)).warnings
    with pytest.raises(ValueError, match="ordered manifests"):
        compare_runs(result, evaluate(one[::-1]))

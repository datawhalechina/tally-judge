"""Ragas ec00339 _string, _context_precision and _context_recall numeric baselines.

Revision 1 requires an actual candidate even at threshold zero and reports empty
reference recall as undefined. Duplicate passages retain their original ranks.
"""

import asyncio
from threading import Event

import pytest

from tallyjudge import (
    Dataset,
    Evaluator,
    MetricBinding,
    compare_runs,
    create_cache_backend,
    create_metric,
    create_storage_backend,
)
from tallyjudge.core.errors import PluginError
from tallyjudge.plugins.metrics.text.string_similarity import DistanceMeasure

pytest.importorskip("rapidfuzz")

STRING = "metric.text.string_similarity"
PRECISION = "metric.retrieval.text_context_precision"
RECALL = "metric.retrieval.text_context_recall"


@pytest.mark.parametrize(
    "measure,response,reference,expected",
    [
        ("levenshtein", "kitten", "sitting", 4 / 7),
        ("levenshtein", "", "", 1.0),
        ("levenshtein", "A", "a", 0),
        ("levenshtein", "x ", "x", 0.5),
        ("hamming", "abc", "ab", 2 / 3),
        ("jaro", "MARTHA", "MARHTA", 17 / 18),
        ("jaro_winkler", "MARTHA", "MARHTA", 0.9611111111111111),
    ],
)
def test_archive_string_similarity(measure, response, reference, expected):
    metric = create_metric(STRING, distance_measure=measure)
    assert metric.score(response=response, reference=reference).value == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    "retrieved,references,threshold,precision,recall",
    [
        (["abc", "qqq", "abc"], ["abc", "xyz"], 1.0, 5 / 6, 0.5),
        (["ac"], ["ab"], 0.5, 1, 1),
        (["a"], ["a", "a", "b"], 1.0, 1, 2 / 3),
        (["a"], ["z"], 0.0, 1, 1),
        ([], ["a"], 0.0, 0, 0),
        (["a"], [], 0.0, 0, None),
        ([], [], 0.5, 0, None),
    ],
)
def test_rank_threshold_duplicates_and_empty(
    retrieved, references, threshold, precision, recall
):
    bindings = [
        MetricBinding(
            create_metric(key, threshold=threshold),
            input_map={"retrieved_contexts": "hits"},
        )
        for key in (PRECISION, RECALL)
    ]
    result = Evaluator(bindings).run(
        Dataset.from_records(
            [{"case_id": "a", "hits": retrieved, "reference_contexts": references}]
        )
    )
    scores = result.row_for("a").metrics
    assert scores[PRECISION].value == pytest.approx(precision)
    if recall is None:
        assert (scores[RECALL].status, scores[RECALL].error.code) == (
            "undefined",
            "no_reference_contexts",
        )
    else:
        assert scores[RECALL].value == pytest.approx(recall)


def test_equivalent_enum_settings_reuse_scores_and_archive(tmp_path):
    data = Dataset.from_records(
        [{"case_id": "one", "response": "abc", "reference": "ab"}]
    )
    cache = create_cache_backend(
        "cache.local.sqlite", path=tmp_path / "cache.sqlite", namespace="v1"
    )
    store = create_storage_backend("storage.local.jsonl", root=tmp_path)

    def run_with(measure):
        return Evaluator(
            [create_metric(STRING, distance_measure=measure)],
            cache=cache,
            backend=store,
        ).run(data)

    first = run_with("hamming")
    same = run_with(DistanceMeasure.HAMMING)
    assert same.rows[0].metrics[STRING].cache_hit
    assert (
        compare_runs(store.load_run(first.run_id), same).rows[0].metrics[STRING].delta
        == 0.0
    )
    changed = run_with("jaro")
    assert not changed.rows[0].metrics[STRING].cache_hit
    assert (
        "config_snapshot"
        in compare_runs(first, changed).rows[0].metrics[STRING].changes
    )
    with pytest.raises(PluginError):
        create_metric(RECALL, threshold=1.1)


@pytest.mark.asyncio
async def test_distance_work_yields_to_the_event_loop(monkeypatch):
    metric = create_metric(PRECISION)
    started, release = Event(), Event()

    def slow_similarity(left, right):
        started.set()
        release.wait(2)
        return 1.0

    monkeypatch.setattr(metric.distance, "similarity", slow_similarity)
    task = asyncio.create_task(metric.ascore(["a"], ["a"]))
    try:
        assert await asyncio.to_thread(started.wait, 1)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("plugin", [STRING, PRECISION, RECALL])
async def test_long_distance_allows_the_caller_deadline_to_fire(plugin):
    left, right = "a" * 95_000, "b" * 95_000
    inputs = (
        {"response": left, "reference": right}
        if plugin == STRING
        else {"retrieved_contexts": [left], "reference_contexts": [right]}
    )
    metric = create_metric(plugin)
    loop = asyncio.get_running_loop()
    fired = loop.create_future()
    timer = loop.call_later(0.01, lambda: fired.set_result(loop.time()))
    start = loop.time()
    try:
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.03):
                await metric.ascore_inputs(inputs)
        assert (await fired) - start < 0.14
    finally:
        timer.cancel()


def test_long_passages_preserve_distance_and_retrieval_ranks():
    passage = "文" * 17_000
    other = passage[:-1] + "字"
    row = {
        "response": passage,
        "reference": other,
        "retrieved_contexts": ["x", passage, passage],
        "reference_contexts": [other, "y"],
    }
    result = Evaluator(
        [create_metric(plugin) for plugin in (STRING, PRECISION, RECALL)]
    ).run(Dataset.from_records([row]))
    scores = result.rows[0].metrics
    assert scores[STRING].value == pytest.approx(1 - 1 / len(passage))
    assert scores[PRECISION].value == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert scores[RECALL].value == 0.5


def test_direct_retrieval_constructor_enforces_the_factory_threshold():
    from tallyjudge.plugins.metrics.retrieval.text_context_recall import (
        TextContextRecallMetric,
    )

    with pytest.raises(ValueError):
        TextContextRecallMetric(threshold=1.1)

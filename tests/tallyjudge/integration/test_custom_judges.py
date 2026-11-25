import asyncio
import json

import pytest

from tallyjudge import (
    Dataset,
    Evaluator,
    ExecutionConfig,
    MetricBinding,
    RunCollector,
    RunResult,
    compare_runs,
    create_cache_backend,
    create_event_sink,
    create_language_model,
    create_metric,
    create_storage_backend,
)
from tallyjudge.registry import registry
from tallyjudge.sdk.errors import PluginError

NUMERIC = "metric.custom.numeric"
CATEGORY = "metric.custom.categorical"
FREE = "metric.custom.rubric_without_reference"
WITH_REF = "metric.custom.rubric_with_reference"
INSTANCE = "metric.custom.instance_rubric"


@pytest.fixture
def judge(client):
    return create_language_model(
        "model.openai_compatible.chat_completions",
        client=client,
        model="local",
        temperature=0.0,
    )


def vote(client, value, reason):
    client.chat_reply(json.dumps({"value": value, "reason": reason}))


def rubric(text="standard"):
    return {f"score{i}_description": f"{text} {i}" for i in range(1, 6)}


def reply_grade(client, score, feedback):
    client.chat_reply(json.dumps({"score": score, "feedback": feedback}))


def test_describe_needs_settings_but_no_client(judge):
    settings = {"criteria": "quality", "labels": ["yes", "partial", "no"]}
    spec = registry.describe(CATEGORY, **settings)
    assert spec == create_metric(CATEGORY, llm=judge, **settings).spec
    assert spec.labels == tuple(settings["labels"])
    with pytest.raises(PluginError, match=r"metric.custom.categorical.*labels"):
        registry.describe(CATEGORY, criteria="quality")
    for config in (
        {"strictness": 2},
        {"levels": [1, 1]},
        {"levels": [0, 2], "value_range": [0, 2]},
        {"value_range": [2, 0]},
        {"strictness": 0},
        {"levels": [True, False]},
    ):
        with pytest.raises(PluginError):
            registry.describe(NUMERIC, criteria="quality", **config)
    assert registry.describe(
        NUMERIC, criteria="quality", value_range=[-2, 5]
    ).value_range == (-2, 5)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "plugin,bad,config",
    [
        (NUMERIC, 3.0, {"criteria": "grade", "levels": [0, 2, 4]}),
        (FREE, 2.5, {}),
        (WITH_REF, 2.5, {}),
        (INSTANCE, 2.5, {}),
    ],
)
async def test_discrete_contract_applies_to_cache_and_storage(
    plugin, bad, config, judge, client
):
    from tallyjudge import MetricResult

    class Cache:
        namespace = "invalid-grades"

        def get(self, key):
            return MetricResult(bad)

        def put(self, key, score):
            self.saved = score

    if plugin == NUMERIC:
        vote(client, 2, "fresh")
    else:
        reply_grade(client, 2, "fresh")
    metric = create_metric(plugin, llm=judge, **config)
    cache = Cache()
    result = await Evaluator([metric], cache=cache).arun(
        Dataset.from_records(
            [{"response": "answer", "reference": "ref", "rubrics": rubric()}]
        )
    )
    score = result.rows[0].metrics[plugin]
    assert (score.value, score.cache_hit) == (2.0, False)
    assert len(client.calls) == 1 and cache.saved.value == 2.0
    saved = result.to_dict()
    saved["rows"][0]["metrics"][plugin]["value"] = bad
    with pytest.raises(ValueError, match="Invalid stored score"):
        RunResult.from_dict(saved)


@pytest.mark.asyncio
async def test_direct_and_factory_judges_use_the_same_settings(judge, client):
    from tallyjudge.plugins.metrics.custom.judge import CategoricalMetric, NumericMetric
    from tallyjudge.plugins.metrics.custom.rubric import RubricWithReferenceMetric

    for plugin, factory, config in [
        (NUMERIC, NumericMetric, {"levels": [0, 2, 4]}),
        (CATEGORY, CategoricalMetric, {"labels": ["yes", "no"]}),
    ]:
        opts = {"llm": judge, "criteria": "grade", "strictness": 2, **config}
        direct, registered = factory(**opts), create_metric(plugin, **opts)
        assert direct.config_snapshot() == registered.config_snapshot()
        assert direct.strictness == 3
        with pytest.raises(ValueError):
            factory(**{**opts, "strictness": 0})

    rules = rubric("cite evidence")
    inputs = {"response": "answer", "reference": "ref"}
    metrics = [
        RubricWithReferenceMetric(judge, dict(reversed(rules.items()))),
        create_metric(WITH_REF, llm=judge, rubrics=rules),
        RubricWithReferenceMetric(judge),
        create_metric(
            WITH_REF, llm=judge, rubrics=RubricWithReferenceMetric.default_rubrics
        ),
    ]
    for metric in metrics:
        reply_grade(client, 3, "ok")
        await metric.ascore_inputs(inputs)
    prompts = [body["messages"][0]["content"] for _, body in client.calls]
    assert prompts[0] != prompts[1] and all("Paris" in prompt for prompt in prompts)
    assert prompts[2] == prompts[3]
    assert metrics[0].config_snapshot() != metrics[1].config_snapshot()
    assert metrics[2].config_snapshot() == metrics[3].config_snapshot()


@pytest.mark.asyncio
async def test_discrete_votes_retry_cache_and_events(judge, client, tmp_path):
    cache = create_cache_backend(
        "cache.local.sqlite", path=tmp_path / "scores.sqlite", namespace="judges"
    )
    sink = create_event_sink("event.local.jsonl", root=tmp_path)
    store = create_storage_backend("storage.local.jsonl", root=tmp_path)
    data = Dataset.from_records([{"case_id": "a", "answer": "private answer"}])

    def evaluator(rule, count):
        binding = MetricBinding(
            create_metric(
                NUMERIC, llm=judge, criteria=rule, levels=[0, 1], strictness=count
            ),
            "gate",
            {"response": "answer"},
        )
        return Evaluator(
            [binding], cache=cache, backend=store, event_sink=sink, include_inputs=False
        )

    for value, reason in ((3, "invalid"), (0, "first"), (1, "winner"), (1, "also")):
        vote(client, value, reason)
    first = await evaluator("private rule", 2).arun(data)
    saved = store.load_run(first.run_id)
    score = saved.row_for("a").metrics["gate"]
    assert (score.value, score.reason, score.usage.generation.requests) == (
        1.0,
        "winner",
        4,
    )
    assert "private rule" not in saved.to_json() and saved.inputs is None
    observed = [
        json.loads(line)
        for line in (tmp_path / f"events-{first.run_id}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    calls = [e["call_id"] for e in observed if e["kind"] == "request_started"]
    assert len(calls) == 4 and len(set(calls)) == 3 and calls[0] == calls[1]
    warm = await evaluator("private rule", 3).arun(data)
    assert warm.row_for("a").metrics["gate"].cache_hit and len(client.calls) == 4
    for i in range(3):
        vote(client, 0, str(i))
    changed = await evaluator("other rule", 3).arun(data)
    diff = compare_runs(saved, changed).rows[0].metrics["gate"]
    assert diff.delta is None and "config_snapshot" in diff.changes


@pytest.mark.asyncio
async def test_category_tie_and_output_boundaries(client, judge):
    for label in ("partial", "yes", "no"):
        vote(client, label, label + " reason")
    metric = create_metric(
        CATEGORY,
        llm=judge,
        criteria="grade",
        labels=["yes", "partial", "no"],
        strictness=3,
    )
    result = await Evaluator([metric]).arun(
        Dataset.from_records([{"response": "answer"}])
    )
    score = result.rows[0].metrics[CATEGORY]
    assert (score.value, score.reason) == ("partial", "partial reason")
    assert result.summary[CATEGORY]["label_counts"]["partial"] == 1
    vote(client, "unknown", "invalid")
    vote(client, "unknown", "invalid")
    failed = await Evaluator([metric]).arun(
        Dataset.from_records([{"response": "answer"}])
    )
    score = failed.rows[0].metrics[CATEGORY]
    assert (
        score.error.category == "model_output_error"
        and score.usage.generation.requests == 2
    )

    continuous = create_metric(
        NUMERIC, llm=judge, criteria="grade", value_range=(-1, 5)
    )
    vote(client, True, "invalid")
    vote(client, 4.5, "valid")
    result = await Evaluator([continuous]).arun(
        Dataset.from_records([{"response": "answer"}])
    )
    assert result.rows[0].metrics[NUMERIC].value == 4.5
    assert result.rows[0].metrics[NUMERIC].usage.generation.requests == 2
    saved = result.to_dict()
    saved["rows"][0]["metrics"][NUMERIC]["value"] = 7.0
    with pytest.raises(ValueError, match="Invalid stored score"):
        RunResult.from_dict(saved)


@pytest.mark.asyncio
async def test_fixed_rubrics_preserve_custom_order_and_optional_reference(
    judge, client
):
    rules = {"second": "private level 2", "first": "private level 1"}
    original = create_metric(FREE, llm=judge, rubrics=rules)
    reversed_rules = create_metric(
        FREE, llm=judge, rubrics=dict(reversed(rules.items()))
    )
    assert reversed_rules.config_snapshot() != original.config_snapshot()
    assert "private level" not in json.dumps(original.config_snapshot())
    data = Dataset.from_records([{"response": "answer", "reference": "some reference"}])
    for metric in (original, reversed_rules):
        reply_grade(client, "4", "four")
        result = await Evaluator([metric]).arun(data)
        assert result.rows[0].metrics[FREE].value == 4.0
    prompts = [body["messages"][0]["content"] for _, body in client.calls]
    assert all("some reference" in prompt and "Paris" in prompt for prompt in prompts)
    assert prompts[0].index("second: private") < prompts[0].index("first: private")
    assert prompts[1].index("first: private") < prompts[1].index("second: private")
    reply_grade(client, 5, "ok")
    reference_metric = create_metric(WITH_REF, llm=judge)
    missing = await Evaluator([reference_metric]).arun(
        Dataset.from_records([{"response": "answer"}])
    )
    assert missing.rows[0].metrics[WITH_REF].value == 5.0
    reply_grade(client, 6, "outside the declared scoring range")
    bad = await Evaluator([reference_metric]).arun(data)
    assert bad.rows[0].metrics[WITH_REF].status == "invalid_output"


@pytest.mark.asyncio
async def test_per_row_rubric_cache_and_readback(client, judge, tmp_path):
    store = create_storage_backend("storage.local.jsonl", root=tmp_path)
    cache = create_cache_backend(
        "cache.local.sqlite", path=tmp_path / "cache.sqlite", namespace="rubric"
    )
    evaluator = Evaluator(
        [create_metric(INSTANCE, llm=judge)],
        config=ExecutionConfig(max_concurrency=1),
        backend=store,
        cache=cache,
        include_inputs=False,
    )
    rows = [
        {"case_id": i, "response": "answer", "rubrics": rules}
        for i, rules in enumerate((rubric("A"), rubric("B"), {}))
    ]
    reply_grade(client, 2, "A feedback")
    reply_grade(client, 4, "B feedback")
    result = await evaluator.arun(Dataset.from_records(rows))
    read = store.load_run(result.run_id)
    a, b, bad = [read.row_for(i).metrics[INSTANCE] for i in range(3)]
    assert [(a.value, a.reason), (b.value, b.reason)] == [
        (2.0, "A feedback"),
        (4.0, "B feedback"),
    ]
    assert a.inputs_sha256 != b.inputs_sha256
    assert bad.error.field == "rubrics" and bad.usage is None
    rows[0]["rubrics"] = dict(reversed(rubric("A").items()))
    rows[1]["rubrics"]["score1_description"] = "new"
    reply_grade(client, 5, "new order")
    reply_grade(client, 3, "changed rule")
    rerun = await evaluator.arun(Dataset.from_records(rows))
    assert not rerun.row_for(0).metrics[INSTANCE].cache_hit
    assert rerun.row_for(0).metrics[INSTANCE].value == 5.0
    assert rerun.row_for(0).metrics[INSTANCE].inputs_sha256 != a.inputs_sha256
    cached = await evaluator.arun(Dataset.from_records(rows))
    assert cached.row_for(0).metrics[INSTANCE].cache_hit
    assert rerun.row_for(1).metrics[INSTANCE].value == 3.0 and len(client.calls) == 4


@pytest.mark.asyncio
async def test_cancelled_vote_starts_no_further_calls(judge, client):
    started = asyncio.Event()

    async def wait_for_cancel():
        started.set()
        await asyncio.Event().wait()

    client.responses.append(wait_for_cancel)
    binding = create_metric(
        NUMERIC, llm=judge, criteria="gate", levels=[0, 1], strictness=3
    )
    collector = RunCollector()
    task = asyncio.create_task(
        Evaluator([binding]).arun(
            Dataset.from_records([{"response": "a"}]), collector=collector
        )
    )
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert collector.snapshot().run_status == "cancelled" and len(client.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("plugin,count", [(NUMERIC, 1), (NUMERIC, 3), (FREE, 1)])
async def test_direct_vote_preserves_cancellation_when_transport_ignores_it(
    judge, client, plugin, count
):
    started, release, finished = (asyncio.Event() for _ in range(3))
    if plugin == FREE:
        reply_grade(client, 1, "late")
    else:
        vote(client, 1, "late")
    reply = client.responses.pop()

    async def stubborn():
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            await release.wait()
            finished.set()
            return reply

    client.responses.append(stubborn)
    client.responses.extend([reply] * (count - 1))
    config = (
        {"criteria": "grade", "levels": [0, 1], "strictness": count}
        if plugin == NUMERIC
        else {}
    )
    metric = create_metric(plugin, llm=judge, **config)
    task = asyncio.create_task(metric.ascore_inputs({"response": "answer"}))
    await asyncio.wait_for(started.wait(), 2)
    task.cancel("stop grading")
    try:
        done, _ = await asyncio.wait({task}, timeout=0.3)
        assert task in done
        with pytest.raises(asyncio.CancelledError, match="stop grading"):
            await task
    finally:
        release.set()
        await asyncio.wait_for(finished.wait(), 2)
        await asyncio.gather(task, return_exceptions=True)
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_direct_timeout_stops_structured_retry_after_late_invalid_vote(
    judge, client
):
    finished = asyncio.Event()
    client.chat_reply("invalid JSON")
    invalid = client.responses.pop()

    async def late_reply():
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            finished.set()
            return invalid

    client.responses.append(late_reply)
    vote(client, 1, "unneeded retry")
    metric = create_metric(NUMERIC, llm=judge, criteria="grade", levels=[0, 1])
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.03):
            await metric.ascore(response="answer")
    await asyncio.wait_for(finished.wait(), 1)
    assert len(client.calls) == 1

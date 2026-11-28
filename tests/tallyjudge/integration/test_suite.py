import asyncio
import json

import pytest

from tallyjudge import (
    Dataset,
    Evaluator,
    Suite,
    create_language_model,
    create_metric,
    create_storage_backend,
)
from tallyjudge.cli import main

FAITH = "metric.rag.faithfulness"
ROW = {
    "case_id": "hours",
    "user_input": "图书馆什么时候开放？",
    "response": "工作日上午九点开放，周末也开放。",
    "retrieved_contexts": ["图书馆公告：工作日上午九点开放。"],
}
SUITE = """
id = "suite.library.grounding"
revision = "1"

[dataset]
path = "samples.jsonl"

[models.judge]
plugin = "model.replay.jsonl"
path = {recording}

[metrics.grounded]
plugin = "metric.rag.faithfulness"
llm = "judge"

[execution]
max_concurrency = 2
"""


async def _record(client, path):
    client.chat_reply(json.dumps({"statements": ["工作日九点开放", "周末开放"]}))
    client.chat_reply(
        json.dumps(
            {
                "statements": [
                    {"statement": "工作日九点开放", "reason": "公告写明", "verdict": 1},
                    {"statement": "周末开放", "reason": "公告未提", "verdict": 0},
                ]
            }
        )
    )
    live = create_language_model(
        "model.openai_compatible.chat_completions", client=client, model="judge"
    )
    judge = create_language_model("model.replay.jsonl", path=path, llm=live)
    await Evaluator([create_metric(FAITH, llm=judge)]).arun(Dataset.from_records([ROW]))


@pytest.fixture
def suite_file(client, tmp_path):
    asyncio.run(_record(client, tmp_path / "judge.jsonl"))
    (tmp_path / "samples.jsonl").write_text(
        json.dumps(ROW, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    path = tmp_path / "suite.toml"
    path.write_text(
        SUITE.format(recording=json.dumps(str(tmp_path / "judge.jsonl"))),
        encoding="utf-8",
    )
    return path


@pytest.mark.asyncio
async def test_suite_scores_and_records_its_identity(suite_file):
    result = await Suite.load(suite_file).arun()
    assert result.rows[0].metrics["grounded"].value == 0.5
    assert result.metadata["suite"] == {
        "id": "suite.library.grounding",
        "revision": "1",
    }
    assert result.execution_config["max_concurrency"] == 2


@pytest.mark.asyncio
async def test_overlay_replaces_a_named_model_without_touching_the_suite(
    suite_file, tmp_path
):
    other = tmp_path / "other-judge.jsonl"
    other.write_bytes((tmp_path / "judge.jsonl").read_bytes())
    overlay = tmp_path / "overlay.toml"
    overlay.write_text(
        f'[models.judge]\nplugin = "model.replay.jsonl"\npath = {json.dumps(str(other))}\n',
        encoding="utf-8",
    )
    base = await Suite.load(suite_file).arun()
    swapped = Suite.load(suite_file, overlay)
    assert swapped.effective_config()["models"]["judge"]["path"] == str(other)
    result = await swapped.arun()
    assert result.rows[0].metrics["grounded"].value == 0.5
    assert (
        result.metrics["grounded"]["config_snapshot"]
        != base.metrics["grounded"]["config_snapshot"]
    )


@pytest.mark.asyncio
async def test_client_is_created_inside_the_run(suite_file, tmp_path, monkeypatch):
    (tmp_path / "suite_clients.py").write_text(
        "import json\n"
        "from tallyjudge.sdk.testing import ScriptedClient\n"
        "def make():\n"
        "    client = ScriptedClient()\n"
        "    client.chat_reply(json.dumps({'statements': []}))\n"
        "    return client\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    overlay = tmp_path / "live.toml"
    overlay.write_text(
        "[models.judge]\n"
        'plugin = "model.openai_compatible.chat_completions"\n'
        'client = "suite_clients:make"\n'
        'model = "judge"\n',
        encoding="utf-8",
    )
    result = await Suite.load(suite_file, overlay).arun()
    score = result.rows[0].metrics["grounded"]
    assert (score.status, score.error.code) == ("undefined", "no_statements")
    assert score.usage.generation.requests == 1


def test_command_line_shows_and_runs_a_suite(suite_file, tmp_path, capsys):
    assert main(["suite", "show", str(suite_file)]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["metrics"]["grounded"]["llm"] == "judge"
    assert (
        main(["suite", "run", str(suite_file), "--store", str(tmp_path / "runs")]) == 0
    )
    printed = json.loads(capsys.readouterr().out)
    stored = create_storage_backend("storage.local.jsonl", root=tmp_path / "runs")
    assert stored.load_run(printed["run_id"]).summary["grounded"]["mean"] == 0.5

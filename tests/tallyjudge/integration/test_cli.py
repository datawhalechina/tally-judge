import json
import subprocess
import sys

import pytest

from tallyjudge import Dataset, Evaluator, create_metric, create_storage_backend
from tallyjudge.cli import main

EXACT = "metric.text.exact_match"


def test_cli_and_python_agree_across_processes(tmp_path):
    data = [
        {"case_id": "z", "response": "ok", "reference": "ok"},
        {"case_id": "a", "response": 5, "reference": "ok"},
    ]
    path = tmp_path / "input.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in data), encoding="utf-8")

    def cli(*args):
        return subprocess.run(
            [sys.executable, "-m", "tallyjudge.cli", *map(str, args)],
            capture_output=True,
            text=True,
            check=True,
        )

    metrics = json.loads(cli("metrics", "list").stdout)
    assert EXACT in [m["id"] for m in metrics]
    saved = json.loads(
        cli("run", "--input", path, "--metric", EXACT, "--store", tmp_path).stdout
    )
    shown = json.loads(cli("runs", "show", saved["run_id"], "--store", tmp_path).stdout)
    expected = Evaluator([create_metric(EXACT)]).run(Dataset.from_records(data))
    assert shown["rows"] == saved["rows"] == expected.to_dict()["rows"]
    assert shown["summary"] == expected.summary and shown["inputs"] == data
    assert (
        create_storage_backend("storage.local.jsonl", root=tmp_path)
        .load_run(saved["run_id"])
        .rows
        == expected.rows
    )


def test_cli_exit_codes(tmp_path, monkeypatch):
    args = [
        "run",
        "--input",
        str(tmp_path / "missing"),
        "--store",
        str(tmp_path),
        "--metric",
    ]
    assert main([*args, EXACT]) == 1
    with pytest.raises(SystemExit) as error:
        main([*args, "metric.rag.faithfulness"])
    assert error.value.code == 2
    path = tmp_path / "missing"
    path.write_text("{}\n", encoding="utf-8")

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(Evaluator, "run", interrupt)
    assert main([*args, EXACT]) == 130

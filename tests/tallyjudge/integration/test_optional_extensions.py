"""Optional exports, token-aware splitting and explicit experiment source metadata."""

import json
import subprocess

import pytest

from tallyjudge import Dataset, Evaluator, create_event_sink, create_metric, experiment
from tallyjudge.plugins.synthesis.documents import SourceDocument, split_documents
from tallyjudge.plugins.tokenizers import huggingface_counter, tiktoken_counter
from tallyjudge.plugins.versioning.git import git_revision


def test_git_revision_on_selected_checkout(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(tmp_path), *args], text=True
        ).strip()

    git("init", "-q")
    git(
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "initial",
    )
    expected = git("rev-parse", "HEAD")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "unrelated.git"))
    version = git_revision(tmp_path, "HEAD")
    run = experiment(lambda row: {}, metadata={"git_revision": version}).run(
        Dataset.from_records([])
    )
    assert run.metadata["git_revision"] == expected
    with pytest.raises(subprocess.CalledProcessError):
        git_revision(tmp_path, "missing")


def test_local_tokenizers_as_chunk_size(tmp_path, monkeypatch):
    tk = pytest.importorskip("tiktoken")
    hf = pytest.importorskip("tokenizers")
    codec = hf.Tokenizer(
        hf.models.WordLevel(
            {"[UNK]": 0, "one": 1, "two": 2, "three": 3}, unk_token="[UNK]"
        )
    )
    codec.pre_tokenizer = hf.pre_tokenizers.Whitespace()
    path = tmp_path / "tokenizer.json"
    codec.save(str(path))
    # A real byte-level tiktoken encoding needs no model download.
    local = tk.Encoding(
        name="bytes",
        pat_str=r"(?s).",
        mergeable_ranks={bytes([i]): i for i in range(256)},
        special_tokens={},
    )
    monkeypatch.setattr(tk, "get_encoding", lambda name: local)
    doc = SourceDocument("id", "one two three")
    for counter, limit in (
        (huggingface_counter(path), 2),
        (tiktoken_counter("bytes"), 7),
    ):
        rows = list(split_documents([doc], max_size=limit, measure=counter))
        assert [r["text"] for r in rows] == ["one two", "three"]


def test_mlflow_artifact_and_export_failure(tmp_path, monkeypatch):
    tracking = pytest.importorskip("mlflow.tracking")
    uri = (tmp_path / "tracking").as_uri()
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    client = tracking.MlflowClient(tracking_uri=uri)
    exp = client.create_experiment("offline")
    sink = create_event_sink(
        "event.mlflow.tracking", tracking_uri=uri, experiment_id=exp
    )
    data = Dataset.from_records(
        [{"case_id": "a", "response": "sensitive", "reference": "sensitive"}]
    )
    metric = create_metric("metric.text.exact_match")
    evaluator = Evaluator([metric], event_sink=sink)
    result = evaluator.run(data)
    assert result.events_complete
    (remote,) = client.search_runs([exp])
    assert (
        remote.data.tags["tallyjudge.run_id"] == result.run_id
        and remote.info.status == "FINISHED"
    )
    path = client.download_artifacts(remote.info.run_id, "tallyjudge/events.jsonl")
    from pathlib import Path

    contents = Path(path).read_text(encoding="utf-8")
    events = [json.loads(line) for line in contents.splitlines()]
    assert [e["sequence"] for e in events] == list(range(1, len(events) + 1))
    assert "sensitive" not in contents

    def fail(*args, **kwargs):
        raise OSError("export unavailable")

    monkeypatch.setattr(sink.client, "log_text", fail)
    failed = evaluator.run(data)
    assert (
        failed.events_complete is False
        and failed.rows[0].metrics[metric.spec.id].value == 1.0
    )

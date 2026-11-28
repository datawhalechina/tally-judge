import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tallyjudge import (
    Dataset,
    MetricBinding,
    create_metric,
    create_runner,
    experiment,
)
from tallyjudge.sdk.errors import PluginError

EXACT = "metric.text.exact_match"
ROWS = [
    {"case_id": "a", "user_input": "ping", "reference": "PING"},
    {
        "case_id": "b",
        "user_input": "pong",
        "reference": "nope",
        "reference_contexts": ["gold evidence"],
        "expected_tools": ["lookup"],
    },
]
ECHO = (
    "import json,sys; row=json.load(sys.stdin); "
    "print(json.dumps({'response': row['user_input'].upper(), 'received': row}))"
)


class _Upper(BaseHTTPRequestHandler):
    def do_POST(self):
        row = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        body = json.dumps(
            {"response": row["user_input"].upper(), "received": row}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def endpoint():
    server = HTTPServer(("127.0.0.1", 0), _Upper)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/"
    server.shutdown()
    server.server_close()


@pytest.fixture(params=["runner.http.json", "runner.command.json"])
def runner_settings(request):
    if request.param == "runner.http.json":
        return request.param, {"url": request.getfixturevalue("endpoint")}
    return request.param, {"command": [sys.executable, "-c", ECHO]}


@pytest.mark.parametrize(
    ("options", "expected_input"),
    [
        ({}, {"user_input": "pong"}),
        (
            {"input_fields": ["user_input", "case_id"]},
            {"user_input": "pong", "case_id": "b"},
        ),
    ],
)
def test_runner_sends_selected_fields_and_scores_against_dataset(
    runner_settings, options, expected_input
):
    plugin, settings = runner_settings
    runner = create_runner(plugin, **settings, **options)
    result = experiment(runner).run(
        Dataset.from_records(ROWS), metrics=[create_metric(EXACT)]
    )
    assert result.row_for("b").task.output["received"] == expected_input
    assert [row.metrics[EXACT].value for row in result.rows] == [1.0, 0.0]


def test_failed_command_is_recorded_as_an_experiment_error():
    runner = create_runner(
        "runner.command.json", command=[sys.executable, "-c", "raise SystemExit(3)"]
    )
    result = experiment(runner).run(
        Dataset.from_records(ROWS[:1]), metrics=[create_metric(EXACT)]
    )
    row = result.rows[0]
    assert (row.task.status, row.task.error.category) == ("error", "experiment_error")
    assert row.metrics[EXACT].status == "not_started"


def test_runner_settings_are_validated():
    with pytest.raises(PluginError, match="url"):
        create_runner("runner.http.json", url="ftp://example")
    with pytest.raises(PluginError, match="command"):
        create_runner("runner.command.json", command=[])


def test_each_row_runs_in_a_fresh_copy_of_the_template_and_reports_its_files(tmp_path):
    template = tmp_path / "template"
    template.mkdir()
    (template / "orders.txt").write_text("1042:paid\n", encoding="utf-8")
    script = (
        "import json,sys; row=json.load(sys.stdin); f=open('orders.txt','a');"
        "f.write(row['user_input']+':refunded\\n'); f.close();"
        "print(json.dumps({'response': 'done'}))"
    )
    runner = create_runner(
        "runner.command.json",
        command=[sys.executable, "-c", script],
        template=str(template),
        collect=["orders.txt", "missing.txt"],
    )
    metric = create_metric("metric.workflow.final_state")
    binding = MetricBinding(metric, input_map={"final_state": "workspace_files"})
    rows = [
        {
            "case_id": name,
            "user_input": name,
            "expected_state": {"orders.txt": f"1042:paid\n{name}:refunded\n"},
        }
        for name in ("a", "b")
    ]
    result = experiment(runner).run(Dataset.from_records(rows), metrics=[binding])
    assert [row.metrics[binding.key].value for row in result.rows] == [1.0, 1.0]
    assert result.row_for("a").task.output["workspace_files"]["missing.txt"] is None
    assert (template / "orders.txt").read_text(encoding="utf-8") == "1042:paid\n"
    with pytest.raises(PluginError, match="Invalid plugin configuration"):
        create_runner("runner.command.json", command=["x"], collect=["a"])

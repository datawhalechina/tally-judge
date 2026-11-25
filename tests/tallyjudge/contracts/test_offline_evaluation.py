import subprocess
import sys
import textwrap


def test_default_entry_points_do_not_open_external_connections():
    script = """
        import sys
        contacts = []
        def audit(event, args):
            if event in ("socket.connect", "socket.getaddrinfo"):
                address = args[1][0] if event=="socket.connect" else args[0]
                if address in ("localhost","127.0.0.1","::1"): return
                contacts.append(str(address))
                raise AssertionError("external network access")
        sys.addaudithook(audit)
        from tallyjudge import Dataset, Evaluator, create_metric
        from tallyjudge.registry import registry
        assert registry.list()
        run=Evaluator([create_metric("metric.text.exact_match")]).run(Dataset.from_records([{"reference":"x","response":"x"}]))
        assert run.rows[0].metrics["metric.text.exact_match"].value==1.0
        assert contacts==[], contacts
        assert not {"dspy","mlflow","langchain_core","llama_index","sknetwork","networkx","tiktoken","tokenizers","ragas"}.intersection(sys.modules)
    """
    process = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)], capture_output=True, text=True
    )
    assert process.returncode == 0, process.stderr

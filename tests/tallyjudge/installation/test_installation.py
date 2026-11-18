"""Import isolation and a real wheel-install check, runnable from local CI.

uv run python tests/tallyjudge/installation/test_installation.py --wheel-dir <dir>
The wheel check creates a fresh venv outside the checkout and installs only the
wheel's runtime dependencies. It does not import pytest or the source package.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROBE = """
import importlib.metadata as metadata
import json
import sys
from pathlib import Path
import tallyjudge
from tallyjudge import Dataset, Evaluator, MetricBinding, create_metric
from tallyjudge.registry import registry
assert sys.version_info >= (3, 12)
assert Path(tallyjudge.__file__).with_name('py.typed').is_file()
assert len(registry.list()) == 2
assert not any(name.startswith('tallyjudge.plugins.') for name in sys.modules)
metrics = [create_metric(item.id) for item in registry.list()]
result = Evaluator(metrics).run(Dataset.from_records([{'response': 'cat', 'reference': 'cat'}, {'response': 'dog', 'reference': 'cat'}]))
assert len(result.rows) == 2
assert all(item['mean'] == 0.5 for item in result.summary.values())
assert json.loads(result.to_json())['schema_version'] == 1
assert result.run_status == 'completed' and result.stop_reason is None
assert not {'ragas', 'langchain', 'langchain_core', 'openai', 'instructor', 'datasets', 'PIL', 'networkx', 'sknetwork'}.intersection(sys.modules)
if len(sys.argv) > 1:
    checkout = Path(sys.argv[1]).resolve()
    assert not Path(tallyjudge.__file__).resolve().is_relative_to(checkout)
    assert not Path.cwd().resolve().is_relative_to(checkout)
    installed = {item.metadata['Name'].lower().replace('_', '-') for item in metadata.distributions()}
    assert not {'ragas', 'langchain', 'langchain-core', 'openai', 'instructor', 'datasets', 'pillow', 'networkx', 'scikit-network', 'pytest'}.intersection(installed)
    assert Path(tallyjudge.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    print('Wheel installation verified:', tallyjudge.__file__)
"""


def test_import_is_independent_of_legacy_and_provider_packages():
    process = subprocess.run(
        [sys.executable, "-c", PROBE], check=False, capture_output=True, text=True
    )
    assert process.returncode == 0, process.stdout + process.stderr


def verify_wheel(wheel_dir: Path) -> None:
    wheels = list(wheel_dir.resolve().glob("tallyjudge-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Expected exactly one TallyJudge wheel in wheel-dir")
    checkout = Path(__file__).resolve().parents[3]
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
    }
    with tempfile.TemporaryDirectory(prefix="tallyjudge-wheel-check-") as temp:
        directory = Path(temp)
        venv = directory / "venv"
        subprocess.run(
            ["uv", "venv", "--python", sys.executable, str(venv)],
            cwd=directory,
            env=env,
            check=True,
        )
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), str(wheels[0])],
            cwd=directory,
            env=env,
            check=True,
        )
        subprocess.run(
            [str(python), "-I", "-c", PROBE, str(checkout)],
            cwd=directory,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", required=True, type=Path)
    verify_wheel(parser.parse_args().wheel_dir)

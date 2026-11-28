import json
import re
import subprocess
import sys
from pathlib import Path
from textwrap import indent

from tallyjudge.registry import registry

ROOT = Path(__file__).resolve().parents[3]


def test_readme_names_every_registered_plugin():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    missing = [spec.id for spec in registry.list() if spec.id not in readme]
    assert not missing


def test_field_conventions_cover_every_metric_input():
    document = (ROOT / "docs" / "evaluate" / "reference" / "fields.md").read_text(
        encoding="utf-8"
    )
    settings = {
        "metric.custom.numeric": {"criteria": "x"},
        "metric.custom.categorical": {"criteria": "x", "labels": ["a", "b"]},
        "metric.conversation.criteria": {"criteria": "x"},
    }
    for spec in registry.list(kind="metric"):
        described = registry.describe(spec.id, **settings.get(spec.id, {}))
        for name in described.inputs:
            assert f"| `{name}` |" in document, (spec.id, name)
        assert f"`{spec.id.removeprefix('metric.')}`" in document, spec.id


def test_factory_examples_type_check(tmp_path):
    blocks = re.findall(
        r"```python\n(.*?)\n```", (ROOT / "README.md").read_text(encoding="utf-8"), re.S
    )
    faithfulness = next(
        block for block in blocks if "faithfulness = create_metric(" in block
    )
    grading = next(block for block in blocks if "grade = create_metric(" in block)
    synthesis = next(block for block in blocks if "async def make_testset(" in block)
    example = tmp_path / "examples.py"
    example.write_text(
        faithfulness
        + "\n\nfrom tallyjudge.sdk import LanguageModel\n"
        + "async def grade_answer(judge: LanguageModel):\n"
        + indent(grading, "    ")
        + "\n\n"
        + synthesis,
        encoding="utf-8",
    )
    (tmp_path / "openai.pyi").write_text(
        "from typing import Any\ndef AsyncOpenAI() -> Any: ...\n", encoding="utf-8"
    )
    settings = tmp_path / "pyrightconfig.json"
    settings.write_text(
        json.dumps(
            {
                "include": [example.name],
                "extraPaths": [str(ROOT)],
                "pythonVersion": "3.12",
                "typeCheckingMode": "basic",
            }
        ),
        encoding="utf-8",
    )
    checked = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyright",
            "--project",
            str(settings),
            "--pythonpath",
            sys.executable,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr

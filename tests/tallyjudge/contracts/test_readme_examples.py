"""Type-check the user-facing model scoring examples against the public package."""

import json
import re
import subprocess
import sys
from pathlib import Path
from textwrap import indent


def test_factory_examples_type_check(tmp_path):
    root = Path(__file__).resolve().parents[3]
    blocks = re.findall(
        r"```python\n(.*?)\n```", (root / "README.md").read_text(encoding="utf-8"), re.S
    )
    faithfulness = next(
        block for block in blocks if "faithfulness = create_metric(" in block
    )
    grading = next(block for block in blocks if "grade = create_metric(" in block)
    synthesis = next(block for block in blocks if "async def make_testset(" in block)
    example = tmp_path / "examples.py"
    example.write_text(
        faithfulness
        + "\n\nfrom tallyjudge.interfaces.models import LanguageModel\n"
        + "async def grade_answer(judge: LanguageModel):\n"
        + indent(grading, "    ")
        + "\n\n"
        + synthesis,
        encoding="utf-8",
    )
    # The OpenAI SDK is optional. The metric factory and protocol always come from this checkout.
    (tmp_path / "openai.pyi").write_text(
        "from typing import Any\ndef AsyncOpenAI() -> Any: ...\n", encoding="utf-8"
    )
    settings = tmp_path / "pyrightconfig.json"
    settings.write_text(
        json.dumps(
            {
                "include": [example.name],
                "extraPaths": [str(root)],
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

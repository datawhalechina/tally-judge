"""Run ONLY with the old ec00339 environment, never with the new package.

The baseline checkout may be a Git worktree or an exact git archive export.
This tool verifies that the imported old source matches that export before scoring.
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
from pathlib import Path

BASELINE = "ec003395bb37c223fc981e6567162513aec52a71"
CASES = [
    ("equal", {"reference": "Paris", "response": "Paris"}),
    ("different", {"reference": "Paris", "response": "Lyon"}),
    ("substring", {"reference": "cat", "response": "the cat sat"}),
    ("case_sensitive", {"reference": "Cat", "response": "cat"}),
    ("whitespace", {"reference": " cat", "response": "cat"}),
    ("both_empty", {"reference": "", "response": ""}),
    ("empty_reference", {"reference": "", "response": "text"}),
    ("empty_response", {"reference": "text", "response": ""}),
    ("unicode", {"reference": "北京", "response": "我在北京"}),
    ("newlines", {"reference": "a\nb", "response": "a\nb"}),
    ("integer_and_string", {"reference": 1, "response": "1"}),
    ("equal_integers", {"reference": 1, "response": 1}),
    ("booleans", {"reference": True, "response": True}),
    ("lists", {"reference": ["a"], "response": ["a"]}),
    ("null", {"reference": None, "response": None}),
    ("missing_reference", {"response": "a"}),
    ("missing_response", {"reference": "a"}),
]


async def generate(source_root):
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    from ragas.metrics.collections._string import ExactMatch, StringPresence
    source = source_root / "src/ragas/metrics/collections/_string.py"
    imported = Path(inspect.getfile(ExactMatch))
    if imported.read_bytes() != source.read_bytes():
        raise RuntimeError("Imported legacy metric differs from the baseline export")
    output = {
        "source_commit": BASELINE,
        "source_file": "src/ragas/metrics/collections/_string.py",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "optimized": not __debug__,
        "environment": dict(sorted((item.metadata["Name"], item.version) for item in importlib.metadata.distributions())),
        "metrics": {},
    }
    if not __debug__:
        raise RuntimeError("Generate the baseline without python -O")
    for cls, plugin_id in [(ExactMatch, "metric.text.exact_match"), (StringPresence, "metric.text.string_presence")]:
        cases = []
        for name, inputs in CASES:
            try:
                result = await cls().ascore(**inputs)
                old = {"value": result.value, "value_type": type(result.value).__name__}
            except Exception as error:
                old = {"exception_type": type(error).__name__}
            difference = None
            if set(inputs) != {"reference", "response"}:
                difference = "missing_field_validation"
            elif not all(isinstance(value, str) for value in inputs.values()):
                difference = "strict_string_validation"
            cases.append({"name": name, "inputs": inputs, "old": old, "intentional_difference": difference})
        output["metrics"][plugin_id] = {"old_symbol": f"ragas.metrics.collections._string.{cls.__name__}", "new_revision": "1", "cases": cases}
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = asyncio.run(generate(args.source_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

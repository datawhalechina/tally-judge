import dataclasses
import inspect
import json
import os
from pathlib import Path

import tallyjudge.sdk as sdk

RECORD = Path(__file__).with_name("sdk_surface.json")
CONTRACTS = {
    "Metric": ("ascore_inputs", "config_snapshot"),
    "MetricBase": ("ascore", "ascore_inputs", "config_snapshot", "score"),
    "LanguageModel": ("agenerate", "config_snapshot"),
    "EmbeddingModel": ("aembed_texts", "config_snapshot"),
    "StorageBackend": ("save_run", "load_run"),
    "ScoreCache": ("get", "put"),
    "EventSink": ("emit",),
    "Importer": ("convert",),
    "Runner": ("__call__",),
    "Analyzer": ("analyze",),
}
RECORDS = ("Plugin", "PluginSpec", "KindSpec", "MetricSpec", "MetricResult")


def surface():
    return {
        "names": sorted(sdk.__all__),
        "contracts": {
            name: {
                method: str(inspect.signature(getattr(getattr(sdk, name), method)))
                for method in methods
            }
            for name, methods in CONTRACTS.items()
        },
        "records": {
            name: [
                field.name
                for field in dataclasses.fields(getattr(sdk, name))
                if not field.name.startswith("_")
            ]
            for name in RECORDS
        },
    }


def test_sdk_surface_matches_the_recorded_version_1_surface():
    current = surface()
    if os.environ.get("TALLYJUDGE_UPDATE_SURFACE"):
        RECORD.write_text(
            json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    assert current == json.loads(RECORD.read_text(encoding="utf-8")), (
        "The plugin-facing surface changed. If intended, rerun with "
        "TALLYJUDGE_UPDATE_SURFACE=1 and review the diff of sdk_surface.json."
    )

import hashlib
import json

import pytest

from tallyjudge import Dataset


def test_freeze_and_canonical_manifest():
    rows = [{"case_id": 7, "response": "北京", "metadata": {"tags": ["a"]}}]
    dataset = Dataset.from_records(rows)
    rows[0]["metadata"]["tags"].append("b")
    detached = dataset.records
    detached[0]["metadata"]["tags"].clear()
    assert dataset.record(0)["metadata"]["tags"] == ["a"]
    manifest = dataset.manifest[0]
    canonical = json.dumps(
        dataset.record(0), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    assert manifest == {
        "case_id": "7",
        "input_index": 0,
        "id_source": "provided",
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }
    assert (
        Dataset.from_records([dict(reversed(list(dataset.record(0).items())))]).manifest
        == dataset.manifest
    )


@pytest.mark.parametrize(
    "rows",
    [
        [{"case_id": 1}, {"case_id": "1"}],
        [{"case_id": "a"}, {}],
        [{}, {"case_id": "row-0"}],
        [{"case_id": None}],
        [{"case_id": True}],
        [{"case_id": 1.5}],
        [{"response": float("nan")}],
        [{"response": object()}],
        [{1: "x"}],
        ["text"],
    ],
)
def test_invalid_dataset(rows):
    with pytest.raises((ValueError, TypeError)):
        Dataset.from_records(rows)


def test_generated_custom_and_empty_ids():
    assert [item["case_id"] for item in Dataset.from_records([{}, {}]).manifest] == [
        "row-0",
        "row-1",
    ]
    assert (
        Dataset.from_records([{"id": -2}], id_field="id").manifest[0]["case_id"] == "-2"
    )
    assert Dataset.from_records([]).manifest == []

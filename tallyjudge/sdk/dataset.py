import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


def encode_json(value: Any, *, sort_keys: bool = False) -> str:
    def check(item: Any) -> None:
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("JSON object keys must be strings")
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        elif item is not None and type(item) not in (str, int, float, bool):
            raise ValueError("Inputs and snapshots must contain only JSON values")

    check(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=sort_keys,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_json(value: Any) -> str:
    return encode_json(value, sort_keys=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def json_sha256(value: Any, *, sort_keys: bool) -> str | None:
    try:
        encoded = encode_json(value, sort_keys=sort_keys).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(encoded).hexdigest()


def arguments_sha256(inputs: Mapping[str, Any]) -> str | None:
    return json_sha256(dict(sorted(inputs.items())), sort_keys=False)


@dataclass(frozen=True)
class Dataset:
    """Immutable records. Accessors return detached copies, never internal data."""

    _records: tuple[str, ...]
    _ids: tuple[str, ...]
    id_source: str

    @classmethod
    def from_records(
        cls, records: Iterable[Mapping[str, Any]], *, id_field: str = "case_id"
    ) -> "Dataset":
        if not isinstance(id_field, str) or not id_field:
            raise ValueError("id_field must be a non-empty string")
        frozen = []
        supplied = []
        ids = []
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise ValueError("Each record must be a mapping")
            row = dict(record)
            supplied.append(id_field in row)
            case_id = row.get(id_field, f"row-{index}")
            if type(case_id) not in (str, int):
                raise ValueError("case_id must be a string or integer (not bool)")
            ids.append(str(case_id))
            frozen.append(encode_json(row))
        if any(supplied) and not all(supplied):
            raise ValueError("IDs must be supplied for every row or for none")
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate case_id after string normalization")
        return cls(
            tuple(frozen),
            tuple(ids),
            "provided" if supplied and all(supplied) else "generated",
        )

    def __len__(self) -> int:
        return len(self._records)

    def record(self, index: int) -> dict[str, Any]:
        return json.loads(self._records[index])

    @property
    def records(self) -> list[dict[str, Any]]:
        return [self.record(index) for index in range(len(self))]

    @property
    def manifest(self) -> list[dict[str, Any]]:
        return self.manifest_of(map(json.loads, self._records))

    def manifest_of(self, records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "case_id": case_id,
                "input_index": index,
                "id_source": self.id_source,
                "sha256": canonical_sha256(record),
            }
            for index, (case_id, record) in enumerate(
                zip(self._ids, records, strict=True)
            )
        ]

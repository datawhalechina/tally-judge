import copy
from typing import Any

SCHEMA_VERSION = 4


def _v1_to_v2(data: dict[str, Any]) -> None:
    for binding in data["metrics"].values():
        binding["output_kind"] = "numeric"


def _v2_to_v3(data: dict[str, Any]) -> None:
    for binding in data["metrics"].values():
        config = binding.get("config_snapshot", {})
        if "value_range" not in binding and config.keys() <= {
            "function",
            "value_range",
        }:
            binding["value_range"] = config.get("value_range")


def _v3_to_v4(data: dict[str, Any]) -> None:
    pass


_STEPS = (_v1_to_v2, _v2_to_v3, _v3_to_v4)


def upgrade(data: dict[str, Any]) -> dict[str, Any]:
    version = data.get("schema_version")
    if type(version) is not int or not 1 <= version <= SCHEMA_VERSION:
        raise ValueError(f"Unsupported run schema version: {version!r}")
    if version == SCHEMA_VERSION:
        return data
    data = copy.deepcopy(data)
    for step in _STEPS[version - 1 :]:
        step(data)
    data["schema_version"] = SCHEMA_VERSION
    return data

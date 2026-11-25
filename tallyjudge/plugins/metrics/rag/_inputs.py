from typing import Annotated, Any

from pydantic import Field


def required_text_inputs(*names: str, passages: bool = False) -> dict[str, Any]:
    inputs: dict[str, Any] = dict.fromkeys(names, Annotated[str, Field(min_length=1)])
    if passages:
        inputs["retrieved_contexts"] = Annotated[list[str], Field(min_length=1)]
    return inputs

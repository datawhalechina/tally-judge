"""Text prompt rendering migrated from Ragas BasePrompt.to_string (ec00339, Apache-2.0)."""

import json
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class PromptInput:
    messages: tuple[Message, ...]

    @classmethod
    def from_text(cls, text: str) -> "PromptInput":
        return cls((Message("user", text),))


class StructuredPrompt[I: BaseModel, O: BaseModel]:
    input_model: type[I]
    output_model: type[O]
    instruction: str
    examples: list[tuple[I, O]]

    def render(self, data: I) -> PromptInput:
        """Render the revisioned instruction, output schema, examples and sample in order."""
        schema = json.dumps(self.output_model.model_json_schema())
        examples = "\n\n".join(
            f"Example {i}\nInput: {example.model_dump_json(indent=4)}\nOutput: {output.model_dump_json(indent=4)}"
            for i, (example, output) in enumerate(self.examples, 1)
        )
        if examples:
            examples = "--------EXAMPLES-----------\n" + examples
        return PromptInput.from_text(
            f"{self.instruction}\n"
            "Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:\n"
            f"{schema}Do not use single quotes in your response but double quotes,properly escaped with a backslash.\n\n"
            f"{examples}\n-----------------------------\n\n"
            f"Now perform the same with the following input\ninput: {data.model_dump_json(indent=4, exclude_none=True)}\nOutput: "
        )

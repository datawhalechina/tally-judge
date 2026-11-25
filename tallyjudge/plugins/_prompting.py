"""Text prompt rendering migrated from Ragas BasePrompt.to_string (ec00339, Apache-2.0)."""

import json

from pydantic import BaseModel

from tallyjudge.sdk.prompts import PromptInput


class StructuredPrompt[I: BaseModel, O: BaseModel]:
    input_model: type[I]
    output_model: type[O]
    instruction: str
    examples: list[tuple[I, O]]

    _empty_example_prefix = "\n"

    def render(self, data: I) -> PromptInput:
        schema = json.dumps(self.output_model.model_json_schema())
        examples = "\n\n".join(
            f"Example {i}\nInput: {example.model_dump_json(indent=4)}\nOutput: {output.model_dump_json(indent=4)}"
            for i, (example, output) in enumerate(self.examples, 1)
        )
        if examples:
            examples = "\n--------EXAMPLES-----------\n" + examples
        else:
            examples = self._empty_example_prefix
        return PromptInput.from_text(
            f"{self.instruction}\n"
            "Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:\n"
            f"{schema}Do not use single quotes in your response but double quotes,properly escaped with a backslash.\n"
            f"{examples}\n-----------------------------\n\n"
            f"Now perform the same with the following input\ninput: {data.model_dump_json(indent=4, exclude_none=True)}\nOutput: "
        )

from dataclasses import dataclass
from typing import Literal


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

"""Optional token counters for source splitting. Counts are independent of provider billing."""

from collections.abc import Callable
from importlib import import_module
from pathlib import Path


def tiktoken_counter(encoding: str) -> Callable[[str], int]:
    """Count ordinary text tokens using a named tiktoken encoding (cache it for offline use)."""
    codec = import_module("tiktoken").get_encoding(encoding)
    return lambda text: len(codec.encode_ordinary(text))


def huggingface_counter(tokenizer_file: str | Path) -> Callable[[str], int]:
    """Load a local tokenizer.json and count content tokens without adding delimiters."""
    codec = import_module("tokenizers").Tokenizer.from_file(str(tokenizer_file))
    return lambda text: len(codec.encode(text, add_special_tokens=False).ids)

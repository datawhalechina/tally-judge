from collections.abc import Callable
from importlib import import_module
from pathlib import Path


def tiktoken_counter(encoding: str) -> Callable[[str], int]:
    codec = import_module("tiktoken").get_encoding(encoding)
    return lambda text: len(codec.encode_ordinary(text))


def huggingface_counter(tokenizer_file: str | Path) -> Callable[[str], int]:
    codec = import_module("tokenizers").Tokenizer.from_file(str(tokenizer_file))
    return lambda text: len(codec.encode(text, add_special_tokens=False).ids)

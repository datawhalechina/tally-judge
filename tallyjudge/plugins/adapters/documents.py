"""Input adapters for LangChain and LlamaIndex documents."""

from collections.abc import Iterable, Iterator
from typing import Any

from tallyjudge.interfaces.synthesis import SourceDocument


def langchain_documents(documents: Iterable[Any]) -> Iterator[SourceDocument]:
    """Retain explicit IDs or identify each page within its source file."""
    for doc in documents:
        source = doc.id or doc.metadata.get("source")
        if not source:
            raise ValueError("LangChain documents need id or metadata['source']")
        yield SourceDocument(
            str(source),
            doc.page_content,
            tuple(doc.metadata.get("entities", ())),
            page=None if doc.id else doc.metadata.get("page"),
        )


def llamaindex_documents(documents: Iterable[Any]) -> Iterator[SourceDocument]:
    """Read LlamaIndex content without rendering metadata into the source text."""
    for doc in documents:
        yield SourceDocument(
            doc.doc_id, doc.text, tuple(doc.metadata.get("entities", ()))
        )

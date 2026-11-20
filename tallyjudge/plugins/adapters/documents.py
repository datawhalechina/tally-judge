"""Input adapters for LangChain and LlamaIndex documents."""

from collections.abc import Iterable, Iterator
from typing import Any

from tallyjudge.interfaces.synthesis import SourceDocument


def langchain_documents(documents: Iterable[Any]) -> Iterator[SourceDocument]:
    """Read LangChain page content and retain an explicit ID or source metadata."""
    for doc in documents:
        source = doc.id or doc.metadata.get("source")
        if not source:
            raise ValueError("LangChain documents need id or metadata['source']")
        yield SourceDocument(
            str(source), doc.page_content, tuple(doc.metadata.get("entities", ()))
        )


def llamaindex_documents(documents: Iterable[Any]) -> Iterator[SourceDocument]:
    """Read LlamaIndex content without rendering metadata into the source text."""
    for doc in documents:
        yield SourceDocument(
            doc.doc_id, doc.text, tuple(doc.metadata.get("entities", ()))
        )

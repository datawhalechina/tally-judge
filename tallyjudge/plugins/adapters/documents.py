from collections.abc import Iterable, Iterator
from typing import Any

from tallyjudge.sdk.documents import SourceDocument
from tallyjudge.sdk.plugin import Plugin


class LangChainDocuments:
    plugin = Plugin("importer", produces="documents")

    plugin_id = "importer.langchain.documents"

    def convert(self, source: Iterable[Any]) -> Iterator[SourceDocument]:
        for doc in source:
            origin = doc.id or doc.metadata.get("source")
            if not origin:
                raise ValueError("LangChain documents need id or metadata['source']")
            yield SourceDocument(
                str(origin),
                doc.page_content,
                tuple(doc.metadata.get("entities", ())),
                page=None if doc.id else doc.metadata.get("page"),
            )


class LlamaIndexDocuments:
    plugin = Plugin("importer", produces="documents")

    plugin_id = "importer.llamaindex.documents"

    def convert(self, source: Iterable[Any]) -> Iterator[SourceDocument]:
        for doc in source:
            yield SourceDocument(
                doc.doc_id, doc.text, tuple(doc.metadata.get("entities", ()))
            )

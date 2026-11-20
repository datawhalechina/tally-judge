"""Stable document and chunk identities used by synthesis plugins."""

from collections.abc import Callable, Iterable, Iterator

from tallyjudge.interfaces.synthesis import SourceDocument as SourceDocument


def split_documents(
    documents: Iterable[SourceDocument],
    *,
    max_size: int = 1200,
    measure: Callable[[str], int] = len,
) -> Iterator[dict]:
    """Split original text at word boundaries where possible, retaining character offsets."""
    import re

    if max_size < 1:
        raise ValueError("max_size must be positive")
    seen = set()
    for doc in documents:
        if not doc.document_id.strip() or doc.document_id in seen:
            raise ValueError("Document IDs must be nonblank and unique")
        seen.add(doc.document_id)
        # Offsets refer to the original text, even when the size is measured in tokens.
        start = end = 0
        for word in re.finditer(r"\S+", doc.text):
            if end > start and measure(doc.text[start : word.end()]) > max_size:
                yield _record(doc, start, end)
                start = word.start()
            if end == start:
                start = word.start()
            end = word.end()
            while end > start and measure(doc.text[start:end]) > max_size:
                lo, hi = start + 1, end
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if measure(doc.text[start:mid]) <= max_size:
                        lo = mid
                    else:
                        hi = mid - 1
                if measure(doc.text[start:lo]) > max_size:
                    raise ValueError(
                        "One character exceeds max_size in the selected tokenizer"
                    )
                yield _record(doc, start, lo)
                start = lo
        if end > start:
            yield _record(doc, start, end)


def _record(doc: SourceDocument, start: int, end: int) -> dict:
    return {
        "document_id": doc.document_id,
        "chunk_id": f"{doc.document_id}:{start}-{end}",
        "text": doc.text[start:end],
        "start": start,
        "end": end,
        "entities": list(doc.entities),
    }

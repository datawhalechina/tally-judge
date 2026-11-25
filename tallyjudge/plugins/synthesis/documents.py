import re
from collections.abc import Callable, Iterable, Iterator

from tallyjudge.sdk.documents import SourceDocument as SourceDocument


def split_documents(
    documents: Iterable[SourceDocument],
    *,
    max_size: int = 1200,
    measure: Callable[[str], int] = len,
) -> Iterator[dict]:
    if max_size < 1:
        raise ValueError("max_size must be positive")
    seen = set()
    for doc in documents:
        source = (
            doc.document_id
            if doc.page is None
            else f"{doc.document_id}#page={doc.page}"
        )
        if not doc.document_id.strip() or source in seen:
            raise ValueError(
                "Document IDs must be nonblank and each source page must be unique"
            )
        seen.add(source)
        start = end = 0
        for word in re.finditer(r"\S+", doc.text):
            if end > start and measure(doc.text[start : word.end()]) > max_size:
                yield _record(doc, source, start, end)
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
                yield _record(doc, source, start, lo)
                start = lo
        if end > start:
            yield _record(doc, source, start, end)


def _mentioned_entities(text: str, labels: Iterable[str]) -> list[str]:
    folded = text.casefold()
    mentions = []
    for label in labels:
        term = label.strip().casefold()
        if not term:
            continue
        pattern = re.escape(term)
        if re.match(r"[a-z0-9_]", term):
            pattern = rf"(?<![a-z0-9_]){pattern}"
        if re.match(r"[a-z0-9_]", term[-1]):
            pattern += r"(?![a-z0-9_])"
        if re.search(pattern, folded):
            mentions.append(label)
    return mentions


def _record(doc: SourceDocument, source: str, start: int, end: int) -> dict:
    text = doc.text[start:end]
    record = {
        "document_id": doc.document_id,
        "chunk_id": f"{source}:{start}-{end}",
        "text": text,
        "start": start,
        "end": end,
        "entities": _mentioned_entities(text, doc.entities),
    }
    if doc.page is not None:
        record["page"] = doc.page
    return record

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceDocument:
    """Source text with a stable document ID, optional entity labels and page locator."""

    document_id: str
    text: str
    entities: tuple[str, ...] = ()
    page: int | str | None = None

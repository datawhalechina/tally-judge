from dataclasses import dataclass


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    text: str
    entities: tuple[str, ...] = ()
    page: int | str | None = None

from typing import Any, Protocol, runtime_checkable

PRODUCTS = ("documents", "trace", "records")


@runtime_checkable
class Importer(Protocol):
    def convert(self, source: Any) -> Any: ...

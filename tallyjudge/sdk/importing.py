from typing import Any, Protocol, runtime_checkable

PRODUCTS = ("documents", "trace", "records")


@runtime_checkable
class Importer(Protocol):
    """Convert another tool's objects into the product the plugin declares."""

    def convert(self, source: Any) -> Any: ...

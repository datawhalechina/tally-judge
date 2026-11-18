"""Public errors without provider or legacy-framework dependencies."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tallyjudge.core.results import RunResult


class InputValidationError(ValueError):
    """A row does not satisfy its metric's input contract."""

    def __init__(
        self, message: str, *, field: str | None = None, expected: str | None = None
    ) -> None:
        super().__init__(message)
        self.field = field
        self.expected = expected


class PluginError(ValueError):
    """Invalid plugin registration, configuration or construction."""


class EvaluationAborted(RuntimeError):
    """Fail-fast termination with a complete, ordered partial result."""

    def __init__(self, partial_result: "RunResult") -> None:
        super().__init__(
            "Evaluation stopped after an error or timeout; see partial_result"
        )
        self.partial_result = partial_result

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


class RunSaveError(RuntimeError):
    """An opted-in automatic save failed; the finished result is still available."""

    def __init__(self, result: "RunResult") -> None:
        super().__init__(f"Could not save run {result.run_id}")
        self.result = result


class ModelError(RuntimeError):
    """A model transport failed. The provider exception is retained as the cause."""


class ClientLoopError(RuntimeError):
    """A caller-owned client was used outside the event loop it is bound to.

    This is a configuration error for the whole run, not a per-row failure: the
    scheduler aborts the run and re-raises it instead of recording each row.
    """


class ModelOutputError(ValueError):
    """A model response cannot satisfy the requested output contract."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class UndefinedScore(ValueError):
    """The metric ran but its mathematical result is undefined."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class UsageScopeClosed(RuntimeError):
    """A finished score can no longer start a model request."""

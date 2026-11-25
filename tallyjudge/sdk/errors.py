from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tallyjudge.sdk.results import RunResult


class InputValidationError(ValueError):
    def __init__(
        self, message: str, *, field: str | None = None, expected: str | None = None
    ) -> None:
        super().__init__(message)
        self.field = field
        self.expected = expected


class PluginError(ValueError): ...


class EvaluationAborted(RuntimeError):
    def __init__(self, partial_result: "RunResult") -> None:
        super().__init__(
            "Evaluation stopped after an error or timeout; see partial_result"
        )
        self.partial_result = partial_result


class RunSaveError(RuntimeError):
    def __init__(self, result: "RunResult") -> None:
        super().__init__(f"Could not save run {result.run_id}")
        self.result = result


class ModelError(RuntimeError): ...


class ClientLoopError(RuntimeError): ...


class ModelOutputError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class UndefinedScore(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class UsageScopeClosed(RuntimeError): ...

"""Dataset fields reserved for questions and evaluation labels."""

CONTROL_FIELDS = ("user_input", "reference")
CONTROL_PREFIXES = ("reference_", "expected_", "forbidden_")


def dataset_bound(parameter: str, column: str) -> bool:
    """Resolve protected parameters and mapped columns from the original sample."""
    return any(
        field in CONTROL_FIELDS or field.startswith(CONTROL_PREFIXES)
        for field in (parameter, column)
    )

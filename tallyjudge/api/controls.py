CONTROL_FIELDS = ("user_input", "reference")
CONTROL_PREFIXES = ("reference_", "expected_", "forbidden_")


def dataset_bound(parameter: str, column: str) -> bool:
    return any(
        field in CONTROL_FIELDS or field.startswith(CONTROL_PREFIXES)
        for field in (parameter, column)
    )

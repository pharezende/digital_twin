import pytest

from digital_twin.config import parse_boolean


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("yes", True),
        ("on", True),
        ("1", True),
        ("False", False),
        ("FALSE", False),
        ("false", False),
        ("no", False),
        ("off", False),
        ("0", False),
        (" false ", False),
    ],
)
def test_parse_boolean(raw_value: str, expected: bool) -> None:
    assert parse_boolean(raw_value) is expected


def test_parse_boolean_rejects_invalid_value() -> None:
    with pytest.raises(
        ValueError,
        match="Invalid Boolean value: '_true'. Use true/false, yes/no, on/off, or 1/0.",
    ):
        parse_boolean("_true")

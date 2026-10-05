import pytest

from keepout.pc.result import PcResult


def test_result_is_read_only() -> None:
    details = {"order": 64.0}
    result = PcResult(0.25, "elrod", ["note"], details)  # type: ignore[arg-type]
    details["order"] = 128.0
    assert result.details["order"] == 64.0
    assert result.diagnostics == ("note",)
    with pytest.raises(TypeError, match="does not support item assignment"):
        result.details["order"] = 1.0  # type: ignore[index]


@pytest.mark.parametrize("value", [-1e-18, 1.0 + 1e-12, float("nan")])
def test_value_must_be_a_probability(value: float) -> None:
    with pytest.raises(ValueError, match=r"Pc must be in \[0, 1\]"):
        PcResult(value, "foster")

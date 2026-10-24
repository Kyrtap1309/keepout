import pickle

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


def test_a_result_pickles() -> None:
    result = PcResult(
        0.25, "montecarlo", ("note",), {"samples": 100.0}, interval=(0.2, 0.3)
    )
    copy = pickle.loads(pickle.dumps(result))
    assert (copy.value, copy.method, copy.diagnostics, copy.kind) == (
        0.25,
        "montecarlo",
        ("note",),
        "estimate",
    )
    assert copy.details == {"samples": 100.0}
    assert copy.interval == (0.2, 0.3)
    bound = PcResult(0.5, "dilution", kind="upper_bound", diluted=True)
    assert pickle.loads(pickle.dumps(bound)).kind == "upper_bound"
    assert pickle.loads(pickle.dumps(bound)).diluted is True


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"kind": "guess"}, "unsupported kind"),
        ({"interval": (0.3, 0.2)}, "low <= high"),
        ({"interval": (-0.1, 0.2)}, r"interval low must be in \[0, 1\]"),
        ({"interval": (0.1, 1.5)}, r"interval high must be in \[0, 1\]"),
    ],
)
def test_kind_and_interval_are_checked(fields: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PcResult(0.25, "montecarlo", **fields)  # type: ignore[arg-type]

import math

import pytest

from keepout.validation.criteria import Criterion, Verdict


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({}, "needs a relative or an absolute bound"),
        ({"interval": True, "relative": 0.1}, "takes no error bounds"),
        ({"relative": -1e-3}, "finite and >= 0"),
        ({"absolute": math.inf}, "finite and >= 0"),
        ({"relative": 1e-3, "small": 1e-10}, "small needs the absolute bound"),
    ],
)
def test_meaningless_criteria_are_rejected(
    options: dict[str, float | bool], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Criterion(**options)  # type: ignore[arg-type]


def test_an_interval_criterion_needs_the_interval() -> None:
    with pytest.raises(ValueError, match="needs the result's interval"):
        Criterion(interval=True).judge(0.1, 0.1)


def test_a_relative_criterion_needs_an_exact_zero_without_an_absolute_bound() -> None:
    criterion = Criterion(relative=1e-3)
    assert criterion.judge(0.0, 0.0) == Verdict(0.0, False, True)
    assert not criterion.judge(1e-30, 0.0).passed

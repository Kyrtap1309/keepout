import math

import pytest

from keepout.validation.criteria import (
    REPRODUCIBILITY_FRACTION,
    TWO_D,
    Criterion,
    Verdict,
)


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


@pytest.mark.parametrize(
    ("criterion", "reference", "interval", "margin"),
    [
        (TWO_D, 2e-3, None, 2e-9),
        (TWO_D, 1e-12, None, 1e-15),
        (Criterion(relative=1e-3), 0.0, None, 0.0),
        (Criterion(interval=True), 0.5, (0.2, 0.6), 0.2),
    ],
)
def test_margin_is_the_bound_that_applies_at_the_reference(
    criterion: Criterion,
    reference: float,
    interval: tuple[float, float] | None,
    margin: float,
) -> None:
    assert criterion.margin(reference, interval) == pytest.approx(margin)


def test_reproduces_within_a_fraction_of_the_margin() -> None:
    step = REPRODUCIBILITY_FRACTION * 2e-9
    assert TWO_D.reproduces(2e-3, 2e-3 + 0.9 * step, 2e-3)
    assert not TWO_D.reproduces(2e-3, 2e-3 - 1.1 * step, 2e-3)


def test_reproduces_an_interval_result_within_its_half_width() -> None:
    criterion = Criterion(interval=True)
    step = REPRODUCIBILITY_FRACTION * 0.2
    assert criterion.reproduces(0.4, 0.4 + 0.9 * step, 0.5, (0.2, 0.6))
    assert not criterion.reproduces(0.4, 0.4 + 1.1 * step, 0.5, (0.2, 0.6))
    with pytest.raises(ValueError, match="needs the result's interval"):
        criterion.reproduces(0.4, 0.4, 0.5)

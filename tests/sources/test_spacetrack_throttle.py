import pytest
from hypothesis import given
from hypothesis import strategies as st
from spacetrack_builders import HOUR

from keepout.sources.spacetrack import Throttle

CLOCK_ROUNDING = 1e-9
"""Seconds of rounding in a fake clock's sum."""


class FakeTime:
    """A monotonic clock that a fake sleep advances."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_the_throttle_waits_only_past_a_limit() -> None:
    fake = FakeTime()
    throttle = Throttle(clock=fake.clock, sleep=fake.sleep)
    for _ in range(20):
        throttle.wait()
    assert fake.slept == []
    fake.now = 10.0
    throttle.wait()
    assert fake.slept == [50.0]
    assert fake.now == 60.0


def test_the_throttle_keeps_the_hourly_limit() -> None:
    fake = FakeTime()
    throttle = Throttle(clock=fake.clock, sleep=fake.sleep)
    for _ in range(200):
        throttle.wait()
    assert fake.now == 9 * 60.0  # 20 a minute
    throttle.wait()
    assert fake.now == HOUR


@pytest.mark.parametrize(
    ("per_minute", "per_hour"), [(0, 200), (20, 0)], ids=["minute", "hour"]
)
def test_a_throttle_limit_below_one_is_refused(per_minute: int, per_hour: int) -> None:
    with pytest.raises(ValueError, match="limits must be at least 1"):
        Throttle(per_minute=per_minute, per_hour=per_hour)


@given(
    gaps=st.lists(st.floats(0.0, 400.0), min_size=1, max_size=60),
    per_minute=st.integers(1, 5),
    per_hour=st.integers(1, 12),
)
def test_throttled_requests_never_exceed_a_limit(
    gaps: list[float], per_minute: int, per_hour: int
) -> None:
    fake = FakeTime()
    throttle = Throttle(
        per_minute=per_minute, per_hour=per_hour, clock=fake.clock, sleep=fake.sleep
    )
    sent: list[float] = []
    for gap in gaps:
        fake.now += gap
        asked = fake.now
        throttle.wait()
        sent.append(fake.now)
        # It waits no longer than a limit needs (to rounding: the fake
        # clock adds the wait to the time it was asked at).
        assert fake.now == asked or any(
            fake.now
            == pytest.approx(sent[-limit - 1] + window, rel=0, abs=CLOCK_ROUNDING)
            for window, limit in ((60.0, per_minute), (HOUR, per_hour))
            if len(sent) > limit
        )
    for index, time in enumerate(sent):
        later = sent[index:]
        assert sum(t < time + 60.0 - CLOCK_ROUNDING for t in later) <= per_minute
        assert sum(t < time + HOUR - CLOCK_ROUNDING for t in later) <= per_hour

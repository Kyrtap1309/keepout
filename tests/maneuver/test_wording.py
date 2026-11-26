"""What an outcome says in words, shared by the command line and the dashboard."""

import astropy.units as u
from astropy.time import Time

from keepout.maneuver import ManeuverOption, RescreenResult
from keepout.maneuver.wording import event_line

START = Time("2026-10-06T00:00:00", scale="utc")


def rescreen_without_the_event() -> RescreenResult:
    """A re-screen that screened the event's secondary and found it nowhere
    near the event's TCA."""
    option = ManeuverOption(
        burn_lead=3600.0,
        delta_v=0.1,
        pc=0.0,
        range_at_tca=0.0,
        method="foster",
        meets_target=True,
    )
    return RescreenResult(
        option=option,
        new_approaches=(),
        closer_approaches=(),
        event_returns=(),
        event_range=None,
        window=(START, START + 7 * u.day),
        catalog_objects=10,
        secondary_screened=True,
        skipped_objects=0,
    )


def test_event_line_gives_the_tca_tolerance_the_rescreen_used() -> None:
    line = event_line(rescreen_without_the_event(), "47826", 300.0)
    assert line.endswith("no approach within 5 min of the CDM's TCA after the burn.")

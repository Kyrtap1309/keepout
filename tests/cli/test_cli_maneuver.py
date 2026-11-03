"""``keepout maneuver``: real plans and re-screens of the fixture's crossing event.

52939 crosses 47826 at 6 km/s, 19.6 h after the CDM's creation (1029 m, green
at the CDM's hard-body radii). A radius of 100 m puts it above the target and
of 1 km in the red. The plans use the small grid below, a run a fraction of a
second, unless a test restores the default one, and most re-screens a catalog of
the objects near that orbit.
"""

import dataclasses
import functools
import json
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final, NamedTuple

import pytest
from astropy.time import Time
from cli_builders import (
    DAYS,
    FIXTURE_CSV,
    FLEET_TOML,
    START,
    FakeWeb,
    celestrak_pages,
    keepout,
)
from inline_snapshot import snapshot
from typer.testing import Result

from keepout import maneuver
from keepout.cdm import kvn
from keepout.cli.output import ExitCode, minute
from keepout.events import ConjunctionEvent
from keepout.maneuver import (
    BurnGrid,
    ManeuverOption,
    ManeuverPlan,
    NewApproach,
    RescreenResult,
)
from keepout.omm import Omm, load
from keepout.omm.celestrak import dumps_csv
from keepout.risk import RiskLevel
from keepout.screening import CloseApproach, screen
from keepout.synthetic import read_fleet, run


class _Operator(NamedTuple):
    crossing: str
    drifting: str
    crossing_id: str


@functools.cache
def operator_cdms() -> _Operator:
    """The synthetic operator's two CDMs, as KVN: 52939 against 47826
    (maneuverable), and 44476 against 55841 (not); computed once, on first
    use rather than on import."""
    result = run(
        read_fleet(FLEET_TOML),
        load(FIXTURE_CSV),
        Time(START, scale="utc"),
        float(DAYS) * 86400.0,
    )
    crossing, drifting = (kvn.dumps(cdm) for cdm in result.cdms)
    return _Operator(crossing, drifting, result.cdms[0].message_id)


PRIMARY: Final = 52939
SECONDARY: Final = 47826
NEIGHBOURS: Final = frozenset({PRIMARY, SECONDARY, 57087, 62118, 100000})
"""The objects whose approaches the burns of these tests change."""
SMALL: Final = BurnGrid(steps_per_orbit=1, delta_v_per_direction=6)
FULL_PLAN: Final = maneuver.plan_maneuver
"""The planner on its default grid, 8 s here."""

pytestmark = pytest.mark.usefixtures("small_grid")


@pytest.fixture
def small_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plans on :data:`SMALL`."""
    small = functools.partial(FULL_PLAN, grid=SMALL)
    monkeypatch.setattr(maneuver, "plan_maneuver", small)


def write(folder: Path, name: str, text: str) -> str:
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def crossing(folder: Path) -> str:
    return write(folder, "crossing.kvn", operator_cdms().crossing)


def catalog_of(folder: Path, numbers: frozenset[int] = NEIGHBOURS) -> str:
    """A catalog file of the fixture's element sets of ``numbers``."""
    chosen = [omm for omm in load(FIXTURE_CSV) if omm.catalog_number in numbers]
    return write(folder, "catalog.csv", dumps_csv(chosen))


def replace_rescreen(
    monkeypatch: pytest.MonkeyPatch, change: Callable[[RescreenResult], RescreenResult]
) -> None:
    """Re-screens that are real up to ``change``."""
    real = maneuver.rescreen

    def rescreen(
        plan: ManeuverPlan,
        option: ManeuverOption,
        omms: Sequence[Omm],
        *,
        workers: int | None,
    ) -> RescreenResult:
        return change(real(plan, option, omms, workers=workers))

    monkeypatch.setattr(maneuver, "rescreen", rescreen)


def test_plan_prints_options_and_exits_zero(tmp_path: Path) -> None:
    result = keepout("maneuver", crossing(tmp_path), "--hbr", "1000")
    assert result.exit_code == ExitCode.OK
    assert result.stderr == ""
    assert result.stdout == snapshot("""\
Event 52939/47826/2026-10-06T19:36:42.717332: 52939 / 47826, TCA 2026-10-06 19:36 UTC, target Pc 1.00e-05.
Burns of the primary
Option      Before TCA (h)  Δv (m/s)        Pc  Range (m)  Method  Target
─────────────────────────────────────────────────────────────────────────
no burn                  -         -  4.25e-02       1029  foster  missed
prograde               7.0    +0.251  2.12e-06      17005  foster  met   \n\
retrograde            11.7    -0.251  2.55e-06      29552  foster  met   \n\
Note: every Pc rests on an assumed covariance (COVARIANCE_METHOD = DEFAULT)
Note: drag is off: the CDM gives the primary no positive CD_AREA_OVER_MASS
Not re-screened: nothing is known of the approaches a burn brings (--catalog or --source screen for them).
""")


def test_json_has_the_documented_keys(tmp_path: Path) -> None:
    catalog = catalog_of(tmp_path)
    args = (crossing(tmp_path), "--hbr", "1000", "--json")
    result = keepout("maneuver", *args, "--catalog", catalog, "--workers", "1")
    assert result.exit_code == ExitCode.OK
    payload = json.loads(result.stdout)
    assert list(payload) == [
        "event",
        "primary",
        "secondary",
        "target_pc",
        "method",
        "assumed_covariance",
        "baseline",
        "options",
        "notes",
    ]
    assert payload["event"] == "52939/47826/2026-10-06T19:36:42.717332"
    assert payload["primary"] == str(PRIMARY)
    assert payload["secondary"] == str(SECONDARY)
    assert payload["target_pc"] == 1e-5
    assert payload["method"] == "foster"
    assert list(payload["baseline"]) == [
        "pc",
        "range_at_tca_m",
        "method",
        "meets_target",
    ]
    assert payload["baseline"]["pc"] == pytest.approx(0.04254, rel=1e-3)
    assert payload["baseline"]["meets_target"] is False
    prograde, retrograde = payload["options"]
    assert list(prograde) == [
        "direction",
        "burn_lead_s",
        "delta_v_m_s",
        "pc",
        "range_at_tca_m",
        "method",
        "meets_target",
        "rescreen",
    ]
    assert prograde["direction"] == "prograde"
    assert retrograde["direction"] == "retrograde"
    assert prograde["delta_v_m_s"] > 0.0 > retrograde["delta_v_m_s"]
    assert prograde["meets_target"] is True
    assert list(prograde["rescreen"]) == [
        "conflict",
        "secondary_screened",
        "event_range_m",
        "catalog_objects",
        "skipped_objects",
        "window_start",
        "window_end",
        "new_approaches",
        "closer_approaches",
        "event_returns",
    ]
    assert prograde["rescreen"]["conflict"] is False
    assert prograde["rescreen"]["secondary_screened"] is True
    assert prograde["rescreen"]["catalog_objects"] == len(NEIGHBOURS)
    assert prograde["rescreen"]["skipped_objects"] == 0
    # The window starts at the burn and lasts 7 days, in UTC as every JSON time.
    window = [prograde["rescreen"][key] for key in ("window_start", "window_end")]
    assert all(re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+", t) for t in window)
    start, end = (Time(text, scale="utc") for text in window)
    tca = Time("2026-10-06T19:36:42.717332", scale="utc")
    assert (tca - start).sec == pytest.approx(prograde["burn_lead_s"], abs=1e-3)
    assert (end - start).sec == pytest.approx(7 * 86400.0, abs=1e-3)
    assert prograde["rescreen"]["event_returns"] == []
    (new,) = retrograde["rescreen"]["new_approaches"]
    assert list(new) == [
        "secondary",
        "tca",
        "range_at_tca_m",
        "relative_speed_m_s",
        "pc",
        "level",
        "assumed_covariance",
    ]
    assert new["secondary"] == 62118
    assert new["level"] == "yellow"
    assert new["assumed_covariance"] is True
    assert new["pc"] == pytest.approx(3.646e-6, rel=1e-3)


def test_without_catalog_says_not_rescreened(tmp_path: Path) -> None:
    args = (crossing(tmp_path), "--hbr", "1000")
    text = keepout("maneuver", *args)
    assert (
        "Not re-screened: nothing is known of the approaches a burn brings "
        "(--catalog or --source screen for them)."
    ) in text.stdout.splitlines()
    options = json.loads(keepout("maneuver", *args, "--json").stdout)["options"]
    assert [option["rescreen"] for option in options] == [None, None]


def test_no_option_exits_four(tmp_path: Path) -> None:
    args = (crossing(tmp_path), "--hbr", "500000")
    result = keepout("maneuver", *args)
    assert result.exit_code == ExitCode.NO_SAFE_BURN
    assert result.stdout == snapshot("""\
Event 52939/47826/2026-10-06T19:36:42.717332: 52939 / 47826, TCA 2026-10-06 19:36 UTC, target Pc 1.00e-05.
Burns of the primary
Option   Before TCA (h)  Δv (m/s)        Pc  Range (m)  Method  Target
──────────────────────────────────────────────────────────────────────
no burn               -         -  1.00e+00       1029  foster  missed
Note: 2-D Pc assumptions questionable: the encounter lasts about 172 s, 3% of the primary's orbital period (limit 1%); a 3-D Pc may differ
Note: the covariances of primary (52939) and secondary (47826) are known only in position (zero velocity block): the 3-D methods need the velocity uncertainty, so the 2-D Pc is used
Note: every Pc rests on an assumed covariance (COVARIANCE_METHOD = DEFAULT)
Note: drag is off: the CDM gives the primary no positive CD_AREA_OVER_MASS
Not re-screened: nothing is known of the approaches a burn brings (--catalog or --source screen for them).
No burn on the grid meets the target Pc 1.00e-05 (exit status 4).
""")
    as_json = keepout("maneuver", *args, "--json")
    assert as_json.exit_code == ExitCode.NO_SAFE_BURN
    assert json.loads(as_json.stdout)["options"] == []


def test_conflict_exits_four(tmp_path: Path) -> None:
    catalog = catalog_of(tmp_path)
    burn = ("--burn-before", "19", "--dv", "1")
    args = (crossing(tmp_path), "--hbr", "1000", *burn, "--workers", "1")
    result = keepout("maneuver", *args, "--catalog", catalog)
    assert result.exit_code == ExitCode.NO_SAFE_BURN
    assert result.stdout.replace(catalog, "<catalog>") == snapshot("""\
Event 52939/47826/2026-10-06T19:36:42.717332: 52939 / 47826, TCA 2026-10-06 19:36 UTC, target Pc 1.00e-05.
Burns of the primary
Option    Before TCA (h)  Δv (m/s)        Pc  Range (m)  Method  Target
───────────────────────────────────────────────────────────────────────
no burn                -         -  4.25e-02       1029  foster  missed
prograde            19.0        +1  0.00e+00     188388  foster  met   \n\
Note: every Pc rests on an assumed covariance (COVARIANCE_METHOD = DEFAULT)
Note: drag is off: the CDM gives the primary no positive CD_AREA_OVER_MASS
prograde re-screened against 5 objects (<catalog>) from 2026-10-06T00:36:42 to 2026-10-13T00:36:42 UTC; 0 catalog objects left out, of which it says nothing.
  Event secondary 47826 on the element sets (SGP4, not the CDM's orbit): no approach within 10 min of the CDM's TCA after the burn.
  CONFLICT: a new or closer approach is at red.
  1 new approach:
    100000 at 2026-10-10 17:28 UTC: range 1997 m, assumed Pc 1.02e-02, red
  Covariances are assumed (keepout's model of GP element set errors), not orbit determination.
This burn brings an approach at red (exit status 4).
""")
    as_json = keepout("maneuver", *args, "--catalog", catalog, "--json")
    assert as_json.exit_code == ExitCode.NO_SAFE_BURN
    (option,) = json.loads(as_json.stdout)["options"]
    assert option["meets_target"] is True
    assert option["rescreen"]["conflict"] is True
    assert option["rescreen"]["new_approaches"][0]["level"] == "red"
    # The burn leaves the event's secondary outside the volume: screened, no range.
    assert option["rescreen"]["secondary_screened"] is True
    assert option["rescreen"]["event_range_m"] is None


def test_one_clean_option_and_one_conflict_exit_zero_and_show_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The default grid, and the whole fixture with what screening leaves out.
    monkeypatch.setattr(maneuver, "plan_maneuver", FULL_PLAN)
    args = (crossing(tmp_path), "--hbr", "100", "--catalog", str(FIXTURE_CSV))
    result = keepout("maneuver", *args, "--workers", "1")
    assert result.exit_code == ExitCode.OK
    shown = result.stdout.replace(str(FIXTURE_CSV), "<catalog>")
    assert shown == snapshot("""\
Event 52939/47826/2026-10-06T19:36:42.717332: 52939 / 47826, TCA 2026-10-06 19:36 UTC, target Pc 1.00e-05.
Burns of the primary
Option      Before TCA (h)  Δv (m/s)        Pc  Range (m)  Method  Target
─────────────────────────────────────────────────────────────────────────
no burn                  -         -  1.40e-05       1029  foster  missed
prograde              16.5  +0.00794  9.96e-06       1330  foster  met   \n\
retrograde            17.2   -0.0316  9.76e-06       5960  foster  met   \n\
Note: every Pc rests on an assumed covariance (COVARIANCE_METHOD = DEFAULT)
Note: drag is off: the CDM gives the primary no positive CD_AREA_OVER_MASS
prograde re-screened against 217 objects (<catalog>) from 2026-10-06T03:09:04 to 2026-10-13T03:09:04 UTC; 5 catalog objects left out, of which it says nothing.
  Event secondary 47826 on the element sets (SGP4, not the CDM's orbit): range 1330 m after the burn.
  No new or closer approach at red among the objects screened.
  1 new approach:
    57087 at 2026-10-11 15:48 UTC: range 3386 m, assumed Pc 2.12e-08, green
  Covariances are assumed (keepout's model of GP element set errors), not orbit determination.
retrograde re-screened against 217 objects (<catalog>) from 2026-10-06T02:21:48 to 2026-10-13T02:21:48 UTC; 5 catalog objects left out, of which it says nothing.
  Event secondary 47826 on the element sets (SGP4, not the CDM's orbit): no approach within 10 min of the CDM's TCA after the burn.
  CONFLICT: a new or closer approach is at red.
  1 new approach:
    57087 at 2026-10-11 22:04 UTC: range 2472 m, assumed Pc 4.12e-04, red
  Covariances are assumed (keepout's model of GP element set errors), not orbit determination.
""")


@functools.cache
def later_approach() -> CloseApproach:
    """52939's approach of 57087, days after the event."""
    omms = [omm for omm in load(FIXTURE_CSV) if omm.catalog_number in NEIGHBOURS]
    (primary,) = (omm for omm in omms if omm.catalog_number == PRIMARY)
    found = screen([primary], omms, Time(START, scale="utc"), 7 * 86400.0)
    return next(item for item in found.approaches if item.secondary_number == 57087)


def back(pc: float = 3e-5) -> NewApproach:
    """The event's secondary back days after the event, above the target."""
    approach = dataclasses.replace(later_approach(), secondary_number=SECONDARY)
    return NewApproach(approach=approach, pc=pc, level=RiskLevel.YELLOW)


def red() -> NewApproach:
    """A new approach at red, of another object."""
    return NewApproach(approach=later_approach(), pc=1e-2, level=RiskLevel.RED)


def only(
    *, returns: Sequence[NewApproach] = (), new: Sequence[NewApproach] = ()
) -> Callable[[RescreenResult], RescreenResult]:
    """A change of a re-screen to nothing but these approaches."""

    def change(result: RescreenResult) -> RescreenResult:
        return dataclasses.replace(
            result,
            new_approaches=(*new, *returns),
            closer_approaches=(),
            event_returns=tuple(returns),
        )

    return change


def per_direction(
    prograde: Callable[[RescreenResult], RescreenResult],
    retrograde: Callable[[RescreenResult], RescreenResult],
) -> Callable[[RescreenResult], RescreenResult]:
    def change(result: RescreenResult) -> RescreenResult:
        chosen = prograde if result.option.delta_v > 0.0 else retrograde
        return chosen(result)

    return change


RETURN_LINE: Final = (
    "  The burn brings this pair back at 2026-10-11 {time} UTC at Pc 3.00e-05 "
    "(assumed), above the target."
)


def test_every_burn_bringing_the_pair_back_above_the_target_exits_four(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replace_rescreen(monkeypatch, only(returns=[back()]))
    args = (crossing(tmp_path), "--hbr", "1000", "--workers", "1")
    result = keepout("maneuver", *args, "--catalog", catalog_of(tmp_path))
    assert result.exit_code == ExitCode.NO_SAFE_BURN
    lines = result.stdout.splitlines()
    time = minute(later_approach().tca)[11:]
    assert lines.count(RETURN_LINE.format(time=time)) == 2
    assert (
        lines.count("  No new or closer approach at red among the objects screened.")
        == 2
    )
    assert lines[-1] == (
        "Every burn brings this pair back above the target Pc 1.00e-05 (exit status 4)."
    )
    as_json = keepout("maneuver", *args, "--catalog", catalog_of(tmp_path), "--json")
    assert as_json.exit_code == ExitCode.NO_SAFE_BURN
    options = json.loads(as_json.stdout)["options"]
    assert len(options) == 2
    for option in options:
        assert option["meets_target"] is True
        assert option["rescreen"]["conflict"] is False
        (returned,) = option["rescreen"]["event_returns"]
        assert returned["secondary"] == SECONDARY
        assert returned["pc"] == 3e-5
        assert returned["level"] == "yellow"
        assert returned["assumed_covariance"] is True


def test_a_return_below_the_target_is_not_shown_as_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Below the target the library gives no event return: a new approach only.
    def change(result: RescreenResult) -> RescreenResult:
        return dataclasses.replace(only()(result), new_approaches=(back(3e-6),))

    replace_rescreen(monkeypatch, change)
    args = (crossing(tmp_path), "--hbr", "1000", "--workers", "1")
    result = keepout("maneuver", *args, "--catalog", catalog_of(tmp_path))
    assert result.exit_code == ExitCode.OK
    assert "brings this pair back" not in result.stdout


def test_one_clean_burn_and_one_bringing_the_pair_back_exit_zero_and_show_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replace_rescreen(monkeypatch, per_direction(only(returns=[back()]), only()))
    args = (crossing(tmp_path), "--hbr", "1000", "--workers", "1")
    result = keepout("maneuver", *args, "--catalog", catalog_of(tmp_path))
    assert result.exit_code == ExitCode.OK
    lines = result.stdout.splitlines()
    time = minute(later_approach().tca)[11:]
    assert lines.count(RETURN_LINE.format(time=time)) == 1
    prograde = next(i for i, line in enumerate(lines) if "prograde re-sc" in line)
    retrograde = next(i for i, line in enumerate(lines) if "retrograde re-sc" in line)
    assert prograde < lines.index(RETURN_LINE.format(time=time)) < retrograde
    assert "exit status" not in result.stdout


def test_a_red_burn_and_a_burn_bringing_the_pair_back_exit_four(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replace_rescreen(
        monkeypatch, per_direction(only(new=[red()]), only(returns=[back()]))
    )
    args = (crossing(tmp_path), "--hbr", "1000", "--workers", "1")
    result = keepout("maneuver", *args, "--catalog", catalog_of(tmp_path))
    assert result.exit_code == ExitCode.NO_SAFE_BURN
    assert result.stdout.splitlines()[-1] == (
        "Every burn brings an approach at red or this pair back above the target "
        "Pc 1.00e-05 (exit status 4)."
    )


def test_an_explicit_burn_bringing_the_pair_back_exits_four(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replace_rescreen(monkeypatch, only(returns=[back()]))
    burn = ("--burn-before", "12.5", "--dv", "-1", "--workers", "1")
    args = (crossing(tmp_path), "--hbr", "1000", *burn)
    result = keepout("maneuver", *args, "--catalog", catalog_of(tmp_path))
    assert result.exit_code == ExitCode.NO_SAFE_BURN
    assert result.stdout.splitlines()[-1] == (
        "This burn brings this pair back above the target Pc 1.00e-05 (exit status 4)."
    )


def test_explicit_burn_replaces_the_minimum_options(tmp_path: Path) -> None:
    burn = ("--burn-before", "12.5", "--dv", "-1")
    args = (crossing(tmp_path), "--hbr", "1000", *burn)
    result = keepout("maneuver", *args)
    assert result.exit_code == ExitCode.OK
    assert result.stdout == snapshot("""\
Event 52939/47826/2026-10-06T19:36:42.717332: 52939 / 47826, TCA 2026-10-06 19:36 UTC, target Pc 1.00e-05.
Burns of the primary
Option      Before TCA (h)  Δv (m/s)        Pc  Range (m)  Method  Target
─────────────────────────────────────────────────────────────────────────
no burn                  -         -  4.25e-02       1029  foster  missed
retrograde            12.5        -1  0.00e+00     126692  foster  met   \n\
Note: every Pc rests on an assumed covariance (COVARIANCE_METHOD = DEFAULT)
Note: drag is off: the CDM gives the primary no positive CD_AREA_OVER_MASS
Not re-screened: nothing is known of the approaches a burn brings (--catalog or --source screen for them).
""")
    (option,) = json.loads(keepout("maneuver", *args, "--json").stdout)["options"]
    assert option["direction"] == "retrograde"
    assert option["burn_lead_s"] == 12.5 * 3600.0
    assert option["delta_v_m_s"] == -1.0
    assert option["meets_target"] is True


def test_explicit_burn_that_misses_the_target_exits_four(tmp_path: Path) -> None:
    burn = ("--burn-before", "19", "--dv", "0.1")
    args = (crossing(tmp_path), "--hbr", "1000", *burn)
    result = keepout("maneuver", *args)
    assert result.exit_code == ExitCode.NO_SAFE_BURN
    assert result.stdout.splitlines()[-1] == (
        "This burn misses the target Pc 1.00e-05 (exit status 4)."
    )
    (option,) = json.loads(keepout("maneuver", *args, "--json").stdout)["options"]
    assert option["direction"] == "prograde"
    assert option["meets_target"] is False


def test_burn_before_without_dv_is_a_usage_error(tmp_path: Path) -> None:
    alone = (("--burn-before", "19"), ("--dv", "0.1"))
    for option in alone:
        result = keepout("maneuver", crossing(tmp_path), *option)
        assert result.exit_code == ExitCode.USAGE
        assert "give --burn-before and --dv together" in result.stderr


def test_catalog_and_source_together_are_a_usage_error(tmp_path: Path) -> None:
    both = ("--catalog", str(FIXTURE_CSV), "--source", "celestrak")
    result = keepout("maneuver", crossing(tmp_path), *both)
    assert result.exit_code == ExitCode.USAGE
    assert "give either --catalog or --source, not both" in result.stderr


def test_two_events_exit_one_listing_the_pairs(tmp_path: Path) -> None:
    files = (
        crossing(tmp_path),
        write(tmp_path, "drifting.kvn", operator_cdms().drifting),
    )
    result = keepout("maneuver", *files)
    assert result.exit_code == ExitCode.ERROR
    assert result.stdout == ""
    assert result.stderr == (
        "Error: these CDMs hold 2 events, and a burn is planned for one: "
        "52939 / 47826 at 2026-10-06 19:36, 44476 / 55841 at 2026-10-08 12:03\n"
    )


def test_cdms_of_one_event_are_planned_from_the_latest(tmp_path: Path) -> None:
    # 19 h before TCA comes before a CDM created 18.6 h before it.
    later = (
        operator_cdms()
        .crossing.replace(operator_cdms().crossing_id, "UPDATE")
        .replace(
            "CREATION_DATE                  = 2026-10-06T00",
            "CREATION_DATE                  = 2026-10-06T01",
        )
    )
    assert "= 2026-10-06T01" in later
    files = (crossing(tmp_path), write(tmp_path, "later.kvn", later))
    burn = ("--burn-before", "19", "--dv", "1")
    result = keepout("maneuver", *files, "--hbr", "1000", *burn)
    assert result.exit_code == ExitCode.ERROR
    assert "19.0 h before TCA comes before the latest CDM" in result.stderr


def test_assumed_covariance_is_labelled(tmp_path: Path) -> None:
    args = (crossing(tmp_path), "--hbr", "1000")
    text = keepout("maneuver", *args).stdout
    label = (
        "Note: every Pc rests on an assumed covariance (COVARIANCE_METHOD = DEFAULT)"
    )
    assert label in text.splitlines()
    payload = json.loads(keepout("maneuver", *args, "--json").stdout)
    assert payload["assumed_covariance"] is True
    assert label.removeprefix("Note: ") in payload["notes"]


def test_a_calculated_covariance_is_not_labelled_assumed(tmp_path: Path) -> None:
    calculated = operator_cdms().crossing.replace(
        "COVARIANCE_METHOD              = DEFAULT",
        "COVARIANCE_METHOD              = CALCULATED",
    )
    assert "= DEFAULT" not in calculated
    args = (write(tmp_path, "calculated.kvn", calculated), "--hbr", "1000")
    assert "assumed" not in keepout("maneuver", *args).stdout
    assert (
        json.loads(keepout("maneuver", *args, "--json").stdout)["assumed_covariance"]
        is False
    )


def test_pc_of_an_approach_of_the_catalog_is_labelled_assumed(tmp_path: Path) -> None:
    # Even for a CDM with covariances of its own: element sets have none.
    calculated = operator_cdms().crossing.replace(
        "COVARIANCE_METHOD              = DEFAULT",
        "COVARIANCE_METHOD              = CALCULATED",
    )
    burn = ("--burn-before", "19", "--dv", "1", "--workers", "1")
    args = (write(tmp_path, "c.kvn", calculated), "--hbr", "1000", *burn)
    result = keepout("maneuver", *args, "--catalog", catalog_of(tmp_path))
    lines = result.stdout.splitlines()
    assert any("assumed Pc" in line and "100000" in line for line in lines)
    assert (
        "  Covariances are assumed (keepout's model of GP element set errors), "
        "not orbit determination."
    ) in lines


def test_missing_hbr_exits_one(tmp_path: Path) -> None:
    lines = operator_cdms().crossing.splitlines(keepends=True)
    without = "".join(line for line in lines if not line.startswith("HBR"))
    result = keepout("maneuver", write(tmp_path, "crossing.kvn", without))
    assert result.exit_code == ExitCode.ERROR
    assert result.stderr.startswith("Error: no hard-body radius for 52939, 47826")


def test_help_loads_no_numerical_library() -> None:
    code = (
        "import sys\n"
        "from keepout.cli.app import app\n"
        "try:\n"
        "    app(['maneuver', '--help'], prog_name='keepout')\n"
        "except SystemExit:\n"
        "    print(sorted(m for m in ('numpy', 'scipy', 'astropy', 'erfa', 'sgp4')"
        " if m in sys.modules))\n"
    )
    loaded = subprocess.run(  # noqa: S603 (this interpreter, fixed code)
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert "keepout maneuver" in loaded.stdout
    assert loaded.stdout.endswith("[]\n")


def test_help_gives_the_exit_status_and_the_library_defaults() -> None:
    result = keepout("maneuver", "--help", env={"COLUMNS": "200"})
    assert result.exit_code == ExitCode.OK
    words = " ".join(result.stdout.split())
    for expected in (
        "CDMS",
        "--target-pc",
        "--burn-before",
        "--dv",
        "--hbr",
        "--catalog",
        "--source",
        "--workers",
        "--cache-dir",
        "--json",
        "from 2 days to 6 hours before TCA",
        "for 7 days from the burn",
        "status 4",
    ):
        assert expected in words


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--target-pc", "0"),
        ("--target-pc", "1"),
        ("--target-pc", "nan"),
        ("--dv", "0"),
        ("--dv", "nan"),
        ("--dv", "inf"),
        ("--burn-before", "0"),
        ("--burn-before", "-1"),
        ("--burn-before", "nan"),
        ("--hbr", "0"),
        ("--workers", "0"),
    ],
)
def test_bad_options_are_usage_errors(tmp_path: Path, option: str, value: str) -> None:
    result = keepout("maneuver", crossing(tmp_path), option, value)
    assert result.exit_code == ExitCode.USAGE
    assert option in result.stderr


def test_no_cdm_is_a_usage_error() -> None:
    result = keepout("maneuver")
    assert result.exit_code == ExitCode.USAGE
    assert "Missing argument 'CDMS'" in result.stderr


def test_a_file_that_is_no_cdm_is_named(tmp_path: Path) -> None:
    bad = write(tmp_path, "bad.kvn", "CCSDS_CDM_VERS = 1.0\nNOT A LINE\n")
    result = keepout("maneuver", bad)
    assert result.exit_code == ExitCode.ERROR
    assert result.stderr.startswith(f"Error: {bad}: ")


def test_a_primary_that_cannot_manoeuvre_exits_one(tmp_path: Path) -> None:
    result = keepout(
        "maneuver", write(tmp_path, "drifting.kvn", operator_cdms().drifting)
    )
    assert result.exit_code == ExitCode.ERROR
    assert "cannot manoeuvre (MANEUVERABLE = NO)" in result.stderr


def test_a_catalog_that_cannot_be_read_is_reported_before_planning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("planned before reading the catalog")

    monkeypatch.setattr(maneuver, "plan_maneuver", refuse)
    bad = write(tmp_path, "bad.csv", "not a catalog\n")
    result = keepout("maneuver", crossing(tmp_path), "--catalog", bad)
    assert result.exit_code == ExitCode.ERROR
    assert result.stderr.startswith("Error: not OMM data")


def test_a_primary_missing_from_the_catalog_exits_one(tmp_path: Path) -> None:
    catalog = catalog_of(tmp_path, frozenset({SECONDARY}))
    args = (crossing(tmp_path), "--hbr", "1000", "--catalog", catalog)
    result = keepout("maneuver", *args, "--workers", "1")
    assert result.exit_code == ExitCode.ERROR
    assert "the primary 52939 has no element set in the catalog" in result.stderr


def test_a_direction_without_an_option_is_said(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def one_way(
        event: ConjunctionEvent,
        *,
        target_pc: float,
        combined_hard_body_radius: float | None,
    ) -> ManeuverPlan:
        plan = FULL_PLAN(
            event,
            grid=SMALL,
            target_pc=target_pc,
            combined_hard_body_radius=combined_hard_body_radius,
        )
        return dataclasses.replace(plan, retrograde=None)

    monkeypatch.setattr(maneuver, "plan_maneuver", one_way)
    result = keepout("maneuver", crossing(tmp_path), "--hbr", "1000")
    assert result.exit_code == ExitCode.OK
    assert "No retrograde burn on the grid meets the target." in result.stdout
    assert "No prograde" not in result.stdout


def rescreen_lines(result: Result, catalog: str) -> list[str]:
    """What ``result`` prints from the first re-screen on."""
    lines = result.stdout.replace(catalog, "<catalog>").splitlines()
    (first,) = (line for line in lines if " re-screened against " in line)
    return lines[lines.index(first) :]


def test_secondary_still_near_after_the_burn_gives_its_range(tmp_path: Path) -> None:
    catalog = catalog_of(tmp_path)
    burn = ("--burn-before", "16", "--dv", "0.008", "--workers", "1")
    args = (crossing(tmp_path), "--hbr", "100", *burn, "--catalog", catalog)
    result = keepout("maneuver", *args)
    assert rescreen_lines(result, catalog) == snapshot(
        [
            "prograde re-screened against 5 objects (<catalog>) from 2026-10-06T03:36:42 to 2026-10-13T03:36:42 UTC; 0 catalog objects left out, of which it says nothing.",
            "  Event secondary 47826 on the element sets (SGP4, not the CDM's orbit): range 1288 m after the burn.",
            "  No new or closer approach at red among the objects screened.",
            "  1 new approach:",
            "    57087 at 2026-10-11 15:48 UTC: range 3276 m, assumed Pc 4.85e-08, green",
            "  Covariances are assumed (keepout's model of GP element set errors), not orbit determination.",
            "This burn misses the target Pc 1.00e-05 (exit status 4).",
        ]
    )
    (option,) = json.loads(keepout("maneuver", *args, "--json").stdout)["options"]
    assert option["rescreen"]["event_range_m"] == pytest.approx(1288.0, abs=1.0)


def test_a_secondary_not_screened_says_nothing_of_the_event(tmp_path: Path) -> None:
    catalog = catalog_of(tmp_path, NEIGHBOURS - {SECONDARY})
    burn = ("--burn-before", "19", "--dv", "0.1", "--workers", "1")
    goal = ("--target-pc", "0.01", "--hbr", "1000")
    args = (crossing(tmp_path), *goal, *burn, "--catalog", catalog)
    result = keepout("maneuver", *args)
    assert result.exit_code == ExitCode.OK
    assert rescreen_lines(result, catalog) == snapshot(
        [
            "prograde re-screened against 4 objects (<catalog>) from 2026-10-06T00:36:42 to 2026-10-13T00:36:42 UTC; 0 catalog objects left out, of which it says nothing.",
            "  Event secondary 47826 was NOT screened: this re-screen says nothing of the event.",
            "  No new or closer approach at red among the objects screened.",
            "  1 new approach:",
            "    57087 at 2026-10-11 03:14 UTC: range 4068 m, assumed Pc 1.23e-15, green",
            "  Covariances are assumed (keepout's model of GP element set errors), not orbit determination.",
        ]
    )
    (option,) = json.loads(keepout("maneuver", *args, "--json").stdout)["options"]
    rescreen = option["rescreen"]
    assert rescreen["conflict"] is False
    assert rescreen["secondary_screened"] is False
    assert rescreen["event_range_m"] is None


def test_objects_left_out_are_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replace_rescreen(
        monkeypatch, lambda result: dataclasses.replace(result, skipped_objects=1)
    )
    burn = ("--burn-before", "19", "--dv", "1", "--workers", "1")
    args = (crossing(tmp_path), "--hbr", "1000", *burn)
    result = keepout("maneuver", *args, "--catalog", catalog_of(tmp_path))
    assert "; 1 catalog object left out, of which it says nothing." in result.stdout


def test_a_long_list_of_approaches_is_cut_in_the_text_not_in_the_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Chronological, Pc rising: the highest are last in time, first in the text.
    def crowded(result: RescreenResult) -> RescreenResult:
        one = result.new_approaches[0]
        many = tuple(
            dataclasses.replace(one, pc=(index + 1) * 1e-7) for index in range(28)
        )
        return dataclasses.replace(result, new_approaches=(), closer_approaches=many)

    replace_rescreen(monkeypatch, crowded)
    burn = ("--burn-before", "19", "--dv", "1", "--workers", "1")
    args = (crossing(tmp_path), "--hbr", "1000", *burn)
    result = keepout("maneuver", *args, "--catalog", catalog_of(tmp_path))
    lines = result.stdout.splitlines()
    start = lines.index("  28 closer approaches, the 5 of highest Pc:")
    pcs = [line.split("assumed Pc ")[1][:8] for line in lines[start + 1 : start + 6]]
    assert pcs == ["2.80e-06", "2.70e-06", "2.60e-06", "2.50e-06", "2.40e-06"]
    assert lines[start + 6] == "    and 23 more (--json lists them)."
    as_json = keepout("maneuver", *args, "--catalog", catalog_of(tmp_path), "--json")
    (option,) = json.loads(as_json.stdout)["options"]
    assert len(option["rescreen"]["closer_approaches"]) == 28


def test_source_is_read_as_a_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    web = FakeWeb(monkeypatch, celestrak_pages())
    seen: list[tuple[int, int | None]] = []
    real = maneuver.rescreen

    def rescreen(
        plan: ManeuverPlan,
        option: ManeuverOption,
        omms: Sequence[Omm],
        *,
        workers: int | None,
    ) -> RescreenResult:
        seen.append((len(omms), workers))
        near = [omm for omm in omms if omm.catalog_number in NEIGHBOURS]
        return real(plan, option, near, workers=workers)

    monkeypatch.setattr(maneuver, "rescreen", rescreen)
    burn = ("--burn-before", "19", "--dv", "1", "--workers", "1")
    args = (crossing(tmp_path), "--hbr", "1000", *burn, "--source", "celestrak")
    result = keepout("maneuver", *args, "--cache-dir", str(tmp_path / "cache"))
    assert result.exit_code == ExitCode.NO_SAFE_BURN
    assert seen == [(222, 1)]
    assert web.urls
    assert "re-screened against 5 objects (CelesTrak" in result.stdout

"""``keepout maneuver``: real plans and re-screens of the fixture's crossing event.

52939 crosses 47826 at 6 km/s, 19.6 h after the CDM's creation (1029 m, green
at the CDM's hard-body radii). A radius of 100 m puts it above the target and
of 1 km in the red. The plans use the small grid below, a fraction of a second
a plan, unless a test restores the default one. A real re-screen, of a week,
takes seconds: most screen a catalog of the objects near that orbit, a command
line run as text and as JSON re-screens once (``rescreen_once``), and the tests
of what is shown of a re-screen make its result without screening
(:func:`replace_rescreen`).
"""

import dataclasses
import functools
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final

import astropy.units as u
import pytest
from astropy.time import Time
from builders import SMALL_GRID
from cli_builders import (
    FIXTURE_CSV,
    START,
    FakeWeb,
    block_store_extra,
    calculated,
    celestrak_pages,
    crossing,
    keepout,
    maneuver_event,
    operator_kvn,
    operator_run,
    queue,
    stored_crossing,
    text_and_json,
    usage_error,
    without_hbr,
    write,
)
from inline_snapshot import snapshot
from typer.testing import Result
from worker_jobs import small_grid_maneuver

from keepout import maneuver
from keepout._defaults import SCREENING_DURATION
from keepout.cli.output import ExitCode, minute
from keepout.cli.store import STORE_HINT
from keepout.errors import StoreError
from keepout.events import ConjunctionEvent
from keepout.maneuver import (
    ManeuverOption,
    ManeuverPlan,
    NewApproach,
    RescreenResult,
    outcome,
)
from keepout.omm import Omm, load
from keepout.omm.celestrak import dumps_csv
from keepout.risk import RiskLevel
from keepout.screening import CloseApproach, screen
from keepout.store import Store, planning

PRIMARY: Final = 52939
SECONDARY: Final = 47826
NEIGHBOURS: Final = frozenset({PRIMARY, SECONDARY, 57087, 62118, 100000})
"""The objects whose approaches the burns of these tests change."""
FULL_PLAN: Final = maneuver.plan_maneuver
"""The planner on its default grid, 8 s here."""

pytestmark = pytest.mark.usefixtures("small_grid")


@pytest.fixture
def small_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plans on :data:`SMALL_GRID`, whatever grid the command asks for."""
    monkeypatch.setattr(outcome, "plan_maneuver", small_plan)


def small_plan(event: ConjunctionEvent, **settings: Any) -> ManeuverPlan:
    """The plan of ``event`` on :data:`SMALL_GRID` instead of the grid given."""
    return FULL_PLAN(event, **{**settings, "grid": SMALL_GRID})


def catalog_of(folder: Path, numbers: frozenset[int] = NEIGHBOURS) -> str:
    """A catalog file of the fixture's element sets of ``numbers``."""
    chosen = [omm for omm in load(FIXTURE_CSV) if omm.catalog_number in numbers]
    return write(folder, "catalog.csv", dumps_csv(chosen))


def quiet_rescreen(
    plan: ManeuverPlan, option: ManeuverOption, omms: Sequence[Omm]
) -> RescreenResult:
    """A re-screen of ``option`` against ``omms`` that finds no approach and
    leaves the event's secondary outside the volume, made without screening."""
    start = plan.conjunction.tca - option.burn_lead * u.s
    return RescreenResult(
        option=option,
        new_approaches=(),
        closer_approaches=(),
        event_returns=(),
        event_range=None,
        window=(start, start + SCREENING_DURATION * u.s),
        catalog_objects=len(omms),
        secondary_screened=True,
        skipped_objects=0,
    )


def patch_rescreen(
    monkeypatch: pytest.MonkeyPatch, rescreen: Callable[..., RescreenResult]
) -> None:
    """``rescreen`` in place of the library's, where the command looks it up:
    :mod:`keepout.maneuver` for one burn, :mod:`keepout.maneuver.outcome`
    for a plan."""
    monkeypatch.setattr(maneuver, "rescreen", rescreen)
    monkeypatch.setattr(outcome, "rescreen", rescreen)


def replace_rescreen(
    monkeypatch: pytest.MonkeyPatch, change: Callable[[RescreenResult], RescreenResult]
) -> None:
    """Re-screens that are :func:`quiet_rescreen`'s up to ``change``: none
    screens, the tests of what the command shows of a result take seconds."""

    def rescreen(
        plan: ManeuverPlan,
        option: ManeuverOption,
        omms: Sequence[Omm],
        *,
        workers: int | None,
    ) -> RescreenResult:
        return change(quiet_rescreen(plan, option, omms))

    patch_rescreen(monkeypatch, rescreen)


@pytest.fixture
def rescreen_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real re-screens, each burn's run once in the test: the same command
    line run again (as text, then as JSON) takes its results."""
    real = maneuver.rescreen
    done: dict[tuple[float, float, tuple[int, ...]], RescreenResult] = {}

    def rescreen(
        plan: ManeuverPlan,
        option: ManeuverOption,
        omms: Sequence[Omm],
        *,
        workers: int | None,
    ) -> RescreenResult:
        key = (
            option.burn_lead,
            option.delta_v,
            tuple(omm.catalog_number for omm in omms),
        )
        if key not in done:
            done[key] = real(plan, option, omms, workers=workers)
        return done[key]

    patch_rescreen(monkeypatch, rescreen)


def test_plan_prints_options_and_exits_zero(tmp_path: Path) -> None:
    result = keepout("maneuver", crossing(tmp_path), "--hbr", "1000")
    assert result.exit_code == ExitCode.OK
    assert result.stderr == ""
    assert result.stdout == snapshot("""\
Event 2a85ed50c4e6: 52939 / 47826, TCA 2026-10-06 19:36 UTC, target Pc 1.00e-05.
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
        "refusal",
    ]
    assert payload["event"] == "2a85ed50c4e6"
    assert payload["primary"] == str(PRIMARY)
    assert payload["secondary"] == str(SECONDARY)
    assert payload["target_pc"] == 1e-5
    assert payload["method"] == "foster"
    assert list(payload["baseline"]) == [
        "pc",
        "range_at_tca_m",
        "method",
        "meets_target",
        "notes",
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
        "notes",
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
Event 2a85ed50c4e6: 52939 / 47826, TCA 2026-10-06 19:36 UTC, target Pc 1.00e-05.
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


def test_json_names_the_refusal_on_exit_4(tmp_path: Path) -> None:
    result = keepout("maneuver", crossing(tmp_path), "--hbr", "500000", "--json")
    payload = json.loads(result.stdout)
    assert result.exit_code == ExitCode.NO_SAFE_BURN
    assert payload["refusal"] == (
        "No burn on the grid meets the target Pc 1.00e-05 (exit status 4)."
    )


def test_json_refusal_is_null_with_a_safe_option(tmp_path: Path) -> None:
    result = keepout("maneuver", crossing(tmp_path), "--hbr", "1000", "--json")
    payload = json.loads(result.stdout)
    assert result.exit_code == ExitCode.OK
    assert payload["refusal"] is None


@pytest.mark.usefixtures("rescreen_once")
def test_conflict_exits_four(tmp_path: Path) -> None:
    catalog = catalog_of(tmp_path)
    burn = ("--burn-before", "19", "--dv", "1")
    args = (crossing(tmp_path), "--hbr", "1000", *burn, "--workers", "1")
    result = keepout("maneuver", *args, "--catalog", catalog)
    assert result.exit_code == ExitCode.NO_SAFE_BURN
    assert result.stdout.replace(catalog, "<catalog>") == snapshot("""\
Event 2a85ed50c4e6: 52939 / 47826, TCA 2026-10-06 19:36 UTC, target Pc 1.00e-05.
Burns of the primary
Option    Before TCA (h)  Δv (m/s)        Pc  Range (m)  Method  Target
───────────────────────────────────────────────────────────────────────
no burn                -         -  4.25e-02       1029  foster  missed
prograde            19.0        +1  0.00e+00     188388  foster  met   \n\
Note: every Pc rests on an assumed covariance (COVARIANCE_METHOD = DEFAULT)
Note: drag is off: the CDM gives the primary no positive CD_AREA_OVER_MASS
prograde re-screened against 5 objects (<catalog>) from 2026-10-06T00:36:42 to 2026-10-13T00:36:42 UTC; 0 catalog objects left out (not checked).
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


@pytest.mark.slow
def test_one_clean_option_and_one_conflict_exit_zero_and_show_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The default grid, and the whole fixture with what screening leaves out.
    monkeypatch.setattr(outcome, "plan_maneuver", FULL_PLAN)
    args = (crossing(tmp_path), "--hbr", "100", "--catalog", str(FIXTURE_CSV))
    result = keepout("maneuver", *args, "--workers", "1")
    assert result.exit_code == ExitCode.OK
    shown = result.stdout.replace(str(FIXTURE_CSV), "<catalog>")
    assert shown == snapshot("""\
Event 2a85ed50c4e6: 52939 / 47826, TCA 2026-10-06 19:36 UTC, target Pc 1.00e-05.
Burns of the primary
Option      Before TCA (h)  Δv (m/s)        Pc  Range (m)  Method  Target
─────────────────────────────────────────────────────────────────────────
no burn                  -         -  1.40e-05       1029  foster  missed
prograde              16.5  +0.00794  9.96e-06       1330  foster  met   \n\
retrograde            17.2   -0.0316  9.76e-06       5960  foster  met   \n\
Note: every Pc rests on an assumed covariance (COVARIANCE_METHOD = DEFAULT)
Note: drag is off: the CDM gives the primary no positive CD_AREA_OVER_MASS
prograde re-screened against 217 objects (<catalog>) from 2026-10-06T03:09:04 to 2026-10-13T03:09:04 UTC; 5 catalog objects left out (not checked).
  Event secondary 47826 on the element sets (SGP4, not the CDM's orbit): range 1330 m after the burn.
  No new or closer approach at red among the objects screened.
  1 new approach:
    57087 at 2026-10-11 15:48 UTC: range 3386 m, assumed Pc 2.12e-08, green
  Covariances are assumed (keepout's model of GP element set errors), not orbit determination.
retrograde re-screened against 217 objects (<catalog>) from 2026-10-06T02:21:48 to 2026-10-13T02:21:48 UTC; 5 catalog objects left out (not checked).
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
Event 2a85ed50c4e6: 52939 / 47826, TCA 2026-10-06 19:36 UTC, target Pc 1.00e-05.
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


@pytest.mark.parametrize("alone", [("--burn-before", "19"), ("--dv", "0.1")])
def test_burn_before_without_dv_is_a_usage_error(
    tmp_path: Path, alone: tuple[str, str]
) -> None:
    result = keepout("maneuver", crossing(tmp_path), *alone)
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
        write(tmp_path, "drifting.kvn", operator_kvn()[1]),
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
        operator_kvn()[0]
        .replace(operator_run().cdms[0].message_id, "UPDATE")
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
    file = write(tmp_path, "calculated.kvn", calculated(operator_kvn()[0]))
    args = (file, "--hbr", "1000")
    assert "assumed" not in keepout("maneuver", *args).stdout
    assert (
        json.loads(keepout("maneuver", *args, "--json").stdout)["assumed_covariance"]
        is False
    )


def test_pc_of_an_approach_of_the_catalog_is_labelled_assumed(tmp_path: Path) -> None:
    # Even for a CDM with covariances of its own: element sets have none.
    file = write(tmp_path, "c.kvn", calculated(operator_kvn()[0]))
    burn = ("--burn-before", "19", "--dv", "1", "--workers", "1")
    args = (file, "--hbr", "1000", *burn)
    result = keepout("maneuver", *args, "--catalog", catalog_of(tmp_path))
    lines = result.stdout.splitlines()
    assert any("assumed Pc" in line and "100000" in line for line in lines)
    assert (
        "  Covariances are assumed (keepout's model of GP element set errors), "
        "not orbit determination."
    ) in lines


def test_missing_hbr_exits_one(tmp_path: Path) -> None:
    file = write(tmp_path, "crossing.kvn", without_hbr(operator_kvn()[0]))
    result = keepout("maneuver", file)
    assert result.exit_code == ExitCode.ERROR
    assert result.stderr == (
        "Error: no hard-body radius for the primary (52939) and the secondary "
        "(47826): pass --hbr METRES or set hbr_m in the --settings file\n"
    )


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
        "[CDM...]",
        "--target-pc PC",
        "--burn-before",
        "--dv",
        "--hbr",
        "--event ID",
        "--store FILE",
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
    ("option", "value", "reason"),
    [
        ("--target-pc", "0", "must be in (0, 1), got 0.0"),
        ("--target-pc", "1", "must be in (0, 1), got 1.0"),
        ("--target-pc", "nan", "must be in (0, 1), got nan"),
        ("--dv", "0", "must be a finite, non-zero number, got 0.0"),
        ("--dv", "nan", "must be a finite, non-zero number, got nan"),
        ("--dv", "inf", "must be a finite, non-zero number, got inf"),
        ("--burn-before", "0", "must be a positive number, got 0.0"),
        ("--burn-before", "-1", "must be a positive number, got -1.0"),
        ("--burn-before", "nan", "must be a positive number, got nan"),
        ("--hbr", "0", "must be a positive number, got 0.0"),
        ("--workers", "0", "0 is not in the range x>=1."),
    ],
)
def test_bad_options_are_usage_errors(
    tmp_path: Path, option: str, value: str, reason: str
) -> None:
    result = keepout("maneuver", crossing(tmp_path), option, value)
    assert result.exit_code == ExitCode.USAGE
    assert f"Invalid value for '{option}': {reason}" in usage_error(result)


def test_no_cdm_is_a_usage_error() -> None:
    result = keepout("maneuver")
    assert result.exit_code == ExitCode.USAGE
    assert "give CDM files or --event" in usage_error(result)


def test_a_file_that_is_no_cdm_is_named(tmp_path: Path) -> None:
    bad = write(tmp_path, "bad.kvn", "CCSDS_CDM_VERS = 1.0\nNOT A LINE\n")
    result = keepout("maneuver", bad)
    assert result.exit_code == ExitCode.ERROR
    assert result.stderr.startswith(f"Error: {bad}: ")


def test_a_primary_that_cannot_manoeuvre_exits_one(tmp_path: Path) -> None:
    result = keepout("maneuver", write(tmp_path, "drifting.kvn", operator_kvn()[1]))
    assert result.exit_code == ExitCode.ERROR
    assert "cannot manoeuvre (MANEUVERABLE = NO)" in result.stderr


def test_a_catalog_that_cannot_be_read_is_reported_before_planning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("planned before reading the catalog")

    monkeypatch.setattr(outcome, "plan_maneuver", refuse)
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
    def one_way(event: ConjunctionEvent, **settings: Any) -> ManeuverPlan:
        plan = small_plan(event, **settings)
        return dataclasses.replace(plan, retrograde=None)

    monkeypatch.setattr(outcome, "plan_maneuver", one_way)
    result = keepout("maneuver", crossing(tmp_path), "--hbr", "1000")
    assert result.exit_code == ExitCode.OK
    assert "No retrograde burn on the grid meets the target." in result.stdout
    assert "No prograde" not in result.stdout


def rescreen_lines(result: Result, catalog: str) -> list[str]:
    """What ``result`` prints from the first re-screen on."""
    lines = result.stdout.replace(catalog, "<catalog>").splitlines()
    (first,) = (line for line in lines if " re-screened against " in line)
    return lines[lines.index(first) :]


@pytest.mark.usefixtures("rescreen_once")
def test_secondary_still_near_after_the_burn_gives_its_range(tmp_path: Path) -> None:
    catalog = catalog_of(tmp_path)
    burn = ("--burn-before", "16", "--dv", "0.008", "--workers", "1")
    args = (crossing(tmp_path), "--hbr", "100", *burn, "--catalog", catalog)
    result = keepout("maneuver", *args)
    assert rescreen_lines(result, catalog) == snapshot(
        [
            "prograde re-screened against 5 objects (<catalog>) from 2026-10-06T03:36:42 to 2026-10-13T03:36:42 UTC; 0 catalog objects left out (not checked).",
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


@pytest.mark.usefixtures("rescreen_once")
def test_a_secondary_not_screened_says_nothing_of_the_event(tmp_path: Path) -> None:
    catalog = catalog_of(tmp_path, NEIGHBOURS - {SECONDARY})
    burn = ("--burn-before", "19", "--dv", "0.1", "--workers", "1")
    goal = ("--target-pc", "0.01", "--hbr", "1000")
    args = (crossing(tmp_path), *goal, *burn, "--catalog", catalog)
    result = keepout("maneuver", *args)
    assert result.exit_code == ExitCode.OK
    assert rescreen_lines(result, catalog) == snapshot(
        [
            "prograde re-screened against 4 objects (<catalog>) from 2026-10-06T00:36:42 to 2026-10-13T00:36:42 UTC; 0 catalog objects left out (not checked).",
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
    assert "; 1 catalog object left out (not checked)." in result.stdout


def test_a_long_list_of_approaches_is_cut_in_the_text_not_in_the_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Chronological, Pc rising: the highest are last in time, first in the text.
    def crowded(result: RescreenResult) -> RescreenResult:
        one = NewApproach(approach=later_approach(), pc=1e-7, level=RiskLevel.GREEN)
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

    def rescreen(
        plan: ManeuverPlan,
        option: ManeuverOption,
        omms: Sequence[Omm],
        *,
        workers: int | None,
    ) -> RescreenResult:
        seen.append((len(omms), workers))
        near = [omm for omm in omms if omm.catalog_number in NEIGHBOURS]
        return dataclasses.replace(
            quiet_rescreen(plan, option, near), new_approaches=(red(),)
        )

    patch_rescreen(monkeypatch, rescreen)
    burn = ("--burn-before", "19", "--dv", "1", "--workers", "1")
    args = (crossing(tmp_path), "--hbr", "1000", *burn, "--source", "celestrak")
    result = keepout("maneuver", *args, "--cache-dir", str(tmp_path / "cache"))
    assert result.exit_code == ExitCode.NO_SAFE_BURN
    assert seen == [(222, 1)]
    assert web.urls
    assert "re-screened against 5 objects (CelesTrak" in result.stdout


ONE_SATELLITE: Final = """\
catalog_hard_body_radius = 1.0

[[satellites]]
catalog_number = 52939
hard_body_radius = 5.0
maneuverable = true
"""
"""A fleet of 52939 alone: one approach in :func:`catalog_of`'s catalog."""


@pytest.fixture(scope="module")
def screened_store(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    """A store of :data:`ONE_SATELLITE` screened against :func:`catalog_of`'s
    catalog (deleted since: the store keeps a copy), and the crossing
    event's id; screened once, tests copy it (:func:`copied_store`)."""
    folder = tmp_path_factory.mktemp("screened")
    fleet = write(folder, "fleet.toml", ONE_SATELLITE)
    catalog = catalog_of(folder)
    db = folder / "keepout.db"
    window = ("--start", START, "--days", "3", "--workers", "1")
    args = ("--fleet", fleet, "--catalog", catalog, *window, "--store", str(db))
    screened = keepout("screen", *args, "--json")
    assert screened.exit_code == ExitCode.OK
    (event_id,) = json.loads(screened.stdout)["store"]["events"]
    Path(catalog).unlink()
    return db, event_id


def copied_store(screened: tuple[Path, str], folder: Path) -> tuple[Path, str]:
    """The screened store and its catalog copies, copied into ``folder``."""
    db, event_id = screened
    shutil.copy(db, folder / db.name)
    catalogs = f"{db.name}.catalogs"
    shutil.copytree(db.parent / catalogs, folder / catalogs)
    return folder / db.name, event_id


def test_event_from_the_store_plans_like_its_files(tmp_path: Path) -> None:
    db, event_id = stored_crossing(tmp_path)
    options = ("--hbr", "1000", "--burn-before", "19", "--dv", "1")
    stored = ("maneuver", "--event", event_id, "--store", str(db), *options)
    from_files = keepout("maneuver", crossing(tmp_path), *options, "--json")
    from_store = keepout(*stored, "--json")
    assert from_store.exit_code == from_files.exit_code == ExitCode.OK
    planned = json.loads(from_store.stdout)
    assert planned["event"] == event_id
    assert planned | {"event": ""} == json.loads(from_files.stdout) | {"event": ""}
    text = keepout(*stored).stdout
    assert text.startswith(f"Event {event_id}: 52939 / 47826, TCA ")
    # Screened by no one: no catalog to re-screen against.
    assert text.splitlines()[-1].startswith("Not re-screened: nothing is known")


def test_event_rescreens_against_its_stored_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    screened_store: tuple[Path, str],
) -> None:
    db, event_id = copied_store(screened_store, tmp_path)
    seen: list[set[int]] = []

    def rescreen(
        plan: ManeuverPlan,
        option: ManeuverOption,
        omms: Sequence[Omm],
        *,
        workers: int | None,
    ) -> RescreenResult:
        seen.append({omm.catalog_number for omm in omms})
        return quiet_rescreen(plan, option, omms)

    patch_rescreen(monkeypatch, rescreen)
    burn = ("--hbr", "1000", "--burn-before", "19", "--dv", "1")
    stored = ("maneuver", "--event", event_id, "--store", str(db), *burn)
    result = keepout(*stored)
    assert result.exit_code == ExitCode.OK
    in_fixture = {omm.catalog_number for omm in load(FIXTURE_CSV)}
    assert seen == [NEIGHBOURS & in_fixture]
    (reference,) = list((tmp_path / f"{db.name}.catalogs").iterdir())
    screened_with = screened_store[0].parent / "catalog.csv"
    assert (
        f"prograde re-screened against {len(seen[0])} objects (stored copy of "
        f"{screened_with}) from "
    ) in result.stdout
    assert reference.suffix == ".csv"
    assert json.loads(keepout(*stored, "--json").stdout)["event"] == event_id
    # --catalog still names the catalog to re-screen against.
    seen.clear()
    other = catalog_of(tmp_path, frozenset({PRIMARY, SECONDARY}))
    assert keepout(*stored, "--catalog", other).exit_code == ExitCode.OK
    assert seen == [{PRIMARY, SECONDARY}]


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda path: path.write_text("changed\n"), "changed since it was referenced"),
        (Path.unlink, "cannot read the catalog file"),
    ],
    ids=["changed", "deleted"],
)
def test_event_with_a_changed_catalog_file_fails_naming_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    screened_store: tuple[Path, str],
    change: Callable[[Path], object],
    reason: str,
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("planned without the event's catalog")

    monkeypatch.setattr(outcome, "plan_maneuver", refuse)
    db, event_id = copied_store(screened_store, tmp_path)
    (copy,) = list((tmp_path / f"{db.name}.catalogs").iterdir())
    change(copy)
    result = keepout("maneuver", "--event", event_id, "--store", str(db))
    assert result.exit_code == ExitCode.ERROR
    assert result.stderr.startswith("Error: ")
    assert str(copy) in result.stderr
    assert reason in result.stderr


@pytest.mark.parametrize(
    ("given", "reason"),
    [
        (("{cdm}", "--event", "deadbeef", "--store", "{db}"), "not both"),
        ((), "give CDM files or --event"),
        (("--event", "deadbeef"), "--event needs --store"),
    ],
    ids=["both", "neither", "no-store"],
)
def test_files_and_event_together_is_a_usage_error(
    tmp_path: Path, given: tuple[str, ...], reason: str
) -> None:
    names = {"cdm": crossing(tmp_path), "db": str(tmp_path / "keepout.db")}
    args = [item.format_map(names) for item in given]
    result = keepout("maneuver", *args, env={"KEEPOUT_STORE": None})
    assert result.exit_code == ExitCode.USAGE
    assert reason in usage_error(result)
    assert not (tmp_path / "keepout.db").exists()


def test_unknown_event_id_is_named(tmp_path: Path) -> None:
    db, _ = stored_crossing(tmp_path)
    result = keepout("maneuver", "--event", "deadbeef", "--store", str(db))
    assert result.exit_code == ExitCode.ERROR
    assert result.stderr == "Error: no event deadbeef in the store\n"
    missing = tmp_path / "missing.db"
    result = keepout("maneuver", "--event", "deadbeef", "--store", str(missing))
    assert result.exit_code == ExitCode.ERROR
    assert result.stderr == f"Error: no store at {missing}\n"
    assert not missing.exists()


def test_store_is_read_from_the_environment(tmp_path: Path) -> None:
    db, event_id = stored_crossing(tmp_path)
    burn = ("--hbr", "1000", "--burn-before", "19", "--dv", "1", "--json")
    env = {"KEEPOUT_STORE": str(db)}
    result = keepout("maneuver", "--event", event_id, *burn, env=env)
    assert json.loads(result.stdout)["event"] == event_id
    # Files ignore it.
    assert keepout("maneuver", crossing(tmp_path), *burn, env=env).exit_code == 0


def test_the_store_extra_is_needed_only_with_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "keepout.db"
    burn = ("--hbr", "1000", "--burn-before", "19", "--dv", "1", "--json")
    env = {"KEEPOUT_STORE": str(db)}
    block_store_extra("sqlalchemy", monkeypatch)
    files = keepout("maneuver", crossing(tmp_path), *burn, env=env)
    assert files.exit_code == ExitCode.OK
    assert files.stderr == ""
    stored = ("--event", "deadbeef")
    for extra in (burn, ("--hbr", "1000"), ("--queue",)):
        event = keepout("maneuver", *stored, *extra, env=env)
        assert event.exit_code == ExitCode.ERROR, extra
        assert event.stderr == f"Error: {STORE_HINT}\n", extra
    for command in (("jobs", "list"), ("jobs", "cancel", "1"), ("worker",)):
        result = keepout(*command, env=env)
        assert result.exit_code == ExitCode.ERROR, command
        assert result.stderr == f"Error: {STORE_HINT}\n", command
    assert not db.exists()


def test_settings_give_the_target_and_target_pc_wins(tmp_path: Path) -> None:
    settings = write(
        tmp_path,
        "settings.toml",
        "[pc]\nhbr_m = 1000.0\n[maneuver]\ntarget_pc = 1e-3\n",
    )
    burn = (crossing(tmp_path), "--burn-before", "7", "--dv", "0.251", "--json")

    def target(*options: str) -> object:
        result = keepout("maneuver", *burn, "--settings", settings, *options)
        return json.loads(result.stdout)["target_pc"]

    assert target() == 1e-3
    assert target("--target-pc", "1e-4") == 1e-4


@pytest.fixture
def planned(monkeypatch: pytest.MonkeyPatch) -> list[ConjunctionEvent]:
    """The events the planner is called on (on :data:`SMALL_GRID`), in order."""
    seen: list[ConjunctionEvent] = []

    def counted(event: ConjunctionEvent, **settings: Any) -> ManeuverPlan:
        seen.append(event)
        return small_plan(event, **settings)

    monkeypatch.setattr(outcome, "plan_maneuver", counted)
    return seen


def test_other_settings_miss_the_cache(
    tmp_path: Path, planned: list[ConjunctionEvent]
) -> None:
    db, event_id = stored_crossing(tmp_path)
    assert keepout(*maneuver_event(db, event_id)).exit_code == ExitCode.OK
    other = maneuver_event(db, event_id, hbr="1001")
    computed = keepout(*other, "--json")
    assert len(planned) == 2
    # Then cached for them.
    assert keepout(*other, "--json").stdout == computed.stdout
    assert len(planned) == 2


def test_cache_gives_the_output_of_a_computed_plan(
    tmp_path: Path, planned: list[ConjunctionEvent]
) -> None:
    db, event_id = stored_crossing(tmp_path)
    stored = maneuver_event(db, event_id)
    computed = text_and_json(*stored)
    cached = text_and_json(*stored)
    # The first run cached its plan: the second, as JSON, read it.
    assert len(planned) == 1
    assert [r.stdout for r in cached] == [r.stdout for r in computed]
    # A copy of the store without the cache computes the same JSON.
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    db2, _ = stored_crossing(fresh)
    recomputed = keepout(*maneuver_event(db2, event_id), "--json")
    assert len(planned) == 2
    assert recomputed.stdout == cached[1].stdout


def test_a_cached_plan_without_a_safe_burn_exits_four(
    tmp_path: Path, planned: list[ConjunctionEvent]
) -> None:
    db, event_id = stored_crossing(tmp_path)
    stored = maneuver_event(db, event_id, hbr="500000")
    computed = text_and_json(*stored)
    cached = text_and_json(*stored)
    assert len(planned) == 1
    assert [r.exit_code for r in computed + cached] == [ExitCode.NO_SAFE_BURN] * 4
    assert [r.stdout for r in cached] == [r.stdout for r in computed]
    assert "(exit status 4)." in cached[0].stdout


def test_a_cached_plan_shows_its_rescreens_without_the_catalog_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    planned: list[ConjunctionEvent],
    screened_store: tuple[Path, str],
) -> None:
    replace_rescreen(
        monkeypatch, per_direction(only(returns=[back()]), only(new=[red()]))
    )
    db, event_id = copied_store(screened_store, tmp_path)
    stored = maneuver_event(db, event_id)
    computed = text_and_json(*stored)
    assert "prograde re-screened against" in computed[0].stdout
    assert "CONFLICT: a new or closer approach is at red." in computed[0].stdout
    # The cache is read without the store's copy of the catalog.
    (copy,) = list((tmp_path / f"{db.name}.catalogs").iterdir())
    copy.unlink()
    cached = text_and_json(*stored)
    assert len(planned) == 1
    assert [r.stdout for r in cached] == [r.stdout for r in computed]
    assert [r.exit_code for r in cached] == [r.exit_code for r in computed]
    # Other settings miss the cache: then the copy is needed.
    missed = keepout(*stored, "--target-pc", "1e-6")
    assert missed.exit_code == ExitCode.ERROR
    assert missed.stderr.startswith("Error: ")
    assert str(copy) in missed.stderr
    assert len(planned) == 1


def test_a_plan_not_cached_is_still_shown(
    tmp_path: Path, planned: list[ConjunctionEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    db, event_id = stored_crossing(tmp_path)
    stored = maneuver_event(db, event_id)

    def refuse(*args: object) -> None:
        raise StoreError("database is locked")

    with monkeypatch.context() as patched:
        patched.setattr(Store, "save_maneuver", refuse)
        result = keepout(*stored)
    assert result.exit_code == ExitCode.OK
    assert result.stderr == "Warning: the plan was not cached: database is locked\n"
    assert len(planned) == 1
    # Not cached: computed again, the same plan shown.
    again = keepout(*stored)
    assert len(planned) == 2
    assert (again.exit_code, again.stdout) == (result.exit_code, result.stdout)


def test_a_catalog_given_is_not_cached(
    tmp_path: Path, planned: list[ConjunctionEvent], monkeypatch: pytest.MonkeyPatch
) -> None:
    replace_rescreen(monkeypatch, lambda result: result)
    db, event_id = stored_crossing(tmp_path)
    stored = maneuver_event(db, event_id)
    other = catalog_of(tmp_path, frozenset({PRIMARY, SECONDARY}))
    for _ in range(2):
        assert keepout(*stored, "--catalog", other).exit_code == ExitCode.OK
    assert len(planned) == 2
    # Nor was it cached under the stored catalog's key.
    assert keepout(*stored).exit_code == ExitCode.OK
    assert len(planned) == 3


def later_crossing(folder: Path) -> str:
    """A CDM of the crossing event created an hour after :func:`crossing`'s."""
    later = (
        operator_kvn()[0]
        .replace(operator_run().cdms[0].message_id, "UPDATE")
        .replace(
            "CREATION_DATE                  = 2026-10-06T00",
            "CREATION_DATE                  = 2026-10-06T01",
        )
    )
    return write(folder, "later.kvn", later)


def test_new_cdm_misses_the_cache(
    tmp_path: Path, planned: list[ConjunctionEvent]
) -> None:
    db, event_id = stored_crossing(tmp_path)
    stored = maneuver_event(db, event_id)
    assert keepout(*stored).exit_code == ExitCode.OK
    assert keepout(*stored).exit_code == ExitCode.OK
    assert len(planned) == 1
    added = keepout("store", "add", later_crossing(tmp_path), "--store", str(db))
    assert added.exit_code == ExitCode.OK
    assert keepout(*stored, "--json").exit_code == ExitCode.OK
    assert len(planned) == 2
    assert planned[1].latest.message_id == "UPDATE"
    assert keepout(*stored).exit_code == ExitCode.OK
    assert len(planned) == 2


@pytest.mark.timeout(60)
def test_queue_then_worker_then_cached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    planned: list[ConjunctionEvent],
) -> None:
    db, event_id = stored_crossing(tmp_path)
    queued = queue(db, event_id)
    assert queued.exit_code == ExitCode.OK
    assert queued.stdout == f"Job 1 queued for event {event_id}.\n"
    listed = keepout("jobs", "list", "--store", str(db), "--json")
    assert [(j["job"], j["status"]) for j in json.loads(listed.stdout)["jobs"]] == [
        (1, "queued")
    ]
    monkeypatch.setattr(planning, "run_maneuver_job", small_grid_maneuver)
    ran = keepout("worker", "--store", str(db), "--until-idle", "--workers", "1")
    assert ran.exit_code == ExitCode.OK
    assert ran.stdout == "Ran 1 job.\n"
    assert "job 1 succeeded" in ran.stderr
    stored = maneuver_event(db, event_id)
    from_worker = text_and_json(*stored)
    assert planned == []
    assert json.loads(queue(db, event_id, "--json").stdout) == {
        "event": event_id,
        "job": None,
        "status": None,
        "cached": True,
    }
    assert queue(db, event_id).stdout == (
        f"Event {event_id} is already planned for these settings: "
        f"keepout maneuver --event {event_id} shows it.\n"
    )
    # The same plan, computed here in a store without the cache.
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    db2, _ = stored_crossing(fresh)
    computed = text_and_json(*maneuver_event(db2, event_id))
    assert [r.stdout for r in from_worker] == [r.stdout for r in computed]
    assert [r.exit_code for r in from_worker] == [r.exit_code for r in computed]


def test_queue_twice_returns_the_same_job(tmp_path: Path) -> None:
    db, event_id = stored_crossing(tmp_path)
    first = json.loads(queue(db, event_id, "--json").stdout)
    assert first == {"event": event_id, "job": 1, "status": "queued", "cached": False}
    again = queue(db, event_id)
    assert again.exit_code == ExitCode.OK
    assert again.stdout == f"Job 1 is already queued for event {event_id}.\n"
    assert json.loads(queue(db, event_id, "--json").stdout) == first
    # Other settings are another job.
    other = queue(db, event_id, "--target-pc", "1e-6", "--json")
    assert json.loads(other.stdout)["job"] == 2


def test_queue_needs_event(tmp_path: Path) -> None:
    db = tmp_path / "keepout.db"
    result = keepout("maneuver", crossing(tmp_path), "--store", str(db), "--queue")
    assert result.exit_code == ExitCode.USAGE
    assert "--queue needs --event" in usage_error(result)
    assert not db.exists()


@pytest.mark.parametrize(
    ("given", "named"),
    [
        (("--catalog", "{catalog}"), "--catalog"),
        (("--source", "celestrak"), "--source"),
        (("--burn-before", "19", "--dv", "1"), "--burn-before"),
        (("--workers", "2"), "--workers"),
    ],
    ids=["catalog", "source", "burn", "workers"],
)
def test_queue_refuses_options_of_a_sync_run(
    tmp_path: Path, given: tuple[str, ...], named: str
) -> None:
    db, event_id = stored_crossing(tmp_path)
    names = {"catalog": catalog_of(tmp_path)}
    args = [item.format_map(names) for item in given]
    result = queue(db, event_id, *args)
    assert result.exit_code == ExitCode.USAGE
    assert f"--queue cannot take {named}" in usage_error(result)
    listed = keepout("jobs", "list", "--store", str(db), "--json")
    assert json.loads(listed.stdout)["jobs"] == []

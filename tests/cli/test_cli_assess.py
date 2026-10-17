import json
import re
from pathlib import Path
from typing import get_args

import pytest
from astropy.time import Time
from cli_builders import DAYS, FIXTURE_CSV, FLEET_TOML, START, keepout
from inline_snapshot import snapshot

from keepout.cdm import kvn, xml
from keepout.cli.assess import AUTO_METHOD, PcMethod, method_choice
from keepout.cli.output import ExitCode
from keepout.omm import load
from keepout.pc import METHODS, Method
from keepout.synthetic import read_fleet
from keepout.synthetic.operator import run

RESULT = run(
    read_fleet(FLEET_TOML),
    load(FIXTURE_CSV),
    Time(START, scale="utc"),
    float(DAYS) * 86400.0,
)
"""Two CDMs: 52939 against a Starlink 0.82 d after creation (green), and
Intelsat 39 against Luch-5X 2.50 d after it (yellow)."""

GREEN, YELLOW = (kvn.dumps(cdm) for cdm in RESULT.cdms)


def write(folder: Path, name: str, text: str) -> str:
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_assess_prints_each_event_and_why(tmp_path: Path) -> None:
    files = [write(tmp_path, "green.kvn", GREEN), write(tmp_path, "yellow.kvn", YELLOW)]
    result = keepout("assess", *files)
    assert result.exit_code == ExitCode.OK
    assert result.stdout == snapshot("""\
Conjunction events
TCA (UTC)         Objects        Miss (m)  CDMs        Pc  Level   Action \n\
──────────────────────────────────────────────────────────────────────────
2026-10-06 19:36  52939 / 47826      1029     1  4.37e-08  green   ignore \n\
2026-10-08 12:03  44476 / 55841      3647     1  7.38e-06  yellow  monitor
2 events from 2 CDMs: 1 monitor, 1 ignore.
2 of 2 events rest on assumed covariances (COVARIANCE_METHOD = DEFAULT), not orbit determination.
44476/55841/2026-10-08T12:03:02.771792: monitor (latest Pc 7.38e-06 is yellow, 2.50 d before TCA; trend undetermined; diagnostic: 2-D Pc assumptions questionable: the encounter lasts about 2.51e+03 s, 2.9% of the primary's orbital period (limit 1%); a 3-D Pc may differ; diagnostic: the covariances of primary (44476) and secondary (55841) are known only in position (zero velocity block): the 3-D methods need the velocity uncertainty, so the 2-D Pc is used)
""")


def test_assess_exits_with_3_when_an_event_needs_a_manoeuvre(tmp_path: Path) -> None:
    files = [write(tmp_path, "green.kvn", GREEN), write(tmp_path, "yellow.xml", "")]
    Path(files[1]).write_text(xml.dumps(RESULT.cdms[1]), encoding="utf-8")
    result = keepout("assess", *files, "--hbr", "1000")
    assert result.exit_code == ExitCode.MANOEUVRE
    assert result.stdout == snapshot("""\
Conjunction events
TCA (UTC)         Objects        Miss (m)  CDMs        Pc  Level  Action        \n\
────────────────────────────────────────────────────────────────────────────────
2026-10-06 19:36  52939 / 47826      1029     1  4.25e-02  red    manoeuvre     \n\
2026-10-08 12:03  44476 / 55841      3647     1  1.73e-02  red    prepare       \n\
                                                                  manoeuvre     \n\
2 events from 2 CDMs: 1 manoeuvre, 1 prepare manoeuvre.
2 of 2 events rest on assumed covariances (COVARIANCE_METHOD = DEFAULT), not orbit determination.
52939/47826/2026-10-06T19:36:42.717332: manoeuvre (latest Pc 4.25e-02 is red, 0.82 d before TCA; trend undetermined; within the decision point (2.00 d before TCA))
44476/55841/2026-10-08T12:03:02.771792: prepare manoeuvre (latest Pc 1.73e-02 is red, 2.50 d before TCA; trend undetermined; before the decision point (2.00 d before TCA): prepare a manoeuvre; diagnostic: 2-D Pc assumptions questionable: the encounter lasts about 3.5e+03 s, 4.1% of the primary's orbital period (limit 1%); a 3-D Pc may differ; diagnostic: the covariances of primary (44476) and secondary (55841) are known only in position (zero velocity block): the 3-D methods need the velocity uncertainty, so the 2-D Pc is used)
At least one event needs a manoeuvre (exit status 3).
""")
    assert result.stderr == ""
    as_json = keepout("assess", *files, "--hbr", "1000", "--json")
    assert as_json.exit_code == ExitCode.MANOEUVRE
    payload = json.loads(as_json.stdout)
    assert payload["cdms"] == 2
    assert payload["manoeuvre"] is True
    assert [event["action"] for event in payload["events"]] == [
        "manoeuvre",
        "prepare_manoeuvre",
    ]


def test_assess_follows_the_trend_of_an_event_over_its_cdms(tmp_path: Path) -> None:
    later = RESULT.cdms[1]
    newer = re.sub(
        r"(CREATION_DATE *= )2026-10-06",
        r"\g<1>2026-10-07",
        re.sub(
            r"(HBR *= )20.0", r"\g<1>60.0", YELLOW.replace(later.message_id, "UPDATE")
        ),
    )
    assert newer.count("UPDATE") == 1
    files = [write(tmp_path, "first.kvn", YELLOW), write(tmp_path, "second.kvn", newer)]
    result = keepout("assess", *files, "--json")
    assert result.exit_code == ExitCode.OK
    (event,) = json.loads(result.stdout)["events"]
    assert event["cdms"] == [later.message_id, "UPDATE"]
    first, second = event["observations"]
    assert second["pc"] > first["pc"]
    assert second["time_to_tca"] == pytest.approx(first["time_to_tca"] - 86400.0)
    assert event["trend"] == "rising"


def test_assess_computes_with_the_method_given(tmp_path: Path) -> None:
    file = write(tmp_path, "green.kvn", GREEN)
    result = keepout("assess", file, "--method", "elrod", "--json")
    assert result.exit_code == ExitCode.OK
    (event,) = json.loads(result.stdout)["events"]
    assert event["observations"][0]["method"] == "elrod"


def test_assess_names_a_file_it_cannot_read(tmp_path: Path) -> None:
    good = write(tmp_path, "green.kvn", GREEN)
    bad = write(tmp_path, "bad.kvn", "CCSDS_CDM_VERS = 1.0\nNOT A LINE\n")
    result = keepout("assess", good, bad)
    assert result.exit_code == ExitCode.ERROR
    assert result.stdout == ""
    assert result.stderr.startswith(f"Error: {bad}: ")
    assert result.stderr.count("\n") == 1


def test_assess_reports_cdms_that_do_not_make_an_event(tmp_path: Path) -> None:
    files = [write(tmp_path, "a.kvn", GREEN), write(tmp_path, "b.kvn", GREEN)]
    result = keepout("assess", *files)
    assert result.exit_code == ExitCode.ERROR
    assert result.stderr == f"Error: MESSAGE_ID repeated: {RESULT.cdms[0].message_id}\n"


def test_assess_needs_both_hard_body_radii_or_hbr(tmp_path: Path) -> None:
    lines = GREEN.splitlines(keepends=True)
    without = "".join(line for line in lines if not line.startswith("HBR"))
    file = write(tmp_path, "green.kvn", without)
    result = keepout("assess", file)
    assert result.exit_code == ExitCode.ERROR
    assert result.stderr.startswith("Error: no HBR for both objects in ")
    assert keepout("assess", file, "--hbr", "6").exit_code == ExitCode.OK


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((), "Missing argument 'files'"),
        (("missing.kvn",), "does not exist"),
        ((".",), "is a directory"),
        (("green.kvn", "--hbr", "0"), "must be a positive number"),
        (("green.kvn", "--method", "nonsense"), "'nonsense' is not one of"),
    ],
)
def test_assess_refuses_a_wrong_command_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: tuple[str, ...],
    expected: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    write(tmp_path, "green.kvn", GREEN)
    result = keepout("assess", *args)
    assert result.exit_code == ExitCode.USAGE
    assert expected in result.stderr


def test_the_method_option_offers_every_pc_method() -> None:
    # PcMethod is built from keepout.pc's names for Typer, which needs an enum.
    assert [m.value for m in PcMethod] == ["auto", *get_args(Method)]
    assert set(get_args(Method)) == set(METHODS)
    assert [method_choice(m.value) for m in PcMethod] == list(PcMethod)
    assert AUTO_METHOD is PcMethod("auto")


def test_a_name_that_is_no_pc_method_is_refused() -> None:
    with pytest.raises(ValueError, match="not a Pc method: 'best'"):
        method_choice("best")


def with_velocity_of_object1(text: str) -> str:
    """The CDM with OBJECT2 given OBJECT1's velocity: no encounter plane."""
    marker = "\nOBJECT                         = OBJECT2"
    first, second = text.split(marker)
    for keyword in ("X_DOT", "Y_DOT", "Z_DOT"):
        line = re.search(rf"^{keyword} .*$", first, re.MULTILINE)
        assert line is not None
        second = re.sub(rf"^{keyword} .*$", line.group(), second, flags=re.MULTILINE)
    return first + marker + second


def test_assess_shows_a_pc_that_failed_or_is_missing(tmp_path: Path) -> None:
    failed = write(tmp_path, "failed.kvn", with_velocity_of_object1(GREEN))
    late = write(
        tmp_path,
        "late.kvn",
        re.sub(r"(CREATION_DATE *= )2026-10-06", r"\g<1>2026-10-09", YELLOW),
    )
    result = keepout("assess", failed, late, "--method", "foster")
    assert result.exit_code == ExitCode.OK
    assert result.stdout == snapshot("""\
Conjunction events
TCA (UTC)         Objects        Miss (m)  CDMs      Pc  Level  Action \n\
───────────────────────────────────────────────────────────────────────
2026-10-06 19:36  52939 / 47826      1029     1  failed  -      monitor
2026-10-08 12:03  44476 / 55841      3647     1       -  -      monitor
2 events from 2 CDMs: 2 monitor.
2 of 2 events rest on assumed covariances (COVARIANCE_METHOD = DEFAULT), not orbit determination.
52939/47826/2026-10-06T19:36:42.717332: monitor (latest observation has no Pc: GeometryError: relative velocity is zero: the encounter plane is undefined)
44476/55841/2026-10-08T12:03:02.771792: monitor (no observation before TCA: nothing to decide on)
""")
    payload = json.loads(keepout("assess", late, "--json").stdout)
    (event,) = payload["events"]
    assert event["pc"] is None
    assert event["level"] is None
    assert event["notes"][0] == "1 observation(s) issued after TCA left out"


def test_assess_counts_in_the_singular(tmp_path: Path) -> None:
    result = keepout("assess", write(tmp_path, "green.kvn", GREEN))
    assert "1 event from 1 CDM: 1 ignore." in result.stdout


def test_assess_labels_events_whose_covariances_are_assumed(tmp_path: Path) -> None:
    calculated = YELLOW.replace(
        "COVARIANCE_METHOD              = DEFAULT",
        "COVARIANCE_METHOD              = CALCULATED",
    )
    assert "= DEFAULT" not in calculated
    files = [
        write(tmp_path, "green.kvn", GREEN),
        write(tmp_path, "yellow.kvn", calculated),
    ]
    result = keepout("assess", *files, "--hbr", "1000")
    assert result.exit_code == ExitCode.MANOEUVRE
    assert (
        "1 of 2 events rests on assumed covariances (COVARIANCE_METHOD = "
        "DEFAULT), not orbit determination."
    ) in result.stdout.splitlines()
    as_json = keepout("assess", *files, "--hbr", "1000", "--json")
    assert as_json.exit_code == ExitCode.MANOEUVRE
    payload = json.loads(as_json.stdout)
    assert payload["assumed_covariances"] == 1
    assert [e["assumed_covariances"] for e in payload["events"]] == [True, False]
    only = keepout("assess", files[1], "--hbr", "1000")
    assert "assumed covariances" not in only.stdout
    payload = json.loads(keepout("assess", files[1], "--json").stdout)
    assert payload["assumed_covariances"] == 0

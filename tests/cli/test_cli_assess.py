import json
import re
from pathlib import Path
from typing import Any, get_args

import pytest
from cli_builders import (
    calculated,
    keepout,
    operator_kvn,
    operator_run,
    without_hbr,
    write,
)
from inline_snapshot import snapshot

from keepout.cdm import xml
from keepout.cli.assess import PcMethod, method_choice
from keepout.cli.output import ExitCode
from keepout.pc import METHODS, Method


def green() -> str:
    """52939 against a Starlink 0.82 d after creation (green), as KVN."""
    return operator_kvn()[0]


def yellow() -> str:
    """Intelsat 39 against Luch-5X 2.50 d after creation (yellow), as KVN."""
    return operator_kvn()[1]


def test_assess_prints_each_event_and_why(tmp_path: Path) -> None:
    files = [
        write(tmp_path, "green.kvn", green()),
        write(tmp_path, "yellow.kvn", yellow()),
    ]
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
3befe6f1305d: monitor (latest Pc 7.38e-06 is yellow, 2.50 d before TCA; trend undetermined; diagnostic: 2-D Pc assumptions questionable: the encounter lasts about 2.51e+03 s, 2.9% of the primary's orbital period (limit 1%); a 3-D Pc may differ; diagnostic: the covariances of primary (44476) and secondary (55841) are known only in position (zero velocity block): the 3-D methods need the velocity uncertainty, so the 2-D Pc is used)
""")


def test_assess_exits_with_3_when_an_event_needs_a_manoeuvre(tmp_path: Path) -> None:
    files = [write(tmp_path, "green.kvn", green()), write(tmp_path, "yellow.xml", "")]
    Path(files[1]).write_text(xml.dumps(operator_run().cdms[1]), encoding="utf-8")
    result = keepout("assess", *files, "--hbr", "1000")
    assert result.exit_code == ExitCode.MANEUVER
    assert result.stdout == snapshot("""\
Conjunction events
TCA (UTC)         Objects        Miss (m)  CDMs        Pc  Level  Action        \n\
────────────────────────────────────────────────────────────────────────────────
2026-10-06 19:36  52939 / 47826      1029     1  4.25e-02  red    maneuver      \n\
2026-10-08 12:03  44476 / 55841      3647     1  1.73e-02  red    prepare       \n\
                                                                  maneuver      \n\
2 events from 2 CDMs: 1 maneuver, 1 prepare maneuver.
2 of 2 events rest on assumed covariances (COVARIANCE_METHOD = DEFAULT), not orbit determination.
2a85ed50c4e6: maneuver (latest Pc 4.25e-02 is red, 0.82 d before TCA; trend undetermined; within the decision point (2.00 d before TCA))
3befe6f1305d: prepare maneuver (latest Pc 1.73e-02 is red, 2.50 d before TCA; trend undetermined; before the decision point (2.00 d before TCA): prepare a manoeuvre; diagnostic: 2-D Pc assumptions questionable: the encounter lasts about 3.5e+03 s, 4.1% of the primary's orbital period (limit 1%); a 3-D Pc may differ; diagnostic: the covariances of primary (44476) and secondary (55841) are known only in position (zero velocity block): the 3-D methods need the velocity uncertainty, so the 2-D Pc is used)
At least one event needs a manoeuvre (exit status 3).
""")
    assert result.stderr == ""
    as_json = keepout("assess", *files, "--hbr", "1000", "--json")
    assert as_json.exit_code == ExitCode.MANEUVER
    payload = json.loads(as_json.stdout)
    assert payload["cdm_count"] == 2
    assert payload["maneuver"] is True
    assert [event["action"] for event in payload["events"]] == [
        "maneuver",
        "prepare_maneuver",
    ]


def test_assess_follows_the_trend_of_an_event_over_its_cdms(tmp_path: Path) -> None:
    later = operator_run().cdms[1]
    newer = re.sub(
        r"(CREATION_DATE *= )2026-10-06",
        r"\g<1>2026-10-07",
        re.sub(
            r"(HBR *= )20.0", r"\g<1>60.0", yellow().replace(later.message_id, "UPDATE")
        ),
    )
    assert newer.count("UPDATE") == 1
    files = [
        write(tmp_path, "first.kvn", yellow()),
        write(tmp_path, "second.kvn", newer),
    ]
    result = keepout("assess", *files, "--json")
    assert result.exit_code == ExitCode.OK
    (event,) = json.loads(result.stdout)["events"]
    assert event["cdms"] == [later.message_id, "UPDATE"]
    first, second = event["observations"]
    assert second["pc"] > first["pc"]
    assert second["time_to_tca_s"] == pytest.approx(first["time_to_tca_s"] - 86400.0)
    assert event["trend"] == "rising"


def test_assess_computes_with_the_method_given(tmp_path: Path) -> None:
    file = write(tmp_path, "green.kvn", green())
    result = keepout("assess", file, "--method", "elrod", "--json")
    assert result.exit_code == ExitCode.OK
    (event,) = json.loads(result.stdout)["events"]
    assert event["observations"][0]["method"] == "elrod"


def test_assess_names_a_file_it_cannot_read(tmp_path: Path) -> None:
    good = write(tmp_path, "green.kvn", green())
    bad = write(tmp_path, "bad.kvn", "CCSDS_CDM_VERS = 1.0\nNOT A LINE\n")
    result = keepout("assess", good, bad)
    assert result.exit_code == ExitCode.ERROR
    assert result.stdout == ""
    assert result.stderr.startswith(f"Error: {bad}: ")
    assert result.stderr.count("\n") == 1


def test_assess_reports_cdms_that_do_not_make_an_event(tmp_path: Path) -> None:
    files = [write(tmp_path, "a.kvn", green()), write(tmp_path, "b.kvn", green())]
    result = keepout("assess", *files)
    assert result.exit_code == ExitCode.ERROR
    assert (
        result.stderr
        == f"Error: MESSAGE_ID repeated: {operator_run().cdms[0].message_id}\n"
    )


def test_assess_needs_both_hard_body_radii_or_hbr(tmp_path: Path) -> None:
    file = write(tmp_path, "green.kvn", without_hbr(green()))
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
    write(tmp_path, "green.kvn", green())
    result = keepout("assess", *args)
    assert result.exit_code == ExitCode.USAGE
    assert expected in result.stderr


def test_the_method_option_offers_every_pc_method() -> None:
    # PcMethod is built from keepout.pc's names for Typer, which needs an enum.
    assert [m.value for m in PcMethod] == ["auto", *get_args(Method)]
    assert set(get_args(Method)) == set(METHODS)
    assert [method_choice(m.value) for m in PcMethod] == list(PcMethod)


def test_a_name_that_is_no_pc_method_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown Pc method 'best'"):
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
    failed = write(tmp_path, "failed.kvn", with_velocity_of_object1(green()))
    late = write(
        tmp_path,
        "late.kvn",
        re.sub(r"(CREATION_DATE *= )2026-10-06", r"\g<1>2026-10-09", yellow()),
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
2a85ed50c4e6: monitor (latest observation has no Pc: GeometryError: relative velocity is zero: the encounter plane is undefined)
3befe6f1305d: monitor (no observation before TCA: nothing to decide on)
""")
    payload = json.loads(keepout("assess", late, "--json").stdout)
    (event,) = payload["events"]
    assert event["pc"] is None
    assert event["level"] is None
    assert event["notes"][0] == "1 observation(s) issued after TCA left out"


def test_assess_counts_in_the_singular(tmp_path: Path) -> None:
    result = keepout("assess", write(tmp_path, "green.kvn", green()))
    assert "1 event from 1 CDM: 1 ignore." in result.stdout


def test_assess_labels_events_whose_covariances_are_assumed(tmp_path: Path) -> None:
    files = [
        write(tmp_path, "green.kvn", green()),
        write(tmp_path, "yellow.kvn", calculated(yellow())),
    ]
    result = keepout("assess", *files, "--hbr", "1000")
    assert result.exit_code == ExitCode.MANEUVER
    assert (
        "1 of 2 events rests on assumed covariances (COVARIANCE_METHOD = "
        "DEFAULT), not orbit determination."
    ) in result.stdout.splitlines()
    as_json = keepout("assess", *files, "--hbr", "1000", "--json")
    assert as_json.exit_code == ExitCode.MANEUVER
    payload = json.loads(as_json.stdout)
    assert payload["assumed_covariance_events"] == 1
    assert [e["assumed_covariance"] for e in payload["events"]] == [True, False]
    only = keepout("assess", files[1], "--hbr", "1000")
    assert "assumed covariances" not in only.stdout
    payload = json.loads(keepout("assess", files[1], "--json").stdout)
    assert payload["assumed_covariance_events"] == 0


def test_assess_with_store_shows_the_same_ids_as_store_add(tmp_path: Path) -> None:
    db = tmp_path / "keepout.db"
    files = [
        write(tmp_path, "green.kvn", green()),
        write(tmp_path, "yellow.kvn", yellow()),
    ]
    plain = json.loads(keepout("assess", *files, "--json").stdout)
    assert plain["store"] is None
    stored = keepout("assess", *files, "--store", str(db), "--json")
    assert stored.exit_code == ExitCode.OK
    payload = json.loads(stored.stdout)
    ids = [event["event_id"] for event in payload["events"]]
    assert all(re.fullmatch(r"[0-9a-f]{12}", event_id) for event_id in ids)
    assert payload["store"]["new_events"] == ids
    # The same Pc, risk and action as without the store.
    for key in ("pc", "level", "action", "cdms", "trend"):
        assert [e[key] for e in payload["events"]] == [e[key] for e in plain["events"]]
    added = json.loads(
        keepout("store", "add", *files, "--store", str(db), "--json").stdout
    )
    assert [event["event_id"] for event in added["events"]] == ids
    assert added["store"]["already_stored"] == payload["store"]["added"]
    text = keepout("assess", *files, "--store", str(db))
    assert f"Event ids, row by row: {ids[0]}, {ids[1]}.\n" in text.stdout
    assert f"{ids[1]}: monitor (latest Pc 7.38e-06" in text.stdout


def test_assess_with_store_computes_with_the_method_and_hbr_given(
    tmp_path: Path,
) -> None:
    db = tmp_path / "keepout.db"
    files = [write(tmp_path, "green.kvn", without_hbr(green()))]
    args = ("assess", *files, "--store", str(db), "--json")
    refused = keepout(*args)
    assert refused.exit_code == ExitCode.ERROR
    assert "no HBR for both objects" in refused.stderr
    result = keepout(*args, "--method", "elrod", "--hbr", "1000")
    assert result.exit_code == ExitCode.MANEUVER
    (event,) = json.loads(result.stdout)["events"]
    assert event["observations"][0]["method"] == "elrod"
    assert event["action"] == "maneuver"


def test_settings_set_the_pc_and_risk_and_options_win_over_them(
    tmp_path: Path,
) -> None:
    files = [write(tmp_path, "green.kvn", green())]
    settings = write(
        tmp_path,
        "settings.toml",
        "[pc]\nmethod = 'alfano'\nhbr_m = 1000.0\n[risk]\nred_from = 0.9\n",
    )

    def observation(*options: str) -> dict[str, Any]:
        result = keepout("assess", *files, "--settings", settings, *options, "--json")
        (event,) = json.loads(result.stdout)["events"]
        (seen,) = event["observations"]
        return {**seen, "level": event["level"]}

    from_file = observation()
    assert (from_file["method"], from_file["level"]) == ("alfano", "yellow")
    overridden = observation("--method", "foster", "--hbr", "1000")
    assert overridden["method"] == "foster"
    assert overridden["pc"] == pytest.approx(4.25e-2, rel=1e-2)
    assert overridden["level"] == "yellow"
    assert observation("--method", "foster", "--hbr", "1")["pc"] < 1e-4

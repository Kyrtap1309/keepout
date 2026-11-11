import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import typer
from builders import base_revision_store
from cli_builders import (
    FIXTURE_CSV,
    block_store_extra,
    keepout,
    operator_kvn,
    usage_error,
    without_hbr,
    write,
    write_fleet,
)

import keepout.cdm as keepout_cdm
from keepout.cli.output import ExitCode
from keepout.cli.store import STORE_HINT, open_cli_store, store_upgrade
from keepout.store import open_store
from keepout.store.db import HEAD

EVENT_ID = re.compile(r"[0-9a-f]{12}")


def cdm_files(folder: Path) -> list[str]:
    """The operator's two CDMs, green then yellow, as KVN files."""
    green, yellow = operator_kvn()
    return [write(folder, "green.kvn", green), write(folder, "yellow.kvn", yellow)]


def stored_ids(path: Path) -> list[str]:
    with open_store(path, create=False) as store:
        return [event.event_id for event in store.events()]


def test_store_add_prints_store_event_ids(tmp_path: Path) -> None:
    db = tmp_path / "keepout.db"
    files = cdm_files(tmp_path)
    result = keepout("store", "add", *files, "--store", str(db), "--json")
    assert result.exit_code == ExitCode.OK
    payload = json.loads(result.stdout)
    ids = [event["event_id"] for event in payload["events"]]
    assert all(EVENT_ID.fullmatch(event_id) for event_id in ids)
    assert ids == stored_ids(db)
    assert payload["cdm_count"] == 2
    assert payload["store"] == {
        "path": str(db),
        "added": [event["cdms"][0] for event in payload["events"]],
        "already_stored": [],
        "events": ids,
        "new_events": ids,
    }
    assert [event["level"] for event in payload["events"]] == ["green", "yellow"]
    text = keepout("store", "add", *files, "--store", str(db))
    assert text.exit_code == ExitCode.OK
    assert (
        "2 events in the store for the 2 CDMs read: 1 monitor, 1 ignore." in text.stdout
    )
    assert f"Event ids, row by row: {ids[0]}, {ids[1]}.\n" in text.stdout
    assert f"{ids[1]}: monitor (latest Pc 7.38e-06" in text.stdout


def test_store_add_reads_no_other_stored_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "keepout.db"
    green, yellow = cdm_files(tmp_path)
    assert keepout("store", "add", green, "--store", str(db)).exit_code == 0
    with open_store(db, create=False) as store:
        (unrelated,) = store.events()
    (unrelated_cdm,) = unrelated.cdms
    parsed: list[str] = []
    loads = keepout_cdm.loads

    def counted(text: str) -> keepout_cdm.Cdm:
        made = loads(text)
        parsed.append(made.message_id)
        return made

    monkeypatch.setattr(keepout_cdm, "loads", counted)
    result = keepout("store", "add", yellow, "--store", str(db), "--json")
    assert result.exit_code == ExitCode.OK
    assert len(json.loads(result.stdout)["events"]) == 1
    assert unrelated_cdm.message_id not in parsed


def test_store_add_twice_reports_already_stored(tmp_path: Path) -> None:
    db = tmp_path / "keepout.db"
    files = cdm_files(tmp_path)
    first = keepout("store", "add", *files, "--store", str(db))
    assert first.exit_code == ExitCode.OK
    assert f"2 CDMs added to {db} (0 already stored), 2 new events.\n" in first.stdout
    again = keepout("store", "add", *files, "--store", str(db), "--json")
    assert again.exit_code == ExitCode.OK
    payload = json.loads(again.stdout)
    assert payload["store"]["added"] == []
    assert len(payload["store"]["already_stored"]) == 2
    assert payload["store"]["new_events"] == []
    assert [event["event_id"] for event in payload["events"]] == stored_ids(db)
    text = keepout("store", "add", files[0], "--store", str(db))
    assert f"0 CDMs added to {db} (1 already stored), 0 new events.\n" in text.stdout


def test_store_add_exits_with_3_as_assess(tmp_path: Path) -> None:
    db = tmp_path / "keepout.db"
    settings = write(tmp_path, "settings.toml", "[pc]\nhbr_m = 1000.0\n")
    files = cdm_files(tmp_path)
    result = keepout("store", "add", *files, "--store", str(db), "--settings", settings)
    assert result.exit_code == ExitCode.MANEUVER
    assert (
        "2 events in the store for the 2 CDMs read: 1 maneuver, 1 prepare maneuver."
        in result.stdout
    )
    assert result.stdout.endswith(
        "At least one event needs a manoeuvre (exit status 3).\n"
    )


def test_store_add_names_a_bad_cdm_file(tmp_path: Path) -> None:
    bad = write(tmp_path, "bad.kvn", "CCSDS_CDM_VERS = 1.0\n")
    result = keepout("store", "add", bad, "--store", str(tmp_path / "keepout.db"))
    assert result.exit_code == ExitCode.ERROR
    assert result.stderr.startswith(f"Error: {bad}: ")


def test_store_add_reports_a_bad_settings_file(tmp_path: Path) -> None:
    settings = write(tmp_path, "settings.toml", "[pc]\npc_method = 'foster'\n")
    files = cdm_files(tmp_path)
    db = tmp_path / "keepout.db"
    result = keepout("store", "add", *files, "--store", str(db), "--settings", settings)
    assert result.exit_code == ExitCode.ERROR
    assert "pc.pc_method" in result.stderr


def test_store_commands_need_a_store(tmp_path: Path) -> None:
    files = cdm_files(tmp_path)
    result = keepout("store", "add", *files, env={"KEEPOUT_STORE": None})
    assert result.exit_code == ExitCode.USAGE
    assert "Missing option '--store'" in usage_error(result)


def test_store_option_reads_keepout_store_env(tmp_path: Path) -> None:
    db = tmp_path / "from-env.db"
    files = cdm_files(tmp_path)
    result = keepout("store", "add", *files, "--json", env={"KEEPOUT_STORE": str(db)})
    assert result.exit_code == ExitCode.OK
    ids = [event["event_id"] for event in json.loads(result.stdout)["events"]]
    assert ids == stored_ids(db)


@pytest.mark.parametrize("command", ["assess", "screen"])
def test_assess_and_screen_use_a_store_only_given_with_store(
    tmp_path: Path, command: str
) -> None:
    exported = tmp_path / "exported.db"
    (cdm, _) = cdm_files(tmp_path)
    fleet = ["--fleet", write_fleet(tmp_path), "--catalog", str(FIXTURE_CSV)]
    window = ["--start", "2026-10-06", "--days", "1", "--workers", "1"]
    args = [cdm] if command == "assess" else [*fleet, *window]
    result = keepout(command, *args, "--json", env={"KEEPOUT_STORE": str(exported)})
    assert result.exit_code == ExitCode.OK
    assert json.loads(result.stdout)["store"] is None
    assert not exported.exists()


@pytest.mark.parametrize("module", ["sqlalchemy", "alembic"])
@pytest.mark.parametrize(
    "command",
    [
        ("store", "upgrade"),
        ("store", "add", "{cdm}"),
        ("store", "purge", "--before", "2030-01-01", "--yes"),
        ("assess", "{cdm}"),
        (
            "screen",
            "--fleet",
            "{fleet}",
            "--catalog",
            "{catalog}",
            "--start",
            "2026-10-06",
            "--days",
            "1",
        ),
    ],
)
def test_store_without_the_extra_prints_the_install_hint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module: str,
    command: tuple[str, ...],
) -> None:
    (cdm, _) = cdm_files(tmp_path)
    block_store_extra(module, monkeypatch)
    fleet = write_fleet(tmp_path)
    args = [arg.format(cdm=cdm, fleet=fleet, catalog=FIXTURE_CSV) for arg in command]
    result = keepout(*args, "--store", str(tmp_path / "keepout.db"))
    assert result.exit_code == ExitCode.ERROR
    assert result.stderr == f"Error: {STORE_HINT}\n"
    assert result.stdout == ""
    assert not (tmp_path / "keepout.db").exists()


def test_the_hint_names_the_extra() -> None:
    assert STORE_HINT == "--store needs the store extra: pip install 'keepout[store]'"


@pytest.mark.parametrize("module", ["sqlalchemy", "alembic"])
def test_called_without_the_option_the_functions_give_the_hint_too(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    module: str,
) -> None:
    # Without --store's check: what another command calling them gets.
    block_store_extra(module, monkeypatch)
    db = tmp_path / "keepout.db"
    for call in (lambda: open_cli_store(db), lambda: store_upgrade(store=db)):
        with pytest.raises(typer.Exit) as exit_:
            call()
        assert exit_.value.exit_code == ExitCode.ERROR
        assert capsys.readouterr().err == f"Error: {STORE_HINT}\n"
    assert not db.exists()


def test_keepout_store_without_the_extra_blocks_no_command_without_a_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (cdm, _) = cdm_files(tmp_path)
    block_store_extra("sqlalchemy", monkeypatch)
    db = tmp_path / "keepout.db"
    result = keepout("assess", cdm, env={"KEEPOUT_STORE": str(db)})
    assert result.exit_code == ExitCode.OK
    assert result.stderr == ""
    store = keepout("store", "upgrade", env={"KEEPOUT_STORE": str(db)})
    assert store.exit_code == ExitCode.ERROR
    assert store.stderr == f"Error: {STORE_HINT}\n"
    assert not db.exists()


def test_a_missing_module_outside_the_extra_is_not_hidden(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "keepout.store", None)
    with pytest.raises(ModuleNotFoundError, match=r"keepout\.store"):
        keepout("store", "upgrade", "--store", str(tmp_path / "keepout.db"))


def test_store_on_an_old_database_says_upgrade(tmp_path: Path) -> None:
    db = tmp_path / "keepout.db"
    base_revision_store(db)
    files = cdm_files(tmp_path)
    result = keepout("store", "add", *files, "--store", str(db))
    assert result.exit_code == ExitCode.ERROR
    assert result.stderr == (
        f"Error: {db} was made by an older keepout: run keepout store upgrade\n"
    )
    assess = keepout("assess", *files, "--store", str(db))
    assert assess.exit_code == ExitCode.ERROR
    assert "run keepout store upgrade" in assess.stderr
    upgraded = keepout("store", "upgrade", "--store", str(db))
    assert upgraded.exit_code == ExitCode.OK
    assert upgraded.stdout == f"Upgraded {db}: base -> {HEAD}.\n"
    again = keepout("store", "upgrade", "--store", str(db), "--json")
    assert json.loads(again.stdout) == {"path": str(db), "from": HEAD, "to": HEAD}
    same = keepout("store", "upgrade", "--store", str(db))
    assert same.stdout == f"{db} is up to date (revision {HEAD}).\n"
    assert keepout("store", "add", *files, "--store", str(db)).exit_code == 0


def test_upgrade_of_no_store_is_an_error(tmp_path: Path) -> None:
    db = tmp_path / "keepout.db"
    result = keepout("store", "upgrade", "--store", str(db))
    assert result.exit_code == ExitCode.ERROR
    assert result.stderr == f"Error: no store at {db}\n"
    assert not db.exists()


def test_rebuild_computes_what_the_settings_miss(tmp_path: Path) -> None:
    db = tmp_path / "keepout.db"
    files = cdm_files(tmp_path)
    assert keepout("store", "add", *files, "--store", str(db)).exit_code == 0
    same = keepout("store", "rebuild", "--store", str(db))
    assert same.exit_code == ExitCode.OK
    assert same.stdout == f"Computed 0 Pc observations in {db}.\n"
    settings = write(tmp_path, "settings.toml", "[pc]\nmethod = 'foster'\n")
    foster = keepout(
        "store", "rebuild", "--store", str(db), "--settings", settings, "--json"
    )
    assert foster.exit_code == ExitCode.OK
    assert json.loads(foster.stdout) == {"path": str(db), "computed": 2, "skipped": []}


def test_rebuild_skips_the_cdms_without_hbr_and_says_so(tmp_path: Path) -> None:
    db = tmp_path / "keepout.db"
    green, yellow = operator_kvn()
    files = [
        write(tmp_path, "green.kvn", green),
        write(tmp_path, "bare.kvn", without_hbr(yellow)),
    ]
    radius = write(tmp_path, "radius.toml", "[pc]\nhbr_m = 20.0\n")
    added = keepout("store", "add", *files, "--store", str(db), "--settings", radius)
    assert added.exit_code == ExitCode.OK
    rebuilt = keepout("store", "rebuild", "--store", str(db))
    assert rebuilt.exit_code == ExitCode.OK
    assert rebuilt.stdout == (
        f"Computed 1 Pc observation in {db}.\n"
        "1 CDM without HBR skipped: set hbr_m in the settings file.\n"
    )
    payload = json.loads(
        keepout("store", "rebuild", "--store", str(db), "--json").stdout
    )
    bare = keepout_cdm.loads(without_hbr(yellow)).message_id
    assert payload == {"path": str(db), "computed": 0, "skipped": [bare]}


def test_rebuild_and_purge_do_not_create_a_store(tmp_path: Path) -> None:
    db = tmp_path / "keepout.db"
    for command in (("rebuild",), ("purge", "--before", "2030-01-01", "--yes")):
        result = keepout("store", *command, "--store", str(db))
        assert result.exit_code == ExitCode.ERROR
        assert result.stderr == f"Error: no store at {db}\n"
    assert not db.exists()


def test_purge_asks_and_no_keeps_everything(tmp_path: Path) -> None:
    db = tmp_path / "keepout.db"
    files = cdm_files(tmp_path)
    assert keepout("store", "add", *files, "--store", str(db)).exit_code == 0
    ids = stored_ids(db)
    purge = ("store", "purge", "--before", "2030-01-01", "--store", str(db))
    kept = keepout(*purge, stdin="n\n")
    assert kept.exit_code == ExitCode.OK
    assert kept.stderr.startswith(
        "Delete 2 events with TCA before 2030-01-01T00:00:00 UTC? [y/N]"
    )
    # The runner echoes the answer to standard output; a terminal does not.
    assert kept.stdout == " n\nNothing deleted.\n"
    assert stored_ids(db) == ids
    by_default = keepout(*purge, stdin="\n")
    assert by_default.stdout == " \nNothing deleted.\n"
    assert stored_ids(db) == ids
    yes = keepout(*purge, stdin="y\n")
    assert yes.exit_code == ExitCode.OK
    assert yes.stdout == (
        f" y\nDeleted 2 events with TCA before 2030-01-01T00:00:00 UTC from {db}.\n"
    )
    assert stored_ids(db) == []


def test_purge_keeps_the_later_events_and_yes_does_not_ask(tmp_path: Path) -> None:
    db = tmp_path / "keepout.db"
    files = cdm_files(tmp_path)
    assert keepout("store", "add", *files, "--store", str(db)).exit_code == 0
    later = stored_ids(db)[1]
    result = keepout(
        "store", "purge", "--before", "2026-10-07", "--yes", "--store", str(db)
    )
    assert result.exit_code == ExitCode.OK
    assert result.stderr == ""
    assert result.stdout.startswith("Deleted 1 event with TCA before 2026-10-07")
    assert stored_ids(db) == [later]
    none = keepout(
        "store", "purge", "--before", "2026-10-07", "--store", str(db), "--json"
    )
    assert none.stderr == ""
    assert json.loads(none.stdout) == {
        "path": str(db),
        "before": "2026-10-07T00:00:00.000",
        "deleted": 0,
    }


def test_purge_needs_a_time(tmp_path: Path) -> None:
    result = keepout(
        "store", "purge", "--before", "soon", "--store", str(tmp_path / "x.db")
    )
    assert result.exit_code == ExitCode.USAGE
    assert "not an ISO 8601 UTC time: 'soon'" in usage_error(result)


def test_help_does_not_import_the_store() -> None:
    code = (
        "import sys\n"
        "from keepout.cli.app import app\n"
        "for args in (['--help'], ['store', '--help'], ['store', 'add', '--help'],"
        " ['assess', '--help'], ['screen', '--help']):\n"
        "    try:\n"
        "        app(args, prog_name='keepout')\n"
        "    except SystemExit:\n"
        "        pass\n"
        "print(sorted(m for m in sys.modules if m.partition('.')[0] in"
        " ('sqlalchemy', 'alembic') or m.startswith('keepout.store')))\n"
    )
    loaded = subprocess.run(  # noqa: S603 (this interpreter, fixed code)
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert "keepout store" in loaded.stdout
    assert loaded.stdout.endswith("[]\n")


@pytest.mark.parametrize("command", ["add", "upgrade", "rebuild", "purge"])
def test_store_help_names_the_file_as_the_docs_do(command: str) -> None:
    result = keepout("store", command, "--help", env={"COLUMNS": "200"})
    assert "--store FILE" in " ".join(result.stdout.split())
    assess = keepout("assess", "--help", env={"COLUMNS": "200"})
    assert "--store FILE" in " ".join(assess.stdout.split())


def test_the_tests_never_use_a_store_from_the_environment() -> None:
    # conftest.py unsets it: a store exported in the shell is not written to.
    assert "KEEPOUT_STORE" not in os.environ

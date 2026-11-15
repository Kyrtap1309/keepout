"""``keepout dashboard``: Streamlit run on the dashboard's app, bound to
localhost, after the store is checked."""

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from alembic import command
from builders import loaded_after
from cli_builders import keepout, usage_error
from sqlalchemy import create_engine

from keepout.cli.dashboard import EXPOSED_WARNING
from keepout.cli.output import ExitCode
from keepout.store import db as store_db
from keepout.store import open_store

SETTINGS = Path(__file__).parents[2] / "examples" / "dashboard" / "settings.toml"


class FakeRun:
    """``subprocess.run`` recording its calls and returning ``status``."""

    def __init__(self, status: int = 0) -> None:
        self.status = status
        self.calls: list[tuple[list[str], dict[str, str]]] = []

    def __call__(
        self, args: Sequence[str], *, env: dict[str, str], check: bool
    ) -> subprocess.CompletedProcess[bytes]:
        assert not check
        self.calls.append((list(args), env))
        return subprocess.CompletedProcess(list(args), self.status)


@pytest.fixture
def fake_run(monkeypatch: pytest.MonkeyPatch) -> FakeRun:
    fake = FakeRun()
    monkeypatch.setattr(subprocess, "run", fake)
    return fake


@pytest.fixture
def store(tmp_path: Path) -> Path:
    path = tmp_path / "keepout.db"
    open_store(path).close()
    return path


def store_at(path: Path, revision: str) -> None:
    """Make ``path`` a store at ``revision`` of the schema, as an older
    keepout made it."""
    engine = create_engine(f"sqlite:///{path}")
    try:
        with engine.begin() as connection:
            command.upgrade(store_db._config(connection), revision)
    finally:
        engine.dispose()


def test_cli_dashboard_runs_streamlit_bound_to_localhost(
    fake_run: FakeRun, store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(store.parent)
    result = keepout(
        "dashboard",
        "--store",
        store.name,
        "--settings",
        str(SETTINGS),
        "--port",
        "8600",
        "--as-of",
        "2026-10-07T12:00",
    )
    assert result.exit_code == ExitCode.OK, result.stderr
    [(args, env)] = fake_run.calls
    assert args[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert Path(args[4]).parts[-3:] == ("keepout", "dashboard", "app.py")
    assert Path(args[4]).is_file()
    assert args[5:] == [
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "8600",
        "--server.maxUploadSize",
        "1",
        "--browser.gatherUsageStats",
        "false",
        "--server.headless",
        "true",
        "--client.showErrorDetails",
        "type",
    ]
    # Absolute paths: Streamlit may run from another directory.
    assert env["KEEPOUT_STORE"] == str(store.resolve())
    assert env["KEEPOUT_SETTINGS"] == str(SETTINGS.resolve())
    assert env["KEEPOUT_AS_OF"] == "2026-10-07T12:00:00.000"


def test_cli_dashboard_defaults(
    fake_run: FakeRun, store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Variables left in the shell do not reach the app: the options decide.
    monkeypatch.setenv("KEEPOUT_SETTINGS", "elsewhere.toml")
    monkeypatch.setenv("KEEPOUT_AS_OF", "2020-01-01")
    result = keepout("dashboard", "--store", str(store))
    assert result.exit_code == ExitCode.OK, result.stderr
    [(args, env)] = fake_run.calls
    assert args[args.index("--server.port") + 1] == "8501"
    assert env["KEEPOUT_STORE"] == str(store.resolve())
    assert "KEEPOUT_SETTINGS" not in env
    assert "KEEPOUT_AS_OF" not in env


def test_cli_dashboard_reads_the_store_from_the_environment(
    fake_run: FakeRun, store: Path
) -> None:
    result = keepout("dashboard", env={"KEEPOUT_STORE": str(store)})
    assert result.exit_code == ExitCode.OK, result.stderr
    [(_, env)] = fake_run.calls
    assert env["KEEPOUT_STORE"] == str(store.resolve())


def test_cli_dashboard_exits_with_streamlits_status(
    fake_run: FakeRun, store: Path
) -> None:
    fake_run.status = 2
    result = keepout("dashboard", "--store", str(store))
    assert result.exit_code == 2


def test_cli_dashboard_refuses_old_store(fake_run: FakeRun, tmp_path: Path) -> None:
    db = tmp_path / "keepout.db"
    store_at(db, "0002")
    before = db.read_bytes()
    result = keepout("dashboard", "--store", str(db))
    assert result.exit_code == ExitCode.ERROR
    assert result.stderr.startswith("Error: ")
    assert "keepout store upgrade" in result.stderr
    assert db.read_bytes() == before
    assert fake_run.calls == []


def test_cli_dashboard_refuses_a_missing_store(
    fake_run: FakeRun, tmp_path: Path
) -> None:
    result = keepout("dashboard", "--store", str(tmp_path / "none.db"))
    assert result.exit_code == ExitCode.ERROR
    assert result.stderr == (
        f"Error: no store at {tmp_path / 'none.db'}: make one by adding CDMs to it, "
        f"keepout store add CDM... --store {tmp_path / 'none.db'}\n"
    )
    assert not (tmp_path / "none.db").exists()
    assert fake_run.calls == []


@pytest.mark.parametrize(
    ("option", "value"), [("--as-of", "soon"), ("--port", "0"), ("--port", "70000")]
)
def test_cli_dashboard_refuses_bad_options(
    fake_run: FakeRun, store: Path, option: str, value: str
) -> None:
    result = keepout("dashboard", "--store", str(store), option, value)
    assert result.exit_code == ExitCode.USAGE
    assert option in usage_error(result)
    assert fake_run.calls == []


def test_default_host_is_loopback(fake_run: FakeRun, store: Path) -> None:
    result = keepout("dashboard", "--store", str(store))
    assert result.exit_code == ExitCode.OK, result.stderr
    [(args, _)] = fake_run.calls
    assert args[args.index("--server.address") + 1] == "127.0.0.1"
    assert result.stderr == ""


def test_host_any_address_warns(fake_run: FakeRun, store: Path) -> None:
    result = keepout("dashboard", "--store", str(store), "--host", "0.0.0.0")  # noqa: S104 (the option under test)
    assert result.exit_code == ExitCode.OK, result.stderr
    [(args, _)] = fake_run.calls
    assert args[args.index("--server.address") + 1] == "0.0.0.0"  # noqa: S104
    assert result.stderr == EXPOSED_WARNING.format(host="0.0.0.0") + "\n"  # noqa: S104


def test_host_ipv6_loopback_quiet(fake_run: FakeRun, store: Path) -> None:
    result = keepout("dashboard", "--store", str(store), "--host", "::1")
    assert result.exit_code == ExitCode.OK, result.stderr
    [(args, _)] = fake_run.calls
    assert args[args.index("--server.address") + 1] == "::1"
    assert result.stderr == ""


@pytest.mark.parametrize("value", ["example.com", "0.0.0.0:80", ""])
def test_host_not_an_address(fake_run: FakeRun, store: Path, value: str) -> None:
    result = keepout("dashboard", "--store", str(store), "--host", value)
    assert result.exit_code == ExitCode.USAGE
    assert "--host" in usage_error(result)
    assert fake_run.calls == []


def test_cli_dashboard_without_the_extra(
    fake_run: FakeRun, store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "streamlit", None)
    result = keepout("dashboard", "--store", str(store))
    assert result.exit_code == ExitCode.ERROR
    assert result.stderr == (
        "Error: keepout dashboard needs the dashboard extra: pip install "
        '"keepout[cli,dashboard] @ git+https://github.com/Kyrtap1309/keepout"\n'
    )
    assert fake_run.calls == []


def test_cli_dashboard_help() -> None:
    result = keepout("dashboard", "--help", env={"COLUMNS": "200"})
    assert result.exit_code == ExitCode.OK
    assert "Run the dashboard on a store (needs the dashboard extra)." in result.stdout
    assert "127.0.0.1" in result.stdout


def test_help_does_not_import_dashboard() -> None:
    code = (
        "from keepout.cli.app import app\n"
        "for args in (['--help'], ['dashboard', '--help']):\n"
        "    try:\n"
        "        app(args, prog_name='keepout')\n"
        "    except SystemExit:\n"
        "        pass\n"
    )
    shown = loaded_after(code, ["streamlit", "plotly", "keepout.dashboard"])
    assert "dashboard" in shown.printed
    assert shown.loaded == []

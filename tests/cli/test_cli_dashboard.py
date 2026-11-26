"""``keepout dashboard``: Streamlit run on the dashboard's app, bound to
localhost, after the store is checked."""

import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Self

import pytest
from builders import loaded_after, store_at
from cli_builders import keepout, usage_error

from keepout.cli.dashboard import EXPOSED_WARNING
from keepout.cli.output import ExitCode
from keepout.store import open_store

SETTINGS = Path(__file__).parents[2] / "examples" / "dashboard" / "settings.toml"


class FakeRun:
    """``subprocess.Popen`` recording its calls; each child exits at once
    with ``status``."""

    def __init__(self, status: int = 0) -> None:
        self.status = status
        self.calls: list[tuple[list[str], dict[str, str]]] = []

    def __call__(self, args: Sequence[str], *, env: dict[str, str]) -> Self:
        self.calls.append((list(args), env))
        return self

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def wait(self) -> int:
        return self.status

    def terminate(self) -> None:
        raise AssertionError("not terminated in these tests")


@pytest.fixture
def fake_run(monkeypatch: pytest.MonkeyPatch) -> FakeRun:
    fake = FakeRun()
    monkeypatch.setattr(subprocess, "Popen", fake)
    return fake


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    path = tmp_path / "keepout.db"
    open_store(path).close()
    return path


def test_cli_dashboard_runs_streamlit_bound_to_localhost(
    fake_run: FakeRun, store_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(store_path.parent)
    result = keepout(
        "dashboard",
        "--store",
        store_path.name,
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
        "--browser.serverAddress",
        "127.0.0.1",
        "--server.headless",
        "true",
        "--client.showErrorDetails",
        "type",
        "--client.toolbarMode",
        "minimal",
    ]
    # Absolute paths: Streamlit may run from another directory.
    assert env["KEEPOUT_STORE"] == str(store_path.resolve())
    assert env["KEEPOUT_SETTINGS"] == str(SETTINGS.resolve())
    assert env["KEEPOUT_AS_OF"] == "2026-10-07T12:00:00.000"


def test_cli_dashboard_defaults(
    fake_run: FakeRun, store_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # KEEPOUT_SETTINGS left in the shell does not reach the app: the
    # option decides. (--as-of reads KEEPOUT_AS_OF: tested below.)
    monkeypatch.setenv("KEEPOUT_SETTINGS", "elsewhere.toml")
    monkeypatch.delenv("KEEPOUT_AS_OF", raising=False)
    result = keepout("dashboard", "--store", str(store_path))
    assert result.exit_code == ExitCode.OK, result.stderr
    [(args, env)] = fake_run.calls
    assert args[args.index("--server.port") + 1] == "8501"
    assert env["KEEPOUT_STORE"] == str(store_path.resolve())
    assert "KEEPOUT_SETTINGS" not in env
    assert "KEEPOUT_AS_OF" not in env


def test_cli_dashboard_reads_the_store_from_the_environment(
    fake_run: FakeRun, store_path: Path
) -> None:
    result = keepout("dashboard", env={"KEEPOUT_STORE": str(store_path)})
    assert result.exit_code == ExitCode.OK, result.stderr
    [(_, env)] = fake_run.calls
    assert env["KEEPOUT_STORE"] == str(store_path.resolve())


def test_cli_dashboard_reads_as_of_from_the_environment(
    fake_run: FakeRun, store_path: Path
) -> None:
    env = {"KEEPOUT_STORE": str(store_path), "KEEPOUT_AS_OF": "2026-10-07T18:00:00Z"}
    result = keepout("dashboard", env=env)
    assert result.exit_code == ExitCode.OK, result.stderr
    [(_, child_env)] = fake_run.calls
    assert child_env["KEEPOUT_AS_OF"] == "2026-10-07T18:00:00.000"


def test_cli_dashboard_empty_as_of_is_the_clock(
    fake_run: FakeRun, store_path: Path
) -> None:
    result = keepout(
        "dashboard", env={"KEEPOUT_STORE": str(store_path), "KEEPOUT_AS_OF": ""}
    )
    assert result.exit_code == ExitCode.OK, result.stderr
    [(_, child_env)] = fake_run.calls
    assert "KEEPOUT_AS_OF" not in child_env


class RunInProcess:
    """``keepout.cli.dashboard._run`` in a Python process of its own, on a
    child that marks ``started`` and then sleeps: the tests signal that
    process alone, not the test runner, and not the child."""

    def __init__(self, folder: Path, *, on_sigterm: str, stop_timeout: float = 10.0):
        self.started = folder / "started"
        child = (
            "import pathlib, signal, sys, time\n"
            "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
            f"signal.signal(signal.SIGTERM, {on_sigterm})\n"
            f"pathlib.Path({str(self.started)!r}).touch()\n"
            "time.sleep(60)\n"
        )
        parent = (
            "import os, sys\n"
            "from keepout.cli import dashboard\n"
            f"dashboard._STOP_TIMEOUT = {stop_timeout!r}\n"
            f"status = dashboard._run([sys.executable, '-c', {child!r}], os.environ)\n"
            "print(status)\n"
        )
        self.process = subprocess.Popen(  # noqa: S603 (this interpreter, fixed code)
            [sys.executable, "-c", parent],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 30
        while not self.started.exists():
            assert time.monotonic() < deadline, "the child never started"
            assert self.process.poll() is None, self.process.communicate()
            time.sleep(0.01)

    def signal(self, signum: signal.Signals) -> None:
        self.process.send_signal(signum)

    def finish(self) -> tuple[str, str]:
        """Its standard output and error once it has exited (within 20 s)."""
        try:
            return self.process.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.communicate()
            raise AssertionError("_run did not return") from None


EXIT_3: Final = "lambda *_: sys.exit(3)"
"""A child's SIGTERM handler: exit with status 3."""

IGNORED: Final = "signal.SIG_IGN"
"""A child's SIGTERM handler: none, it keeps running."""


def test_run_passes_sigterm_on(tmp_path: Path) -> None:
    run = RunInProcess(tmp_path, on_sigterm=EXIT_3)
    run.signal(signal.SIGTERM)
    out, _ = run.finish()
    assert out == "3\n"


def test_ctrl_c_on_the_command_alone_stops_streamlit(tmp_path: Path) -> None:
    # SIGINT to this process only (not its group): the child must be told.
    run = RunInProcess(tmp_path, on_sigterm=EXIT_3)
    run.signal(signal.SIGINT)
    out, err = run.finish()
    assert (out, err) == ("0\n", "")


def test_a_child_that_does_not_stop_is_killed(tmp_path: Path) -> None:
    run = RunInProcess(tmp_path, on_sigterm=IGNORED, stop_timeout=0.5)
    run.signal(signal.SIGINT)
    out, err = run.finish()
    assert (out, err) == ("0\n", "")


def test_a_second_ctrl_c_kills_the_child_without_a_traceback(tmp_path: Path) -> None:
    run = RunInProcess(tmp_path, on_sigterm=IGNORED, stop_timeout=60.0)
    run.signal(signal.SIGINT)
    time.sleep(0.5)
    run.signal(signal.SIGINT)
    out, err = run.finish()
    assert (out, err) == ("0\n", "")


def test_cli_dashboard_exits_with_streamlits_status(
    fake_run: FakeRun, store_path: Path
) -> None:
    fake_run.status = 2
    result = keepout("dashboard", "--store", str(store_path))
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
    fake_run: FakeRun, store_path: Path, option: str, value: str
) -> None:
    result = keepout("dashboard", "--store", str(store_path), option, value)
    assert result.exit_code == ExitCode.USAGE
    assert option in usage_error(result)
    assert fake_run.calls == []


@pytest.mark.parametrize(
    ("host", "warns"),
    [
        (None, False),
        ("::1", False),
        ("0.0.0.0", True),  # noqa: S104 (the option under test)
        ("::", True),
        ("192.168.1.5", True),
    ],
)
def test_host_binds_and_warns_off_loopback(
    fake_run: FakeRun, store_path: Path, host: str | None, warns: bool
) -> None:
    option = [] if host is None else ["--host", host]
    result = keepout("dashboard", "--store", str(store_path), *option)
    assert result.exit_code == ExitCode.OK, result.stderr
    [(args, _)] = fake_run.calls
    bound = args[args.index("--server.address") + 1]
    assert bound == (host or "127.0.0.1")
    assert result.stderr == (EXPOSED_WARNING.format(host=host) + "\n" if warns else "")


@pytest.mark.parametrize(
    ("host", "server_address"),
    [
        ("127.0.0.1", "127.0.0.1"),
        ("::1", "[::1]"),
        ("0.0.0.0", "localhost"),  # noqa: S104 (the option under test)
        ("::", "localhost"),
    ],
)
def test_server_address_avoids_public_ip_lookup(
    fake_run: FakeRun, store_path: Path, host: str, server_address: str
) -> None:
    # Streamlit looks the public IP up over the network (to print an
    # "External URL") unless browser.serverAddress is given.
    result = keepout("dashboard", "--store", str(store_path), "--host", host)
    assert result.exit_code == ExitCode.OK, result.stderr
    [(args, _)] = fake_run.calls
    assert args[args.index("--browser.serverAddress") + 1] == server_address


@pytest.mark.parametrize("value", ["example.com", "0.0.0.0:80", ""])
def test_host_not_an_address(fake_run: FakeRun, store_path: Path, value: str) -> None:
    result = keepout("dashboard", "--store", str(store_path), "--host", value)
    assert result.exit_code == ExitCode.USAGE
    assert "--host" in usage_error(result)
    assert fake_run.calls == []


def test_cli_dashboard_without_the_extra(
    fake_run: FakeRun, store_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "streamlit", None)
    result = keepout("dashboard", "--store", str(store_path))
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


def test_a_bad_as_of_from_the_environment_names_the_variable(
    fake_run: FakeRun, store_path: Path
) -> None:
    env = {"KEEPOUT_STORE": str(store_path), "KEEPOUT_AS_OF": "soon"}
    result = keepout("dashboard", env=env)
    assert result.exit_code == ExitCode.USAGE
    error = usage_error(result)
    assert "KEEPOUT_AS_OF" in error
    assert "--as-of" not in error
    assert fake_run.calls == []


def test_no_exposure_warning_when_the_command_stops_before_streamlit(
    fake_run: FakeRun, tmp_path: Path
) -> None:
    missing = tmp_path / "none.db"
    result = keepout("dashboard", "--store", str(missing), "--host", "0.0.0.0")  # noqa: S104 (the option under test)
    assert result.exit_code == ExitCode.ERROR
    assert "Warning" not in result.stderr
    assert fake_run.calls == []


def test_a_scoped_ipv6_address_is_a_usage_error(
    fake_run: FakeRun, store_path: Path
) -> None:
    result = keepout("dashboard", "--store", str(store_path), "--host", "fe80::1%eth0")
    assert result.exit_code == ExitCode.USAGE
    assert "scoped" in usage_error(result)
    assert fake_run.calls == []

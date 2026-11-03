import json
import math
import re

import pytest
import typer
from astropy.time import Time
from cli_builders import keepout
from rich.table import Table
from typer.testing import CliRunner

import keepout as package
from keepout.cli.output import (
    ExitCode,
    JsonOut,
    duration_words,
    fail,
    minute,
    plural,
    reported,
    show,
    utc,
)
from keepout.errors import CdmParseError, SourceUnavailable


def test_exit_codes() -> None:
    assert {code.name: int(code) for code in ExitCode} == {
        "OK": 0,
        "ERROR": 1,
        "USAGE": 2,
        "MANOEUVRE": 3,
        "NO_SAFE_BURN": 4,
    }


def test_version() -> None:
    result = keepout("--version")
    assert result.exit_code == ExitCode.OK
    assert result.stdout == f"keepout {package.__version__}\n"


def commands(help_text: str) -> list[str]:
    """The command names in the Commands box of ``help_text``."""
    box = help_text.split("─ Commands ─", 1)[1]
    return re.findall(r"^│ (\w+) ", box, re.MULTILINE)


def test_help_lists_the_commands() -> None:
    result = keepout("--help")
    assert result.exit_code == ExitCode.OK
    assert commands(result.stdout) == [
        "screen",
        "assess",
        "maneuver",
        "validate",
        "fetch",
    ]
    fetch = keepout("fetch", "--help")
    assert fetch.exit_code == ExitCode.OK
    assert commands(fetch.stdout) == ["catalog", "cdm", "cara", "kelvins"]


@pytest.mark.parametrize(
    "args", [(), ("fetch",), ("nonsense",), ("fetch", "nonsense"), ("--nonsense",)]
)
def test_a_wrong_or_missing_command_is_a_usage_error(args: tuple[str, ...]) -> None:
    result = keepout(*args)
    assert result.exit_code == ExitCode.USAGE
    assert "Usage: keepout" in result.stdout + result.stderr


def run(command: typer.Typer) -> tuple[int, str, str]:
    result = CliRunner().invoke(command, [], catch_exceptions=False)
    return result.exit_code, result.stdout, result.stderr


def app_raising(error: Exception) -> typer.Typer:
    app = typer.Typer()

    def broken() -> None:
        print("partial output")
        raise error

    app.command()(reported(broken))
    return app


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (
            SourceUnavailable("cannot download x: HTTP 503"),
            "cannot download x: HTTP 503",
        ),
        (CdmParseError("bad number", line=4), "bad number (line 4)"),
        (FileNotFoundError(2, "No such file or directory", "fleet.toml"), "fleet.toml"),
    ],
)
def test_keepout_and_file_errors_end_with_status_1_and_no_traceback(
    error: Exception, message: str
) -> None:
    status, out, err = run(app_raising(error))
    assert status == ExitCode.ERROR
    assert out == "partial output\n"
    assert err.startswith("Error: ")
    assert message in err
    assert err.count("\n") == 1
    assert "Traceback" not in err


def test_a_bug_keeps_its_traceback() -> None:
    with pytest.raises(ZeroDivisionError):
        run(app_raising(ZeroDivisionError("bug")))


def test_reported_keeps_the_command_signature_and_result() -> None:
    def command(*, count: int = 1) -> int:
        """Doc."""
        return count + 1

    wrapped = reported(command)
    assert wrapped(count=2) == 3
    assert wrapped.__doc__ == "Doc."
    assert wrapped.__wrapped__ is command  # type: ignore[attr-defined]


def test_fail_prints_the_message_and_exits_with_the_status(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(typer.Exit) as exit_:
        fail("it broke", ExitCode.MANOEUVRE)
    assert exit_.value.exit_code == ExitCode.MANOEUVRE
    assert capsys.readouterr().err == "Error: it broke\n"


def test_show_prints_json_or_text(capsys: pytest.CaptureFixture[str]) -> None:
    payload: dict[str, JsonOut] = {
        "name": "[bold]x[/]",
        "values": [1, 2.5, None],
        "ok": True,
    }
    show(payload, "text [bold]x[/]", as_json=True)
    assert json.loads(capsys.readouterr().out) == payload
    table = Table("A", "B")
    table.add_row("1", "2")
    long = "/very/long/path/" * 10
    show(payload, "text [bold]x[/] :smile:", table, long, as_json=False)
    out = capsys.readouterr().out
    assert out.startswith("text [bold]x[/] :smile:\n")
    assert "│ 1 │ 2 │" in out
    assert out.endswith(long + "\n")


def test_show_refuses_values_json_cannot_hold() -> None:
    with pytest.raises(ValueError, match="JSON compliant"):
        show({"pc": math.nan}, as_json=True)


def test_plural() -> None:
    assert plural(1, "event") == "1 event"
    assert plural(0, "event") == "0 events"
    assert plural(2, "approach", "approaches") == "2 approaches"
    assert plural(1, "approach", "approaches") == "1 approach"


@pytest.mark.parametrize(
    ("seconds", "words"),
    [
        (86400.0, "a day"),
        (172800.0, "2 days"),
        (3600.0, "an hour"),
        (21600.0, "6 hours"),
        (60.0, "a minute"),
        (600.0, "10 minutes"),
        (90.0, "90 s"),
    ],
)
def test_duration_words(seconds: float, words: str) -> None:
    assert duration_words(seconds) == words


def test_times_in_utc() -> None:
    time = Time("2026-10-06T19:36:42.717", scale="tai")
    assert utc(time) == "2026-10-06T19:36:05"
    assert minute(time) == "2026-10-06 19:36"

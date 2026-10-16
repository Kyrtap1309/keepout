import runpy
import subprocess
import sys
from importlib.metadata import entry_points

import pytest

import keepout
from keepout import cli


def test_the_keepout_script_runs_main() -> None:
    (script,) = entry_points(group="console_scripts", name="keepout")
    assert script.value == "keepout.cli:main"
    assert script.load() is cli.main


def test_main_runs_the_command_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["keepout", "--version"])
    with pytest.raises(SystemExit) as exit_:
        cli.main()
    assert exit_.value.code == 0
    assert capsys.readouterr().out == f"keepout {keepout.__version__}\n"


@pytest.mark.parametrize("module", ["typer", "rich"])
def test_main_without_the_extra_says_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch, module: str
) -> None:
    for name in [m for m in sys.modules if m.startswith("keepout.cli.")]:
        monkeypatch.delitem(sys.modules, name)
    for name in [m for m in sys.modules if m.partition(".")[0] == module]:
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(SystemExit) as exit_:
        cli.main()
    assert exit_.value.code == cli.INSTALL_HINT
    assert "pip install 'keepout[cli]'" in cli.INSTALL_HINT


def test_main_does_not_hide_other_missing_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "keepout.cli.app", None)
    with pytest.raises(ModuleNotFoundError, match=r"keepout\.cli\.app"):
        cli.main()


def test_python_m_keepout_cli_runs_main(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["keepout", "--version"])
    with pytest.raises(SystemExit) as exit_:
        runpy.run_module("keepout.cli", run_name="__main__")
    assert exit_.value.code == 0


def test_the_library_does_not_import_the_extra() -> None:
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, keepout, keepout.cli, keepout.synthetic, keepout.validation;"
            "print(sorted({'typer', 'rich'} & set(sys.modules)))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout == "[]\n"

"""The README GIF's recorder: its ffmpeg command (the recording itself needs
a running dashboard and is made by hand)."""

from pathlib import Path

import dashboard_gif as gif
import pytest


def test_ffmpeg_args(tmp_path: Path) -> None:
    video, out = tmp_path / "in.webm", tmp_path / "out.gif"
    args = gif.ffmpeg_args(video, out)
    assert Path(args[0]).name.startswith("ffmpeg")
    assert args[args.index("-i") + 1] == str(video)
    assert args[-1] == str(out)
    assert f"fps={gif.FPS}" in args[args.index("-filter_complex") + 1]
    assert "-y" in args


def test_gifsicle_optimises_in_place(tmp_path: Path) -> None:
    gif_path = tmp_path / "out.gif"
    args = gif.gifsicle_args(gif_path)
    assert Path(args[0]).name == "gifsicle"
    assert args[1:] == ["--batch", "-O3", str(gif_path)]


def test_main_refuses_an_unknown_option() -> None:
    with pytest.raises(SystemExit) as exit_info:
        gif.main(["--nope"])
    assert exit_info.value.code == 2

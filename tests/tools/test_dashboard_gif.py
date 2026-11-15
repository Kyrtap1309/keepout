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
    assert args[args.index("-filter_complex") + 1] == (
        f"fps={gif.FPS},scale={gif.WIDTH}:-1:flags=lanczos,split[a][b];"
        "[a]palettegen[p];[b][p]paletteuse"
    )
    assert "-y" in args


def test_gif_path_is_in_the_docs_images() -> None:
    assert gif.GIF.parts[-3:] == ("img", "dashboard", "dashboard.gif")
    assert gif.VIEWPORT == (1280, 800)


def test_main_refuses_an_unknown_option() -> None:
    with pytest.raises(SystemExit) as exit_info:
        gif.main(["--nope"])
    assert exit_info.value.code == 2

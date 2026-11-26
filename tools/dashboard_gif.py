"""The README's dashboard GIF: the demo scenario (``tools/dashboard_scenario.py``)
recorded in a browser, converted with ffmpeg and optimised with gifsicle.

Usage: ``uv run python tools/dashboard_gif.py [--url URL]``, with the
dashboard of this tree running on a fresh volume under its own project name
(``docker compose -p keepout-docs down -v && docker compose -p keepout-docs
up --build --wait``, so the default project's volume is kept): the scenario
uploads the demo's update, which a second run would find already stored.
Writes :data:`GIF`. Run by hand; CI runs the scenario (``tests/e2e``) but
never compares the GIF, whose pixels differ between runs. ffmpeg comes from
the ``imageio-ffmpeg`` package of the development environment; gifsicle from
the system (``apt install gifsicle``), and so does a colour emoji font for
the risk column (``apt install fonts-noto-color-emoji``).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Final

import dashboard_seed as seed
import imageio_ffmpeg
from dashboard_scenario import VIEWPORT, run_scenario
from generated import REPO
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

if TYPE_CHECKING:
    from collections.abc import Sequence

TOOL: Final = "tools/dashboard_gif.py"
GIF: Final = REPO / "docs/img/dashboard/dashboard.gif"
"""The GIF the README shows."""

DEFAULT_URL: Final = "http://127.0.0.1:8501"
"""Where ``docker compose up`` serves the dashboard."""

FPS: Final = 8
"""The GIF's frames per second."""

WIDTH: Final = 960
"""The GIF's width [px]; the height keeps the viewport's ratio."""

PAUSE_S: Final = 1.5
"""How long [s] the recording holds each view."""


def ffmpeg_args(video: Path, gif: Path) -> list[str]:
    """The ffmpeg command converting ``video`` to ``gif``: :data:`FPS`
    frames per second, :data:`WIDTH` pixels wide, with a palette made from
    the video itself (two passes in one filter graph). The palette favours
    what moves (``stats_mode=diff``), and only the changed rectangle of a
    frame is redrawn, with an ordered dither: flat UI colours stay flat."""
    graph = (
        f"fps={FPS},scale={WIDTH}:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=stats_mode=diff[p];"
        "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
    )
    return [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-filter_complex",
        graph,
        str(gif),
    ]


def gifsicle_args(gif: Path) -> list[str]:
    """The gifsicle command optimising ``gif`` in place, losslessly.

    Raises
    ------
    RuntimeError
        If gifsicle is not installed.
    """
    gifsicle = shutil.which("gifsicle")
    if gifsicle is None:
        raise RuntimeError("gifsicle is not installed: sudo apt install gifsicle")
    return [gifsicle, "--batch", "-O3", str(gif)]


def record(url: str, directory: Path) -> Path:
    """Run the scenario on ``url`` in a recorded browser.

    Returns
    -------
    pathlib.Path
        The video (WebM) in ``directory``.

    Raises
    ------
    AssertionError
        If a step of the scenario fails.
    RuntimeError
        If the browser recorded no video.
    """
    upload = seed.demo_file()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            context = browser.new_context(
                viewport=VIEWPORT,
                record_video_dir=str(directory),
                record_video_size=VIEWPORT,
            )
            page = context.new_page()
            run_scenario(page, url.rstrip("/"), upload=upload, pause_s=PAUSE_S)
            context.close()  # Writes the video.
            video = page.video
            if video is None:  # Not with record_video_dir set.
                raise RuntimeError("the browser recorded no video")
            return Path(video.path())
        finally:
            browser.close()


def main(argv: Sequence[str] | None = None) -> int:
    """The entry point recording :data:`GIF`.

    Returns
    -------
    int
        0 on success; 1 if the browser, the scenario, ffmpeg or gifsicle
        fails.
    """
    parser = argparse.ArgumentParser(
        prog=TOOL,
        description=(
            "Record the dashboard GIF. Needs the dashboard on a fresh volume: "
            "docker compose -p keepout-docs down -v && "
            "docker compose -p keepout-docs up --build --wait."
        ),
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="the dashboard's URL")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as directory:
        try:
            optimise = gifsicle_args(GIF)  # Before the recording, if missing.
            video = record(args.url, Path(directory))
        except (AssertionError, RuntimeError, ValueError, PlaywrightError) as exc:
            print(f"{TOOL}: {exc}", file=sys.stderr)
            return 1
        GIF.parent.mkdir(parents=True, exist_ok=True)
        for command in (ffmpeg_args(video, GIF), optimise):
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                name = Path(command[0]).name
                print(
                    f"{TOOL}: {name} failed ({completed.returncode})", file=sys.stderr
                )
                return 1
    print(f"wrote {GIF} ({GIF.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

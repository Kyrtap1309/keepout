"""The README's dashboard GIF: the demo scenario (``tools/dashboard_scenario.py``)
recorded in a browser and converted with ffmpeg.

Usage: ``uv run python tools/dashboard_gif.py [--url URL]``, with the
dashboard running on a fresh volume (``docker compose down -v && docker
compose up --wait``): the scenario uploads the demo's update, which a
second run would find already stored. Writes :data:`GIF`. Run by hand; CI
runs the scenario (``tests/e2e``) but never compares the GIF, whose pixels
differ between runs. ffmpeg comes from the ``imageio-ffmpeg`` package of the
development environment.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Final

import dashboard_seed as seed
import imageio_ffmpeg
from dashboard_scenario import run_scenario
from generated import REPO
from playwright.sync_api import sync_playwright

if TYPE_CHECKING:
    from collections.abc import Sequence

TOOL: Final = "tools/dashboard_gif.py"
GIF: Final = REPO / "docs/img/dashboard/dashboard.gif"
"""The GIF the README shows."""

DEFAULT_URL: Final = "http://127.0.0.1:8501"
"""Where ``docker compose up`` serves the dashboard."""

VIEWPORT: Final = (1280, 800)
"""The browser's size [px], and the video's."""

FPS: Final = 6
"""The GIF's frames per second."""

WIDTH: Final = 880
"""The GIF's width [px]; the height keeps the viewport's ratio."""

PAUSE_S: Final = 1.5
"""How long [s] the recording holds each view."""


def ffmpeg_args(video: Path, gif: Path) -> list[str]:
    """The ffmpeg command converting ``video`` to ``gif``: :data:`FPS`
    frames per second, :data:`WIDTH` pixels wide, with a palette made from
    the video itself (two passes in one filter graph)."""
    graph = (
        f"fps={FPS},scale={WIDTH}:-1:flags=lanczos,split[a][b];"
        "[a]palettegen[p];[b][p]paletteuse"
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
    """
    width, height = VIEWPORT
    upload = min(seed.DEMO_DIR.glob("*.kvn"))
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            context = browser.new_context(
                viewport={"width": width, "height": height},
                record_video_dir=str(directory),
                record_video_size={"width": width, "height": height},
            )
            page = context.new_page()
            run_scenario(page, url.rstrip("/"), upload=upload, pause_s=PAUSE_S)
            context.close()  # Writes the video.
            video = page.video
            if video is None:  # Not with record_video_dir set.
                raise AssertionError("the browser recorded no video")
            return Path(video.path())
        finally:
            browser.close()


def main(argv: Sequence[str] | None = None) -> int:
    """The entry point recording :data:`GIF`.

    Returns
    -------
    int
        0 on success; 1 if the scenario or ffmpeg fails.
    """
    parser = argparse.ArgumentParser(
        prog=TOOL,
        description=(
            "Record the dashboard GIF. Needs the dashboard on a fresh volume: "
            "docker compose down -v && docker compose up --wait."
        ),
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="the dashboard's URL")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as directory:
        try:
            video = record(args.url, Path(directory))
        except AssertionError as exc:
            print(f"{TOOL}: {exc}", file=sys.stderr)
            return 1
        GIF.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(ffmpeg_args(video, GIF), check=False)
    if completed.returncode:
        print(f"{TOOL}: ffmpeg failed ({completed.returncode})", file=sys.stderr)
        return 1
    print(f"wrote {GIF} ({GIF.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""``keepout fetch ...``: download data into keepout's cache."""

from __future__ import annotations

from keepout.cli.options import CacheDir, JsonFlag
from keepout.cli.output import JsonValue, show
from keepout.sources import cara, kelvins

__all__ = ["fetch_cara", "fetch_kelvins"]


def fetch_cara(*, cache_dir: CacheDir = None, as_json: JsonFlag = False) -> None:
    """Download NASA CARA's validation files (pinned commit, verified by SHA-256).

    Files already in the cache with the right digest are not downloaded
    again. keepout validate uses them.
    """
    cara.prefetch(cara.MANIFEST, cache=cache_dir)
    folder = cara.commit_dir(cache_dir)
    payload: dict[str, JsonValue] = {
        "source": "cara",
        "commit": cara.CARA_COMMIT,
        "files": len(cara.MANIFEST),
        "directory": str(folder),
    }
    show(
        payload,
        f"NASA CARA Analysis Tools, commit {cara.CARA_COMMIT[:12]}: "
        f"{len(cara.MANIFEST)} files in {folder}",
        as_json=as_json,
    )


def fetch_kelvins(*, cache_dir: CacheDir = None, as_json: JsonFlag = False) -> None:
    """Download ESA's Kelvins collision avoidance data set (221 MB, CC BY 4.0).

    Keeps only the raw table, verified by SHA-256; nothing is downloaded if
    it is already in the cache.
    """
    path = kelvins.fetch_raw(cache=cache_dir)
    payload: dict[str, JsonValue] = {
        "source": "kelvins",
        "doi": kelvins.DOI,
        "path": str(path),
    }
    show(
        payload,
        f"ESA Kelvins collision avoidance data set (doi:{kelvins.DOI}): {path}",
        as_json=as_json,
    )

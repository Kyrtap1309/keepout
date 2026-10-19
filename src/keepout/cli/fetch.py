"""``keepout fetch ...``: download data into keepout's cache."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, TypeAlias

import typer

from keepout.cli.options import CacheDir, JsonFlag
from keepout.cli.output import (
    JsonValue,
    isot,
    minute,
    pc_text,
    plural,
    show,
    table,
    utc,
)
from keepout.sources.catalog import CATALOG_SOURCES, DEFAULT_SOURCE, CatalogSourceName
from keepout.units import HOUR

# The sources are imported in the commands: keepout --help need not load
# the numerical libraries.
if TYPE_CHECKING:
    from keepout.sources.catalog import Catalog, CatalogPart, Retrieval
    from keepout.sources.spacetrack import PublicCdm

__all__ = [
    "fetch_cara",
    "fetch_catalog",
    "fetch_cdm",
    "fetch_kelvins",
    "load_catalog",
]

SourceOption: TypeAlias = Annotated[
    CatalogSourceName,
    typer.Option(
        "--source",
        help=" ".join(
            f"{source.name}: {source.summary}." for source in CATALOG_SOURCES.values()
        ),
    ),
]
"""``--source``: where ``fetch catalog`` gets the catalog."""


def load_catalog(source: CatalogSourceName, cache: Path | None) -> Catalog:
    """The catalog of ``source``: its cached copy while that is current, else a
    download (:meth:`keepout.sources.catalog.CatalogSource.fetch`).

    Raises
    ------
    KeepoutError
        As :func:`keepout.sources.celestrak.fetch_catalog` or
        :func:`keepout.sources.spacetrack.fetch_catalog`.
    """
    return CATALOG_SOURCES[source].fetch(cache)


def _hours(seconds: float) -> str:
    """A whole number of hours in words: ``an hour``, ``2 hours``."""
    return "an hour" if seconds == HOUR else f"{seconds / HOUR:g} hours"


def fetch_cara(*, cache_dir: CacheDir = None, as_json: JsonFlag = False) -> None:
    """Download NASA CARA's validation files (pinned commit, verified by SHA-256).

    Files already in the cache with the right digest are not downloaded
    again. keepout validate uses them.
    """
    from keepout.sources import cara

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
    from keepout.sources import kelvins

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


def fetch_catalog(
    *,
    source: SourceOption = DEFAULT_SOURCE,
    cache_dir: CacheDir = None,
    as_json: JsonFlag = False,
) -> None:
    """Download the GP element sets of every object in orbit, for keepout screen.

    From CelesTrak (default): its public groups, about 20 000 objects, no
    account; a copy is reused for {celestrak} (CelesTrak's update interval).
    From Space-Track: the full catalog, about 32 000 objects, with the
    account in SPACETRACK_IDENTITY and SPACETRACK_PASSWORD; a copy is
    reused for {spacetrack}, and Space-Track is asked at most once {spacetrack}.
    The data is basic SSA data of USSPACECOM (18th Space Defense
    Squadron); cite it as such.
    """
    catalog = load_catalog(source, cache_dir)
    payload: dict[str, JsonValue] = {
        "source": str(source),
        "objects": len(catalog.omms),
        "duplicates": len(catalog.duplicates),
        "parts": [_part_json(part) for part in catalog.parts],
    }
    described = CATALOG_SOURCES[source]
    parts = table(described.title)
    parts.add_column(described.part_title)
    parts.add_column("Objects", justify="right")
    parts.add_column("Retrieved (UTC)")
    for part in catalog.parts:
        parts.add_row(part.name, str(len(part.omms)), utc(part.retrieval.retrieved))
    show(
        payload,
        parts,
        f"{len(catalog.omms)} objects, one element set each "
        f"({len(catalog.duplicates)} older sets of the same objects left out); "
        f"cached in {catalog.parts[0].retrieval.path.parent}",
        as_json=as_json,
    )


# The reuse times in the help are the sources' own (none under python -OO).
fetch_catalog.__doc__ = (fetch_catalog.__doc__ or "").format_map(
    {name: _hours(source.reuse) for name, source in CATALOG_SOURCES.items()}
)


def fetch_cdm(*, cache_dir: CacheDir = None, as_json: JsonFlag = False) -> None:
    """Download Space-Track's public CDM summaries of approaches to come.

    Needs the account in SPACETRACK_IDENTITY and SPACETRACK_PASSWORD; a copy
    is reused for 8 hours, and Space-Track is asked at most once in 8 hours.
    The summaries (cdm_public) give the objects, TCA, minimum range and Pc,
    but no states or covariances, so keepout assess cannot read them. They
    may not be redistributed without Space-Track's approval.
    """
    from keepout.sources import spacetrack

    public = spacetrack.fetch_public_cdms(cache=cache_dir)
    rows = sorted(public.cdms, key=lambda cdm: (cdm.tca.jd1, cdm.tca.jd2, cdm.cdm_id))
    payload: dict[str, JsonValue] = {
        "source": "spacetrack",
        "retrieval": _retrieval_json(public.retrieval),
        "cdms": [_public_cdm_json(cdm) for cdm in rows],
    }
    summaries = table("Space-Track public CDMs (cdm_public)")
    summaries.add_column("TCA (UTC)", no_wrap=True)
    summaries.add_column("Objects", no_wrap=True)
    for column in ("Min range (m)", "Pc"):
        summaries.add_column(column, justify="right", no_wrap=True)
    summaries.add_column("Emergency")
    for cdm in rows:
        summaries.add_row(
            minute(cdm.tca),
            f"{cdm.object1.catalog_number} / {cdm.object2.catalog_number}",
            f"{cdm.min_range:.0f}",
            pc_text(cdm.probability),
            "yes" if cdm.emergency_reportable else "no",
        )
    show(
        payload,
        summaries,
        f"{plural(len(rows), 'CDM summary', 'CDM summaries')}, retrieved "
        f"{utc(public.retrieval.retrieved)} UTC; cached in {public.retrieval.path}",
        as_json=as_json,
    )


def _retrieval_json(retrieval: Retrieval) -> dict[str, JsonValue]:
    return {
        "url": retrieval.url,
        "path": str(retrieval.path),
        "retrieved": isot(retrieval.retrieved),
        "checked": isot(retrieval.checked),
    }


def _part_json(part: CatalogPart) -> dict[str, JsonValue]:
    return {
        "name": part.name,
        "objects": len(part.omms),
        **_retrieval_json(part.retrieval),
    }


def _public_cdm_json(cdm: PublicCdm) -> dict[str, JsonValue]:
    return {
        "cdm_id": cdm.cdm_id,
        "created": isot(cdm.created),
        "tca": isot(cdm.tca),
        "emergency_reportable": cdm.emergency_reportable,
        "min_range": cdm.min_range,
        "pc": cdm.probability,
        "object1": dataclasses.asdict(cdm.object1),
        "object2": dataclasses.asdict(cdm.object2),
    }

"""``keepout fetch ...``: download data into keepout's cache."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Final, TypeAlias

import typer

from keepout.cli.options import CacheDir, CatalogSource, JsonFlag
from keepout.cli.output import JsonValue, isot, minute, pc_text, show, table, utc
from keepout.sources import cara, celestrak, kelvins, spacetrack
from keepout.sources.celestrak import Catalog, GpGroup, Retrieval
from keepout.sources.spacetrack import PublicCdm, PublicCdmObject

__all__ = [
    "SOURCE_NAMES",
    "fetch_cara",
    "fetch_catalog",
    "fetch_cdm",
    "fetch_kelvins",
    "load_catalog",
]

SOURCE_NAMES: Final[Mapping[CatalogSource, str]] = MappingProxyType(
    {
        CatalogSource.CELESTRAK: "CelesTrak public GP groups",
        CatalogSource.SPACETRACK: "Space-Track full GP catalog",
    }
)
"""Each catalog source as the commands name it."""

SourceOption: TypeAlias = Annotated[
    CatalogSource,
    typer.Option(
        "--source",
        help=(
            "celestrak: public groups, no account. spacetrack: the full "
            "catalog, needs SPACETRACK_IDENTITY and SPACETRACK_PASSWORD."
        ),
    ),
]
"""``--source``: where ``fetch catalog`` gets the catalog."""


def load_catalog(source: CatalogSource, cache: Path | None) -> Catalog:
    """The catalog of ``source``: its cached copy while that is current (2 h
    for CelesTrak, 1 h for Space-Track), else a download.

    Raises
    ------
    KeepoutError
        As :func:`keepout.sources.celestrak.fetch_catalog` or
        :func:`keepout.sources.spacetrack.fetch_catalog`.
    """
    if source is CatalogSource.SPACETRACK:
        return spacetrack.fetch_catalog(cache=cache)
    return celestrak.fetch_catalog(cache=cache)


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


def fetch_catalog(
    *,
    source: SourceOption = CatalogSource.CELESTRAK,
    cache_dir: CacheDir = None,
    as_json: JsonFlag = False,
) -> None:
    """Download the GP element sets of every object in orbit, for keepout screen.

    From CelesTrak (default): its public groups, about 20 000 objects, no
    account; a copy is reused for 2 hours (CelesTrak's update interval).
    From Space-Track: the full catalog, about 32 000 objects, with the
    account in SPACETRACK_IDENTITY and SPACETRACK_PASSWORD; a copy is
    reused for an hour, and Space-Track is asked at most once an hour.
    The data is basic SSA data of USSPACECOM (18th Space Defense
    Squadron); cite it as such.
    """
    catalog = load_catalog(source, cache_dir)
    payload: dict[str, JsonValue] = {
        "source": str(source),
        "objects": len(catalog.omms),
        "duplicates": len(catalog.duplicates),
        "groups": [_group_json(group) for group in catalog.groups],
    }
    groups = table(SOURCE_NAMES[source])
    groups.add_column("Group")
    groups.add_column("Objects", justify="right")
    groups.add_column("Retrieved (UTC)")
    for group in catalog.groups:
        groups.add_row(group.name, str(len(group.omms)), utc(group.retrieval.retrieved))
    show(
        payload,
        groups,
        f"{len(catalog.omms)} objects, one element set each "
        f"({len(catalog.duplicates)} older sets of objects in several groups "
        f"left out); cached in {catalog.groups[0].retrieval.path.parent}",
        as_json=as_json,
    )


def fetch_cdm(*, cache_dir: CacheDir = None, as_json: JsonFlag = False) -> None:
    """Download Space-Track's public CDM summaries of approaches to come.

    Needs the account in SPACETRACK_IDENTITY and SPACETRACK_PASSWORD; a copy
    is reused for 8 hours, and Space-Track is asked at most once in 8 hours.
    The summaries (cdm_public) give the objects, TCA, minimum range and Pc,
    but no states or covariances, so keepout assess cannot read them. They
    may not be redistributed without Space-Track's approval.
    """
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
        f"{len(rows)} CDM summaries, retrieved {utc(public.retrieval.retrieved)} "
        f"UTC; cached in {public.retrieval.path}",
        as_json=as_json,
    )


def _retrieval_json(retrieval: Retrieval) -> dict[str, JsonValue]:
    return {
        "url": retrieval.url,
        "path": str(retrieval.path),
        "retrieved": isot(retrieval.retrieved),
        "checked": isot(retrieval.checked),
    }


def _group_json(group: GpGroup) -> dict[str, JsonValue]:
    return {
        "name": group.name,
        "objects": len(group.omms),
        **_retrieval_json(group.retrieval),
    }


def _public_object_json(cdm_object: PublicCdmObject) -> dict[str, JsonValue]:
    return dataclasses.asdict(cdm_object)


def _public_cdm_json(cdm: PublicCdm) -> dict[str, JsonValue]:
    return {
        "cdm_id": cdm.cdm_id,
        "created": isot(cdm.created),
        "tca": isot(cdm.tca),
        "emergency_reportable": cdm.emergency_reportable,
        "min_range": cdm.min_range,
        "pc": cdm.probability,
        "object1": _public_object_json(cdm.object1),
        "object2": _public_object_json(cdm.object2),
    }

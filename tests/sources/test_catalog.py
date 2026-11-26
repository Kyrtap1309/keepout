import dataclasses
import importlib
from pathlib import Path
from typing import get_args

import astropy.units as u
import pytest
from astropy.time import Time
from celestrak_builders import SAMPLE

from keepout.omm import Omm
from keepout.sources.cache import CopyMeta
from keepout.sources.catalog import (
    CATALOG_SOURCES,
    DEFAULT_SOURCE,
    Catalog,
    CatalogPart,
    CatalogSourceName,
    Retrieval,
)
from keepout.units import HOUR

NOW = 1_791_000_000.0
URL = "https://example.org/gp"


def part(name: str, omms: tuple[Omm, ...], tmp_path: Path) -> CatalogPart:
    now = Time(NOW, format="unix", scale="utc")
    retrieval = Retrieval(URL, tmp_path / name, now, now)
    return CatalogPart(name, omms, retrieval)


def test_a_retrieval_takes_its_times_from_the_copys_metadata(tmp_path: Path) -> None:
    meta = CopyMeta(url=URL, sha256="0" * 64, retrieved=NOW, checked=NOW + 60.0)
    retrieval = Retrieval.of(tmp_path / "copy", meta)
    assert (retrieval.url, retrieval.path) == (URL, tmp_path / "copy")
    assert retrieval.retrieved.unix == NOW
    assert retrieval.checked.unix == NOW + 60.0
    assert retrieval.retrieved.scale == "utc"


def test_a_catalog_keeps_each_objects_newest_set_by_catalog_number(
    tmp_path: Path,
) -> None:
    newest, other = SAMPLE[0], SAMPLE[1]
    older = dataclasses.replace(newest, epoch=newest.epoch - 1.0 * u.day)
    first = part("a", (other, older), tmp_path)
    second = part("b", (newest,), tmp_path)
    catalog = Catalog.of([first, second])
    assert catalog.omms == tuple(
        sorted((newest, other), key=lambda omm: omm.catalog_number)
    )
    assert catalog.duplicates == (older,)
    assert catalog.parts == (first, second)


def test_the_catalog_sources_are_named_by_their_keys() -> None:
    assert tuple(CATALOG_SOURCES) == get_args(CatalogSourceName)
    assert all(name == source.name for name, source in CATALOG_SOURCES.items())
    assert DEFAULT_SOURCE in CATALOG_SOURCES


def test_a_catalog_is_reused_as_long_as_its_source_allows() -> None:
    # CelesTrak updates every 2 h; Space-Track allows a gp query an hour.
    assert CATALOG_SOURCES["celestrak"].reuse == 2 * HOUR
    assert CATALOG_SOURCES["spacetrack"].reuse == HOUR


@pytest.mark.parametrize("name", get_args(CatalogSourceName))
def test_a_catalog_source_fetches_with_its_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: CatalogSourceName
) -> None:
    source = CATALOG_SOURCES[name]
    fetched = Catalog((), (), ())
    asked: list[object] = []

    def fetch_catalog(*, cache: Path | None) -> Catalog:
        asked.append(cache)
        return fetched

    module = importlib.import_module(f"keepout.sources.{name}")
    monkeypatch.setattr(module, "fetch_catalog", fetch_catalog)
    assert source.fetch(tmp_path) is fetched
    assert asked == [tmp_path]

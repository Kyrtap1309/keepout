import dataclasses
from pathlib import Path

import astropy.units as u
from astropy.time import Time
from celestrak_builders import SAMPLE

from keepout.omm import Omm
from keepout.sources.cache import CopyMeta, read_copy, save_copy
from keepout.sources.catalog import Catalog, CatalogPart, Retrieval

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


def test_a_saved_copy_reads_back_with_its_metadata(tmp_path: Path) -> None:
    target = tmp_path / "gp" / "catalog.json"
    meta = save_copy(target, URL, b"[]", NOW)
    assert (meta.url, meta.retrieved, meta.checked) == (URL, NOW, NOW)
    cached = read_copy(target, URL)
    assert cached is not None
    assert cached.data == b"[]"
    assert cached.meta == meta

"""The published Kelvins table (downloads 221 MB on first run)."""

import csv
import gzip
from pathlib import Path

import pytest
from builders import KELVINS_EXCERPT as EXCERPT

from keepout.sources import kelvins

pytestmark = pytest.mark.validation


@pytest.fixture(scope="module")
def table() -> Path:
    return kelvins.fetch_raw()


def test_the_table_has_every_event(table: Path) -> None:
    events = kelvins.load(table)
    assert len(events) == 15_321
    assert sum(len(event) for event in events) == 199_082


def test_the_excerpt_is_unchanged_rows_of_the_table(table: Path) -> None:
    with EXCERPT.open(encoding="utf-8", newline="") as stream:
        excerpt = list(csv.reader(stream))
    wanted = {row[0] for row in excerpt[1:]}
    with gzip.open(table, "rt", encoding="utf-8", newline="") as stream:
        rows = csv.reader(stream)
        published = [next(rows)] + [row for row in rows if row[0] in wanted]
    assert excerpt == published

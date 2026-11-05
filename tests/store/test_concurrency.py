"""Two processes ingesting into one store at once."""

import multiprocessing
from pathlib import Path

from store_builders import cdm_text

import keepout.cdm
from keepout.store import CdmInput, open_store

COUNT = 200


def _ingest(path: str, texts: list[str]) -> int:
    """Ingest ``texts`` one CDM per transaction; the number added."""
    added = 0
    with open_store(path) as store:
        for text in texts:
            given = CdmInput.of(keepout.cdm.loads(text), origin="test")
            added += len(store.ingest([given]).added)
    return added


def test_two_processes_ingest_at_once(tmp_path: Path) -> None:
    path = tmp_path / "keepout.db"
    with open_store(path):
        pass
    # Disjoint sets: other secondaries, each CDM a few seconds after the last.
    sets = [
        [
            cdm_text(f"{secondary}-{number}", tca=3.0 * number, secondary=secondary)
            for number in range(COUNT)
        ]
        for secondary in ("40000", "50000")
    ]
    context = multiprocessing.get_context("spawn")
    with context.Pool(2) as pool:
        added = pool.starmap(_ingest, [(str(path), texts) for texts in sets])
    assert added == [COUNT, COUNT]
    with open_store(path) as store:
        events = store.events()
    assert sum(len(event.cdms) for event in events) == 2 * COUNT

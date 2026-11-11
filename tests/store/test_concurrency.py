"""Two processes ingesting into one store at once."""

import multiprocessing
import threading
from pathlib import Path

import pytest
from store_builders import cdm_text

import keepout.cdm
from keepout.store import CdmInput, open_store

COUNT = 200


def _ingest(path: str, texts: list[str], start: threading.Barrier) -> int:
    """Ingest ``texts`` one CDM per transaction, once both processes are
    ready; the number added."""
    added = 0
    with open_store(path) as store:
        start.wait()
        for text in texts:
            given = CdmInput.of(keepout.cdm.loads(text), origin="test")
            added += len(store.ingest([given]).added)
    return added


@pytest.mark.slow
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
    with context.Manager() as manager, context.Pool(2) as pool:
        start = manager.Barrier(2)
        added = pool.starmap(_ingest, [(str(path), texts, start) for texts in sets])
    assert added == [COUNT, COUNT]
    with open_store(path) as store:
        events = store.events()
    assert sum(len(event.cdms) for event in events) == 2 * COUNT

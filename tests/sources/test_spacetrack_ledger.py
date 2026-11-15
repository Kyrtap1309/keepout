"""The Space-Track request ledger's lock: one process reserves at a time."""

import threading
from pathlib import Path

from keepout.sources.spacetrack._ledger import LOCK_SUFFIX, RequestLedger


def test_the_ledger_lock_makes_a_second_holder_wait(tmp_path: Path) -> None:
    # flock locks an open file, not a process: a second open in another
    # thread waits as another process would.
    ledger = RequestLedger(tmp_path / "spacetrack" / "requests.json")
    entered = threading.Event()

    def second() -> None:
        with RequestLedger(ledger.path).locked():
            entered.set()

    with ledger.locked():
        thread = threading.Thread(target=second)
        thread.start()
        assert not entered.wait(0.3)
    thread.join(5.0)
    assert entered.is_set()
    assert (tmp_path / "spacetrack" / f"requests.json{LOCK_SUFFIX}").exists()


def test_the_ledger_lock_keeps_the_record(tmp_path: Path) -> None:
    ledger = RequestLedger(tmp_path / "requests.json")
    with ledger.locked():
        ledger.write({"gp": 1.5})
    with ledger.locked():
        assert ledger.read() == {"gp": 1.5}

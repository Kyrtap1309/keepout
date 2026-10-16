import os

import pytest

from keepout._workers import ONE_THREAD, process_pool, resolve_workers


def test_workers_given_are_kept() -> None:
    assert resolve_workers(1) == 1
    assert resolve_workers(32) == 32


@pytest.mark.parametrize(("cpus", "expected"), [(None, 1), (2, 2), (4, 4), (20, 4)])
def test_the_default_is_one_process_per_cpu_up_to_the_cap(
    monkeypatch: pytest.MonkeyPatch, cpus: int | None, expected: int
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: cpus)
    assert resolve_workers(None) == expected


@pytest.mark.parametrize("workers", [0, -1])
def test_fewer_than_one_worker_is_refused(workers: int) -> None:
    with pytest.raises(ValueError, match=f"workers must be at least 1, got {workers}"):
        resolve_workers(workers)


def _environment(names: tuple[str, ...]) -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in names}


def test_the_pool_runs_numerical_libraries_on_one_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = tuple(ONE_THREAD)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    before = _environment(names)
    with process_pool(1) as pool:
        seen = pool.submit(_environment, names).result()
    assert seen == dict(ONE_THREAD)
    assert _environment(names) == before

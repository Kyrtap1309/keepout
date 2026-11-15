"""Process pools of keepout's parallel computations (screening, validation).

The pools ``spawn`` their processes (the same on every platform) and run the
numerical libraries in each on one thread: the processes already use the
cores, so BLAS and OpenMP threads would only compete for them.
"""

from __future__ import annotations

import contextlib
import os
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from keepout._validation import require_at_least

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from concurrent.futures import ProcessPoolExecutor

__all__ = [
    "DEFAULT_WORKERS",
    "ONE_THREAD",
    "environment",
    "process_pool",
    "resolve_workers",
]

DEFAULT_WORKERS: Final = 4
"""Most processes used when their number is not given. Screening a fleet
against the full catalog takes as long on 4 processes as on 20, at less
than half the CPU time: each process is sent its share of the catalog and
imports keepout again."""

ONE_THREAD: Final[Mapping[str, str]] = MappingProxyType(
    {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
)
"""Environment of the pool's processes: one thread per numerical library."""


def resolve_workers(workers: int | None) -> int:
    """The number of processes to use: ``workers``, or if ``None`` one per
    CPU up to :data:`DEFAULT_WORKERS`.

    Raises
    ------
    ValueError
        If ``workers`` is below 1.
    """
    if workers is None:
        return min(DEFAULT_WORKERS, os.cpu_count() or 1)
    require_at_least(workers, 1, "workers")
    return workers


@contextlib.contextmanager
def environment(values: Mapping[str, str], *, override: bool) -> Generator[None]:
    """``values`` in this process's environment while in the context;
    restored on the way out.

    Parameters
    ----------
    values
        The variables to set.
    override
        Whether a variable set already takes its value from ``values``;
        if ``False`` it keeps its own.
    """
    saved = {name: os.environ.get(name) for name in values}
    for name, value in values.items():
        if override:
            os.environ[name] = value
        else:
            os.environ.setdefault(name, value)
    try:
        yield
    finally:
        for name, before in saved.items():
            if before is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = before


@contextlib.contextmanager
def process_pool(workers: int) -> Generator[ProcessPoolExecutor]:
    """A pool of ``workers`` spawned processes, each with :data:`ONE_THREAD`.

    The variables are set in this process's environment while the pool
    lives, as its processes start when work is submitted, and restored
    afterwards; this process's own libraries are already loaded, so their
    threads do not change.
    """
    # Imported here: ``import keepout.cli`` reads ONE_THREAD only.
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    with environment(ONE_THREAD, override=True):
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            yield pool

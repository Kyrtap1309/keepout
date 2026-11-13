"""The sample event in ``tests/cdm/data/event/``: three CDMs of one
encounter, built from ``tracss_v2.kvn``, and the example of ``docs/risk.md``
that reads them.

Run this file as a script to write the CDMs again::

    uv run python tests/cdm/test_sample_event.py
"""

import contextlib
import dataclasses
import io
import re
from pathlib import Path
from typing import Final

import pytest
from astropy.time import Time

from keepout.cdm import Cdm, Maneuverable, ObjectMetadata, kvn, load, make_cdm
from keepout.conjunction import Conjunction
from keepout.covariance import make_covariance
from keepout.objects import ObjectState

REPO: Final = Path(__file__).parents[2]
SAMPLE: Final = REPO / "tests" / "cdm" / "data" / "tracss_v2.kvn"
EVENT: Final = REPO / "tests" / "cdm" / "data" / "event"
RISK_PAGE: Final = REPO / "docs" / "risk.md"

UPDATES: Final = (
    ("2025-05-29T08:00:00.000", 1000.0),
    ("2025-05-30T20:00:00.000", 25.0),
    ("2025-06-01T12:00:00.000", 1.0),
)
"""Each CDM's creation date and the factor its covariances are multiplied by:
the covariances shrink as the TCA nears, so the Pc rises."""

HARD_BODY_RADIUS: Final = 10.0
"""Each object's ``HBR`` [m]: 20 m combined."""


def sample_event() -> list[Cdm]:
    """The three CDMs, oldest first.

    Both objects' states are those of ``tracss_v2.kvn`` (in GCRF), with its
    covariances multiplied by the factors of :data:`UPDATES` and an ``HBR``
    of 10 m each.
    """
    source = load(SAMPLE)
    primary, secondary = source.object_states()
    metadata = [
        ObjectMetadata(
            item.name,
            item.international_designator,
            maneuverable=Maneuverable(item.maneuverable),
        )
        for item in (source.object1, source.object2)
    ]

    def scaled(item: ObjectState, factor: float) -> ObjectState:
        covariance = item.covariance
        assert covariance is not None
        matrix = covariance.matrix * factor
        return dataclasses.replace(
            item,
            covariance=make_covariance(matrix, "GCRF", covariance.epoch),
            hard_body_radius=HARD_BODY_RADIUS,
        )

    return [
        make_cdm(
            Conjunction(
                scaled(primary, factor),
                scaled(secondary, factor),
                2 * HARD_BODY_RADIUS,
            ),
            *metadata,
            created=Time(created, scale="utc", precision=3),
            comments=("keepout sample event, built from tracss_v2.kvn",),
        )
        for created, factor in UPDATES
    ]


def write_sample_event() -> None:
    """Write :func:`sample_event` to :data:`EVENT`, one KVN file per CDM."""
    EVENT.mkdir(exist_ok=True)
    for message in sample_event():
        kvn.dump(message, EVENT / f"{message.message_id}.kvn")


def test_the_committed_cdms_are_the_sample_event() -> None:
    committed = sorted(EVENT.glob("*.kvn"))
    built = sample_event()
    assert [path.stem for path in committed] == [cdm.message_id for cdm in built]
    for path, message in zip(committed, built, strict=True):
        assert path.read_text(encoding="utf-8") == kvn.dumps(message)


def risk_example() -> tuple[str, str]:
    """The first ``python`` block of ``docs/risk.md`` and the ``text`` block
    after it."""
    page = RISK_PAGE.read_text(encoding="utf-8")
    blocks = re.findall(r"^```(\w*)\n(.*?)^```$", page, re.DOTALL | re.MULTILINE)
    (language, code), (next_language, shown) = blocks[:2]
    assert (language, next_language) == ("python", "text")
    return code, shown


def test_the_risk_page_example_prints_what_the_page_shows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, shown = risk_example()
    assert "tests/cdm/data/event" in code
    monkeypatch.chdir(REPO)
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        exec(compile(code, str(RISK_PAGE), "exec"), {"__name__": "docs_example"})  # noqa: S102
    assert printed.getvalue() == shown


if __name__ == "__main__":
    write_sample_event()

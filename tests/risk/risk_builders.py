"""Builders and strategies for the risk tests."""

import re

from builders import CDM_DATA
from hypothesis import strategies as st

import keepout.cdm
from keepout.cdm import Cdm
from keepout.risk import DAY, PcObservation


def at(days: float, pc: float | None, **changes: object) -> PcObservation:
    """An observation ``days`` before TCA; ``pc=None`` makes it a failed one."""
    if pc is None:
        changes.setdefault("error", "GeometryError: no plane")
    return PcObservation(
        time_to_tca=days * DAY,
        source="test",
        pc=pc,
        **changes,  # type: ignore[arg-type]
    )


def series(*pairs: tuple[float, float | None]) -> list[PcObservation]:
    """Observations from ``(days before TCA, Pc)`` pairs."""
    return [at(days, pc) for days, pc in pairs]


pcs = st.one_of(st.just(0.0), st.floats(1e-12, 1.0))
"""Probabilities across the levels, including exactly 0."""

observations = st.builds(
    lambda days, pc, failed: at(days, None if failed else pc),
    st.floats(-0.5, 7.0),
    pcs,
    st.booleans(),
)
"""Any observation, also failed ones and ones issued after TCA."""


B1_TEXT = (CDM_DATA / "b1_example.kvn").read_text(encoding="utf-8")
"""The CCSDS 508.0-B-1 annex example: created 2010-03-12T22:31:12.000, TCA
2010-03-13T22:37:52.618, objects 12345 and 30337, no HBR."""


def _set(text: str, keyword: str, value: str) -> str:
    """``text`` with the first ``keyword = ...`` line set to ``value``."""
    line = re.compile(rf"^{keyword}\s*=.*$", re.MULTILINE)
    assert line.search(text), keyword
    return line.sub(f"{keyword} = {value}", text, count=1)


def b1_cdm(
    message_id: str = "B1-1",
    *,
    created: str = "2010-03-12T22:31:12.000",
    tca: str = "2010-03-13T22:37:52.618",
    secondary: str = "30337",
    conjunction_id: str | None = None,
    reported_pc: str = "4.835E-05",
) -> Cdm:
    """The annex example with its header, TCA and secondary changed."""
    text = _set(B1_TEXT, "MESSAGE_ID", message_id)
    text = _set(text, "CREATION_DATE", created)
    text = _set(text, "TCA", tca)
    text = _set(text, "COLLISION_PROBABILITY", reported_pc)
    assert text.count("= 30337") == 1
    text = text.replace("= 30337", f"= {secondary}")
    if conjunction_id is not None:
        text = text.replace(
            f"MESSAGE_ID = {message_id}",
            f"MESSAGE_ID = {message_id}\nCONJUNCTION_ID = {conjunction_id}",
        )
    return keepout.cdm.loads(text)

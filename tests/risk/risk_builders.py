"""Builders and strategies for the risk tests."""

import re

from builders import CDM_DATA, at
from hypothesis import strategies as st

import keepout.cdm
from keepout.cdm import Cdm
from keepout.risk import PcObservation

pcs = st.one_of(st.just(0.0), st.floats(1e-12, 1.0))
"""Probabilities across the levels, including exactly 0."""

days = st.one_of(st.sampled_from([3.0, 2.0, 1.0, 0.0, -0.1]), st.floats(-0.5, 7.0))
"""Times before TCA [d]; the few fixed ones make simultaneous observations
likely."""


@st.composite
def observations(draw: st.DrawFn) -> PcObservation:
    """Any observation: also failed ones, ones issued after TCA, upper bounds,
    ones with diagnostics or a Monte Carlo interval."""
    when = draw(days)
    if draw(st.booleans()):
        return at(when, None)
    pc = draw(pcs)
    return at(
        when,
        pc,
        kind=draw(st.sampled_from(["estimate", "upper_bound"])),
        diagnostics=draw(st.sampled_from([(), ("covariance repaired",)])),
        ci_high=draw(st.one_of(st.none(), st.floats(pc, 1.0))),
    )


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
    assumed_covariance: bool = False,
) -> Cdm:
    """The annex example with its header, TCA and secondary changed; with
    ``assumed_covariance``, the secondary's covariance is assumed
    (``COVARIANCE_METHOD = DEFAULT``)."""
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
    if assumed_covariance:
        head, keyword, secondary_part = text.rpartition("COVARIANCE_METHOD")
        text = head + _set(keyword + secondary_part, keyword, "DEFAULT")
    return keepout.cdm.loads(text)

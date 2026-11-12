"""CDMs for the store tests: the CCSDS annex example with its header, TCA
and secondary object changed; settings and row counts."""

import re
from pathlib import Path

import astropy.units as u
from astropy.time import Time
from builders import CDM_DATA
from sqlalchemy import func, select

import keepout.cdm
from keepout.cdm import Cdm
from keepout.pc import computation
from keepout.settings import Settings
from keepout.store import CdmInput, Store, models

B1_TEXT = (CDM_DATA / "b1_example.kvn").read_text(encoding="utf-8")
"""Created 2010-03-12T22:31:12.000, TCA 2010-03-13T22:37:52.618, objects
12345 and 30337, no HBR."""

TCA = Time("2010-03-13T22:37:52.618", scale="utc")
"""The example's TCA."""

FOSTER = Settings(compute=computation("foster"), combined_hard_body_radius=20.0)
"""Settings of a fast Pc, with the HBR the example lacks."""


def _set(text: str, keyword: str, value: str) -> str:
    """``text`` with the first ``keyword = ...`` line set to ``value``."""
    line = re.compile(rf"^{keyword}\s*=.*$", re.MULTILINE)
    assert line.search(text), keyword
    return line.sub(f"{keyword} = {value}", text, count=1)


def cdm_text(
    message_id: str,
    *,
    tca: float = 0.0,
    secondary: str = "30337",
    conjunction_id: str | None = None,
    created: float = -86_400.0,
) -> str:
    """The KVN of the example as ``message_id``, its TCA ``tca`` seconds
    after the example's and created ``created`` seconds from that TCA."""
    when = TCA + tca * u.s
    text = _set(B1_TEXT, "MESSAGE_ID", message_id)
    text = _set(text, "CREATION_DATE", (when + created * u.s).isot)
    text = _set(text, "TCA", when.isot)
    text = text.replace("= 30337", f"= {secondary}")
    if conjunction_id is not None:
        text = text.replace(
            f"MESSAGE_ID = {message_id}",
            f"MESSAGE_ID = {message_id}\nCONJUNCTION_ID = {conjunction_id}",
        )
    return text


def cdm(
    message_id: str,
    *,
    tca: float = 0.0,
    secondary: str = "30337",
    conjunction_id: str | None = None,
    created: float = -86_400.0,
) -> Cdm:
    """The CDM of :func:`cdm_text`."""
    return keepout.cdm.loads(
        cdm_text(
            message_id,
            tca=tca,
            secondary=secondary,
            conjunction_id=conjunction_id,
            created=created,
        )
    )


def message(
    message_id: str,
    *,
    tca: float = 0.0,
    secondary: str = "30337",
    conjunction_id: str | None = None,
    created: float = -86_400.0,
) -> CdmInput:
    """The CDM of :func:`cdm_text`, to ingest."""
    made = cdm(
        message_id,
        tca=tca,
        secondary=secondary,
        conjunction_id=conjunction_id,
        created=created,
    )
    return CdmInput.of(made, origin="test")


def written(path: Path, text: str) -> Path:
    """``path`` with ``text`` written to it."""
    path.write_text(text, encoding="utf-8")
    return path


def count_rows(store: Store, table: type[models.Base]) -> int:
    """How many rows ``table`` of ``store`` has."""
    with store.read() as session:
        return session.scalar(select(func.count()).select_from(table)) or 0

"""CDMs for the store tests: the CCSDS annex example with its header, TCA
and secondary object changed; settings and row counts."""

from pathlib import Path

import astropy.units as u
from builders import B1_TEXT, EPOCH, set_keyword
from sqlalchemy import delete, func, select

import keepout.cdm
from keepout.cdm import Cdm
from keepout.pc import computation
from keepout.settings import Settings
from keepout.store import CdmInput, Store, models
from keepout.store.codec import result_key

TCA = EPOCH
"""The example's TCA."""

FOSTER = Settings(compute=computation("foster"), combined_hard_body_radius=20.0)
"""Settings of a fast Pc, with the HBR the example lacks."""


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
    text = set_keyword(B1_TEXT, "MESSAGE_ID", message_id)
    text = set_keyword(text, "CREATION_DATE", (when + created * u.s).isot)
    text = set_keyword(text, "TCA", when.isot)
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


def drop_result(store: Store, message_id: str, settings: Settings) -> None:
    """Forget the Pc of a CDM cached for ``settings``."""
    with store.write() as session:
        session.execute(
            delete(models.Result).where(
                models.Result.message_id == message_id,
                models.Result.key == result_key(settings),
            )
        )


def missing_pc(store: Store, event_id: str, settings: Settings) -> tuple[str, ...]:
    """The CDMs of a stored event whose Pc is not cached for ``settings``,
    oldest first."""
    pairs = store.cached_observations(event_id, settings)
    return tuple(message_id for message_id, item in pairs if item is None)

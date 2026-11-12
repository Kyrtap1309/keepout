"""Errors for downloads that are not in the format a source should send."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, NoReturn

from keepout.errors import OmmParseError, SourceFormatError

if TYPE_CHECKING:
    from collections.abc import Callable

    from keepout.omm.model import Omm

__all__ = ["FORMAT_EXCERPT_BYTES", "omm_reader", "refuse_format"]

FORMAT_EXCERPT_BYTES: Final = 60
"""Bytes of a download quoted when it is not what was expected."""


def refuse_format(
    source: str, expected: str, data: bytes, cause: Exception | None = None
) -> NoReturn:
    """A refusal of ``data`` from ``source`` for not being ``expected``.

    Parameters
    ----------
    source
        Who sent the data: a URL or a service name.
    expected
        What it should have sent, e.g. ``"OMM JSON"``.
    data
        The download; its first :data:`FORMAT_EXCERPT_BYTES` bytes are quoted.
    cause
        Why reading it failed, if an error says; it is chained as the
        refusal's cause.

    Raises
    ------
    SourceFormatError
        Always: ``<source> did not send <expected> (<cause>); it starts
        with b'...'``.
    """
    because = "" if cause is None else f" ({cause})"
    error = SourceFormatError(
        f"{source} did not send {expected}{because}; "
        f"it starts with {data[:FORMAT_EXCERPT_BYTES]!r}"
    )
    if cause is None:
        raise error
    raise error from cause


def omm_reader(
    loads: Callable[[str], tuple[Omm, ...]], source: str, expected: str
) -> Callable[[bytes], tuple[Omm, ...]]:
    """A reader of the element sets in a download from ``source``: UTF-8
    text parsed by ``loads``.

    The reader raises :class:`~keepout.errors.SourceFormatError`
    (:func:`refuse_format`) if the download is not UTF-8 or ``loads``
    refuses it.

    Parameters
    ----------
    loads
        The OMM reader of the format, e.g.
        :func:`~keepout.omm.celestrak.loads_json`.
    source
        Who sends the data, for the error.
    expected
        The format, for the error.
    """

    def read(data: bytes) -> tuple[Omm, ...]:
        try:
            return loads(data.decode("utf-8"))
        except (UnicodeDecodeError, OmmParseError) as exc:
            refuse_format(source, expected, data, exc)

    return read

"""Space-Track test data: a fake server, and ``gp`` and ``cdm_public``
answers in its JSON layout.

The ``gp`` answers are made from :data:`celestrak_builders.SAMPLE`
(CelesTrak data, which Space-Track's user agreement allows to redistribute
with citation): every value a string and Space-Track's further keywords
added, as the ``gp`` class's model definition lists them. The
``cdm_public`` rows are invented (Space-Track's may not be redistributed):
their layout is the class's model definition, every value a string. None
was downloaded from Space-Track.
"""

import json
import urllib.parse
from collections.abc import Iterable, Mapping
from typing import Final, TypeAlias

from celestrak_builders import SAMPLE

from keepout.omm import Omm
from keepout.omm.celestrak import dumps_json
from keepout.sources.spacetrack import (
    LOGIN_URL,
    LOGOUT_URL,
    Credentials,
    Request,
    Response,
)

IDENTITY: Final = "operator@example.org"
# Made up, for the fake server.
PASSWORD: Final = "s3cret P@ss&w0rd%"  # noqa: S105
"""A password with characters that URL encoding changes."""

CREDENTIALS: Final = Credentials(IDENTITY, PASSWORD)

HOUR: Final = 3600.0

Answer: TypeAlias = bytes | Response | Exception
"""A fake server's answer to a query: a body (HTTP 200), a response, or a
failure to raise."""

LOGGED_OUT: Final = Response(
    401, b'{"error":"You must be logged in to complete this action"}'
)
LOGIN_FAILED: Final = b'{"Login":"Failed"}'
RATE_LIMITED: Final = Response(
    500, b"<html>You've violated your query rate limit.</html>"
)


def space_track_gp(omms: Iterable[Omm] = SAMPLE) -> bytes:
    """A ``gp`` answer in Space-Track's JSON: strings, more keywords, nulls."""
    records = []
    for item in json.loads(dumps_json(omms)):
        record: dict[str, str | None] = {
            "CCSDS_OMM_VERS": "3.0",
            "COMMENT": "GENERATED VIA SPACE-TRACK.ORG API",
            "CREATION_DATE": "2026-10-15T12:00:00",
            "ORIGINATOR": "18 SPCS",
            "CENTER_NAME": "EARTH",
            "REF_FRAME": "TEME",
            "TIME_SYSTEM": "UTC",
            "MEAN_ELEMENT_THEORY": "SGP4",
            "DECAY_DATE": None,
            "RCS_SIZE": None,
        }
        record.update({key: str(value) for key, value in item.items()})
        records.append(record)
    return json.dumps(records).encode()


def public_cdm(**values: str | None) -> dict[str, str | None]:
    """An invented ``cdm_public`` row: an approach of two of the sample's
    objects (Calsphere 1 and the ISS), with ``values`` replacing columns."""
    row: dict[str, str | None] = {
        "CDM_ID": "1000001",
        "CREATED": "2026-10-15 06:12:44.000000",
        "EMERGENCY_REPORTABLE": "Y",
        "TCA": "2026-10-17T03:25:31.250000",
        "MIN_RNG": "640",
        "PC": "2.5e-05",
        "SAT_1_ID": "900",
        "SAT_1_NAME": "CALSPHERE 1",
        "SAT1_OBJECT_TYPE": "PAYLOAD",
        "SAT1_RCS": "MEDIUM",
        "SAT_1_EXCL_VOL": "5.00",
        "SAT_2_ID": "25544",
        "SAT_2_NAME": "ISS (ZARYA)",
        "SAT2_OBJECT_TYPE": "PAYLOAD",
        "SAT2_RCS": "LARGE",
        "SAT_2_EXCL_VOL": "5.00",
    }
    row.update(values)
    return row


def space_track_cdm_public(*rows: Mapping[str, object]) -> bytes:
    """A ``cdm_public`` answer holding ``rows``."""
    return json.dumps(list(rows)).encode()


class FakeSpaceTrack:
    """Answers like Space-Track: a login with :data:`CREDENTIALS` opens a
    session, a query without one gets :data:`LOGGED_OUT`, a logout ends it.

    ``answers`` maps each query URL to one answer, or a list of answers
    given in turn. ``login`` replaces the answer to every login, which then
    opens a session only if ``login_opens_session``. Every request is
    recorded.
    """

    def __init__(
        self,
        answers: Mapping[str, Answer | list[Answer]] | None = None,
        *,
        login: Response | None = None,
        logout: Answer | None = None,
        login_opens_session: bool = False,
    ) -> None:
        self.answers = {
            url: list(answer) if isinstance(answer, list) else [answer]
            for url, answer in (answers or {}).items()
        }
        self.login = login
        self.login_opens_session = login_opens_session
        self.logout = logout
        self.requests: list[Request] = []
        self.session = False

    @property
    def calls(self) -> list[str]:
        """The requests, as ``login``, ``logout`` or the query URL."""
        names = {LOGIN_URL: "login", LOGOUT_URL: "logout"}
        return [names.get(request.url, request.url) for request in self.requests]

    def __call__(self, request: Request, /) -> Response:
        self.requests.append(request)
        if request.url == LOGIN_URL:
            return self._login(request)
        if request.url == LOGOUT_URL:
            self.session = False
            return self._answer(self.logout or b'"Successfully logged out"')
        if not self.session:
            return LOGGED_OUT
        queue = self.answers[request.url]
        return self._answer(queue.pop(0) if len(queue) > 1 else queue[0])

    def _login(self, request: Request) -> Response:
        assert request.method == "POST"
        assert request.form is not None
        form = urllib.parse.parse_qs(request.form.decode(), strict_parsing=True)
        if self.login is not None:
            self.session = self.login_opens_session
            return self.login
        if form == {"identity": [IDENTITY], "password": [PASSWORD]}:
            self.session = True
            return Response(200, b'""')
        return Response(200, LOGIN_FAILED)

    @staticmethod
    def _answer(answer: Answer) -> Response:
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, bytes):
            return Response(200, answer)
        return answer

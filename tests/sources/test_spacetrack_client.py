import gzip
import json
import urllib.parse
from collections.abc import Iterator
from pathlib import Path

import pytest
from celestrak_builders import SAMPLE
from spacetrack_builders import (
    CREDENTIALS,
    IDENTITY,
    LOGGED_OUT,
    LOGIN_FAILED,
    PASSWORD,
    RATE_LIMITED,
    Answer,
    FakeSpaceTrack,
    assert_hidden,
    space_track_gp,
)
from typing_extensions import override
from web_builders import START, Clock

from keepout.errors import (
    AuthError,
    HttpStatusError,
    RateLimited,
    SourceFormatError,
    SourceUnavailable,
)
from keepout.omm import deduplicate
from keepout.sources.cache import read_copy, save_copy
from keepout.sources.catalog import Catalog
from keepout.sources.spacetrack import (
    CATALOG,
    HIDDEN,
    IDENTITY_VARIABLE,
    LOGIN_URL,
    PASSWORD_VARIABLE,
    Credentials,
    Query,
    Request,
    Response,
    SpaceTrackClient,
    Throttle,
    Transport,
    fetch_catalog,
    read_catalog,
)
from keepout.units import HOUR

GP = space_track_gp()
CDM_QUERY = Query("cdm_public", "TCA/>now/orderby/TCA asc/format/json", "upcoming")
ISS_CDMS = Query("cdm_public", "SAT_1_ID/25544/format/json", "iss")
ISS_GP = Query("gp", "NORAD_CAT_ID/25544/format/json", "iss")


@pytest.fixture(autouse=True)
def _no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(IDENTITY_VARIABLE, raising=False)
    monkeypatch.delenv(PASSWORD_VARIABLE, raising=False)


class RecordingThrottle(Throttle):
    """A throttle that never waits, recording how many requests ``server``
    had been sent at each wait."""

    def __init__(self, server: FakeSpaceTrack) -> None:
        super().__init__()
        self.server = server
        self.sent_before: list[int] = []

    @override
    def wait(self) -> None:
        self.sent_before.append(len(self.server.requests))


def client_for(
    server: FakeSpaceTrack,
    root: Path,
    clock: Clock,
    credentials: Credentials | None = CREDENTIALS,
    throttle: Throttle | None = None,
) -> SpaceTrackClient:
    return SpaceTrackClient(
        credentials,
        cache=root,
        transport=server,
        throttle=RecordingThrottle(server) if throttle is None else throttle,
        now=clock,
    )


def fetch(
    transport: Transport,
    root: Path,
    clock: Clock,
    *,
    credentials: Credentials | None = CREDENTIALS,
    max_age: float | None = None,
) -> Catalog:
    """:func:`fetch_catalog` sent through ``transport``, cached under ``root``."""
    return fetch_catalog(
        cache=root,
        max_age=max_age,
        credentials=credentials,
        transport=transport,
        now=clock,
    )


def ledger(root: Path) -> object:
    return json.loads((root / "spacetrack" / "requests.json").read_text())


def errors_in_chain(error: BaseException) -> Iterator[BaseException]:
    seen: BaseException | None = error
    while seen is not None:
        yield seen
        seen = seen.__cause__ or seen.__context__


def test_the_catalog_is_downloaded_read_and_cached(
    tmp_path: Path, clock: Clock
) -> None:
    server = FakeSpaceTrack({CATALOG.url: GP})
    catalog = fetch(server, tmp_path, clock)
    assert server.calls == ["login", CATALOG.url, "logout"]
    login = server.requests[0]
    assert login.method == "POST"
    assert login.form == (
        b"identity=operator%40example.org&password=s3cret+P%40ss%26w0rd%25"
    )
    assert [request.method for request in server.requests[1:]] == ["GET", "GET"]
    kept, dropped = deduplicate(SAMPLE)
    assert catalog.omms == tuple(sorted(kept, key=lambda omm: omm.catalog_number))
    assert catalog.duplicates == dropped
    (part,) = catalog.parts
    assert part.name == "gp"
    assert part.omms == SAMPLE
    path = tmp_path / "spacetrack" / "gp" / "catalog.json.gz"
    assert part.retrieval.path == path
    assert part.retrieval.url == CATALOG.url
    assert part.retrieval.retrieved.unix == START
    assert part.retrieval.checked.unix == START
    assert gzip.decompress(path.read_bytes()) == GP
    cached = read_copy(path, CATALOG.url)
    assert cached is not None
    assert cached.meta.retrieved == START
    assert ledger(tmp_path) == {"format": 1, "queried": {"gp": START}}


def test_a_cached_catalog_needs_no_request_and_no_credentials(
    tmp_path: Path, clock: Clock
) -> None:
    server = FakeSpaceTrack({CATALOG.url: GP})
    first = client_for(server, tmp_path, clock).fetch_catalog()
    clock.now += HOUR - 1.0
    again = FakeSpaceTrack()
    with client_for(again, tmp_path, clock, credentials=None) as client:
        assert client.fetch_catalog() == first
    assert again.requests == []


def test_a_copy_is_used_up_to_max_age(tmp_path: Path, clock: Clock) -> None:
    server = FakeSpaceTrack({CATALOG.url: GP})
    fetch(server, tmp_path, clock)
    clock.now += 2 * HOUR
    fetch(server, tmp_path, clock, credentials=None, max_age=3 * HOUR)
    assert server.calls == ["login", CATALOG.url, "logout"]


def test_a_max_age_below_the_class_interval_is_refused(
    tmp_path: Path, clock: Clock
) -> None:
    server = FakeSpaceTrack({CATALOG.url: GP})
    with pytest.raises(ValueError, match=r"max_age must be at least 3600 s .* got 60"):
        fetch(server, tmp_path, clock, credentials=None, max_age=60.0)
    assert server.requests == []


def test_an_old_copy_is_downloaded_again(tmp_path: Path, clock: Clock) -> None:
    newer = space_track_gp(SAMPLE[:2])
    server = FakeSpaceTrack({CATALOG.url: [GP, newer]})
    with client_for(server, tmp_path, clock) as client:
        client.fetch_catalog()
        clock.now += HOUR
        catalog = client.fetch_catalog()
    assert server.calls == ["login", CATALOG.url, CATALOG.url, "logout"]
    assert catalog.parts[0].omms == SAMPLE[:2]
    assert ledger(tmp_path) == {"format": 1, "queried": {"gp": START + HOUR}}


def test_a_class_asked_for_within_its_interval_is_refused_offline(
    tmp_path: Path, clock: Clock
) -> None:
    server = FakeSpaceTrack({CATALOG.url: GP})
    fetch(server, tmp_path, clock)
    (tmp_path / "spacetrack" / "gp" / "catalog.json.gz").unlink()
    clock.now += 20 * 60.0
    with pytest.raises(RateLimited) as caught:
        fetch(server, tmp_path, clock)
    assert caught.value.retry_after == 40 * 60.0
    assert str(caught.value) == (
        "Space-Track allows one gp query per 1 h; the last was sent at "
        f"2026-09-21 14:13:20 UTC (recorded in {tmp_path}/spacetrack/requests.json); "
        "try again in 40 min 00 s"
    )
    assert server.calls == ["login", CATALOG.url, "logout"]


@pytest.mark.parametrize(
    ("query", "elapsed", "wait"),
    [
        (ISS_CDMS, 3 * HOUR, "5 h 00 min"),
        (ISS_CDMS, 8 * HOUR - 0.5, "1 s"),
        (ISS_GP, 3590.0, "10 s"),
        (ISS_GP, 1801.0, "29 min 59 s"),
        # The clock was set back: wait the whole interval from the record.
        (ISS_GP, -HOUR, "2 h 00 min"),
    ],
)
def test_every_query_of_a_class_shares_its_limit(
    tmp_path: Path, clock: Clock, query: Query, elapsed: float, wait: str
) -> None:
    server = FakeSpaceTrack({CATALOG.url: GP, CDM_QUERY.url: b"[]"})
    with client_for(server, tmp_path, clock) as client:
        client.fetch_catalog()
        client.fetch(CDM_QUERY, json.loads)
        clock.now += elapsed
        with pytest.raises(RateLimited, match=f"try again in {wait}$"):
            client.fetch(query, json.loads)
    assert server.calls == ["login", CATALOG.url, CDM_QUERY.url, "logout"]
    assert ledger(tmp_path) == {
        "format": 1,
        "queried": {"cdm_public": START, "gp": START},
    }


def test_credentials_are_read_from_the_environment_when_needed(
    tmp_path: Path, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = FakeSpaceTrack({CATALOG.url: GP})
    with pytest.raises(AuthError, match=f"set {IDENTITY_VARIABLE} and"):
        fetch(server, tmp_path, clock, credentials=None)
    assert server.requests == []
    assert not (tmp_path / "spacetrack" / "requests.json").exists()
    monkeypatch.setenv(IDENTITY_VARIABLE, IDENTITY)
    monkeypatch.setenv(PASSWORD_VARIABLE, PASSWORD)
    fetch(server, tmp_path, clock, credentials=None)
    assert server.calls == ["login", CATALOG.url, "logout"]


@pytest.mark.parametrize(
    ("login", "message"),
    [
        (Response(200, LOGIN_FAILED), r"\(HTTP 200\): Failed$"),
        (
            Response(401, b'{"Login":"Password does not meet minimum length"}'),
            r"\(HTTP 401\): Password does not meet minimum length$",
        ),
        (Response(503, b"<p>Down for maintenance</p>"), r"\(HTTP 503\): <p>Down"),
        (
            Response(200, f'{{"Login":"no user {IDENTITY}"}}'.encode()),
            r"\(HTTP 200\): no user <hidden>$",
        ),
    ],
)
def test_a_refused_login_is_an_auth_error(
    tmp_path: Path, clock: Clock, login: Response, message: str
) -> None:
    server = FakeSpaceTrack({CATALOG.url: GP}, login=login)
    with pytest.raises(AuthError, match=f"^Space-Track refused the login {message}"):
        fetch(server, tmp_path, clock)
    assert server.calls == ["login"]
    # No query was sent, so none is recorded.
    assert not (tmp_path / "spacetrack" / "requests.json").exists()


def test_wrong_credentials_are_refused(tmp_path: Path, clock: Clock) -> None:
    server = FakeSpaceTrack({CATALOG.url: GP})
    wrong = Credentials(IDENTITY, "wrong")
    with pytest.raises(AuthError, match=r"\(HTTP 200\): Failed$"):
        fetch(server, tmp_path, clock, credentials=wrong)


def test_an_expired_session_is_renewed_once(tmp_path: Path, clock: Clock) -> None:
    server = FakeSpaceTrack({CATALOG.url: [LOGGED_OUT, GP]})
    catalog = fetch(server, tmp_path, clock)
    assert server.calls == ["login", CATALOG.url, "login", CATALOG.url, "logout"]
    assert catalog.parts[0].omms == SAMPLE


def test_a_second_401_is_an_auth_error(tmp_path: Path, clock: Clock) -> None:
    server = FakeSpaceTrack({CATALOG.url: LOGGED_OUT})
    with pytest.raises(AuthError, match=r"HTTP 401 right after a login: .*logged in"):
        fetch(server, tmp_path, clock)
    assert server.calls == ["login", CATALOG.url, "login", CATALOG.url, "logout"]


@pytest.mark.parametrize(
    "answer", [RATE_LIMITED, Response(429, b"Too Many Requests")], ids=["500", "429"]
)
def test_a_rate_limit_answer_stops_the_client(
    tmp_path: Path, clock: Clock, answer: Response
) -> None:
    server = FakeSpaceTrack({CATALOG.url: answer, CDM_QUERY.url: b"[]"})
    client = client_for(server, tmp_path, clock)
    with pytest.raises(RateLimited, match="nothing more is sent") as caught:
        client.fetch_catalog()
    assert caught.value.retry_after is None
    assert repr(client) == f"<SpaceTrackClient {tmp_path}/spacetrack (stopped)>"
    with pytest.raises(RateLimited, match="this client stopped after"):
        client.fetch(CDM_QUERY, json.loads)
    client.close()
    # No retry and no logout.
    assert server.calls == ["login", CATALOG.url]


def test_a_rate_limit_answer_to_the_login_stops_the_client(
    tmp_path: Path, clock: Clock
) -> None:
    server = FakeSpaceTrack({CATALOG.url: GP}, login=RATE_LIMITED)
    with pytest.raises(RateLimited, match="violated your query rate limit"):
        fetch(server, tmp_path, clock)
    assert server.calls == ["login"]


def test_another_error_status_is_raised(tmp_path: Path, clock: Clock) -> None:
    body = b"<html>" + b"x" * 300 + b"</html>"
    server = FakeSpaceTrack({CATALOG.url: Response(503, body)})
    with pytest.raises(HttpStatusError) as caught:
        fetch(server, tmp_path, clock)
    assert caught.value.status == 503
    assert caught.value.url == CATALOG.url
    assert str(caught.value) == (
        f"Space-Track answered {CATALOG.url} with HTTP 503: <html>{'x' * 194} ..."
    )
    assert server.calls == ["login", CATALOG.url, "logout"]
    # The query was sent, so it counts.
    assert ledger(tmp_path) == {"format": 1, "queried": {"gp": START}}


def test_a_network_failure_is_raised_and_the_query_counts(
    tmp_path: Path, clock: Clock
) -> None:
    server = FakeSpaceTrack({CATALOG.url: SourceUnavailable("connection reset")})
    with pytest.raises(SourceUnavailable, match="connection reset"):
        fetch(server, tmp_path, clock)
    assert server.calls == ["login", CATALOG.url, "logout"]
    with pytest.raises(RateLimited):
        fetch(server, tmp_path, clock)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b'{"error":"query failed"}', "JSON must be an array of OMM objects"),
        (b"\xff\xfe", "invalid start byte"),
    ],
)
def test_an_answer_that_is_not_omm_json_is_not_cached(
    tmp_path: Path, clock: Clock, body: bytes, message: str
) -> None:
    server = FakeSpaceTrack({CATALOG.url: body})
    with pytest.raises(SourceFormatError, match=f"did not send OMM JSON.*{message}"):
        fetch(server, tmp_path, clock)
    assert not (tmp_path / "spacetrack" / "gp").exists()
    assert ledger(tmp_path) == {"format": 1, "queried": {"gp": START}}


def test_a_cached_copy_that_cannot_be_read_is_downloaded_again(
    tmp_path: Path, clock: Clock
) -> None:
    server = FakeSpaceTrack({CATALOG.url: [b'{"error":"none"}', GP]})
    with client_for(server, tmp_path, clock) as client:
        client.fetch(CATALOG, lambda data: data)
        clock.now += HOUR
        catalog = client.fetch_catalog(max_age=2 * HOUR)
    assert server.calls == ["login", CATALOG.url, CATALOG.url, "logout"]
    assert catalog.parts[0].omms == SAMPLE


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("{not json", "cannot read the Space-Track request record"),
        ("[]", "is not a Space-Track request record"),
        ('{"format": 2, "queried": {}}', "is not a Space-Track request record"),
        ('{"format": 1, "queried": []}', "is not a Space-Track request record"),
        ('{"format": 1, "queried": {"gp": 1}}', "is not a Space-Track request record"),
        (
            '{"format": 1, "queried": {"gp": Infinity}}',
            "is not a Space-Track request record",
        ),
    ],
)
def test_a_damaged_request_record_stops_every_query(
    tmp_path: Path, clock: Clock, text: str, message: str
) -> None:
    record = tmp_path / "spacetrack" / "requests.json"
    record.parent.mkdir()
    record.write_text(text)
    server = FakeSpaceTrack({CATALOG.url: GP})
    with pytest.raises(SourceFormatError, match=message):
        fetch(server, tmp_path, clock)
    assert server.requests == []


def test_a_request_record_that_cannot_be_opened_stops_every_query(
    tmp_path: Path, clock: Clock
) -> None:
    (tmp_path / "spacetrack" / "requests.json").mkdir(parents=True)
    with pytest.raises(SourceFormatError, match="cannot read the Space-Track"):
        fetch(FakeSpaceTrack(), tmp_path, clock, credentials=None)


def test_closing_an_unused_client_sends_nothing(tmp_path: Path, clock: Clock) -> None:
    server = FakeSpaceTrack()
    client_for(server, tmp_path, clock).close()
    assert server.requests == []


def test_a_client_shows_whether_it_is_logged_in(tmp_path: Path, clock: Clock) -> None:
    client = client_for(FakeSpaceTrack({CATALOG.url: GP}), tmp_path, clock)
    assert repr(client) == f"<SpaceTrackClient {tmp_path}/spacetrack>"
    client.fetch_catalog()
    assert repr(client) == f"<SpaceTrackClient {tmp_path}/spacetrack (logged in)>"
    client.close()
    assert repr(client) == f"<SpaceTrackClient {tmp_path}/spacetrack>"


@pytest.mark.parametrize(
    "logout",
    [
        b'"Successfully logged out"',
        SourceUnavailable("timed out"),
        Response(500, b"oops"),
    ],
    ids=["ok", "unreachable", "error-status"],
)
def test_closing_logs_out_once_whatever_the_logout_answers(
    tmp_path: Path, clock: Clock, logout: Answer
) -> None:
    server = FakeSpaceTrack({CATALOG.url: GP}, logout=logout)
    client = client_for(server, tmp_path, clock)
    client.fetch_catalog()
    client.close()
    client.close()
    assert server.calls == ["login", CATALOG.url, "logout"]


ECHO = f"bad form identity={IDENTITY}&password={PASSWORD}".encode()
"""An answer that quotes the login form back."""


@pytest.mark.parametrize(
    ("login", "catalog"),
    [
        (Response(200, b'{"Login":"%s"}' % ECHO), GP),
        (Response(400, ECHO), GP),
        (None, Response(500, ECHO)),
        (None, Response(500, ECHO + b" violated your query rate limit")),
        (None, LOGGED_OUT),
    ],
    ids=["login-refused", "login-error", "query-error", "rate-limited", "second-401"],
)
def test_the_credentials_are_never_shown(
    tmp_path: Path, clock: Clock, login: Response | None, catalog: Answer
) -> None:
    server = FakeSpaceTrack({CATALOG.url: catalog}, login=login)
    client = client_for(server, tmp_path, clock)
    with pytest.raises(SourceUnavailable) as caught:
        client.fetch_catalog()
    client.close()
    for error in errors_in_chain(caught.value):
        assert_hidden(str(error), repr(error))
    assert_hidden(repr(client), *(repr(request) for request in server.requests))
    for path in (tmp_path / "spacetrack").rglob("*"):
        if path.is_file():
            assert_hidden(path.read_text())


@pytest.mark.parametrize(
    "secret",
    [PASSWORD, urllib.parse.quote_plus(PASSWORD), IDENTITY],
    ids=["password", "form-encoded", "identity"],
)
def test_a_secret_across_the_excerpts_end_is_hidden(
    tmp_path: Path, clock: Clock, secret: str
) -> None:
    # The secret starts 10 characters before the excerpt's 200th.
    body = b"x" * 190 + secret.encode() + b"y" * 300
    server = FakeSpaceTrack({CATALOG.url: Response(500, body)})
    with pytest.raises(HttpStatusError) as caught:
        fetch(server, tmp_path, clock)
    assert str(caught.value) == (
        f"Space-Track answered {CATALOG.url} with HTTP 500: {'x' * 190}{HIDDEN}yy ..."
    )


def test_the_start_of_a_secret_at_the_excerpts_end_is_dropped(
    tmp_path: Path, clock: Clock
) -> None:
    # Not the whole password, so redaction alone would leave its start.
    body = b"x" * 195 + PASSWORD[:10].encode() + b" " * 300
    server = FakeSpaceTrack({CATALOG.url: Response(500, body)})
    with pytest.raises(HttpStatusError) as caught:
        fetch(server, tmp_path, clock)
    assert str(caught.value) == (
        f"Space-Track answered {CATALOG.url} with HTTP 500: {'x' * 195} ..."
    )


class LedgerWrittenAtLogin:
    """A fake server during whose login another process records a query and
    the clock moves on 30 s."""

    def __init__(
        self, server: FakeSpaceTrack, root: Path, clock: Clock, queried: object
    ) -> None:
        self.server = server
        self.path = root / "spacetrack" / "requests.json"
        self.clock = clock
        self.queried = queried

    def __call__(self, request: Request, /) -> Response:
        if request.url == LOGIN_URL:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"format": 1, "queried": self.queried}))
            self.clock.now += 30.0
        return self.server(request)


def test_a_record_written_during_the_login_is_kept(
    tmp_path: Path, clock: Clock
) -> None:
    server = FakeSpaceTrack({CATALOG.url: GP})
    other = START - 60.0
    transport = LedgerWrittenAtLogin(server, tmp_path, clock, {"cdm_public": other})
    catalog = fetch(transport, tmp_path, clock)
    assert server.calls == ["login", CATALOG.url, "logout"]
    # The query is recorded at the time it was sent, after the login.
    assert ledger(tmp_path) == {
        "format": 1,
        "queried": {"cdm_public": other, "gp": START + 30.0},
    }
    assert catalog.parts[0].retrieval.retrieved.unix == START + 30.0


def test_a_query_of_the_class_recorded_during_the_login_is_refused(
    tmp_path: Path, clock: Clock
) -> None:
    server = FakeSpaceTrack({CATALOG.url: GP})
    transport = LedgerWrittenAtLogin(server, tmp_path, clock, {"gp": START})
    client = SpaceTrackClient(
        CREDENTIALS, cache=tmp_path, transport=transport, now=clock
    )
    with pytest.raises(RateLimited, match=r"try again in 59 min 30 s$") as caught:
        client.fetch_catalog()
    assert caught.value.retry_after == HOUR - 30.0
    assert server.calls == ["login"]
    assert ledger(tmp_path) == {"format": 1, "queried": {"gp": START}}
    # The session is still ended normally.
    client.close()
    assert server.calls == ["login", "logout"]


def test_read_catalog_reads_space_track_json() -> None:
    assert read_catalog(GP) == SAMPLE


def test_a_cached_copy_that_is_not_gzip_is_downloaded_again(
    tmp_path: Path, clock: Clock
) -> None:
    # E.g. a copy an older keepout kept uncompressed under this name.
    target = tmp_path / "spacetrack" / "gp" / "catalog.json.gz"
    save_copy(target, CATALOG.url, GP, clock.now)
    server = FakeSpaceTrack({CATALOG.url: GP})
    catalog = fetch(server, tmp_path, clock)
    assert server.calls == ["login", CATALOG.url, "logout"]
    assert catalog.parts[0].omms == SAMPLE
    assert gzip.decompress(target.read_bytes()) == GP


def test_a_client_sends_over_urllib_by_default(tmp_path: Path) -> None:
    client = SpaceTrackClient(CREDENTIALS, cache=tmp_path)
    with pytest.raises(AssertionError, match="must not access the network"):
        client.fetch_catalog()


def test_the_request_record_is_in_the_cache(tmp_path: Path) -> None:
    client = SpaceTrackClient(CREDENTIALS, cache=tmp_path)
    assert client.ledger == tmp_path / "spacetrack" / "requests.json"


def test_the_client_waits_for_its_throttle_before_every_request(
    tmp_path: Path, clock: Clock
) -> None:
    server = FakeSpaceTrack({CATALOG.url: GP})
    throttle = RecordingThrottle(server)
    with client_for(server, tmp_path, clock, throttle=throttle) as client:
        client.fetch_catalog()
    assert server.calls == ["login", CATALOG.url, "logout"]
    assert throttle.sent_before == [0, 1, 2]

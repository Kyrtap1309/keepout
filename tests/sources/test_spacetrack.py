import email
import http.client
import io
import json
import urllib.error
import urllib.parse
import urllib.request
import urllib.response
from collections.abc import Iterator
from pathlib import Path

import pytest
from celestrak_builders import SAMPLE
from hypothesis import given
from hypothesis import strategies as st
from spacetrack_builders import (
    CREDENTIALS,
    IDENTITY,
    LOGGED_OUT,
    LOGIN_FAILED,
    PASSWORD,
    RATE_LIMITED,
    FakeSpaceTrack,
    space_track_gp,
)

from keepout.errors import (
    AuthError,
    HttpStatusError,
    RateLimited,
    SourceFormatError,
    SourceUnavailable,
)
from keepout.omm import deduplicate
from keepout.sources import spacetrack
from keepout.sources.cache import read_copy
from keepout.sources.spacetrack import (
    CATALOG,
    CLASS_INTERVALS,
    GROUP,
    HIDDEN,
    LOGIN_URL,
    LOGOUT_URL,
    Credentials,
    Query,
    Request,
    Response,
    SpaceTrackClient,
    Throttle,
    UrllibTransport,
    fetch_catalog,
    read_catalog,
)

START = 1_790_000_000.0
"""A Unix time in September 2026."""
HOUR = 3600.0
TINY = 1e-9
"""Seconds of rounding in a fake clock's sum."""
GP = space_track_gp()
CDM_QUERY = Query("cdm_public", "TCA/>now/orderby/TCA asc/format/json", "upcoming")
ISS_CDMS = Query("cdm_public", "SAT_1_ID/25544/format/json", "iss")
ISS_GP = Query("gp", "NORAD_CAT_ID/25544/format/json", "iss")


class Clock:
    """A settable Unix time."""

    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture(autouse=True)
def _no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPACETRACK_IDENTITY", raising=False)
    monkeypatch.delenv("SPACETRACK_PASSWORD", raising=False)


def client_for(
    server: FakeSpaceTrack,
    root: Path,
    clock: Clock,
    credentials: Credentials | None = CREDENTIALS,
) -> SpaceTrackClient:
    return SpaceTrackClient(credentials, cache=root, transport=server, now=clock)


def ledger(root: Path) -> object:
    return json.loads((root / "spacetrack" / "requests.json").read_text())


def errors_in_chain(error: BaseException) -> Iterator[BaseException]:
    seen: BaseException | None = error
    while seen is not None:
        yield seen
        seen = seen.__cause__ or seen.__context__


def assert_hidden(*texts: str) -> None:
    for text in texts:
        for secret in (PASSWORD, urllib.parse.quote_plus(PASSWORD), IDENTITY):
            assert secret not in text


# Credentials


def test_credentials_come_from_the_environment() -> None:
    environ = {"SPACETRACK_IDENTITY": IDENTITY, "SPACETRACK_PASSWORD": PASSWORD}
    assert Credentials.from_env(environ) == CREDENTIALS


def test_credentials_default_to_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPACETRACK_IDENTITY", IDENTITY)
    monkeypatch.setenv("SPACETRACK_PASSWORD", PASSWORD)
    assert Credentials.from_env() == CREDENTIALS


@pytest.mark.parametrize(
    ("environ", "missing"),
    [
        ({}, "SPACETRACK_IDENTITY and SPACETRACK_PASSWORD"),
        ({"SPACETRACK_IDENTITY": IDENTITY}, "SPACETRACK_PASSWORD"),
        (
            {"SPACETRACK_IDENTITY": "", "SPACETRACK_PASSWORD": PASSWORD},
            "SPACETRACK_IDENTITY",
        ),
    ],
)
def test_missing_credentials_name_the_variables(
    environ: dict[str, str], missing: str
) -> None:
    with pytest.raises(AuthError, match=f"set {missing} ") as caught:
        Credentials.from_env(environ)
    assert_hidden(str(caught.value))


def test_credentials_are_not_empty() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        Credentials(IDENTITY, "")


def test_credentials_are_not_shown() -> None:
    assert repr(CREDENTIALS) == "Credentials(<hidden>)"
    assert str(CREDENTIALS) == "Credentials(<hidden>)"


def test_credentials_are_redacted_in_any_spelling() -> None:
    text = (
        f"user {IDENTITY} password {PASSWORD} form "
        f"{urllib.parse.urlencode({'identity': IDENTITY, 'password': PASSWORD})} "
        f"path {urllib.parse.quote(PASSWORD, safe='')}"
    )
    redacted = CREDENTIALS.redact(text)
    assert_hidden(redacted)
    assert redacted == (
        "user <hidden> password <hidden> form identity=<hidden>&password=<hidden> "
        "path <hidden>"
    )


def test_a_secret_inside_another_is_hidden_whole() -> None:
    credentials = Credentials("ab", "xaby")
    assert credentials.redact("xaby ab") == f"{HIDDEN} {HIDDEN}"


# Queries


def test_the_catalog_query_is_space_tracks_recommended_one() -> None:
    assert CATALOG.url == (
        "https://www.space-track.org/basicspacedata/query/class/gp/"
        "decay_date/null-val/epoch/%3Enow-10/orderby/norad_cat_id/format/json"
    )
    assert CATALOG.interval == HOUR


def test_the_class_limits_are_the_published_ones() -> None:
    assert dict(CLASS_INTERVALS) == {"gp": HOUR, "cdm_public": 8 * HOUR}
    assert CDM_QUERY.interval == 8 * HOUR
    assert CDM_QUERY.url.endswith(
        "/class/cdm_public/TCA/%3Enow/orderby/TCA%20asc/format/json"
    )
    assert Query("gp", "NORAD_CAT_ID/25544,43873/format/json", "a").url.endswith(
        "/NORAD_CAT_ID/25544,43873/format/json"
    )


@pytest.mark.parametrize(
    ("request_class", "path", "name", "message"),
    [
        ("gp_history", "format/json", "x", "no Space-Track limit known"),
        ("cdm", "format/json", "x", "no Space-Track limit known"),
        ("gp", "format/json", "../x", "not a cache file name"),
        ("gp", "format/json", "Catalog", "not a cache file name"),
        ("gp", "format/json", "", "not a cache file name"),
        ("gp", "", "x", "not a Space-Track query path"),
        ("gp", "/format/json", "x", "not a Space-Track query path"),
        ("gp", "format/json/", "x", "not a Space-Track query path"),
        ("gp", "format/json?x=1", "x", "not a Space-Track query path"),
        ("gp", "format/json#x", "x", "not a Space-Track query path"),
        ("gp", "a/../b", "x", "not a Space-Track query path"),
    ],
)
def test_a_query_is_checked(
    request_class: str, path: str, name: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Query(request_class, path, name)


# Throttle


class FakeTime:
    """A monotonic clock that a fake sleep advances."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_the_throttle_waits_only_past_a_limit() -> None:
    fake = FakeTime()
    throttle = Throttle(clock=fake.clock, sleep=fake.sleep)
    for _ in range(20):
        throttle.wait()
    assert fake.slept == []
    fake.now = 10.0
    throttle.wait()
    assert fake.slept == [50.0]
    assert fake.now == 60.0


def test_the_throttle_keeps_the_hourly_limit() -> None:
    fake = FakeTime()
    throttle = Throttle(clock=fake.clock, sleep=fake.sleep)
    for _ in range(200):
        throttle.wait()
    assert fake.now == 9 * 60.0  # 20 a minute
    throttle.wait()
    assert fake.now == HOUR


def test_the_throttle_limits_are_checked() -> None:
    with pytest.raises(ValueError, match="limits must be at least 1"):
        Throttle(per_minute=0)
    with pytest.raises(ValueError, match="limits must be at least 1"):
        Throttle(per_hour=0)


@given(
    gaps=st.lists(st.floats(0.0, 400.0), min_size=1, max_size=60),
    per_minute=st.integers(1, 5),
    per_hour=st.integers(1, 12),
)
def test_throttled_requests_never_exceed_a_limit(
    gaps: list[float], per_minute: int, per_hour: int
) -> None:
    fake = FakeTime()
    throttle = Throttle(
        per_minute=per_minute, per_hour=per_hour, clock=fake.clock, sleep=fake.sleep
    )
    sent: list[float] = []
    for gap in gaps:
        fake.now += gap
        asked = fake.now
        throttle.wait()
        sent.append(fake.now)
        # It waits no longer than a limit needs (to rounding: the fake
        # clock adds the wait to the time it was asked at).
        assert fake.now == asked or any(
            fake.now == pytest.approx(sent[-limit - 1] + window, rel=0, abs=TINY)
            for window, limit in ((60.0, per_minute), (HOUR, per_hour))
            if len(sent) > limit
        )
    for index, time in enumerate(sent):
        later = sent[index:]
        assert sum(t < time + 60.0 - TINY for t in later) <= per_minute
        assert sum(t < time + HOUR - TINY for t in later) <= per_hour


# The client


def test_the_catalog_is_downloaded_read_and_cached(
    tmp_path: Path, clock: Clock
) -> None:
    server = FakeSpaceTrack({CATALOG.url: GP})
    catalog = fetch_catalog(
        cache=tmp_path, credentials=CREDENTIALS, transport=server, now=clock
    )
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
    (group,) = catalog.groups
    assert group.name == GROUP == "space-track"
    assert group.omms == SAMPLE
    path = tmp_path / "spacetrack" / "gp" / "catalog.json"
    assert group.retrieval.path == path
    assert group.retrieval.url == CATALOG.url
    assert group.retrieval.retrieved.unix == START
    assert group.retrieval.checked.unix == START
    assert path.read_bytes() == GP
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
    fetch_catalog(cache=tmp_path, credentials=CREDENTIALS, transport=server, now=clock)
    clock.now += 2 * HOUR
    fetch_catalog(cache=tmp_path, transport=server, now=clock, max_age=3 * HOUR)
    assert server.calls == ["login", CATALOG.url, "logout"]
    with pytest.raises(ValueError, match=r"max_age must be at least 3600 s .* got 60"):
        fetch_catalog(cache=tmp_path, transport=server, now=clock, max_age=60.0)


def test_an_old_copy_is_downloaded_again(tmp_path: Path, clock: Clock) -> None:
    newer = space_track_gp(SAMPLE[:2])
    server = FakeSpaceTrack({CATALOG.url: [GP, newer]})
    with client_for(server, tmp_path, clock) as client:
        client.fetch_catalog()
        clock.now += HOUR
        catalog = client.fetch_catalog()
    assert server.calls == ["login", CATALOG.url, CATALOG.url, "logout"]
    assert catalog.groups[0].omms == SAMPLE[:2]
    assert ledger(tmp_path) == {"format": 1, "queried": {"gp": START + HOUR}}


def test_a_class_asked_for_within_its_interval_is_refused_offline(
    tmp_path: Path, clock: Clock
) -> None:
    server = FakeSpaceTrack({CATALOG.url: GP})
    fetch_catalog(cache=tmp_path, credentials=CREDENTIALS, transport=server, now=clock)
    (tmp_path / "spacetrack" / "gp" / "catalog.json").unlink()
    clock.now += 20 * 60.0
    with pytest.raises(RateLimited) as caught:
        fetch_catalog(
            cache=tmp_path, credentials=CREDENTIALS, transport=server, now=clock
        )
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
    with pytest.raises(AuthError, match="set SPACETRACK_IDENTITY and"):
        fetch_catalog(cache=tmp_path, transport=server, now=clock)
    assert server.requests == []
    assert not (tmp_path / "spacetrack" / "requests.json").exists()
    monkeypatch.setenv("SPACETRACK_IDENTITY", IDENTITY)
    monkeypatch.setenv("SPACETRACK_PASSWORD", PASSWORD)
    fetch_catalog(cache=tmp_path, transport=server, now=clock)
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
        fetch_catalog(
            cache=tmp_path, credentials=CREDENTIALS, transport=server, now=clock
        )
    assert server.calls == ["login"]
    # No query was sent, so none is recorded.
    assert not (tmp_path / "spacetrack" / "requests.json").exists()


def test_wrong_credentials_are_refused(tmp_path: Path, clock: Clock) -> None:
    server = FakeSpaceTrack({CATALOG.url: GP})
    wrong = Credentials(IDENTITY, "wrong")
    with pytest.raises(AuthError, match=r"\(HTTP 200\): Failed$"):
        fetch_catalog(cache=tmp_path, credentials=wrong, transport=server, now=clock)


def test_an_expired_session_is_renewed_once(tmp_path: Path, clock: Clock) -> None:
    server = FakeSpaceTrack({CATALOG.url: [LOGGED_OUT, GP]})
    catalog = fetch_catalog(
        cache=tmp_path, credentials=CREDENTIALS, transport=server, now=clock
    )
    assert server.calls == ["login", CATALOG.url, "login", CATALOG.url, "logout"]
    assert catalog.groups[0].omms == SAMPLE


def test_a_second_401_is_an_auth_error(tmp_path: Path, clock: Clock) -> None:
    server = FakeSpaceTrack({CATALOG.url: LOGGED_OUT})
    with pytest.raises(AuthError, match=r"HTTP 401 right after a login: .*logged in"):
        fetch_catalog(
            cache=tmp_path, credentials=CREDENTIALS, transport=server, now=clock
        )
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
        fetch_catalog(
            cache=tmp_path, credentials=CREDENTIALS, transport=server, now=clock
        )
    assert server.calls == ["login"]


def test_another_error_status_is_raised(tmp_path: Path, clock: Clock) -> None:
    body = b"<html>" + b"x" * 300 + b"</html>"
    server = FakeSpaceTrack({CATALOG.url: Response(503, body)})
    with pytest.raises(HttpStatusError) as caught:
        fetch_catalog(
            cache=tmp_path, credentials=CREDENTIALS, transport=server, now=clock
        )
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
        fetch_catalog(
            cache=tmp_path, credentials=CREDENTIALS, transport=server, now=clock
        )
    assert server.calls == ["login", CATALOG.url, "logout"]
    with pytest.raises(RateLimited):
        fetch_catalog(
            cache=tmp_path, credentials=CREDENTIALS, transport=server, now=clock
        )


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
        fetch_catalog(
            cache=tmp_path, credentials=CREDENTIALS, transport=server, now=clock
        )
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
    assert catalog.groups[0].omms == SAMPLE


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
        fetch_catalog(
            cache=tmp_path, credentials=CREDENTIALS, transport=server, now=clock
        )
    assert server.requests == []


def test_a_request_record_that_cannot_be_opened_stops_every_query(
    tmp_path: Path, clock: Clock
) -> None:
    (tmp_path / "spacetrack" / "requests.json").mkdir(parents=True)
    with pytest.raises(SourceFormatError, match="cannot read the Space-Track"):
        fetch_catalog(cache=tmp_path, transport=FakeSpaceTrack(), now=clock)


def test_closing_logs_out_once_and_ignores_a_failed_logout(
    tmp_path: Path, clock: Clock
) -> None:
    unused = FakeSpaceTrack()
    client_for(unused, tmp_path, clock).close()
    assert unused.requests == []
    for logout in (SourceUnavailable("timed out"), Response(500, b"oops")):
        server = FakeSpaceTrack({CATALOG.url: GP}, logout=logout)
        clock.now += HOUR
        client = client_for(server, tmp_path, clock)
        assert repr(client) == f"<SpaceTrackClient {tmp_path}/spacetrack>"
        client.fetch_catalog()
        assert repr(client) == f"<SpaceTrackClient {tmp_path}/spacetrack (logged in)>"
        client.close()
        client.close()
        assert server.calls == ["login", CATALOG.url, "logout"]


def test_the_password_is_never_shown(tmp_path: Path, clock: Clock) -> None:
    echo = f"bad form identity={IDENTITY}&password={PASSWORD}".encode()
    failures = [
        FakeSpaceTrack(
            {CATALOG.url: GP}, login=Response(200, b'{"Login":"%s"}' % echo)
        ),
        FakeSpaceTrack({CATALOG.url: GP}, login=Response(400, echo)),
        FakeSpaceTrack({CATALOG.url: Response(500, echo)}),
        FakeSpaceTrack(
            {CATALOG.url: Response(500, echo + b" violated your query rate limit")}
        ),
        FakeSpaceTrack({CATALOG.url: LOGGED_OUT}),
    ]
    for server in failures:
        clock.now += HOUR
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
        fetch_catalog(
            cache=tmp_path, credentials=CREDENTIALS, transport=server, now=clock
        )
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
        fetch_catalog(
            cache=tmp_path, credentials=CREDENTIALS, transport=server, now=clock
        )
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
    catalog = fetch_catalog(
        cache=tmp_path, credentials=CREDENTIALS, transport=transport, now=clock
    )
    assert server.calls == ["login", CATALOG.url, "logout"]
    # The query is recorded at the time it was sent, after the login.
    assert ledger(tmp_path) == {
        "format": 1,
        "queried": {"cdm_public": other, "gp": START + 30.0},
    }
    assert catalog.groups[0].retrieval.retrieved.unix == START + 30.0


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


# The urllib transport, on a fake HTTPS handler


class BrokenBody(io.BytesIO):
    """A body the connection drops while it is read."""

    def read(self, size: int | None = -1, /) -> bytes:
        raise http.client.IncompleteRead(b"")


class FakeHttps(urllib.request.BaseHandler):
    """Answers HTTPS requests from ``answers`` (status, body, headers) and
    records what was sent: method, URL, body and headers.

    It comes before urllib's own HTTPS handler, so that is never asked.
    """

    handler_order = 100

    def __init__(
        self, answers: dict[str, tuple[int, bytes | BrokenBody, str] | Exception]
    ) -> None:
        super().__init__()
        self.answers = answers
        self.sent: list[tuple[str, str, bytes | None, dict[str, str]]] = []

    def https_open(self, req: urllib.request.Request) -> urllib.response.addinfourl:
        data = req.data if isinstance(req.data, bytes) else None
        self.sent.append(
            (req.get_method(), req.full_url, data, dict(req.header_items()))
        )
        answer = self.answers[req.full_url]
        if isinstance(answer, Exception):
            raise answer
        status, body, headers = answer
        message = email.message_from_string(headers, _class=http.client.HTTPMessage)
        stream = io.BytesIO(body) if isinstance(body, bytes) else body
        response = urllib.response.addinfourl(stream, message, req.full_url, status)
        # urllib's error processor reads the reason phrase.
        response.msg = "fake"  # type: ignore[attr-defined]
        return response


def test_the_transport_keeps_the_session_cookie() -> None:
    handler = FakeHttps(
        {
            LOGIN_URL: (
                200,
                b'""',
                "Set-Cookie: chocolatechip=abc123; path=/; HttpOnly\n",
            ),
            CATALOG.url: (200, GP, ""),
        }
    )
    transport = UrllibTransport(handlers=[handler])
    form = urllib.parse.urlencode({"identity": IDENTITY, "password": PASSWORD})
    assert transport(Request("POST", LOGIN_URL, form.encode())) == Response(200, b'""')
    assert transport(Request("GET", CATALOG.url)) == Response(200, GP)
    (login_method, _, login_body, login_headers), query = handler.sent
    assert login_method == "POST"
    assert login_body == form.encode()
    assert login_headers["Content-type"] == "application/x-www-form-urlencoded"
    assert login_headers["User-agent"].startswith("keepout/")
    assert "Cookie" not in login_headers
    assert query[0] == "GET"
    assert query[3]["Cookie"] == "chocolatechip=abc123"
    assert "Content-type" not in query[3]
    assert [cookie.name for cookie in transport.cookies] == ["chocolatechip"]


def test_the_transport_returns_error_answers() -> None:
    handler = FakeHttps({CATALOG.url: (401, b'{"error":"log in"}', "")})
    transport = UrllibTransport(handlers=[handler])
    assert transport(Request("GET", CATALOG.url)) == Response(
        401, b'{"error":"log in"}'
    )


def test_an_error_body_that_cannot_be_read_is_empty() -> None:
    handler = FakeHttps({CATALOG.url: (500, BrokenBody(), "")})
    transport = UrllibTransport(handlers=[handler])
    assert transport(Request("GET", CATALOG.url)) == Response(500, b"")


def test_a_body_that_cannot_be_read_is_source_unavailable() -> None:
    handler = FakeHttps({CATALOG.url: (200, BrokenBody(), "")})
    transport = UrllibTransport(handlers=[handler])
    with pytest.raises(SourceUnavailable, match="IncompleteRead"):
        transport(Request("GET", CATALOG.url))


@pytest.mark.parametrize(
    "failure",
    [urllib.error.URLError("no route"), http.client.RemoteDisconnected("closed")],
)
def test_a_transport_failure_is_source_unavailable(failure: Exception) -> None:
    transport = UrllibTransport(handlers=[FakeHttps({CATALOG.url: failure})])
    with pytest.raises(SourceUnavailable, match=f"cannot reach {CATALOG.url}: "):
        transport(Request("GET", CATALOG.url))


def test_the_transport_sends_only_https() -> None:
    with pytest.raises(ValueError, match="not an HTTPS URL"):
        UrllibTransport()(Request("GET", "http://www.space-track.org/"))


def test_the_default_transport_is_offline_in_tests() -> None:
    with pytest.raises(AssertionError, match="must not access the network"):
        UrllibTransport()(Request("GET", LOGOUT_URL))


def test_a_client_has_a_urllib_transport_by_default(tmp_path: Path) -> None:
    client = SpaceTrackClient(CREDENTIALS, cache=tmp_path)
    assert isinstance(client._transport, UrllibTransport)
    assert client.ledger == tmp_path / "spacetrack" / "requests.json"
    assert spacetrack.PER_MINUTE == 20
    assert spacetrack.PER_HOUR == 200

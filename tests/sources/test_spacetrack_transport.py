import gzip
import http.client
import urllib.error
import urllib.parse

import pytest
from spacetrack_builders import IDENTITY, PASSWORD, FakeHttps, space_track_gp
from web_builders import BrokenBody

from keepout.errors import SourceUnavailable
from keepout.sources.spacetrack import (
    CATALOG,
    LOGIN_URL,
    Request,
    Response,
    UrllibTransport,
)

GP = space_track_gp()


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


@pytest.mark.parametrize("status", [200, 500])
def test_the_transport_asks_for_gzip_and_decompresses_it(status: int) -> None:
    body = gzip.compress(GP, mtime=0)
    handler = FakeHttps({CATALOG.url: (status, body, "Content-Encoding: gzip\n")})
    transport = UrllibTransport(handlers=[handler])
    assert transport(Request("GET", CATALOG.url)) == Response(status, GP)
    ((_, _, _, headers),) = handler.sent
    assert headers["Accept-encoding"] == "gzip"


def test_an_answer_sent_plain_is_kept_as_it_is() -> None:
    handler = FakeHttps({CATALOG.url: (200, GP, "")})
    assert UrllibTransport(handlers=[handler])(Request("GET", CATALOG.url)) == (
        Response(200, GP)
    )


def test_a_gzipped_answer_cut_off_is_source_unavailable() -> None:
    cut = gzip.compress(GP, mtime=0)[:20]
    handler = FakeHttps({CATALOG.url: (200, cut, "Content-Encoding: gzip\n")})
    with pytest.raises(SourceUnavailable, match="does not decompress"):
        UrllibTransport(handlers=[handler])(Request("GET", CATALOG.url))


def test_the_transport_sends_only_https() -> None:
    with pytest.raises(ValueError, match="not an HTTPS URL"):
        UrllibTransport()(Request("GET", "http://www.space-track.org/"))

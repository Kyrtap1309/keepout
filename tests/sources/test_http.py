import gzip
import http.client
import io
import urllib.error
from collections.abc import Callable
from importlib.metadata import version

import pytest
from web_builders import BrokenBody, FakeUrlopen, Reply

from keepout.errors import AuthError, HttpStatusError, SourceUnavailable
from keepout.sources import _http

URL = "https://example.invalid/x"


def test_the_user_agent_names_keepout_its_version_and_home_page() -> None:
    assert (
        f"keepout/{version('keepout')} (+https://github.com/Kyrtap1309/keepout)"
    ) == _http.USER_AGENT


def test_http_download_returns_body(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeUrlopen(monkeypatch, b"body")
    assert _http.http_download(URL) == b"body"


def test_every_request_names_keepout_in_the_user_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    web = FakeUrlopen(monkeypatch, b"body")
    _http.http_download(URL)
    _http.http_stream("https://example.invalid/y", io.BytesIO())
    assert web.urls == [URL, "https://example.invalid/y"]
    for request in web.requests:
        assert request.header_items() == [("User-agent", _http.USER_AGENT)]


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        (urllib.error.URLError("offline"), r"example\.invalid.*offline"),
        (BrokenBody(), r"example\.invalid"),
        (ConnectionResetError("reset by peer"), "reset by peer"),
    ],
    ids=["unreachable", "dropped mid-read", "connection reset"],
)
def test_http_download_wraps_network_errors(
    monkeypatch: pytest.MonkeyPatch, reply: Reply, message: str
) -> None:
    web = FakeUrlopen(monkeypatch, reply)
    with pytest.raises(SourceUnavailable, match=message):
        _http.http_download(URL, retry_delays=())
    assert web.urls == [URL]


def test_http_download_wraps_http_404(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeUrlopen(monkeypatch, 404)
    with pytest.raises(HttpStatusError, match="404") as caught:
        _http.http_download(URL)
    assert caught.value.status == 404
    assert caught.value.url == URL


def test_http_download_refuses_a_success_other_than_200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeUrlopen(monkeypatch, 204)
    with pytest.raises(HttpStatusError, match="HTTP 204 No Content") as caught:
        _http.http_download(URL)
    assert caught.value.status == 204


def test_http_stream_writes_the_body_in_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = bytes(range(256)) * 10
    monkeypatch.setattr(_http, "CHUNK_SIZE", 1000)
    FakeUrlopen(monkeypatch, body)
    chunks: list[bytes] = []

    class Collect:
        def write(self, data: bytes, /) -> object:
            chunks.append(data)
            return len(data)

    _http.http_stream(URL, Collect())
    assert b"".join(chunks) == body
    assert [len(chunk) for chunk in chunks] == [1000, 1000, 560]


def test_http_stream_wraps_mid_read_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeUrlopen(monkeypatch, BrokenBody())
    with pytest.raises(SourceUnavailable, match=r"example\.invalid"):
        _http.http_stream(URL, io.BytesIO())


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.invalid/x"])
def test_only_http_urls_are_opened(url: str) -> None:
    with pytest.raises(ValueError, match="not an HTTP or HTTPS URL"):
        _http.http_stream(url, io.BytesIO())


def test_a_request_names_keepout_and_adds_the_headers_given() -> None:
    request = _http.http_request(
        "https://example.org/x",
        data=b"a=1",
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert request.get_method() == "POST"
    assert request.data == b"a=1"
    assert dict(request.header_items()) == {
        "User-agent": _http.USER_AGENT,
        "Content-type": "application/x-www-form-urlencoded",
    }


@pytest.mark.parametrize(
    ("url", "schemes", "message"),
    [
        ("file:///etc/passwd", ("http", "https"), "not an HTTP or HTTPS URL"),
        ("http://example.org/", ("https",), "not an HTTPS URL"),
    ],
)
def test_a_request_refuses_other_schemes(
    url: str, schemes: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _http.http_request(url, schemes=schemes)


@pytest.mark.parametrize("error", [OSError("refused"), http.client.IncompleteRead(b"")])
def test_network_errors_become_source_unavailable(error: Exception) -> None:
    with (
        pytest.raises(SourceUnavailable, match=r"^cannot reach x: ") as raised,
        _http.network_errors("cannot reach x"),
    ):
        raise error
    assert raised.value.__cause__ is error


def test_other_errors_pass_through_network_errors() -> None:
    with pytest.raises(KeyError), _http.network_errors("cannot reach x"):
        raise KeyError("bug")


def test_gunzip_decompresses_gzip_data() -> None:
    assert _http.gunzip(gzip.compress(b"rows")) == b"rows"


@pytest.mark.parametrize("data", [b"not gzip", b"\x1f\x8b\x08truncated"])
def test_gunzip_of_something_else_is_none(data: bytes) -> None:
    assert _http.gunzip(data) is None


def test_a_dropped_download_is_tried_again(monkeypatch: pytest.MonkeyPatch) -> None:
    web = FakeUrlopen(monkeypatch, ConnectionResetError("reset"), b"body")
    assert _http.http_download(URL, retry_delays=(0.0,)) == b"body"
    assert web.urls == [URL, URL]
    assert web.waits == [0.0]


def _failing(*errors: Exception) -> Callable[[], str]:
    remaining = list(errors)

    def attempt() -> str:
        if remaining:
            raise remaining.pop(0)
        return "done"

    return attempt


@pytest.mark.parametrize(
    "error",
    [
        SourceUnavailable("offline"),
        HttpStatusError("busy", url="u", status=503),
        HttpStatusError("slow down", url="u", status=429),
    ],
    ids=["network", "server error", "too many requests"],
)
def test_retrying_waits_and_tries_a_transient_failure_again(error: Exception) -> None:
    waits: list[float] = []
    assert _http.retrying(
        _failing(error, error), delays=(1.0, 2.0), sleep=waits.append
    ) == ("done")
    assert waits == [1.0, 2.0]


@pytest.mark.parametrize(
    "error",
    [
        HttpStatusError("gone", url="u", status=404),
        HttpStatusError("unchanged", url="u", status=403),
        AuthError("refused"),
    ],
    ids=["not found", "forbidden", "login refused"],
)
def test_retrying_raises_a_lasting_failure_at_once(error: Exception) -> None:
    waits: list[float] = []
    with pytest.raises(type(error)):
        _http.retrying(_failing(error), delays=(1.0,), sleep=waits.append)
    assert waits == []


def test_retrying_raises_the_last_failure_when_attempts_run_out() -> None:
    waits: list[float] = []
    attempt = _failing(SourceUnavailable("first"), SourceUnavailable("second"))
    with pytest.raises(SourceUnavailable, match="second"):
        _http.retrying(attempt, delays=(1.0,), sleep=waits.append)
    assert waits == [1.0]

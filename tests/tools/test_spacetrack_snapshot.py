import gzip
import json
import urllib.parse
from pathlib import Path

import pytest
import snapshot_files
import spacetrack_snapshot as snapshot
from celestrak_builders import SAMPLE
from hypothesis import given
from hypothesis import strategies as st
from report_builders import assert_writes_then_checks
from snapshot_raws import (
    CDMS,
    CROSSING,
    GP,
    START,
    space_track,
    spacetrack_download,
)
from spacetrack_builders import CREDENTIALS, IDENTITY, PASSWORD, FakeSpaceTrack
from web_builders import digest

from keepout.errors import AuthError, HttpStatusError, SourceUnavailable
from keepout.omm import load
from keepout.sources.spacetrack import (
    CATALOG,
    HIDDEN,
    IDENTITY_VARIABLE,
    LOGIN_URL,
    LOGOUT_URL,
    PASSWORD_VARIABLE,
    PUBLIC_CDMS,
    Request,
    Response,
    read_public_cdms,
)

CALSPHERE, ISS, GOES, THEMIS, EXPRESS, SARAMAGO, STARLINK, DRAGON = SAMPLE
URLS = [LOGIN_URL, CATALOG.url, PUBLIC_CDMS.url, LOGOUT_URL]


@pytest.fixture
def server() -> FakeSpaceTrack:
    return space_track()


@pytest.fixture
def raw(tmp_path: Path, server: FakeSpaceTrack) -> Path:
    directory = tmp_path / "raw"
    spacetrack_download(directory, server, tmp_path / "cache")
    return directory


def build(raw: Path, data: Path, *options: str) -> int:
    return snapshot.main(["build", str(raw), "--data", str(data), *options])


def test_download_sends_each_request_once_and_keeps_the_answers(
    raw: Path, server: FakeSpaceTrack
) -> None:
    assert [request.url for request in server.requests] == URLS
    index = snapshot.INDEX.read(raw)
    assert [saved.url for saved in index] == URLS
    assert [saved.method for saved in index] == ["POST", "GET", "GET", "GET"]
    assert [saved.file for saved in index] == [
        "login.json",
        "gp.json",
        "cdm_public.json",
        "logout.json",
    ]
    bodies = [b'""', GP, CDMS, b'"Successfully logged out"']
    for saved, body in zip(index, bodies, strict=True):
        assert (raw / saved.file).read_bytes() == body
        assert saved.status == 200
        assert saved.sha256 == digest(body)
        assert saved.size == len(body)
        assert saved.retrieved == "2026-10-03T04:00:00.000Z"
        assert saved.redacted is False
    assert snapshot.saved_answer(raw, PUBLIC_CDMS.url) == CDMS


def test_download_records_the_queries_in_the_clients_cache(
    raw: Path, tmp_path: Path
) -> None:
    ledger = json.loads((tmp_path / "cache/spacetrack/requests.json").read_text())
    assert ledger == {"format": 1, "queried": {"cdm_public": START, "gp": START}}


def test_download_refuses_a_directory_with_a_download(
    raw: Path, server: FakeSpaceTrack, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="already holds a download"):
        spacetrack_download(raw, server, tmp_path / "other-cache")
    assert len(server.requests) == 4


def test_download_refuses_an_answer_from_the_cache(
    raw: Path, server: FakeSpaceTrack, tmp_path: Path
) -> None:
    again = tmp_path / "again"
    with pytest.raises(ValueError, match="answered from the cache"):
        spacetrack_download(again, server, tmp_path / "cache")
    assert len(server.requests) == 4
    assert snapshot.INDEX.read(again) == ()


def test_download_stops_at_an_error_and_keeps_the_answers(tmp_path: Path) -> None:
    server = FakeSpaceTrack({CATALOG.url: Response(500, b"maintenance")})
    raw = tmp_path / "raw"
    with pytest.raises(HttpStatusError, match="HTTP 500"):
        spacetrack_download(raw, server, tmp_path / "cache")
    assert server.calls == ["login", CATALOG.url, "logout"]
    index = snapshot.INDEX.read(raw)
    assert [(saved.url, saved.status) for saved in index] == [
        (LOGIN_URL, 200),
        (CATALOG.url, 500),
        (LOGOUT_URL, 200),
    ]
    assert (raw / "gp.json").read_bytes() == b"maintenance"


def test_download_hides_the_credentials_and_sends_no_second_login(
    tmp_path: Path,
) -> None:
    # A login answer quoting the account, then a session that does not hold:
    # the client would log in again, which the tool refuses.
    echo = f'{{"user": "{IDENTITY}", "note": "{PASSWORD}"}}'.encode()
    server = FakeSpaceTrack({CATALOG.url: GP}, login=Response(200, echo))
    raw = tmp_path / "raw"
    with pytest.raises(SourceUnavailable, match="was sent already"):
        spacetrack_download(raw, server, tmp_path / "cache")
    assert server.calls == ["login", CATALOG.url]
    login, gp = snapshot.INDEX.read(raw)
    assert login.redacted is True
    saved = (raw / login.file).read_text()
    assert saved == '{"user": "<hidden>", "note": "<hidden>"}'
    assert IDENTITY not in saved
    assert PASSWORD not in saved
    assert gp.status == 401


def test_download_reads_the_credentials_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(IDENTITY_VARIABLE, raising=False)
    monkeypatch.delenv(PASSWORD_VARIABLE, raising=False)
    with pytest.raises(AuthError, match=f"{IDENTITY_VARIABLE} and {PASSWORD_VARIABLE}"):
        snapshot.download(tmp_path / "raw", cache=tmp_path)
    assert not (tmp_path / "raw").exists()


def test_the_recorder_sends_only_the_four_urls(tmp_path: Path) -> None:
    recorder = snapshot.Recorder(FakeSpaceTrack(), tmp_path, CREDENTIALS)
    with pytest.raises(SourceUnavailable, match="does not send to"):
        recorder(Request("GET", f"{CATALOG.url}/limit/1"))
    assert recorder.sent == set()


@given(st.binary())
def test_scrub_keeps_bytes_without_credentials(body: bytes) -> None:
    clean = snapshot.scrub(body, CREDENTIALS)
    if IDENTITY.encode() not in body and PASSWORD.encode() not in body:
        assert clean == body


SECRET_FORMS = sorted(
    {
        form.encode()
        for secret in (IDENTITY, PASSWORD)
        for form in (
            secret,
            urllib.parse.quote(secret, safe=""),
            urllib.parse.quote_plus(secret),
        )
    }
)
"""The identity and password as typed, path- and form-encoded: what
:meth:`Credentials.redact` hides."""
NOT_PRINTABLE = st.binary().map(
    lambda data: bytes(byte for byte in data if not 0x20 <= byte <= 0x7E)
)
"""Bytes without printable ASCII, of which every secret form is made: no
part of a secret, and invalid UTF-8 as often as not."""


@given(st.binary(), st.sampled_from(SECRET_FORMS), st.binary())
def test_scrub_leaves_no_secret_form(
    prefix: bytes, secret: bytes, suffix: bytes
) -> None:
    clean = snapshot.scrub(prefix + secret + suffix, CREDENTIALS)
    for form in SECRET_FORMS:
        assert form not in clean


@given(NOT_PRINTABLE, st.sampled_from(SECRET_FORMS), NOT_PRINTABLE)
def test_scrub_keeps_the_bytes_around_a_secret(
    prefix: bytes, secret: bytes, suffix: bytes
) -> None:
    clean = snapshot.scrub(prefix + secret + suffix, CREDENTIALS)
    assert clean == prefix + HIDDEN.encode() + suffix


def test_main_download_reports_the_answers(
    tmp_path: Path,
    server: FakeSpaceTrack,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(IDENTITY_VARIABLE, IDENTITY)
    monkeypatch.setenv(PASSWORD_VARIABLE, PASSWORD)
    monkeypatch.setattr(snapshot, "UrllibTransport", lambda: server)
    raw = tmp_path / "raw"
    assert snapshot.main(["download", str(raw), "--cache", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert f"POST {LOGIN_URL}: HTTP 200, 2 bytes in login.json\n" in out
    assert f"GET {CATALOG.url}: HTTP 200, {len(GP)} bytes in gp.json\n" in out
    assert out.endswith("10 element sets, 3 public CDMs\n")
    assert IDENTITY not in out
    assert PASSWORD not in out


def test_main_download_keeps_an_answer_keepout_cannot_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    echo = Response(200, f'"welcome {IDENTITY}"'.encode())
    server = FakeSpaceTrack(
        {CATALOG.url: GP, PUBLIC_CDMS.url: b'{"error": "later"}'},
        login=echo,
        login_opens_session=True,
    )
    monkeypatch.setenv(IDENTITY_VARIABLE, IDENTITY)
    monkeypatch.setenv(PASSWORD_VARIABLE, PASSWORD)
    monkeypatch.setattr(snapshot, "UrllibTransport", lambda: server)
    raw = tmp_path / "raw"
    assert snapshot.main(["download", str(raw), "--cache", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "in login.json (<hidden> credentials)\n" in out
    assert "saved, but keepout cannot read it: Space-Track did not send an array" in out
    assert (raw / "cdm_public.json").read_bytes() == b'{"error": "later"}'


def test_replay_answers_from_the_saved_answers(raw: Path) -> None:
    replay = snapshot.Replay(raw)
    assert replay(Request("GET", CATALOG.url)) == Response(200, GP)
    assert snapshot.replay_public_cdms(raw) == read_public_cdms(CDMS)


def test_replay_refuses_a_request_not_saved(raw: Path) -> None:
    replay = snapshot.Replay(raw)
    with pytest.raises(SourceUnavailable, match="is not saved"):
        replay(Request("GET", f"{CATALOG.url}/limit/1"))


def test_replay_needs_a_download(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no answers saved"):
        snapshot.Replay(tmp_path)


def test_a_saved_answer_needs_a_download(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no answer to"):
        snapshot.saved_answer(tmp_path, CATALOG.url)


def test_a_changed_answer_is_refused(raw: Path) -> None:
    (raw / "gp.json").write_bytes(b"[]")
    with pytest.raises(ValueError, match="digest"):
        snapshot.replay_catalog(raw)


def test_build_writes_the_data_then_checks_it(raw: Path, tmp_path: Path) -> None:
    data = tmp_path / "data"
    assert_writes_then_checks(snapshot.main, ["build", str(raw), "--data", str(data)])
    assert snapshot.main(["verify", "--data", str(data)]) == 0
    assert load(data / snapshot.SNAPSHOT) == tuple(
        sorted([*SAMPLE, CROSSING], key=lambda omm: omm.catalog_number)
    )
    manifest = json.loads((data / snapshot_files.MANIFEST).read_text(encoding="utf-8"))
    assert manifest["download"] == snapshot.INDEX.read(raw)[1].record()
    assert manifest["catalog"] == {
        "rows": 10,
        "duplicates": 1,
        "objects": 9,
        "six_digit": 3,
        "regimes": {"LEO": 6, "MEO": 1, "GEO": 1, "HEO": 1},
        "oldest_epoch": THEMIS.epoch.isot,
        "newest_epoch": GOES.epoch.isot,
    }
    assert set(manifest["files"]) == {snapshot.SNAPSHOT, snapshot.NOTICE}
    for name, entry in manifest["files"].items():
        stored = (data / name).read_bytes()
        assert entry == {
            "bytes": len(stored),
            "sha256": digest(stored),
        }


def test_nothing_of_the_public_cdms_is_committed(raw: Path, tmp_path: Path) -> None:
    data = tmp_path / "data"
    assert build(raw, data) == 0
    written = b"".join(
        gzip.decompress(path.read_bytes())
        if path.suffix == ".gz"
        else path.read_bytes()
        for path in data.iterdir()
    )
    assert b"cdm_public" not in written
    assert b"CALSPHERE 1" in written
    assert b"123456" not in written


def test_build_refuses_a_catalog_csv_cannot_hold(
    raw: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(snapshot_files, "loads_csv", lambda text: ())
    with pytest.raises(ValueError, match="does not read back"):
        build(raw, tmp_path / "data")


def test_notice_cites_the_source_and_date(raw: Path, tmp_path: Path) -> None:
    data = tmp_path / "data"
    assert build(raw, data) == 0
    notice = (data / snapshot.NOTICE).read_text(encoding="utf-8")
    assert notice.startswith(
        "Data: USSPACECOM/18 SDS via Space-Track.org (www.space-track.org),\n"
        "retrieved on 2026-10-03.\n\n"
        "Retrieved 2026-10-03 04:00:00 UTC by tools/spacetrack_snapshot.py with\n"
    )
    assert "not covered by keepout's MIT licence" in notice


def test_compare_finds_keepouts_approach_of_each_pair(raw: Path) -> None:
    cdms = snapshot.replay_public_cdms(raw)
    comparison = snapshot.compare(cdms, snapshot.replay_catalog(raw).omms)
    assert comparison.outside_catalog == 1
    passing, apart = comparison.matches
    assert passing.cdm == cdms[0]
    assert passing.tca_difference == pytest.approx(-2.0, abs=1e-3)
    assert passing.range_at_tca == pytest.approx(42195.4, abs=0.1)
    assert apart == snapshot.Match(cdms[2], None)
    assert (apart.tca_difference, apart.range_at_tca) == (None, None)
    summary = snapshot.summarise(comparison)
    assert summary["cdms"] == 3
    assert (summary["compared"], summary["found"]) == (2, 1)
    assert summary["tca_difference"]["median"] == pytest.approx(2.0, abs=1e-3)
    assert summary["range_difference"]["p90"] == pytest.approx(195.4, abs=0.1)
    assert summary["range_ratio"]["median"] == pytest.approx(42000 / 42195.4, rel=1e-5)


def test_compare_in_processes_gives_the_same(raw: Path) -> None:
    cdms = snapshot.replay_public_cdms(raw)
    omms = snapshot.replay_catalog(raw).omms
    one, two = snapshot.compare(cdms, omms), snapshot.compare(cdms, omms, workers=2)
    assert [(m.cdm, m.tca_difference, m.range_at_tca) for m in two.matches] == [
        (m.cdm, m.tca_difference, m.range_at_tca) for m in one.matches
    ]
    assert two.outside_catalog == one.outside_catalog


def test_main_compare_prints_the_summary(
    raw: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert snapshot.main(["compare", str(raw)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert (summary["cdms"], summary["outside_catalog"], summary["found"]) == (3, 1, 1)


def test_the_spread_of_nothing_is_unknown() -> None:
    assert snapshot.spread([]) == {"median": None, "p90": None}


def test_the_spread_is_the_median_and_90th_percentile() -> None:
    assert snapshot.spread([3.0, 1.0, 2.0]) == {"median": 2.0, "p90": 3.0}

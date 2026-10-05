import gzip
import hashlib
import io
import math
import zipfile
from pathlib import Path

import numpy as np
import pytest

from keepout.errors import ChecksumMismatch, KelvinsFormatError
from keepout.risk import Action, assess
from keepout.sources import _cache, kelvins
from keepout.sources.kelvins import FEATURES, KelvinsEvent

EXCERPT = Path(__file__).parent / "data" / "kelvins_excerpt.csv"
DAY = 86400.0


@pytest.fixture(scope="module")
def events() -> dict[int, KelvinsEvent]:
    return {event.event_id: event for event in kelvins.load(EXCERPT)}


def test_events_are_read_in_order(events: dict[int, KelvinsEvent]) -> None:
    assert list(events) == [30, 78, 97, 505, 907, 5490]
    assert [len(event) for event in events.values()] == [4, 3, 7, 3, 4, 3]
    assert [event.mission_id for event in events.values()] == [5, 1, 7, 1, 6, 16]
    assert events[78].object_types == ("UNKNOWN",) * 3


def test_features_are_numbers_with_nan_for_empty_cells(
    events: dict[int, KelvinsEvent],
) -> None:
    assert events[30].features.shape == (4, len(FEATURES))
    assert len(FEATURES) == 100
    assert np.isnan(events[907].column("F10")).any()
    assert np.isnan(events[5490].column("c_sigma_rdot")).all()
    assert events[78].column("miss_distance")[0] > 0.0


def test_features_are_read_only(events: dict[int, KelvinsEvent]) -> None:
    with pytest.raises(ValueError, match="read-only"):
        events[30].features[0, 0] = 1.0


def test_unknown_column(events: dict[int, KelvinsEvent]) -> None:
    with pytest.raises(KeyError, match="'event_id' is not a Kelvins feature"):
        events[30].column("event_id")


def test_time_to_tca_is_in_seconds(events: dict[int, KelvinsEvent]) -> None:
    assert events[505].time_to_tca[-1] == pytest.approx(-0.00979340277777778 * DAY)


def test_observations_are_esas_pc(events: dict[int, KelvinsEvent]) -> None:
    seen = events[78].observations()
    assert [o.message_id for o in seen] == ["78/0", "78/1", "78/2"]
    assert seen[2].pc == pytest.approx(10**-4.6311554931741785)
    assert (seen[2].source, seen[2].kind) == ("ESA (Kelvins)", "estimate")
    assert seen[2].time_to_tca == pytest.approx(6.04415412037037 * DAY)


def test_the_floor_is_below_floor(events: dict[int, KelvinsEvent]) -> None:
    seen = events[30].observations()
    assert {(o.pc, o.kind) for o in seen} == {(1e-30, "below_floor")}


@pytest.mark.parametrize(
    ("event_id", "action", "direction"),
    [
        (30, Action.IGNORE, "steady"),
        (78, Action.MONITOR, "rising"),
        (97, Action.IGNORE, "steady"),
        (505, Action.IGNORE, "steady"),
        (907, Action.IGNORE, "steady"),
        (5490, Action.MANOEUVRE, "steady"),
    ],
)
def test_assessment_of_the_excerpt(
    events: dict[int, KelvinsEvent], event_id: int, action: Action, direction: str
) -> None:
    assessment = assess(events[event_id].observations())
    assert (assessment.decision.action, assessment.trend.direction) == (
        action,
        direction,
    )


def test_a_gzipped_table_is_read_too(tmp_path: Path) -> None:
    packed = tmp_path / "table.csv.gz"
    packed.write_bytes(gzip.compress(EXCERPT.read_bytes()))
    assert [e.event_id for e in kelvins.load(packed)] == [30, 78, 97, 505, 907, 5490]


def excerpt_with(tmp_path: Path, old: str, new: str) -> Path:
    text = EXCERPT.read_text(encoding="utf-8")
    assert old in text, old
    changed = tmp_path / "changed.csv"
    changed.write_text(text.replace(old, new, 1), encoding="utf-8")
    return changed


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("event_id,time_to_tca", "event,time_to_tca", r"missing \['event_id'\]"),
        ("30,1.0878610185185187,5,", "30,1.08,5,1,", "line 2: 104 values"),
        ("30,1.0878610185185187,5,", "30,soon,5,", "time_to_tca is not a number"),
        ("30,1.0878610185185187,5,", "30.5,1.0878610185185187,5,", "event_id must be"),
        ("30,1.0878610185185187,5,", "30,,5,", "line 2: time_to_tca is missing"),
        (",-9.683610248926804,", ",0.5,", "line 6: risk .* is above 0"),
        ("78,6.6573216087962965,1,", "78,6.6573216087962965,2,", "changes mission"),
        ("97,0.013183530092592592,", "30,0.013183530092592592,", "not contiguous"),
    ],
)
def test_malformed_tables_are_refused(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    with pytest.raises(KelvinsFormatError, match=message):
        kelvins.load(excerpt_with(tmp_path, old, new))


def test_an_empty_file_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(KelvinsFormatError, match="header is not"):
        kelvins.load(empty)


@pytest.mark.parametrize(
    ("features", "types", "message"),
    [
        (np.zeros((2, 3)), ("DEBRIS", "DEBRIS"), r"shape \(n, 100\)"),
        (np.zeros((2, 100)), ("DEBRIS",), "2 feature rows for 1 object types"),
        (np.zeros((0, 100)), (), "need at least one CDM"),
    ],
)
def test_invalid_events_are_refused(
    features: np.ndarray, types: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        KelvinsEvent(1, 1, features, types)


TABLE = b"raw table bytes"


def archive(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)
    return buffer.getvalue()


class FakeZenodo:
    """Serves ``body`` for the archive URL and counts the downloads."""

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.calls = 0

    def __call__(self, url: str, sink: _cache.Sink) -> None:
        assert url == kelvins.ARCHIVE_URL
        self.calls += 1
        sink.write(self.body)


def pin(monkeypatch: pytest.MonkeyPatch, body: bytes) -> None:
    monkeypatch.setattr(kelvins, "ARCHIVE_SHA256", hashlib.sha256(body).hexdigest())
    monkeypatch.setattr(kelvins, "RAW_SHA256", hashlib.sha256(TABLE).hexdigest())


def folder(root: Path) -> Path:
    return root / "kelvins" / kelvins.ZENODO_RECORD


def test_fetch_raw_extracts_the_table_and_drops_the_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = archive({kelvins.RAW_MEMBER: TABLE, "other.csv": b"x"})
    pin(monkeypatch, body)
    zenodo = FakeZenodo(body)
    path = kelvins.fetch_raw(cache=tmp_path, download=zenodo)
    assert path == folder(tmp_path) / "raw_data_2015-2019.gz"
    assert path.read_bytes() == TABLE
    assert [p.name for p in folder(tmp_path).iterdir()] == [path.name]
    assert kelvins.fetch_raw(cache=tmp_path, download=zenodo) == path
    assert zenodo.calls == 1


def test_fetch_raw_reuses_a_verified_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = archive({kelvins.RAW_MEMBER: TABLE})
    pin(monkeypatch, body)
    folder(tmp_path).mkdir(parents=True)
    (folder(tmp_path) / "dataset.zip").write_bytes(body)
    zenodo = FakeZenodo(b"unused")
    assert kelvins.fetch_raw(cache=tmp_path, download=zenodo).read_bytes() == TABLE
    assert zenodo.calls == 0


def test_fetch_raw_drops_an_archive_left_beside_a_verified_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = archive({kelvins.RAW_MEMBER: TABLE})
    pin(monkeypatch, body)
    folder(tmp_path).mkdir(parents=True)
    table = folder(tmp_path) / "raw_data_2015-2019.gz"
    table.write_bytes(TABLE)
    (folder(tmp_path) / "dataset.zip").write_bytes(body)
    zenodo = FakeZenodo(b"unused")
    assert kelvins.fetch_raw(cache=tmp_path, download=zenodo) == table
    assert [p.name for p in folder(tmp_path).iterdir()] == [table.name]
    assert zenodo.calls == 0


def test_fetch_raw_refuses_a_tampered_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pin(monkeypatch, archive({kelvins.RAW_MEMBER: TABLE}))
    with pytest.raises(ChecksumMismatch, match="expected sha256"):
        kelvins.fetch_raw(cache=tmp_path, download=FakeZenodo(b"tampered"))
    assert list(folder(tmp_path).iterdir()) == []


def test_fetch_raw_refuses_a_tampered_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = archive({kelvins.RAW_MEMBER: b"other table"})
    pin(monkeypatch, body)
    with pytest.raises(ChecksumMismatch, match=r"raw_data_2015-2019\.gz: expected"):
        kelvins.fetch_raw(cache=tmp_path, download=FakeZenodo(body))
    assert [p.name for p in folder(tmp_path).iterdir()] == ["dataset.zip"]


@pytest.mark.parametrize(
    "body", [archive({"elsewhere.gz": TABLE}), b"not a zip file"], ids=["member", "zip"]
)
def test_fetch_raw_refuses_an_archive_without_the_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: bytes
) -> None:
    pin(monkeypatch, body)
    with pytest.raises(KelvinsFormatError, match="no Collision Avoidance"):
        kelvins.fetch_raw(cache=tmp_path, download=FakeZenodo(body))


def test_fetch_raw_uses_the_cache_dir_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = archive({kelvins.RAW_MEMBER: TABLE})
    pin(monkeypatch, body)
    monkeypatch.setenv("KEEPOUT_CACHE_DIR", str(tmp_path))
    path = kelvins.fetch_raw(download=FakeZenodo(body))
    assert path.is_relative_to(tmp_path)


def test_load_fetches_the_table_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(kelvins, "fetch_raw", lambda: EXCERPT)
    assert len(kelvins.load()) == 6


def test_no_nan_in_the_kept_time_and_risk(events: dict[int, KelvinsEvent]) -> None:
    for event in events.values():
        assert not any(map(math.isnan, event.column("risk")))
        assert not any(map(math.isnan, event.time_to_tca))

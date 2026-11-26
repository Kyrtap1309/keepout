"""The dashboard's CDM upload: every file parsed first, nothing ingested on
any error, then a ``refresh`` job per event touched."""

import gzip
import re
from collections.abc import Iterator
from pathlib import Path

import dashboard_seed as seed
import pytest
from builders import job_of, leo_crossing
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import keepout.cdm
from keepout.cdm import kvn
from keepout.dashboard.upload import (
    MAX_FILE_BYTES,
    MAX_FILES,
    UploadError,
    UploadResult,
    ingest_upload,
)
from keepout.errors import StoreError
from keepout.settings import load_settings
from keepout.store import JobKind, JobQueue, JobStatus, Store, open_store

SETTINGS = load_settings(seed.SETTINGS)


@pytest.fixture
def seed_store(seed_copy: Path) -> Iterator[Store]:
    """A writable copy of the seed store, opened."""
    with open_store(seed_copy, create=False) as opened:
        yield opened


def _fresh() -> bytes:
    """A CDM the seed does not hold, as KVN."""
    return kvn.dumps(leo_crossing().cdms[0]).encode()


def _cdm_count(store: Store) -> int:
    """How many CDMs the store holds."""
    page = store.summaries(
        SETTINGS, as_of=seed.seed_as_of(), include_past=True, limit=200
    )
    return sum(item.cdm_count for item in page.items)


def _active_jobs(store: Store) -> list[int]:
    """The ids of the store's queued jobs."""
    queue = JobQueue(store)
    return [job.id for job in queue.jobs() if job.status is JobStatus.QUEUED]


def _nothing_ingested(store: Store, result: UploadResult, before: int) -> None:
    assert result.report is None
    assert result.jobs == ()
    assert _cdm_count(store) == before
    assert _active_jobs(store) == []


def test_upload_rejects_large_file(seed_store: Store) -> None:
    before = _cdm_count(seed_store)
    big = b"x" * (MAX_FILE_BYTES + 1)
    result = ingest_upload(
        seed_store, SETTINGS, [("new.kvn", _fresh()), ("big.kvn", big)]
    )
    assert result.errors == (("big.kvn", f"larger than {MAX_FILE_BYTES} bytes"),)
    _nothing_ingested(seed_store, result, before)


def test_upload_takes_what_streamlit_takes(seed_store: Store) -> None:
    # Streamlit's uploader counts 2^20 bytes per MB: a file of 1,020,000
    # bytes passes it, so it is not refused here as too large.
    big = b"x" * 1_020_000
    result = ingest_upload(seed_store, SETTINGS, [("big.kvn", big)])
    assert not any("larger than" in message for _, message in result.errors)


def test_upload_rejects_a_gzip_bomb(seed_store: Store) -> None:
    before = _cdm_count(seed_store)
    bomb = gzip.compress(b" " * (MAX_FILE_BYTES + 1))
    assert len(bomb) < MAX_FILE_BYTES
    result = ingest_upload(seed_store, SETTINGS, [("bomb.kvn.gz", bomb)])
    assert result.errors == (
        ("bomb.kvn.gz", f"larger than {MAX_FILE_BYTES} bytes once decompressed"),
    )
    _nothing_ingested(seed_store, result, before)


def test_upload_reads_gzip(seed_store: Store) -> None:
    result = ingest_upload(
        seed_store, SETTINGS, [("new.kvn.gz", gzip.compress(_fresh()))]
    )
    assert result.errors == ()
    assert result.report is not None
    assert len(result.report.added) == 1


def test_upload_rejects_non_cdm(seed_store: Store) -> None:
    before = _cdm_count(seed_store)
    result = ingest_upload(seed_store, SETTINGS, [("hello.txt", b"hello")])
    assert [name for name, _ in result.errors] == ["hello.txt"]
    assert "expected 'KEYWORD = value'" in result.errors[0][1]
    _nothing_ingested(seed_store, result, before)


def test_upload_rejects_non_utf8(seed_store: Store) -> None:
    result = ingest_upload(seed_store, SETTINGS, [("bin.kvn", b"\xff\xfe\x00")])
    assert [name for name, _ in result.errors] == ["bin.kvn"]
    assert result.errors[0][1].startswith("not UTF-8 text: ")


def test_upload_conflicting_message_id(seed_store: Store) -> None:
    before = _cdm_count(seed_store)
    first = seed.seed_files()[0]
    changed = _with_miss_distance(first.read_text(encoding="utf-8"))
    result = ingest_upload(
        seed_store,
        SETTINGS,
        [("new.kvn", _fresh()), (first.name, changed.encode())],
    )
    assert [name for name, _ in result.errors] == [first.name]
    assert "is stored with other content" in result.errors[0][1]
    _nothing_ingested(seed_store, result, before)


def test_upload_too_many_files(seed_store: Store) -> None:
    before = _cdm_count(seed_store)
    files = [(f"{i}.kvn", _fresh()) for i in range(MAX_FILES + 1)]
    result = ingest_upload(seed_store, SETTINGS, files)
    assert result.errors == (
        UploadError(None, f"at most {MAX_FILES} files per upload, not {MAX_FILES + 1}"),
    )
    _nothing_ingested(seed_store, result, before)


def test_upload_ingests_and_queues_refresh(seed_store: Store) -> None:
    before = _cdm_count(seed_store)
    result = ingest_upload(seed_store, SETTINGS, [("new.kvn", _fresh())])
    assert result.errors == ()
    assert result.report is not None
    assert len(result.report.added) == 1
    assert _cdm_count(seed_store) == before + 1
    (job_id,) = result.jobs
    job = job_of(seed_store, job_id)
    assert job.kind is JobKind.REFRESH
    assert job.event_id == result.report.events[0]
    assert _active_jobs(seed_store) == [job_id]


def test_upload_ingests_though_a_job_cannot_be_queued(
    seed_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    def locked(*_: object) -> None:
        raise StoreError("database is locked")

    monkeypatch.setattr(JobQueue, "submit_refresh", locked)
    before = _cdm_count(seed_store)
    result = ingest_upload(seed_store, SETTINGS, [("new.kvn", _fresh())])
    assert result.report is not None
    assert len(result.report.added) == 1
    assert _cdm_count(seed_store) == before + 1
    (event_id,) = result.report.events
    assert result.errors == (
        UploadError(None, f"event {event_id}: database is locked"),
    )
    assert result.jobs == ()


def test_upload_of_stored_cdms_queues_nothing(seed_store: Store) -> None:
    files = [(file.name, file.read_bytes()) for file in seed.seed_files()[:2]]
    result = ingest_upload(seed_store, SETTINGS, files)
    assert result.errors == ()
    assert result.report is not None
    assert result.report.added == ()
    assert len(result.report.already_stored) == 2
    # Their Pc is cached: nothing to refresh.
    assert result.jobs == ()


def test_an_empty_upload_does_nothing(seed_store: Store) -> None:
    assert ingest_upload(seed_store, SETTINGS, []) == UploadResult(None, (), ())


def test_upload_blames_the_file_whose_message_id_is_named(seed_store: Store) -> None:
    # A MESSAGE_ID that is part of another's is not the one an error names.
    first = seed.seed_files()[0]
    text = first.read_text(encoding="utf-8")
    stored = keepout.cdm.loads(text).message_id
    changed = _with_miss_distance(text)
    prefix = _with_id(_fresh().decode(), stored[:-4])
    result = ingest_upload(
        seed_store,
        SETTINGS,
        [("prefix.kvn", prefix.encode()), (first.name, changed.encode())],
    )
    assert [name for name, _ in result.errors] == [first.name]


def _with_id(text: str, message_id: str) -> str:
    """``text`` with its ``MESSAGE_ID`` set to ``message_id``."""
    line = re.compile(r"^MESSAGE_ID\s*=.*$", flags=re.MULTILINE)
    return line.sub(f"MESSAGE_ID = {message_id}", text)


def _with_miss_distance(text: str) -> str:
    """``text`` with its ``MISS_DISTANCE`` changed to 1 m."""
    line = re.compile(r"^(MISS_DISTANCE\s*=\s*)\S+", flags=re.MULTILINE)
    changed = line.sub(r"\g<1>1.0", text)
    assert changed != text
    return changed


def _check(result: UploadResult) -> None:
    """An upload's outcome is one of: nothing ingested, with errors naming
    their files; or ingested, with only errors of no file (a refresh job
    that could not be queued)."""
    if result.report is None:
        assert result.errors
        assert result.jobs == ()
    else:
        assert all(error.file is None for error in result.errors)


FUZZ = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@FUZZ
@given(data=st.binary(max_size=4096))
def test_any_upload_is_ingested_or_refused_with_its_errors(
    seed_store: Store, data: bytes
) -> None:
    _check(ingest_upload(seed_store, SETTINGS, [("fuzz.kvn", data)]))


SEED_LINES = seed.seed_files()[0].read_text(encoding="utf-8").splitlines()


@st.composite
def mutated_cdms(draw: st.DrawFn) -> bytes:
    """A seed CDM, its MESSAGE_ID made new, with lines deleted, duplicated
    or with their value changed."""
    lines = _with_id("\n".join(SEED_LINES), f"FUZZ-{draw(st.uuids())}").splitlines()
    for _ in range(draw(st.integers(0, 4))):
        i = draw(st.integers(0, len(lines) - 1))
        action = draw(st.sampled_from(["delete", "duplicate", "value"]))
        if action == "delete":
            del lines[i]
        elif action == "duplicate":
            lines.insert(i, lines[i])
        else:
            keyword = lines[i].split("=", 1)[0]
            value = draw(
                st.one_of(
                    st.text(max_size=20).filter(
                        lambda v: "\n" not in v and "\r" not in v
                    ),
                    st.floats(allow_nan=True).map(str),
                    st.integers().map(str),
                )
            )
            lines[i] = f"{keyword}= {value}"
        if not lines:
            break
    return "\n".join(lines).encode()


@FUZZ
@given(data=mutated_cdms())
def test_upload_fuzz_mutated_cdm(seed_store: Store, data: bytes) -> None:
    _check(ingest_upload(seed_store, SETTINGS, [("mutated.kvn", data)]))


def test_mutated_cdms_reach_ingest(seed_store: Store) -> None:
    # Without mutations, the strategy's CDM is new and ingested.
    text = _with_id("\n".join(SEED_LINES), "FUZZ-plain")
    result = ingest_upload(seed_store, SETTINGS, [("plain.kvn", text.encode())])
    assert result.report is not None
    assert len(result.report.added) == 1

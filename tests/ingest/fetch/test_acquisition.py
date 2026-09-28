"""D46's acquisition directory, journal, immutable record and verification.

Pages are written here the way ``ingest.cli.cache_page`` writes them -- gzipped
canonical JSON named by its digest -- without importing ``ingest.cli``, whose
scipy import would keep this module out of the application job. The two digests
are held equal in ``tests/ingest/test_cli.py``. The directory's conftest closes
the network.
"""

import gzip
import hashlib
import json
from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from ingest.fetch.acquisition import (
    ACQUISITIONS_ROOT,
    JOURNAL_NAME,
    RECORD_NAME,
    REQUIRED_KEYS,
    Acquisition,
    AcquisitionError,
    AcquisitionIncomplete,
    AcquisitionIntegrityError,
    acquisition_dir,
    current_commit,
    is_complete,
    journal_path,
    record_path,
    verify_acquisition,
)
from ingest.fetch.canonical import canonical_bytes, format_timestamp, page_digest
from ingest.fetch.http import ClientIdentity, RequestRecord
from ingest.fetch.registry import FETCHERS

SOURCE = "nyc311"
START, END = date(2024, 1, 1), date(2024, 1, 3)
COMMIT = "a" * 40
CLIENT = ClientIdentity(user_agent="Sentinel-test/0", library="fake", library_version="0")
NOW = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)


def cache(directory, page, source=SOURCE):
    """Store a page as ``cache_page`` does: gzipped canonical JSON under its digest."""
    digest = page_digest(page)
    folder = directory / source
    folder.mkdir(parents=True, exist_ok=True)
    with gzip.open(folder / f"{digest}.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(page, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return digest


def request(n, retrieved="2026-09-28T11:00:00.000000+00:00"):
    return RequestRecord(
        url="https://example.test/api",
        params=(("day", f"2024-01-0{n}"),),
        retrieved_at=retrieved,
        status=200,
        response_sha256=hashlib.sha256(str(n).encode()).hexdigest(),
        response_bytes=n,
    )


def opened(directory, **overrides):
    arguments = {
        "source": SOURCE,
        "start": START,
        "end": END,
        "resolved_start": datetime(2024, 1, 1, 5, tzinfo=UTC),
        "resolved_end": datetime(2024, 1, 4, 4, 59, 59, 999999, tzinfo=UTC),
        "client": CLIENT,
        "sentinel_commit": COMMIT,
        "now": lambda: NOW,
    }
    arguments.update(overrides)
    return Acquisition(directory, **arguments)


def slice_pages(n):
    return [
        {"unique_key": f"{n}-{i}", "created_date": f"2024-01-0{n}T09:00:00.000"} for i in range(2)
    ]


def complete_three_slices(directory):
    acquisition = opened(directory)
    for n in (1, 2, 3):
        digest = cache(directory, slice_pages(n))
        acquisition.record_slice(
            f"2024-01-0{n}", requests=[request(n)], pages=[digest], verification={"rows": 2}
        )
    return acquisition.complete({})


# --- canonical bytes -----------------------------------------------------------------


def test_canonical_bytes_sort_keys_drop_whitespace_keep_utf8_and_end_in_one_line_feed():
    data = canonical_bytes({"b": [1, {"z": None, "a": True}], "a": "café ✓"})
    assert data == '{"a":"café ✓","b":[1,{"a":true,"z":null}]}\n'.encode()
    assert not data.startswith(b"\xef\xbb\xbf")
    assert data.count(b"\n") == 1 and data.endswith(b"\n")


@pytest.mark.parametrize("value", [1.5, {"a": 0.0}, [1, [float("nan")]], {"a": float("inf")}])
def test_canonical_bytes_refuse_every_floating_point_value(value):
    with pytest.raises(ValueError):
        canonical_bytes(value)


def test_timestamps_are_utc_with_six_fractional_digits():
    assert format_timestamp(datetime(2026, 9, 28, 7, 0, tzinfo=UTC)) == (
        "2026-09-28T07:00:00.000000+00:00"
    )
    ist = timezone(timedelta(hours=5, minutes=30))
    assert format_timestamp(datetime(2026, 9, 28, 12, 30, 0, 5, tzinfo=ist)) == (
        "2026-09-28T07:00:00.000005+00:00"
    )
    with pytest.raises(ValueError):
        format_timestamp(datetime(2026, 9, 28))


def test_the_page_digest_ignores_key_order():
    assert page_digest({"b": 1, "a": [2]}) == page_digest({"a": [2], "b": 1})


# --- the directory ---------------------------------------------------------------------


def test_the_acquisition_directory_is_per_source_and_window():
    assert ACQUISITIONS_ROOT.parts == ("data", "acquisitions")
    assert acquisition_dir("cfpb", date(2024, 1, 1), date(2025, 12, 31)).as_posix() == (
        "data/acquisitions/cfpb/2024-01-01_2025-12-31"
    )


def test_task_23_registers_no_source_fetcher():
    assert FETCHERS == {}


def test_the_commit_is_read_from_git_or_is_none(tmp_path):
    commit = current_commit()
    assert commit is None or (len(commit) == 40 and all(c in "0123456789abcdef" for c in commit))
    assert current_commit(tmp_path) is None


@pytest.mark.parametrize("commit", [None, "", "abc", "A" * 40, "g" * 40])
def test_an_undeterminable_commit_means_the_acquisition_does_not_start(tmp_path, commit):
    with pytest.raises(AcquisitionError, match="does not start"):
        opened(tmp_path / "acq", sentinel_commit=commit)
    assert not (tmp_path / "acq").exists()


# --- the record ------------------------------------------------------------------------


def test_the_record_is_canonical_and_its_id_is_the_sha256_of_its_bytes(tmp_path):
    directory = tmp_path / "acq"
    acquisition_id = complete_three_slices(directory)
    data = record_path(directory).read_bytes()
    assert acquisition_id == hashlib.sha256(data).hexdigest()
    assert canonical_bytes(json.loads(data)) == data


def test_the_record_holds_every_frozen_field(tmp_path):
    directory = tmp_path / "acq"
    complete_three_slices(directory)
    record = json.loads(record_path(directory).read_bytes())
    assert REQUIRED_KEYS <= set(record)
    assert record["record_version"] == 1
    assert record["source"] == SOURCE
    assert record["window"] == {
        "start": "2024-01-01",
        "end": "2024-01-03",
        "resolved_start": "2024-01-01T05:00:00.000000+00:00",
        "resolved_end": "2024-01-04T04:59:59.999999+00:00",
    }
    assert record["client"] == {
        "user_agent": "Sentinel-test/0",
        "library": "fake",
        "library_version": "0",
    }
    assert record["policy"] == {
        "backoff_seconds": [2, 4, 8, 16, 32],
        "max_attempts": 6,
        "min_request_interval_ms": 1000,
        "retry_after_cap_seconds": 300,
    }
    assert record["sentinel_commit"] == COMMIT
    assert [s["key"] for s in record["slices"]] == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert record["slices"][0]["requests"][0] == request(1).as_record()
    assert record["slices"][0]["verification"] == {"rows": 2}
    digests = sorted(page_digest(slice_pages(n)) for n in (1, 2, 3))
    assert record["pages"] == digests
    assert record["started_at"] == "2026-09-28T11:00:00.000000+00:00"
    assert record["completed_at"] == "2026-09-28T12:00:00.000000+00:00"
    assert record["source_details"] == {}


def test_slices_are_sorted_by_key_whatever_order_they_completed_in(tmp_path):
    directory = tmp_path / "acq"
    acquisition = opened(directory)
    for n in (3, 1, 2):
        digest = cache(directory, slice_pages(n))
        acquisition.record_slice(f"2024-01-0{n}", requests=[], pages=[digest], verification={})
    acquisition.complete({})
    keys = [s["key"] for s in json.loads(record_path(directory).read_bytes())["slices"]]
    assert keys == ["2024-01-01", "2024-01-02", "2024-01-03"]


def test_started_at_is_the_earliest_retrieval(tmp_path):
    directory = tmp_path / "acq"
    acquisition = opened(directory)
    for n, when in (
        (1, "2026-09-28T11:30:00.000000+00:00"),
        (2, "2026-09-28T10:15:00.000000+00:00"),
    ):
        digest = cache(directory, slice_pages(n))
        acquisition.record_slice(
            f"k{n}", requests=[request(n, when)], pages=[digest], verification={}
        )
    acquisition.complete({})
    record = json.loads(record_path(directory).read_bytes())
    assert record["started_at"] == "2026-09-28T10:15:00.000000+00:00"


def test_identical_inputs_give_identical_record_bytes(tmp_path):
    first = complete_three_slices(tmp_path / "one")
    second = complete_three_slices(tmp_path / "two")
    assert first == second
    assert record_path(tmp_path / "one").read_bytes() == record_path(tmp_path / "two").read_bytes()


# --- incomplete means no record, and no completed_at ---------------------------------


def test_an_incomplete_acquisition_has_no_record_and_no_completed_at(tmp_path):
    directory = tmp_path / "acq"
    acquisition = opened(directory)
    digest = cache(directory, slice_pages(1))
    acquisition.record_slice("k1", requests=[request(1)], pages=[digest], verification={})

    assert not is_complete(directory)
    assert not record_path(directory).exists()
    journal = journal_path(directory).read_text(encoding="utf-8")
    assert "completed_at" not in journal
    for line in journal.splitlines():
        assert set(json.loads(line)) == {"key", "requests", "pages", "verification"}
    with pytest.raises(AcquisitionIncomplete):
        verify_acquisition(directory, source=SOURCE, start=START, end=END)


# --- a completed acquisition is immutable -----------------------------------------


def test_a_completed_acquisition_cannot_be_reopened_or_completed_again(tmp_path):
    directory = tmp_path / "acq"
    complete_three_slices(directory)
    before = {p.name: p.read_bytes() for p in directory.rglob("*") if p.is_file()}

    with pytest.raises(AcquisitionError, match="immutable"):
        opened(directory)

    after = {p.name: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    assert after == before


def test_completion_refuses_when_a_record_already_exists(tmp_path):
    directory = tmp_path / "acq"
    acquisition = opened(directory)
    digest = cache(directory, slice_pages(1))
    acquisition.record_slice("k1", requests=[], pages=[digest], verification={})
    acquisition.complete({})
    data = record_path(directory).read_bytes()
    with pytest.raises(AcquisitionError):
        acquisition.complete({"changed": True})
    with pytest.raises(AcquisitionError):
        acquisition.record_slice("k2", requests=[], pages=[digest], verification={})
    assert record_path(directory).read_bytes() == data


# --- the journal and resuming ------------------------------------------------------


def test_a_reopened_acquisition_resumes_from_its_journal(tmp_path):
    directory = tmp_path / "acq"
    first = opened(directory)
    digest = cache(directory, slice_pages(1))
    first.record_slice("k1", requests=[request(1)], pages=[digest], verification={"rows": 2})

    resumed = opened(directory)
    assert list(resumed.completed_slices()) == ["k1"]
    with pytest.raises(AcquisitionError, match="already journaled"):
        resumed.record_slice("k1", requests=[], pages=[digest], verification={})


def test_a_trailing_incomplete_journal_line_is_removed_before_anything_is_appended(tmp_path):
    directory = tmp_path / "acq"
    acquisition = opened(directory)
    first = cache(directory, slice_pages(1))
    acquisition.record_slice("k1", requests=[], pages=[first], verification={})
    whole = journal_path(directory).read_bytes()
    with open(journal_path(directory), "ab") as stream:
        stream.write(b'{"key":"k2","pa')

    resumed = opened(directory)
    assert list(resumed.completed_slices()) == ["k1"]
    assert journal_path(directory).read_bytes() == whole
    second = cache(directory, slice_pages(2))
    resumed.record_slice("k2", requests=[], pages=[second], verification={})
    lines = journal_path(directory).read_bytes().splitlines(keepends=True)
    assert [json.loads(line)["key"] for line in lines] == ["k1", "k2"]


def test_a_slice_whose_page_is_gone_no_longer_counts_as_completed(tmp_path):
    directory = tmp_path / "acq"
    acquisition = opened(directory)
    digests = [cache(directory, slice_pages(n)) for n in (1, 2)]
    acquisition.record_slice("k1", requests=[], pages=[digests[0]], verification={})
    acquisition.record_slice("k2", requests=[], pages=[digests[1]], verification={})
    (directory / SOURCE / f"{digests[0]}.json.gz").unlink()

    assert opened(directory).completed_slices() == {}, "k1 and everything after it are removed"
    assert journal_path(directory).read_bytes() == b""


def test_a_slice_cannot_list_a_page_that_is_not_cached(tmp_path):
    directory = tmp_path / "acq"
    with pytest.raises(AcquisitionError, match="not cached"):
        opened(directory).record_slice("k1", requests=[], pages=["0" * 64], verification={})


def test_completion_refuses_a_cached_page_no_slice_lists(tmp_path):
    directory = tmp_path / "acq"
    acquisition = opened(directory)
    digest = cache(directory, slice_pages(1))
    acquisition.record_slice("k1", requests=[], pages=[digest], verification={})
    cache(directory, slice_pages(9))
    with pytest.raises(AcquisitionIntegrityError, match="no completed slice lists"):
        acquisition.complete({})
    assert not record_path(directory).exists()


# --- verification --------------------------------------------------------------------


def test_a_completed_acquisition_verifies(tmp_path):
    directory = tmp_path / "acq"
    acquisition_id = complete_three_slices(directory)
    verified = verify_acquisition(directory, source=SOURCE, start=START, end=END)
    assert verified.acquisition_id == acquisition_id
    assert list(verified.pages) == sorted(page_digest(slice_pages(n)) for n in (1, 2, 3))


def rewrite_record(directory, change):
    record = json.loads(record_path(directory).read_bytes())
    change(record)
    record_path(directory).write_bytes(canonical_bytes(record))


@pytest.mark.parametrize(
    "tamper, message",
    [
        (lambda d: record_path(d).write_bytes(b"{not json\n"), "cannot be read"),
        (
            lambda d: record_path(d).write_bytes(
                json.dumps(json.loads(record_path(d).read_bytes()), indent=2).encode()
            ),
            "canonical",
        ),
        (
            lambda d: record_path(d).write_bytes(record_path(d).read_bytes().rstrip(b"\n")),
            "canonical",
        ),
        (lambda d: rewrite_record(d, lambda r: r.pop("sentinel_commit")), "required keys"),
        (lambda d: rewrite_record(d, lambda r: r.update(record_version=2)), "record_version"),
        (lambda d: rewrite_record(d, lambda r: r.update(source="cfpb")), "source"),
        (lambda d: rewrite_record(d, lambda r: r["window"].update(end="2024-01-04")), "window"),
        (lambda d: rewrite_record(d, lambda r: r["pages"].reverse()), "ascending"),
    ],
)
def test_a_damaged_record_refuses(tmp_path, tamper, message):
    directory = tmp_path / "acq"
    complete_three_slices(directory)
    tamper(directory)
    with pytest.raises(AcquisitionIntegrityError, match=message):
        verify_acquisition(directory, source=SOURCE, start=START, end=END)


def test_a_record_holding_a_float_refuses(tmp_path):
    directory = tmp_path / "acq"
    complete_three_slices(directory)
    record = json.loads(record_path(directory).read_bytes())
    record["source_details"] = {"ratio": 0.5}
    text = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    record_path(directory).write_bytes(text.encode() + b"\n")
    with pytest.raises(AcquisitionIntegrityError):
        verify_acquisition(directory, source=SOURCE, start=START, end=END)


def test_a_missing_page_refuses(tmp_path):
    directory = tmp_path / "acq"
    complete_three_slices(directory)
    next((directory / SOURCE).glob("*.json.gz")).unlink()
    with pytest.raises(AcquisitionIntegrityError, match="missing"):
        verify_acquisition(directory, source=SOURCE, start=START, end=END)


def test_an_unlisted_page_refuses(tmp_path):
    directory = tmp_path / "acq"
    complete_three_slices(directory)
    cache(directory, slice_pages(9))
    with pytest.raises(AcquisitionIntegrityError, match="does not list"):
        verify_acquisition(directory, source=SOURCE, start=START, end=END)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda path: path.write_bytes(path.read_bytes()[:10]),
        lambda path: path.write_bytes(gzip.compress(b'[{"unique_key":"other"}]')),
        lambda path: path.write_bytes(b"not gzip"),
    ],
)
def test_a_corrupt_page_refuses(tmp_path, corrupt):
    directory = tmp_path / "acq"
    complete_three_slices(directory)
    corrupt(next((directory / SOURCE).glob("*.json.gz")))
    with pytest.raises(AcquisitionIntegrityError, match="content its name promises"):
        verify_acquisition(directory, source=SOURCE, start=START, end=END)


def test_verification_changes_nothing(tmp_path):
    directory = tmp_path / "acq"
    complete_three_slices(directory)
    before = {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    verify_acquisition(directory, source=SOURCE, start=START, end=END)
    after = {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    assert after == before
    assert {p.name for p in directory.iterdir()} == {RECORD_NAME, JOURNAL_NAME, SOURCE}

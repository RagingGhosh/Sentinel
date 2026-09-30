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
import os
import subprocess
import sys
import textwrap
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from ingest.fetch.acquisition import (
    ACQUISITIONS_ROOT,
    JOURNAL_NAME,
    LOCK_NAME,
    QUARANTINE_DIR,
    RECORD_NAME,
    REQUIRED_KEYS,
    REWINDS_NAME,
    START_NAME,
    Acquisition,
    AcquisitionError,
    AcquisitionIncomplete,
    AcquisitionIntegrityError,
    AcquisitionLocked,
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
    with opened(directory) as acquisition:
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


def test_each_source_registers_its_own_fetcher():
    """D46 (B2), D48 (8): Task 24 registered NYC 311 and Task 25 registered CFPB."""
    assert sorted(FETCHERS) == ["cfpb", "nyc311"]


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
        "connect_timeout_seconds": 30,
        "max_attempts": 6,
        "min_request_interval_ms": 1000,
        "read_timeout_seconds": 300,
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
    first.close()

    resumed = opened(directory)
    assert list(resumed.completed_slices()) == ["k1"]
    with pytest.raises(AcquisitionError, match="already journaled"):
        resumed.record_slice("k1", requests=[], pages=[digest], verification={})


def test_a_trailing_incomplete_journal_line_is_removed_before_anything_is_appended(tmp_path):
    directory = tmp_path / "acq"
    acquisition = opened(directory)
    first = cache(directory, slice_pages(1))
    acquisition.record_slice("k1", requests=[], pages=[first], verification={})
    acquisition.close()
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
    acquisition.close()
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


# --- the journal is checked once, never on every append (D47) --------------------------


@pytest.fixture
def page_reads(monkeypatch):
    import ingest.fetch.acquisition as module

    count = {"reads": 0}
    real = module._read_page_digest

    def counting(path):
        count["reads"] += 1
        return real(path)

    monkeypatch.setattr(module, "_read_page_digest", counting)
    return count


def test_recording_slices_reads_each_new_page_once(tmp_path, page_reads):
    directory = tmp_path / "acq"
    acquisition = opened(directory)
    for n in range(50):
        digest = cache(directory, [{"unique_key": f"page-{n}"}])
        acquisition.record_slice(f"k{n:03d}", requests=[], pages=[digest], verification={})
    assert page_reads["reads"] == 50, "each new page once, never the whole journal again"
    acquisition.complete({})
    assert page_reads["reads"] == 100, "completion verifies every listed page once more"


def test_opening_verifies_the_journal_once(tmp_path, page_reads):
    directory = tmp_path / "acq"
    acquisition = opened(directory)
    for n in range(20):
        digest = cache(directory, [{"unique_key": f"page-{n}"}])
        acquisition.record_slice(f"k{n:03d}", requests=[], pages=[digest], verification={})
    acquisition.close()
    page_reads["reads"] = 0
    resumed = opened(directory)
    assert page_reads["reads"] == 20
    for _ in range(3):
        assert len(resumed.completed_slices()) == 20
    assert page_reads["reads"] == 20, "asking again reads nothing"


def test_completion_refuses_and_writes_nothing_when_a_journaled_page_is_damaged(tmp_path):
    directory = tmp_path / "acq"
    acquisition = opened(directory)
    digests = [cache(directory, slice_pages(n)) for n in (1, 2, 3)]
    for n, digest in zip((1, 2, 3), digests):
        acquisition.record_slice(f"k{n}", requests=[], pages=[digest], verification={})
    journal = journal_path(directory).read_bytes()
    damaged = directory / SOURCE / f"{digests[1]}.json.gz"
    damaged.write_bytes(gzip.compress(b'[{"unique_key":"forged"}]'))

    with pytest.raises(AcquisitionIntegrityError, match="no record"):
        acquisition.complete({})
    assert not record_path(directory).exists()
    assert journal_path(directory).read_bytes() == journal, "completion never truncates"


def test_a_failed_append_leaves_the_journal_as_it_was(tmp_path, monkeypatch):
    import ingest.fetch.acquisition as module

    directory = tmp_path / "acq"
    acquisition = opened(directory)
    acquisition.record_slice(
        "k1", requests=[], pages=[cache(directory, slice_pages(1))], verification={}
    )
    journal = journal_path(directory).read_bytes()
    second = cache(directory, slice_pages(2))

    def disk_full(fd):
        raise OSError("disk full")

    monkeypatch.setattr(module.os, "fsync", disk_full)
    with pytest.raises(OSError, match="disk full"):
        acquisition.record_slice("k2", requests=[], pages=[second], verification={})
    monkeypatch.undo()

    assert journal_path(directory).read_bytes() == journal
    assert list(acquisition.completed_slices()) == ["k1"]
    acquisition.record_slice("k2", requests=[], pages=[second], verification={})
    assert list(acquisition.completed_slices()) == ["k1", "k2"]


# --- D47 (4): orphaned pages are quarantined, never adopted or deleted ------------------

ROOT = Path(__file__).resolve().parents[3]


def one_slice(directory, n=1):
    with opened(directory) as acquisition:
        acquisition.record_slice(
            f"k{n}", requests=[], pages=[cache(directory, slice_pages(n))], verification={}
        )


def quarantined_files(directory):
    return sorted((directory / QUARANTINE_DIR).rglob("*.json.gz"))


def test_an_orphaned_page_is_moved_to_quarantine_by_its_bytes_when_opened(tmp_path):
    directory = tmp_path / "acq"
    one_slice(directory)
    orphan = cache(directory, slice_pages(2))
    path = directory / SOURCE / f"{orphan}.json.gz"
    data = path.read_bytes()
    journal = journal_path(directory).read_bytes()

    with opened(directory) as resumed:
        destination = directory / QUARANTINE_DIR / hashlib.sha256(data).hexdigest() / path.name
        assert resumed.quarantined == (destination,)
        assert not path.exists()
        assert destination.read_bytes() == data, "moved, never deleted"
        assert list(resumed.completed_slices()) == ["k1"], "never adopted into the journal"
        assert journal_path(directory).read_bytes() == journal


def test_a_page_orphaned_by_a_changed_source_no_longer_blocks_completion(tmp_path):
    directory = tmp_path / "acq"
    one_slice(directory)
    cache(directory, [{"unique_key": "2-0", "version": "yesterday"}])

    with opened(directory) as resumed:
        today = cache(directory, [{"unique_key": "2-0", "version": "today"}])
        resumed.record_slice("k2", requests=[], pages=[today], verification={})
        acquisition_id = resumed.complete({})
    verified = verify_acquisition(directory, source=SOURCE, start=START, end=END)
    assert verified.acquisition_id == acquisition_id
    assert len(quarantined_files(directory)) == 1


def test_quarantine_is_deterministic_and_never_overwrites_different_bytes(tmp_path):
    directory = tmp_path / "acq"
    one_slice(directory)
    orphan = slice_pages(5)
    path = directory / SOURCE / f"{cache(directory, orphan)}.json.gz"
    first = path.read_bytes()
    with opened(directory):
        pass

    text = json.dumps(orphan, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    path.write_bytes(gzip.compress(text.encode("utf-8"), mtime=1))
    second = path.read_bytes()
    assert second != first
    with opened(directory):
        pass
    assert sorted(p.read_bytes() for p in quarantined_files(directory)) == sorted([first, second])

    path.write_bytes(second)
    with opened(directory):
        pass
    assert len(quarantined_files(directory)) == 2, "identical bytes land on themselves"


def test_listed_pages_and_temporary_files_are_never_quarantined(tmp_path):
    directory = tmp_path / "acq"
    one_slice(directory)
    listed = page_digest(slice_pages(1))
    leftover = directory / SOURCE / ".page-killed.tmp"
    leftover.write_bytes(b"\x1f\x8b partial")
    with opened(directory) as resumed:
        assert resumed.quarantined == ()
    assert leftover.exists() and (directory / SOURCE / f"{listed}.json.gz").exists()
    assert not (directory / QUARANTINE_DIR).exists()


def test_completion_stays_strict_about_a_page_orphaned_while_open(tmp_path):
    directory = tmp_path / "acq"
    with opened(directory) as acquisition:
        acquisition.record_slice(
            "k1", requests=[], pages=[cache(directory, slice_pages(1))], verification={}
        )
        cache(directory, slice_pages(9))
        with pytest.raises(AcquisitionIntegrityError, match="no completed slice lists"):
            acquisition.complete({})
    assert not record_path(directory).exists()


def test_verification_ignores_the_quarantine(tmp_path):
    directory = tmp_path / "acq"
    one_slice(directory)
    cache(directory, slice_pages(9))
    with opened(directory) as resumed:
        resumed.complete({})
    assert quarantined_files(directory)
    verify_acquisition(directory, source=SOURCE, start=START, end=END)


# --- D47 (5): damage before the journal's last line ------------------------------------


def test_damage_before_the_last_line_truncates_from_the_first_damaged_slice(tmp_path):
    directory = tmp_path / "acq"
    digests = []
    with opened(directory) as acquisition:
        for n in (1, 2, 3, 4):
            digests.append(cache(directory, slice_pages(n)))
            acquisition.record_slice(f"k{n}", requests=[], pages=[digests[-1]], verification={})
    first_line = journal_path(directory).read_bytes().splitlines(keepends=True)[0]
    (directory / SOURCE / f"{digests[1]}.json.gz").write_bytes(b"damaged")

    with opened(directory) as resumed:
        assert list(resumed.completed_slices()) == ["k1"]
        assert journal_path(directory).read_bytes() == first_line
        moved = {path.name for path in resumed.quarantined}
        assert moved == {f"{digests[i]}.json.gz" for i in (1, 2, 3)}, "k3 and k4 are unfinished"
    with opened(directory) as again:
        assert list(again.completed_slices()) == ["k1"], "the resume point is deterministic"
        assert again.quarantined == ()


def test_a_repeated_key_is_where_the_journal_is_cut(tmp_path):
    directory = tmp_path / "acq"
    one_slice(directory)
    line = journal_path(directory).read_bytes()
    with open(journal_path(directory), "ab") as stream:
        stream.write(line)
    with opened(directory) as resumed:
        assert list(resumed.completed_slices()) == ["k1"]
    assert journal_path(directory).read_bytes() == line


# --- D47 (6): one writer per acquisition -----------------------------------------------


def test_a_second_writer_is_refused_while_the_first_is_open(tmp_path):
    directory = tmp_path / "acq"
    first = opened(directory)
    lock = directory / LOCK_NAME
    holder = json.loads(lock.read_bytes())
    assert holder["pid"] == os.getpid() and set(holder) == {"host", "pid", "started_at"}
    with pytest.raises(AcquisitionLocked, match=LOCK_NAME) as caught:
        opened(directory)
    assert str(os.getpid()) in str(caught.value)
    first.close()
    assert not lock.exists()
    with opened(directory):
        pass


def test_the_lock_is_released_when_the_writer_fails(tmp_path):
    directory = tmp_path / "acq"
    with pytest.raises(RuntimeError):
        with opened(directory):
            raise RuntimeError("the fetch failed")
    assert not (directory / LOCK_NAME).exists()
    with opened(directory):
        pass


def test_a_lock_left_by_a_hard_kill_refuses_until_an_operator_removes_it(tmp_path):
    directory = tmp_path / "acq"
    one_slice(directory)
    stale = {"host": "gone", "pid": 1, "started_at": "then"}
    (directory / LOCK_NAME).write_text(json.dumps(stale) + "\n")
    before = {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    for _ in range(2):
        with pytest.raises(AcquisitionLocked, match="gone"):
            opened(directory)
    after = {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    assert after == before, "a refused open touches nothing, lock included"

    (directory / LOCK_NAME).unlink()
    with opened(directory) as resumed:
        assert list(resumed.completed_slices()) == ["k1"], "the evidence survived"


def test_a_writer_never_removes_a_lock_it_did_not_create(tmp_path):
    directory = tmp_path / "acq"
    acquisition = opened(directory)
    (directory / LOCK_NAME).write_text("another writer's lock\n")
    acquisition.close()
    assert (directory / LOCK_NAME).read_text() == "another writer's lock\n"


def test_a_closed_writer_writes_nothing(tmp_path):
    directory = tmp_path / "acq"
    acquisition = opened(directory)
    digest = cache(directory, slice_pages(1))
    acquisition.close()
    acquisition.close()
    with pytest.raises(AcquisitionError, match="closed"):
        acquisition.record_slice("k1", requests=[], pages=[digest], verification={})
    with pytest.raises(AcquisitionError, match="closed"):
        acquisition.complete({})
    assert not journal_path(directory).exists() and not record_path(directory).exists()


HOLDER = textwrap.dedent(
    """
    import sys
    from datetime import UTC, date, datetime
    from pathlib import Path

    from ingest.fetch.acquisition import Acquisition
    from ingest.fetch.http import ClientIdentity

    now = datetime(2026, 9, 28, tzinfo=UTC)
    with Acquisition(
        Path(sys.argv[1]),
        source="nyc311",
        start=date(2024, 1, 1),
        end=date(2024, 1, 3),
        resolved_start=now,
        resolved_end=now,
        client=ClientIdentity("Sentinel-test/0", "fake", "0"),
        sentinel_commit="a" * 40,
        now=lambda: now,
    ):
        print("held", flush=True)
        sys.stdin.read()
    """
)


def test_a_writer_in_another_process_is_refused(tmp_path):
    """The production mechanism itself, across two real processes on one filesystem."""
    directory = tmp_path / "acq"
    environment = {**os.environ, "PYTHONPATH": str(ROOT)}
    child = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(directory)],
        cwd=ROOT,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "held"
        # The recorded pid, not Popen's: on Windows a venv's python.exe is a launcher
        # whose own pid differs from the interpreter's that holds the lock.
        holder = json.loads((directory / LOCK_NAME).read_bytes())
        assert holder["pid"] != os.getpid()
        with pytest.raises(AcquisitionLocked, match=str(holder["pid"])):
            opened(directory)
    finally:
        child.stdin.close()
        child.wait(timeout=120)
    assert child.returncode == 0
    with opened(directory):
        pass


# --- D48 (2): the acquisition's start state ---------------------------------------------

START_STATE = {"snapshot": {"rows_updated_at": 1790559478, "window_count": 7}, "note": "café"}


class TickingClock:
    """A clock that moves one second per reading, so re-fetched lines differ in time."""

    def __init__(self):
        self.moment = NOW

    def __call__(self):
        self.moment += timedelta(seconds=1)
        return self.moment


def journaled(directory, keys, *, shared=None):
    """Open a fresh writer and journal one page per key; returns (acquisition, digests)."""
    acquisition = opened(directory)
    digests = {}
    for index, key in enumerate(keys):
        page = shared if shared is not None and index % 2 else [{"unique_key": key}]
        digests[key] = cache(directory, page)
        acquisition.record_slice(key, requests=[], pages=[digests[key]], verification={})
    return acquisition, digests


def test_a_new_acquisition_has_no_start_state(tmp_path):
    with opened(tmp_path / "acq") as acquisition:
        assert acquisition.start_state is None
    assert not (tmp_path / "acq" / START_NAME).exists()


def test_the_start_state_is_written_once_canonically_and_atomically(tmp_path, monkeypatch):
    import ingest.fetch.acquisition as module

    directory = tmp_path / "acq"
    written = []
    real = module._write_atomically

    def recording(path, data):
        written.append(path.name)
        return real(path, data)

    monkeypatch.setattr(module, "_write_atomically", recording)
    with opened(directory) as acquisition:
        acquisition.record_start_state(START_STATE)
        assert acquisition.start_state == START_STATE
    assert written == [START_NAME], "through the same-directory temporary file and os.replace"
    assert (directory / START_NAME).read_bytes() == canonical_bytes(START_STATE)


def test_the_start_state_survives_reopen_and_is_never_replaced(tmp_path):
    directory = tmp_path / "acq"
    with opened(directory) as acquisition:
        acquisition.record_start_state(START_STATE)
    data = (directory / START_NAME).read_bytes()
    with opened(directory) as resumed:
        assert resumed.start_state == START_STATE
        with pytest.raises(AcquisitionError, match="start state"):
            resumed.record_start_state({"snapshot": "later"})
    assert (directory / START_NAME).read_bytes() == data


def test_a_second_start_state_is_refused_in_the_same_writer(tmp_path):
    with opened(tmp_path / "acq") as acquisition:
        acquisition.record_start_state(START_STATE)
        with pytest.raises(AcquisitionError, match="start state"):
            acquisition.record_start_state(START_STATE)


def test_a_start_state_cannot_be_recorded_after_close_or_completion(tmp_path):
    acquisition = opened(tmp_path / "closed")
    acquisition.close()
    with pytest.raises(AcquisitionError, match="closed"):
        acquisition.record_start_state(START_STATE)

    with opened(tmp_path / "done") as done:
        done.complete({})
        with pytest.raises(AcquisitionError, match="immutable"):
            done.record_start_state(START_STATE)
    assert not (tmp_path / "done" / START_NAME).exists()


def test_the_start_state_must_be_a_float_free_object(tmp_path):
    with opened(tmp_path / "acq") as acquisition:
        with pytest.raises(ValueError):
            acquisition.record_start_state({"ratio": 0.5})
        with pytest.raises(AcquisitionError, match="object"):
            acquisition.record_start_state(["not", "an", "object"])
    assert not (tmp_path / "acq" / START_NAME).exists()


@pytest.mark.parametrize(
    "damage",
    [b"{not json\n", b'{"b":1, "a":2}\n', b'{"a":1}', b"[1,2]\n", b'{"x":1.5}\n'],
)
def test_a_corrupt_start_state_refuses_the_open_and_changes_nothing(tmp_path, damage):
    directory = tmp_path / "acq"
    with opened(directory):
        pass
    (directory / START_NAME).write_bytes(damage)
    with pytest.raises(AcquisitionIntegrityError, match=START_NAME):
        opened(directory)
    assert (directory / START_NAME).read_bytes() == damage
    assert not (directory / LOCK_NAME).exists(), "the refused open released its lock"


def test_the_start_state_is_not_a_journal_slice(tmp_path):
    directory = tmp_path / "acq"
    acquisition, _ = journaled(directory, ["k1"])
    acquisition.record_start_state(START_STATE)
    acquisition.close()
    lines = journal_path(directory).read_bytes().splitlines()
    assert [json.loads(line)["key"] for line in lines] == ["k1"]
    with opened(directory) as resumed:
        assert list(resumed.completed_slices()) == ["k1"]
        assert resumed.quarantined == ()


# --- D48 (1): rewinding from a completed slice ---------------------------------------------


def reason(*changed, **extra):
    return {"changed": list(changed), **extra}


def test_a_rewind_truncates_the_journal_immediately_before_its_slice(tmp_path):
    directory = tmp_path / "acq"
    acquisition, digests = journaled(directory, ["k1", "k2", "k3", "k4"])
    lines = journal_path(directory).read_bytes().splitlines(keepends=True)

    removed = acquisition.rewind("k2", reason=reason("k2", "k4"))

    assert removed == ("k2", "k3", "k4")
    assert journal_path(directory).read_bytes() == lines[0], "no superseding line, only a cut"
    assert list(acquisition.completed_slices()) == ["k1"]
    assert (directory / SOURCE / f"{digests['k1']}.json.gz").exists()
    for key in ("k2", "k3", "k4"):
        assert not (directory / SOURCE / f"{digests[key]}.json.gz").exists()
    acquisition.close()


def test_a_rewind_after_a_reopen_cuts_where_the_journal_holds_its_line(tmp_path):
    directory = tmp_path / "acq"
    acquisition, _ = journaled(directory, ["k1", "k2", "k3"])
    acquisition.close()
    lines = journal_path(directory).read_bytes().splitlines(keepends=True)
    with opened(directory) as resumed:
        assert resumed.rewind("k2", reason=reason("k2")) == ("k2", "k3")
        assert resumed.rewinds[0]["journal_offset"] == len(lines[0])
    assert journal_path(directory).read_bytes() == lines[0]


def test_rewound_pages_are_quarantined_by_their_bytes_and_never_deleted(tmp_path):
    directory = tmp_path / "acq"
    acquisition, digests = journaled(directory, ["k1", "k2", "k3"])
    before = {
        key: (directory / SOURCE / f"{digest}.json.gz").read_bytes()
        for key, digest in digests.items()
    }
    acquisition.rewind("k2", reason=reason("k2"))
    for key in ("k2", "k3"):
        name = f"{digests[key]}.json.gz"
        destination = directory / QUARANTINE_DIR / hashlib.sha256(before[key]).hexdigest() / name
        assert destination.read_bytes() == before[key]
    assert (directory / SOURCE / f"{digests['k1']}.json.gz").read_bytes() == before["k1"]
    acquisition.close()


def test_a_page_a_retained_slice_still_lists_stays_active(tmp_path):
    directory = tmp_path / "acq"
    shared = [{"unique_key": "shared"}]
    acquisition, digests = journaled(directory, ["k1", "k2", "k3", "k4"], shared=shared)
    assert digests["k2"] == digests["k4"]
    acquisition.rewind("k3", reason=reason("k3"))
    assert (directory / SOURCE / f"{digests['k2']}.json.gz").exists(), "k2 still lists it"
    assert not (directory / SOURCE / f"{digests['k3']}.json.gz").exists()
    acquisition.close()


def test_rewound_slices_can_be_journaled_again_with_unique_keys(tmp_path):
    directory = tmp_path / "acq"
    acquisition, digests = journaled(directory, ["k1", "k2", "k3"])
    acquisition.rewind("k2", reason=reason("k2"))
    for key in ("k2", "k3"):
        fresh = cache(directory, [{"unique_key": key, "fetched": "again"}])
        acquisition.record_slice(key, requests=[], pages=[fresh], verification={})
    keys = [json.loads(line)["key"] for line in journal_path(directory).read_bytes().splitlines()]
    assert keys == ["k1", "k2", "k3"]
    acquisition.complete({})
    verify_acquisition(directory, source=SOURCE, start=START, end=END)


def test_the_removed_keys_come_back_in_journal_order(tmp_path):
    directory = tmp_path / "acq"
    acquisition, _ = journaled(directory, ["k3", "k1", "k4", "k2"])
    assert acquisition.rewind("k1", reason=reason("k1", "k2")) == ("k1", "k4", "k2")
    assert list(acquisition.completed_slices()) == ["k3"]
    acquisition.close()


@pytest.mark.parametrize(
    "bad_reason, message",
    [
        ({}, "changed"),
        ({"changed": []}, "changed"),
        ({"changed": ["k3"]}, "k2"),
        ({"changed": ["k2", "k1"]}, "k1"),
        ({"changed": ["k2", "k2"]}, "changed"),
        ({"changed": ["k2", 7]}, "changed"),
        ({"changed": ["k2"], "ratio": 0.5}, "floating"),
    ],
)
def test_a_rewind_reason_names_its_changed_slices_from_the_rewind_point_on(
    tmp_path, bad_reason, message
):
    directory = tmp_path / "acq"
    acquisition, _ = journaled(directory, ["k1", "k2", "k3"])
    journal = journal_path(directory).read_bytes()
    with pytest.raises((AcquisitionError, ValueError), match=message):
        acquisition.rewind("k2", reason=bad_reason)
    assert journal_path(directory).read_bytes() == journal
    assert not (directory / REWINDS_NAME).exists()
    acquisition.close()


def test_a_rewind_from_an_unknown_slice_is_refused(tmp_path):
    directory = tmp_path / "acq"
    acquisition, _ = journaled(directory, ["k1"])
    with pytest.raises(AcquisitionError, match="k9"):
        acquisition.rewind("k9", reason=reason("k9"))
    assert not (directory / REWINDS_NAME).exists()
    acquisition.close()


def test_a_closed_or_completed_acquisition_cannot_rewind(tmp_path):
    acquisition, _ = journaled(tmp_path / "closed", ["k1"])
    acquisition.close()
    with pytest.raises(AcquisitionError, match="closed"):
        acquisition.rewind("k1", reason=reason("k1"))

    done, _ = journaled(tmp_path / "done", ["k1"])
    done.complete({})
    record = record_path(tmp_path / "done").read_bytes()
    with pytest.raises(AcquisitionError, match="immutable"):
        done.rewind("k1", reason=reason("k1"))
    assert record_path(tmp_path / "done").read_bytes() == record
    assert not (tmp_path / "done" / REWINDS_NAME).exists()
    done.close()


def test_a_rewind_never_replaces_the_start_state(tmp_path):
    directory = tmp_path / "acq"
    acquisition, _ = journaled(directory, ["k1", "k2"])
    acquisition.record_start_state(START_STATE)
    acquisition.rewind("k1", reason=reason("k1"))
    assert acquisition.start_state == START_STATE
    acquisition.close()
    with opened(directory) as resumed:
        assert resumed.start_state == START_STATE


# --- D48 (1, 4): the rewind history and the at-most-once rule ------------------------------


def test_the_rewind_event_is_canonical_and_holds_what_d48_names(tmp_path):
    directory = tmp_path / "acq"
    acquisition, _ = journaled(directory, ["k1", "k2", "k3"])
    offset = len(journal_path(directory).read_bytes().splitlines(keepends=True)[0])
    line = journal_path(directory).read_bytes().splitlines(keepends=True)[1]
    acquisition.rewind("k2", reason=reason("k2", why="drift"))
    data = (directory / REWINDS_NAME).read_bytes()
    event = json.loads(data)
    assert canonical_bytes(event) == data
    assert event == {
        "at": "2026-09-28T12:00:00.000000+00:00",
        "from_key": "k2",
        "journal_offset": offset,
        "line_sha256": hashlib.sha256(line).hexdigest(),
        "reason": {"changed": ["k2"], "why": "drift"},
        "removed": ["k2", "k3"],
    }
    assert acquisition.rewinds == (event,)
    acquisition.close()


def test_the_rewind_event_is_durable_before_the_journal_is_cut(tmp_path, monkeypatch):
    import ingest.fetch.acquisition as module

    directory = tmp_path / "acq"
    acquisition, _ = journaled(directory, ["k1", "k2"])
    steps = []
    real_append, real_truncate = module._append_line, module._truncate

    def append(path, line):
        steps.append(("append", path.name))
        return real_append(path, line)

    def truncate(path, size):
        steps.append(("truncate", path.name, (directory / REWINDS_NAME).exists()))
        return real_truncate(path, size)

    monkeypatch.setattr(module, "_append_line", append)
    monkeypatch.setattr(module, "_truncate", truncate)
    acquisition.rewind("k2", reason=reason("k2"))
    assert steps == [("append", REWINDS_NAME), ("truncate", JOURNAL_NAME, True)]
    acquisition.close()


def test_appending_a_line_is_flushed_to_disk_and_taken_back_if_it_fails(tmp_path, monkeypatch):
    import ingest.fetch.acquisition as module

    path = tmp_path / "log.jsonl"
    synced = []
    real_fsync = module.os.fsync
    monkeypatch.setattr(module.os, "fsync", lambda fd: synced.append(fd) or real_fsync(fd))
    module._append_line(path, b'{"a":1}\n')
    assert synced and path.read_bytes() == b'{"a":1}\n'

    def disk_full(fd):
        raise OSError("disk full")

    monkeypatch.setattr(module.os, "fsync", disk_full)
    with pytest.raises(OSError, match="disk full"):
        module._append_line(path, b'{"b":2}\n')
    assert path.read_bytes() == b'{"a":1}\n'


def test_a_cut_is_flushed_to_disk(tmp_path, monkeypatch):
    import ingest.fetch.acquisition as module

    path = tmp_path / "log.jsonl"
    path.write_bytes(b'{"a":1}\n{"b":2}\n')
    synced = []
    real_fsync = module.os.fsync
    monkeypatch.setattr(module.os, "fsync", lambda fd: synced.append(fd) or real_fsync(fd))
    module._truncate(path, 8)
    assert synced and path.read_bytes() == b'{"a":1}\n'


def test_a_failed_event_append_leaves_everything_as_it_was(tmp_path, monkeypatch):
    import ingest.fetch.acquisition as module

    directory = tmp_path / "acq"
    acquisition, _ = journaled(directory, ["k1", "k2"])
    journal = journal_path(directory).read_bytes()

    def fails(path, line):
        raise OSError("disk full")

    monkeypatch.setattr(module, "_append_line", fails)
    with pytest.raises(OSError):
        acquisition.rewind("k2", reason=reason("k2"))
    assert journal_path(directory).read_bytes() == journal
    assert list(acquisition.completed_slices()) == ["k1", "k2"]
    assert acquisition.rewinds == ()
    acquisition.close()


def test_rewind_events_keep_their_order_and_survive_reopen(tmp_path):
    directory = tmp_path / "acq"
    acquisition, _ = journaled(directory, ["k1", "k2", "k3", "k4"])
    acquisition.rewind("k4", reason=reason("k4"))
    acquisition.rewind("k2", reason=reason("k2"))
    acquisition.close()
    with opened(directory) as resumed:
        assert [event["from_key"] for event in resumed.rewinds] == ["k4", "k2"]
        assert list(resumed.completed_slices()) == ["k1"]


def test_a_slice_already_changed_once_cannot_be_rewound_again(tmp_path):
    directory = tmp_path / "acq"
    acquisition, _ = journaled(directory, ["k1", "k2", "k3"])
    acquisition.rewind("k2", reason=reason("k2", "k3"))
    for key in ("k2", "k3"):
        fresh = cache(directory, [{"unique_key": key, "fetched": "again"}])
        acquisition.record_slice(key, requests=[], pages=[fresh], verification={})
    journal = journal_path(directory).read_bytes()
    events = (directory / REWINDS_NAME).read_bytes()

    with pytest.raises(AcquisitionError, match="k3"):
        acquisition.rewind("k3", reason=reason("k3"))
    with pytest.raises(AcquisitionError, match="k2"):
        acquisition.rewind("k2", reason=reason("k2"))
    assert journal_path(directory).read_bytes() == journal
    assert (directory / REWINDS_NAME).read_bytes() == events
    acquisition.close()


def test_the_at_most_once_rule_survives_a_resume(tmp_path):
    directory = tmp_path / "acq"
    acquisition, _ = journaled(directory, ["k1", "k2", "k3"])
    acquisition.rewind("k2", reason=reason("k2"))
    for key in ("k2", "k3"):
        fresh = cache(directory, [{"unique_key": key, "fetched": "again"}])
        acquisition.record_slice(key, requests=[], pages=[fresh], verification={})
    acquisition.close()
    events = (directory / REWINDS_NAME).read_bytes()
    with opened(directory) as resumed:
        # A rewind from k1 would remove k2, found changed once already, a second time.
        for from_key in ("k2", "k1"):
            with pytest.raises(AcquisitionError, match="k2"):
                resumed.rewind(from_key, reason=reason(from_key))
        assert (directory / REWINDS_NAME).read_bytes() == events
        assert list(resumed.completed_slices()) == ["k1", "k2", "k3"]
        # k3 was removed by that rewind but never found changed, so it may be.
        assert resumed.rewind("k3", reason=reason("k3")) == ("k3",)


def test_a_page_quarantined_again_lands_on_itself(tmp_path):
    directory = tmp_path / "acq"
    acquisition, digests = journaled(directory, ["k1", "k2", "k3"])
    path = directory / SOURCE / f"{digests['k3']}.json.gz"
    data = path.read_bytes()
    acquisition.rewind("k2", reason=reason("k2"))
    fresh = cache(directory, [{"unique_key": "k2", "fetched": "again"}])
    acquisition.record_slice("k2", requests=[], pages=[fresh], verification={})
    path.write_bytes(data)
    acquisition.record_slice("k3", requests=[], pages=[digests["k3"]], verification={})
    acquisition.rewind("k3", reason=reason("k3"))
    destination = directory / QUARANTINE_DIR / hashlib.sha256(data).hexdigest() / path.name
    assert destination.read_bytes() == data
    assert len(quarantined_files(directory)) == 2, "k2's first page, and k3's page once"
    acquisition.close()


def test_a_trailing_partial_event_is_removed_but_a_damaged_one_refuses(tmp_path):
    directory = tmp_path / "acq"
    acquisition, _ = journaled(directory, ["k1", "k2"])
    acquisition.rewind("k2", reason=reason("k2"))
    acquisition.close()
    whole = (directory / REWINDS_NAME).read_bytes()

    with open(directory / REWINDS_NAME, "ab") as stream:
        stream.write(b'{"at":"2026')
    with opened(directory) as resumed:
        assert len(resumed.rewinds) == 1
    assert (directory / REWINDS_NAME).read_bytes() == whole

    (directory / REWINDS_NAME).write_bytes(b'{"from_key":"k2"}\n' + whole)
    with pytest.raises(AcquisitionIntegrityError, match=REWINDS_NAME):
        opened(directory)


EVENT_DAMAGE = {
    "not utf-8": lambda event: b"\xff" + canonical_bytes(event),
    "not canonical": lambda event: json.dumps(event).encode() + b"\n",
    "not an object": lambda event: canonical_bytes([event]),
    "offset a string": lambda event: canonical_bytes({**event, "journal_offset": "0"}),
    "offset a boolean": lambda event: canonical_bytes({**event, "journal_offset": True}),
    "offset negative": lambda event: canonical_bytes({**event, "journal_offset": -1}),
    "digest too short": lambda event: canonical_bytes({**event, "line_sha256": "0" * 63}),
    "digest a number": lambda event: canonical_bytes({**event, "line_sha256": 7}),
    "reason a list": lambda event: canonical_bytes({**event, "reason": ["k2"]}),
    "changed a string": lambda event: canonical_bytes({**event, "reason": {"changed": "k2"}}),
    "changed empty": lambda event: canonical_bytes({**event, "reason": {"changed": []}}),
    "changed a number": lambda event: canonical_bytes({**event, "reason": {"changed": [7]}}),
}


@pytest.mark.parametrize("damage", EVENT_DAMAGE.values(), ids=EVENT_DAMAGE.keys())
def test_an_event_an_open_cannot_rely_on_refuses_the_open(tmp_path, damage):
    directory = tmp_path / "acq"
    acquisition, _ = journaled(directory, ["k1", "k2"])
    acquisition.rewind("k2", reason=reason("k2"))
    acquisition.close()
    event = json.loads((directory / REWINDS_NAME).read_bytes())
    (directory / REWINDS_NAME).write_bytes(damage(event))
    journal = journal_path(directory).read_bytes()
    with pytest.raises(AcquisitionIntegrityError, match=REWINDS_NAME):
        opened(directory)
    assert journal_path(directory).read_bytes() == journal
    assert not (directory / LOCK_NAME).exists(), "the refused open released its lock"


# --- D48 (1): an interrupted rewind is completed on the next open ---------------------------


def test_a_rewind_interrupted_after_its_event_is_completed_on_reopen(tmp_path, monkeypatch):
    import ingest.fetch.acquisition as module

    directory = tmp_path / "acq"
    acquisition, digests = journaled(directory, ["k1", "k2", "k3"])
    first_line = journal_path(directory).read_bytes().splitlines(keepends=True)[0]

    def killed(path, size):
        raise RuntimeError("killed before the cut")

    monkeypatch.setattr(module, "_truncate", killed)
    with pytest.raises(RuntimeError):
        acquisition.rewind("k2", reason=reason("k2"))
    monkeypatch.undo()
    with pytest.raises(AcquisitionError, match="closed"):
        acquisition.complete({})
    assert not (directory / LOCK_NAME).exists(), "only an open completes the rewind"
    assert not record_path(directory).exists()
    events = (directory / REWINDS_NAME).read_bytes()
    assert len(events.splitlines()) == 1
    assert len(journal_path(directory).read_bytes().splitlines()) == 3, "not yet cut"

    with opened(directory) as resumed:
        assert list(resumed.completed_slices()) == ["k1"]
        assert journal_path(directory).read_bytes() == first_line
        assert (directory / REWINDS_NAME).read_bytes() == events, "the event is not repeated"
        moved = {path.name for path in resumed.quarantined}
        assert moved == {f"{digests[k]}.json.gz" for k in ("k2", "k3")}
    with opened(directory) as again:
        assert list(again.completed_slices()) == ["k1"]
        assert again.quarantined == ()


def test_only_the_latest_rewind_is_completed_on_reopen(tmp_path, monkeypatch):
    import ingest.fetch.acquisition as module

    directory = tmp_path / "acq"
    acquisition, _ = journaled(directory, ["k1", "k2", "k3"])
    acquisition.rewind("k3", reason=reason("k3"))

    def killed(path, size):
        raise RuntimeError("killed before the cut")

    monkeypatch.setattr(module, "_truncate", killed)
    with pytest.raises(RuntimeError):
        acquisition.rewind("k2", reason=reason("k2"))
    monkeypatch.undo()

    with opened(directory) as resumed:
        assert list(resumed.completed_slices()) == ["k1"]
        assert [event["from_key"] for event in resumed.rewinds] == ["k3", "k2"]


def test_a_rewind_killed_right_after_its_cut_is_finished_on_reopen(tmp_path, monkeypatch):
    import ingest.fetch.acquisition as module

    directory = tmp_path / "acq"
    acquisition, digests = journaled(directory, ["k1", "k2", "k3"])
    first_line = journal_path(directory).read_bytes().splitlines(keepends=True)[0]
    real = module._truncate

    def cut_then_killed(path, size):
        real(path, size)
        raise RuntimeError("killed after the cut")

    monkeypatch.setattr(module, "_truncate", cut_then_killed)
    with pytest.raises(RuntimeError):
        acquisition.rewind("k2", reason=reason("k2"))
    monkeypatch.undo()
    with pytest.raises(AcquisitionError, match="closed"):
        acquisition.record_slice("k2", requests=[], pages=[], verification={})
    assert journal_path(directory).read_bytes() == first_line
    assert not quarantined_files(directory), "no page was moved before the kill"

    with opened(directory) as resumed:
        assert list(resumed.completed_slices()) == ["k1"]
        assert journal_path(directory).read_bytes() == first_line
        assert len(resumed.rewinds) == 1
        moved = {path.name for path in resumed.quarantined}
        assert moved == {f"{digests[k]}.json.gz" for k in ("k2", "k3")}


def test_a_rewind_interrupted_before_its_quarantine_is_finished_on_reopen(tmp_path, monkeypatch):
    import ingest.fetch.acquisition as module

    directory = tmp_path / "acq"
    acquisition, digests = journaled(directory, ["k1", "k2", "k3"])
    moves = []
    real = module._quarantine_page

    def killed_after_one(directory_, path):
        if moves:
            raise RuntimeError("killed mid-quarantine")
        moves.append(path.name)
        return real(directory_, path)

    monkeypatch.setattr(module, "_quarantine_page", killed_after_one)
    with pytest.raises(RuntimeError):
        acquisition.rewind("k2", reason=reason("k2"))
    monkeypatch.undo()
    assert not (directory / LOCK_NAME).exists(), "the interrupted writer let go"

    with opened(directory) as resumed:
        assert list(resumed.completed_slices()) == ["k1"]
        assert len(resumed.rewinds) == 1
    quarantined = sorted(p.name for p in (directory / QUARANTINE_DIR).rglob("*.json.gz"))
    assert quarantined == sorted(f"{digests[k]}.json.gz" for k in ("k2", "k3"))
    assert (directory / SOURCE / f"{digests['k1']}.json.gz").exists()


def test_a_completed_rewind_is_not_repeated_once_its_slices_are_fetched_again(tmp_path):
    directory = tmp_path / "acq"
    acquisition = opened(directory, now=TickingClock())
    for key in ("k1", "k2", "k3"):
        acquisition.record_slice(
            key, requests=[request(1)], pages=[cache(directory, [{"k": key}])], verification={}
        )
    acquisition.rewind("k2", reason=reason("k2"))
    for key in ("k2", "k3"):
        acquisition.record_slice(
            key,
            requests=[request(1, "2026-09-28T13:00:00.000000+00:00")],
            pages=[cache(directory, [{"k": key}])],
            verification={},
        )
    acquisition.close()
    journal = journal_path(directory).read_bytes()
    with opened(directory) as resumed:
        assert list(resumed.completed_slices()) == ["k1", "k2", "k3"]
    assert journal_path(directory).read_bytes() == journal


# --- D48 (1, 2): completion carries the rewinds and checks the start state -----------------


def test_the_record_carries_an_empty_rewind_list_when_none_occurred(tmp_path):
    directory = tmp_path / "acq"
    complete_three_slices(directory)
    record = json.loads(record_path(directory).read_bytes())
    assert record["rewinds"] == []


def test_the_record_carries_every_rewind_in_order(tmp_path):
    directory = tmp_path / "acq"
    acquisition, _ = journaled(directory, ["k1", "k2", "k3"])
    acquisition.rewind("k3", reason=reason("k3"))
    acquisition.rewind("k2", reason=reason("k2"))
    for key in ("k2", "k3"):
        fresh = cache(directory, [{"unique_key": key, "fetched": "again"}])
        acquisition.record_slice(key, requests=[], pages=[fresh], verification={})
    acquisition.complete({})
    acquisition.close()
    record = json.loads(record_path(directory).read_bytes())
    assert [event["from_key"] for event in record["rewinds"]] == ["k3", "k2"]
    assert record["rewinds"] == list(acquisition.rewinds)
    assert record["record_version"] == 1
    verify_acquisition(directory, source=SOURCE, start=START, end=END)


def test_completion_requires_the_start_state_in_source_details(tmp_path):
    directory = tmp_path / "acq"
    acquisition, _ = journaled(directory, ["k1"])
    acquisition.record_start_state(START_STATE)
    for details in ({}, {"start": {"snapshot": "different"}}):
        with pytest.raises(AcquisitionIntegrityError, match="start"):
            acquisition.complete(details)
        assert not record_path(directory).exists()
    acquisition.complete({"start": START_STATE, "end": {"later": True}})
    acquisition.close()
    record = json.loads(record_path(directory).read_bytes())
    assert record["source_details"]["start"] == START_STATE
    verify_acquisition(directory, source=SOURCE, start=START, end=END)


def test_without_a_start_state_completion_checks_nothing_about_it(tmp_path):
    directory = tmp_path / "acq"
    acquisition, _ = journaled(directory, ["k1"])
    acquisition.complete({"start": {"anything": 1}})
    acquisition.close()
    assert json.loads(record_path(directory).read_bytes())["source_details"] == {
        "start": {"anything": 1}
    }

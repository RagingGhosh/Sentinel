"""The acquisition directory, its journal, its immutable record, and its verification.

Addendum D44, D46 and D47. One acquisition per source and window::

    data/acquisitions/<source>/<start>_<end>/
        acquisition.json          the record, written last: its presence means complete
        journal.jsonl             one canonical line per completed slice
        writer.lock               present while one writer holds the acquisition (D47)
        <source>/<sha256>.json.gz the pages -- the directory is the ingest raw root
        quarantine/<sha256 of the file's bytes>/<file name>   orphaned pages (D47)

**One writer at a time (D47).** Opening an acquisition for writing creates
`writer.lock` with an atomic exclusive create, and refuses if it already exists;
closing removes it, and removes nothing else. A lock a hard kill left behind refuses
every open until an operator removes that one file: nothing here guesses whether its
holder is alive. Reading a completed acquisition takes no lock.

**Orphaned pages are quarantined, never adopted or deleted (D47).** When an
incomplete acquisition is opened, after its journal is checked, each page under
`<source>/` that no completed slice lists is moved to the quarantine, at a path fixed
by the file's bytes, so a rerun is idempotent and different bytes never collide.
Completion stays strict: it refuses while any unlisted page lies under `<source>/`.

**An acquisition without a record is incomplete, and resumes from its journal.** A
slice counts as completed only while its line is whole, canonical and parses, its key
is new, and every page it lists is present under its digest. The first line that is
not a completed slice, and everything after it, is removed before anything more is
appended, so a key appears in the journal once. That costs at most a re-fetch of the
slices it removes, never a wrong record. The journal is checked once, when the
acquisition is opened, and each later slice as it is recorded (D47); completion then
verifies every listed page once more and, finding damage, refuses rather than
truncating, so a record never omits a slice its fetcher completed.

**An acquisition with a record is complete and immutable.** Nothing here deletes,
overwrites or appends to it: `Acquisition` refuses to open it for writing,
`Acquisition.complete` refuses to write a second record, and `verify_acquisition`
reads it and changes nothing. ``completed_at`` exists only in the record, which is
written once, at completion; an incomplete acquisition has none.

**`acquisition_id` is the SHA256 of the record's exact bytes**, which
`ingest.fetch.canonical` fixes. A reader serializes the parsed record again and
refuses it unless the bytes match, so the id is never a matter of interpretation.
"""

from __future__ import annotations

import gzip
import json
import os
import platform
import re
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ingest.fetch.canonical import (
    DIGEST,
    canonical_bytes,
    format_timestamp,
    page_digest,
    sha256_hex,
)
from ingest.fetch.http import ClientIdentity, RequestRecord, policy

ACQUISITIONS_ROOT = Path("data") / "acquisitions"
"""Gitignored with the rest of ``data/``: nothing acquired is ever committed."""

RECORD_NAME = "acquisition.json"
JOURNAL_NAME = "journal.jsonl"
LOCK_NAME = "writer.lock"
QUARANTINE_DIR = "quarantine"
PAGE_SUFFIX = ".json.gz"
RECORD_VERSION = 1

REQUIRED_KEYS = frozenset(
    {
        "record_version",
        "source",
        "window",
        "started_at",
        "completed_at",
        "client",
        "policy",
        "sentinel_commit",
        "slices",
        "pages",
        "source_details",
    }
)
"""D46's record keys. A record may carry more; it may not carry fewer."""

SLICE_KEYS = frozenset({"key", "requests", "pages", "verification"})

COMMIT = re.compile(r"[0-9a-f]{40}")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class AcquisitionError(Exception):
    """An acquisition cannot be started, extended, completed or trusted."""


class AcquisitionIncomplete(AcquisitionError):
    """The acquisition has no record, so it is not complete (D46)."""


class AcquisitionIntegrityError(AcquisitionError):
    """A record or page is not what a completed acquisition must hold (D46)."""


class AcquisitionLocked(AcquisitionError):
    """Another writer holds the acquisition, or a crashed one left its lock (D47)."""


@dataclass(frozen=True)
class VerifiedAcquisition:
    acquisition_id: str
    pages: tuple[str, ...]


def acquisition_dir(source: str, start: date, end: date, root: Path = ACQUISITIONS_ROOT) -> Path:
    """``<root>/<source>/<start>_<end>``, the dates as ``YYYY-MM-DD`` (D46)."""
    return Path(root) / source / f"{start.isoformat()}_{end.isoformat()}"


def record_path(directory: Path) -> Path:
    return Path(directory) / RECORD_NAME


def journal_path(directory: Path) -> Path:
    return Path(directory) / JOURNAL_NAME


def pages_dir(directory: Path, source: str) -> Path:
    """Where ``ingest.cli.cache_page`` puts pages when this directory is the raw root."""
    return Path(directory) / source


def is_complete(directory: Path) -> bool:
    """Complete means the record exists (D46). Whether it verifies is a separate question."""
    return record_path(directory).is_file()


def current_commit(repository: Path = REPOSITORY_ROOT) -> str | None:
    """The 40-character commit checked out at `repository`, or `None` when unknown.

    `GIT_*` variables are dropped so an inherited ``GIT_DIR`` cannot answer for some
    other repository.
    """
    environment = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = done.stdout.strip()
    return value if done.returncode == 0 and COMMIT.fullmatch(value) else None


def _read_page_digest(path: Path) -> str | None:
    """The digest of the page stored at `path`, or `None` when it cannot be read."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return page_digest(json.load(handle))
    except (OSError, EOFError, ValueError):
        return None


def _page_present(directory: Path, source: str, digest: str) -> bool:
    path = pages_dir(directory, source) / f"{digest}{PAGE_SUFFIX}"
    return path.is_file() and _read_page_digest(path) == digest


def _pages_on_disk(directory: Path, source: str) -> set[str]:
    folder = pages_dir(directory, source)
    if not folder.is_dir():
        return set()
    return {
        path.name[: -len(PAGE_SUFFIX)]
        for path in folder.iterdir()
        if path.is_file() and path.name.endswith(PAGE_SUFFIX)
    }


def _write_atomically(path: Path, data: bytes) -> None:
    """Binary, beside its destination, moved into place with `os.replace` (D46)."""
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _slice_shape_problem(entry: Any) -> str | None:
    """Why `entry` is not a well-formed slice object, or `None` when it is."""
    if not isinstance(entry, dict) or not SLICE_KEYS <= set(entry):
        return "a slice needs key, requests, pages and verification"
    if not isinstance(entry["key"], str) or not entry["key"]:
        return "a slice key is a non-empty string"
    if not isinstance(entry["requests"], list) or not all(
        isinstance(item, dict) for item in entry["requests"]
    ):
        return "requests is a list of objects"
    if not isinstance(entry["pages"], list) or not all(
        isinstance(item, str) and DIGEST.fullmatch(item) for item in entry["pages"]
    ):
        return "pages is a list of digests"
    if not isinstance(entry["verification"], dict):
        return "verification is an object"
    return None


class Acquisition:
    """An incomplete acquisition, open for its fetcher to extend and then complete."""

    def __init__(
        self,
        directory: Path,
        *,
        source: str,
        start: date,
        end: date,
        resolved_start: datetime,
        resolved_end: datetime,
        client: ClientIdentity,
        sentinel_commit: str | None,
        now: Callable[[], datetime],
    ) -> None:
        if sentinel_commit is None or not COMMIT.fullmatch(sentinel_commit):
            raise AcquisitionError(
                f"{source}: the Sentinel commit could not be determined "
                f"({sentinel_commit!r}), so the acquisition does not start (D46)"
            )
        self.directory = Path(directory)
        if is_complete(self.directory):
            raise AcquisitionError(
                f"{self.directory} is a completed acquisition; it is immutable (D46)"
            )
        self.source = source
        self.start = start
        self.end = end
        self.resolved_start = resolved_start
        self.resolved_end = resolved_end
        self.client = client
        self.sentinel_commit = sentinel_commit
        self._now = now
        self._closed = False
        self.quarantined: tuple[Path, ...] = ()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = self._acquire_lock()
        try:
            # The journal is checked once, here; each later slice is checked as it is
            # recorded (D47). Completion verifies every listed page once more.
            self._slices = self._load()
            self.quarantined = self._quarantine_orphans()
        except BaseException:
            self.close()
            raise

    # --- the single writer (D47) -------------------------------------------------

    def _acquire_lock(self) -> bytes:
        """Create `writer.lock` atomically and exclusively, or refuse."""
        path = self.directory / LOCK_NAME
        holder = canonical_bytes(
            {
                "host": platform.node(),
                "pid": os.getpid(),
                "started_at": format_timestamp(self._now()),
            }
        )
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
        try:
            descriptor = os.open(path, flags)
        except FileExistsError:
            try:
                found = path.read_bytes().decode("utf-8", "replace").strip()
            except OSError:
                found = "unreadable"
            raise AcquisitionLocked(
                f"{path} exists, so another writer holds this acquisition ({found}). If no "
                "writer is running, an operator may remove that one file; nothing else "
                "needs to change (D47)."
            ) from None
        try:
            os.write(descriptor, holder)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return holder

    def close(self) -> None:
        """Give up the acquisition: remove our own lock, and nothing else. Idempotent."""
        if self._closed:
            return
        self._closed = True
        path = self.directory / LOCK_NAME
        try:
            if path.read_bytes() == self._lock:
                path.unlink()
        except (AttributeError, OSError):
            pass

    def __enter__(self) -> Acquisition:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _writable(self) -> None:
        if self._closed:
            raise AcquisitionError(f"{self.directory}: this writer is closed")
        if is_complete(self.directory):
            raise AcquisitionError(f"{self.directory} is complete; it is immutable (D46)")

    def _quarantine_orphans(self) -> tuple[Path, ...]:
        """Move every page no completed slice lists into the quarantine (D47)."""
        listed = {digest for entry in self._slices.values() for digest in entry["pages"]}
        folder = pages_dir(self.directory, self.source)
        if not folder.is_dir():
            return ()
        moved = []
        for path in sorted(folder.iterdir()):
            if not (path.is_file() and path.name.endswith(PAGE_SUFFIX)):
                continue
            if path.name[: -len(PAGE_SUFFIX)] in listed:
                continue
            destination = (
                self.directory / QUARANTINE_DIR / sha256_hex(path.read_bytes()) / path.name
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, destination)
            moved.append(destination)
        return tuple(moved)

    # --- the journal -------------------------------------------------------------

    def _load(self) -> dict[str, dict[str, Any]]:
        """Completed slices by key, after removing the first non-completed line onward."""
        path = journal_path(self.directory)
        if not path.is_file():
            return {}
        data = path.read_bytes()
        completed: dict[str, dict[str, Any]] = {}
        verified: set[str] = set()
        offset = 0
        while offset < len(data):
            end = data.find(b"\n", offset)
            if end < 0:
                break
            line = data[offset : end + 1]
            try:
                entry = json.loads(line.decode("utf-8"))
                whole = canonical_bytes(entry) == line
            except (UnicodeDecodeError, ValueError):
                break
            if not whole or _slice_shape_problem(entry) is not None:
                break
            if entry["key"] in completed:
                break
            unchecked = [d for d in entry["pages"] if d not in verified]
            if not all(_page_present(self.directory, self.source, d) for d in unchecked):
                break
            verified.update(unchecked)
            completed[entry["key"]] = entry
            offset = end + 1
        if offset < len(data):
            with open(path, "r+b") as stream:
                stream.truncate(offset)
                stream.flush()
                os.fsync(stream.fileno())
        return completed

    def completed_slices(self) -> dict[str, dict[str, Any]]:
        """Each completed slice's object, keyed by slice key. Reads nothing from disk."""
        return dict(self._slices)

    def record_slice(
        self,
        key: str,
        *,
        requests: Sequence[RequestRecord],
        pages: Sequence[str],
        verification: Mapping[str, int],
    ) -> None:
        """Journal one completed slice. Its pages must already be cached (D46)."""
        self._writable()
        if key in self._slices:
            raise AcquisitionError(f"slice {key!r} is already journaled")
        entry = {
            "key": key,
            "pages": list(pages),
            "requests": [request.as_record() for request in requests],
            "verification": dict(verification),
        }
        problem = _slice_shape_problem(entry)
        if problem:
            raise AcquisitionError(f"slice {key!r}: {problem}")
        missing = [d for d in pages if not _page_present(self.directory, self.source, d)]
        if missing:
            raise AcquisitionError(f"slice {key!r} lists pages that are not cached: {missing}")
        line = canonical_bytes(entry)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = journal_path(self.directory)
        before = path.stat().st_size if path.exists() else 0
        try:
            with open(path, "ab") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            # The journal is not reread before the next append, so a line that did
            # not land whole is taken back now rather than left for the next open.
            with open(path, "r+b") as stream:
                stream.truncate(before)
            raise
        self._slices[key] = entry

    # --- completion --------------------------------------------------------------

    def complete(self, source_details: Mapping[str, Any]) -> str:
        """Write the record, last and atomically, and return its `acquisition_id`."""
        self._writable()
        slices = [entry for _, entry in sorted(self._slices.items())]
        listed = sorted({digest for entry in slices for digest in entry["pages"]})
        damaged = [d for d in listed if not _page_present(self.directory, self.source, d)]
        if damaged:
            # D47: completion never truncates. The next open finds the first damaged
            # slice and resumes from there.
            raise AcquisitionIntegrityError(
                f"{self.directory}: journaled pages are missing or damaged: {damaged}; "
                "no record was written"
            )
        unlisted = sorted(_pages_on_disk(self.directory, self.source) - set(listed))
        if unlisted:
            raise AcquisitionIntegrityError(
                f"{self.directory}: cached pages no completed slice lists: {unlisted}"
            )
        completed_at = format_timestamp(self._now())
        retrieved = [request["retrieved_at"] for entry in slices for request in entry["requests"]]
        record = {
            "client": self.client.as_record(),
            "completed_at": completed_at,
            "pages": listed,
            "policy": policy(),
            "record_version": RECORD_VERSION,
            "sentinel_commit": self.sentinel_commit,
            "slices": slices,
            "source": self.source,
            "source_details": dict(source_details),
            # The earliest retrieval the journal holds: the first request whose
            # response the acquisition kept. With no request it is the completion.
            "started_at": min(retrieved) if retrieved else completed_at,
            "window": {
                "end": self.end.isoformat(),
                "resolved_end": format_timestamp(self.resolved_end),
                "resolved_start": format_timestamp(self.resolved_start),
                "start": self.start.isoformat(),
            },
        }
        data = canonical_bytes(record)
        self.directory.mkdir(parents=True, exist_ok=True)
        _write_atomically(record_path(self.directory), data)
        return sha256_hex(data)


def _check_record(record: Any, source: str, start: date, end: date) -> None:
    if not isinstance(record, dict):
        raise AcquisitionIntegrityError("the record is not a JSON object")
    missing = sorted(REQUIRED_KEYS - set(record))
    if missing:
        raise AcquisitionIntegrityError(f"the record lacks required keys: {missing}")
    version = record["record_version"]
    if isinstance(version, bool) or version != RECORD_VERSION:
        raise AcquisitionIntegrityError(f"record_version is {version!r}, not {RECORD_VERSION}")
    if record["source"] != source:
        raise AcquisitionIntegrityError(
            f"the record is for source {record['source']!r}, not {source!r}"
        )
    window = record["window"]
    if (
        not isinstance(window, dict)
        or window.get("start") != start.isoformat()
        or window.get("end") != end.isoformat()
    ):
        raise AcquisitionIntegrityError(
            f"the record's window {window!r} is not {start.isoformat()} .. {end.isoformat()}"
        )
    slices = record["slices"]
    if not isinstance(slices, list):
        raise AcquisitionIntegrityError("slices is not a list")
    for entry in slices:
        problem = _slice_shape_problem(entry)
        if problem:
            raise AcquisitionIntegrityError(problem)
    keys = [entry["key"] for entry in slices]
    if keys != sorted(set(keys)):
        raise AcquisitionIntegrityError("slice keys are not unique and in ascending order")
    pages = record["pages"]
    if not isinstance(pages, list) or not all(
        isinstance(d, str) and DIGEST.fullmatch(d) for d in pages
    ):
        raise AcquisitionIntegrityError("pages is not a list of digests")
    if pages != sorted(set(pages)):
        raise AcquisitionIntegrityError("pages are not unique and in ascending order")
    if set(pages) != {digest for entry in slices for digest in entry["pages"]}:
        raise AcquisitionIntegrityError("pages differ from the pages the slices list")


def verify_acquisition(
    directory: Path, *, source: str, start: date, end: date
) -> VerifiedAcquisition:
    """Everything D46 requires of a completed acquisition before any page is normalized.

    Reads, never writes. Raises `AcquisitionIncomplete` when there is no record and
    `AcquisitionIntegrityError` for every other failure.
    """
    directory = Path(directory)
    path = record_path(directory)
    if not path.is_file():
        raise AcquisitionIncomplete(
            f"{source}: {directory} has no {RECORD_NAME}, so the acquisition is incomplete"
        )
    data = path.read_bytes()
    try:
        record = json.loads(data.decode("utf-8"))
        canonical = canonical_bytes(record)
    except (UnicodeDecodeError, ValueError) as exc:
        raise AcquisitionIntegrityError(f"{path} cannot be read as a record: {exc}") from exc
    if canonical != data:
        raise AcquisitionIntegrityError(f"{path} is not in the canonical form D46 fixes")
    _check_record(record, source, start, end)

    listed = set(record["pages"])
    on_disk = _pages_on_disk(directory, source)
    missing = sorted(listed - on_disk)
    if missing:
        raise AcquisitionIntegrityError(f"{source}: listed pages are missing: {missing}")
    unlisted = sorted(on_disk - listed)
    if unlisted:
        raise AcquisitionIntegrityError(f"{source}: pages the record does not list: {unlisted}")
    for digest in sorted(listed):
        if not _page_present(directory, source, digest):
            raise AcquisitionIntegrityError(
                f"{source}: page {digest} does not hold the content its name promises"
            )
    return VerifiedAcquisition(acquisition_id=sha256_hex(data), pages=tuple(sorted(listed)))

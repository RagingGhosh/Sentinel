"""The acquisition directory, its journal, its immutable record, and its verification.

Addendum D44, D46, D47 and D48. One acquisition per source and window::

    data/acquisitions/<source>/<start>_<end>/
        acquisition.json          the record, written last: its presence means complete
        journal.jsonl             one canonical line per completed slice
        start.json                the source's start snapshot, written once (D48)
        rewinds.jsonl             one canonical line per rewind, in order (D48)
        writer.lock               present while one writer holds the acquisition (D47)
        <source>/<sha256>.json.gz the pages -- the directory is the ingest raw root
        quarantine/<sha256 of the file's bytes>/<file name>   orphaned pages (D47)

**A drifted slice is fetched again through a rewind (D48).** `Acquisition.rewind`
names a completed slice; the rewind is first appended, durably, to `rewinds.jsonl`,
then the journal is cut immediately before that slice's line, so it and every later
slice become unfinished, and the pages only they listed go to the quarantine at once.
No line is rewritten and no key is ever repeated. Each rewind records the slices its
caller found changed, and a slice already recorded as changed is never rewound again:
the persisted history, not memory, is the authority, so the rule survives a resume. An
open that finds the latest rewind's own line still at its recorded place in the
journal completes that rewind first, without recording it again.

**The start state is the acquisition's, not a slice's (D48).** `start.json` is written
once and never replaced by a reopen, a resume or a rewind, and completion refuses if
persisted `start.json` and `source_details.start` differ.

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
from typing import Any, TypeGuard

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
START_NAME = "start.json"
REWINDS_NAME = "rewinds.jsonl"
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
EVENT_KEYS = frozenset({"at", "from_key", "journal_offset", "line_sha256", "reason", "removed"})
"""A rewind event: when, from which slice, where that slice's line began and its digest
(so an interrupted rewind is recognised exactly), why, and which slices it removed."""

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


def _append_line(path: Path, line: bytes) -> int:
    """Append one line, flushed to disk, and return the offset it begins at.

    Nothing rereads the file before its next append, so a line that did not land
    whole is taken back at once rather than left for the next open.
    """
    before = path.stat().st_size if path.exists() else 0
    try:
        with open(path, "ab") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        with open(path, "r+b") as stream:
            stream.truncate(before)
        raise
    return before


def _truncate(path: Path, size: int) -> None:
    """Cut `path` to its first `size` bytes, flushed to disk."""
    with open(path, "r+b") as stream:
        stream.truncate(size)
        stream.flush()
        os.fsync(stream.fileno())


def _quarantine_page(directory: Path, path: Path) -> Path:
    """Move a page to ``quarantine/<sha256 of its bytes>/<its name>`` (D47); never delete it."""
    destination = Path(directory) / QUARANTINE_DIR / sha256_hex(path.read_bytes()) / path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(path, destination)
    return destination


def _is_key_list(value: Any) -> TypeGuard[list[str]]:
    """A non-empty list of distinct strings: the slice keys a rewind names changed."""
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) for item in value)
        and len(set(value)) == len(value)
    )


def _event_problem(line: bytes) -> str | None:
    """Why `line` is not a rewind event an open can rely on, or `None` when it is."""
    try:
        event = json.loads(line.decode("utf-8"))
        if canonical_bytes(event) != line:
            return "it is not in canonical form"
    except (UnicodeDecodeError, ValueError) as exc:
        return str(exc)
    if not isinstance(event, dict) or not EVENT_KEYS <= set(event):
        return "an event needs at, from_key, journal_offset, line_sha256, reason and removed"
    offset = event["journal_offset"]
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        return "journal_offset is a byte offset"
    if not isinstance(event["line_sha256"], str) or not DIGEST.fullmatch(event["line_sha256"]):
        return "line_sha256 is a digest"
    if not isinstance(event["reason"], dict) or not _is_key_list(event["reason"].get("changed")):
        return "reason.changed is a list of slice keys"
    return None


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
            self._start = self._load_start()
            self._rewinds = self._load_rewinds()
            self._finish_rewind()
            # The journal is checked once, here; each later slice is checked as it is
            # recorded (D47). Completion verifies every listed page once more.
            self._slices, self._offsets = self._load()
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
            moved.append(_quarantine_page(self.directory, path))
        return tuple(moved)

    # --- the start state (D48) ---------------------------------------------------

    def _load_start(self) -> bytes | None:
        """The persisted start state's bytes, or `None` before one is recorded."""
        path = self.directory / START_NAME
        if not path.is_file():
            return None
        data = path.read_bytes()
        try:
            state = json.loads(data.decode("utf-8"))
            whole = canonical_bytes(state) == data
        except (UnicodeDecodeError, ValueError) as exc:
            raise AcquisitionIntegrityError(
                f"{path} cannot be read as a start state: {exc}"
            ) from exc
        if not whole or not isinstance(state, dict):
            raise AcquisitionIntegrityError(f"{path} is not a canonical JSON object (D48)")
        return data

    @property
    def start_state(self) -> dict[str, Any] | None:
        """The start state as `start.json` holds it, or `None` before it is recorded."""
        return None if self._start is None else json.loads(self._start)

    def record_start_state(self, state: Mapping[str, Any]) -> None:
        """Persist the start state, canonically and atomically. It is never replaced."""
        self._writable()
        if self._start is not None:
            raise AcquisitionError(
                f"{self.directory}: the start state is recorded once and never replaced (D48)"
            )
        if not isinstance(state, Mapping):
            raise AcquisitionError(f"the start state is a JSON object, not {type(state).__name__}")
        data = canonical_bytes(dict(state))
        _write_atomically(self.directory / START_NAME, data)
        self._start = data

    # --- the rewind history (D48) ------------------------------------------------

    def _load_rewinds(self) -> list[bytes]:
        """Every rewind event's line, in order, after taking back a partial last line."""
        path = self.directory / REWINDS_NAME
        if not path.is_file():
            return []
        data = path.read_bytes()
        events: list[bytes] = []
        offset = 0
        while offset < len(data):
            end = data.find(b"\n", offset)
            if end < 0:
                break
            line = data[offset : end + 1]
            problem = _event_problem(line)
            if problem:
                # The at-most-once rule rests on this history, so a damaged event is
                # never cut away: dropping it could let a slice be rewound twice.
                raise AcquisitionIntegrityError(
                    f"{path}: event {len(events) + 1} cannot be trusted ({problem})"
                )
            events.append(line)
            offset = end + 1
        if offset < len(data):
            # Only an append a hard kill interrupted leaves a partial last line. Its
            # rewind never began: the journal is cut only once the event is whole.
            _truncate(path, offset)
        return events

    def _finish_rewind(self) -> None:
        """Complete the latest rewind if the line it cut before is still journaled.

        The event names that line's byte offset and digest, so a line a later fetch
        journaled at the same place is never mistaken for the one the rewind removes.
        """
        if not self._rewinds:
            return
        event = json.loads(self._rewinds[-1])
        path = journal_path(self.directory)
        data = path.read_bytes() if path.is_file() else b""
        offset = event["journal_offset"]
        end = data.find(b"\n", offset)
        if end >= 0 and sha256_hex(data[offset : end + 1]) == event["line_sha256"]:
            _truncate(path, offset)

    @property
    def rewinds(self) -> tuple[dict[str, Any], ...]:
        """Every rewind event, in order, as `rewinds.jsonl` holds it."""
        return tuple(json.loads(line) for line in self._rewinds)

    def rewind(self, from_key: str, *, reason: Mapping[str, Any]) -> tuple[str, ...]:
        """Make `from_key` and every slice journaled after it unfinished (D48).

        ``reason["changed"]`` lists the slices found changed: `from_key`, the earliest,
        and any after it. A slice an earlier rewind listed as changed has been fetched
        again once already, so no rewind may remove it again. The event is made
        durable first; then the journal is cut before `from_key`'s line and the pages
        only the removed slices listed are quarantined. Returns the removed keys, in
        journal order.
        """
        self._writable()
        keys = list(self._slices)
        if from_key not in keys:
            raise AcquisitionError(f"slice {from_key!r} is not journaled, so it cannot be rewound")
        removed = keys[keys.index(from_key) :]
        changed = reason.get("changed")
        if not _is_key_list(changed):
            raise AcquisitionError(
                "a rewind's reason names 'changed', a non-empty list of distinct slice keys"
            )
        if from_key not in changed:
            raise AcquisitionError(f"a rewind from {from_key!r} must list it as changed")
        outside = [key for key in changed if key not in removed]
        if outside:
            raise AcquisitionError(
                f"changed slices {outside} are not journaled at or after {from_key!r}"
            )
        earlier = {key for event in self.rewinds for key in event["reason"]["changed"]}
        again = [key for key in removed if key in earlier]
        if again:
            raise AcquisitionError(
                f"slices {again} were found changed before and fetched again once; no "
                "rewind may remove them again, so the acquisition refuses (D48)"
            )
        offset = self._offsets[from_key]
        line = canonical_bytes(
            {
                "at": format_timestamp(self._now()),
                "from_key": from_key,
                "journal_offset": offset,
                "line_sha256": sha256_hex(canonical_bytes(self._slices[from_key])),
                "reason": dict(reason),
                "removed": removed,
            }
        )
        _append_line(self.directory / REWINDS_NAME, line)
        self._rewinds.append(line)
        try:
            _truncate(journal_path(self.directory), offset)
            gone = [self._slices.pop(key) for key in removed]
            for key in removed:
                del self._offsets[key]
            kept = {digest for entry in self._slices.values() for digest in entry["pages"]}
            folder = pages_dir(self.directory, self.source)
            for digest in sorted({d for entry in gone for d in entry["pages"]} - kept):
                _quarantine_page(self.directory, folder / f"{digest}{PAGE_SUFFIX}")
        except BaseException:
            # The event is durable, so the rewind has begun and only an open completes
            # it deterministically. Until then this writer writes nothing more.
            self.close()
            raise
        return tuple(removed)

    # --- the journal -------------------------------------------------------------

    def _load(self) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
        """Completed slices and their lines' byte offsets, by key.

        Everything from the first line that is not a completed slice is removed.
        """
        path = journal_path(self.directory)
        if not path.is_file():
            return {}, {}
        data = path.read_bytes()
        completed: dict[str, dict[str, Any]] = {}
        offsets: dict[str, int] = {}
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
            offsets[entry["key"]] = offset
            offset = end + 1
        if offset < len(data):
            _truncate(path, offset)
        return completed, offsets

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
        self._offsets[key] = _append_line(journal_path(self.directory), line)
        self._slices[key] = entry

    # --- completion --------------------------------------------------------------

    def complete(self, source_details: Mapping[str, Any]) -> str:
        """Write the record, last and atomically, and return its `acquisition_id`."""
        self._writable()
        if self._start is not None and canonical_bytes(source_details.get("start")) != self._start:
            raise AcquisitionIntegrityError(
                f"{self.directory}: source_details.start differs from the persisted "
                f"{START_NAME}; no record was written (D48)"
            )
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
            "rewinds": list(self.rewinds),
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

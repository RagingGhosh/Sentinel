"""CFPB's fetcher: the Narratives Archive joined to the Consumer Complaint Database API by
exact Complaint ID (D43, D44, D49; plan Task 25).

The reconstruction has two sources, each authoritative for its own fields. The archive
alone gives the narrative; the API alone gives `date_received` and
`date_sent_to_company`; `product` and `timely` must agree between them, and the
archive's `Date received` must equal the UTC calendar date of the API's. Any
disagreement, and any included archive record with no usable API record, refuses the
acquisition (D43).

**The archive.** The reading room's list of exports is read, and each export is
selected by the months its name states: an export whose months meet the window is a
population export; one that meets only the day before or after the window is boundary
only, and holds no row of the population. Each selected export is downloaded through
the context's client, retained in the acquisition directory and pinned by SHA256, and
its single CSV member is streamed from the ZIP, never extracted. Its content, not its
name, decides: the header must be the verified 16 columns, every `Date received` a
bare `YYYY-MM-DD`, every Complaint ID a plain decimal string found once across every
selected export, no boundary export may hold a window row, and every window day must
appear in the content. A row is in the population when it is dated inside the window
and its narrative is not blank after `strip()`; every other row is counted, never
dropped silently (D44 (1)).

**The API, one UTC day at a time.** For each window day and one margin day at each end,
the same-day JSON count, whose `hits.total` must be exact, then the day's CSV. The CSV's
rows must equal the count, its Complaint IDs must be distinct and every row must fall
on the requested UTC day; rows that differ from the count re-request both, once, and a
second disagreement refuses (D49 (2)). A literal `None` in the Complaint ID column
refuses (D49 (4)); a literal `None` in the product, timely or either date of a row the
join uses is resolved by `GET /{complaintId}`, whose missing-record answer is either
HTTP 404 or zero hits. No other by-ID request is made.

**Pages.** One page per window day, in the adapter's `hits.hits[]._source` shape with
its six fields, rows sorted by (`date_received`, `complaint_id`) and every string
copied exactly as served. The margin days are checked and journaled but yield no page.

**Drift is recorded, never acted on (D49 (1)).** Each day's source snapshot is kept, and
the start and end snapshots are provenance only; CFPB historical stability is not
asserted, and reproducibility rests on the retained archive exports and raw API
responses. The pins are persisted in the acquisition's start state, and a resume
refuses a pinned export that is missing or whose bytes have changed (D49 (3)).

**Memory.** No narrative is held for the whole archive: each wanted day's rows are
written to a per-day spill file while the exports stream, and the index of every
archive row is a sorted array of 64-bit integers.

Imports only the standard library and ``ingest.fetch`` (D44).
"""

from __future__ import annotations

import bisect
import csv
import gzip
import heapq
import io
import json
import os
import re
import shutil
import tempfile
import zipfile
from array import array
from calendar import monthrange
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ingest.fetch.acquisition import Acquisition, AcquisitionError
from ingest.fetch.canonical import page_digest, sha256_hex
from ingest.fetch.http import Fetched, FetchFailed, InvalidResponse, RequestRecord, Response

if TYPE_CHECKING:
    from ingest.fetch.registry import FetchContext

SOURCE = "cfpb"
"""The adapter's `SOURCE_SLUG`, restated: this package may not import the adapter."""

SOURCE_API_VERSION = "cfpb-ccdb-v1"
"""The adapter's constant, unchanged (D44 (3)); a test holds both equal."""

ACQUISITION_KIND = "cfpb-archive-api-reconstruction-v1"

READING_ROOM = (
    "https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/"
    "cfpb-consumer-complaint-database-narratives-archive/"
)
API_URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
BY_ID = API_URL + "{complaintId}"
EXPORT_LINK = re.compile(
    r"https://files\.consumerfinance\.gov/f/documents/CCDB_Export_(\d+)_([A-Za-z0-9_]+)\.zip"
)
NAMED_MONTHS = re.compile(r"([A-Z][a-z]+)_(\d{4})(?:_through_([A-Z][a-z]+)_(\d{4}))?")
MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)  # fmt: skip

ARCHIVE_HEADER = (
    "Date received", "Product", "Sub-product", "Issue", "Sub-issue",
    "Consumer complaint narrative", "Company public response", "Company", "State",
    "ZIP code", "Tags", "Submitted via", "Date sent to company",
    "Company response to consumer", "Timely response?", "Complaint ID",
)  # fmt: skip
"""Export #8's verified header; every export must carry exactly these 16 columns."""

API_HEADER = (
    "Date received", "Product", "Sub-product", "Issue", "Sub-issue",
    "Company public response", "Company", "State", "ZIP code", "Tags", "Submitted via",
    "Date sent to company", "Company response to consumer", "Timely response?",
    "Complaint ID",
)  # fmt: skip
"""The API CSV export's 15 columns, as served."""

NONE = "None"
"""How the API's CSV spells a JSON null: ambiguous with a value, so never converted."""

FALLBACK = {
    "Product": "product",
    "Timely response?": "timely",
    "Date received": "date_received",
    "Date sent to company": "date_sent_to_company",
}
"""The adapter-relevant columns a `None` in which is resolved by Complaint ID (D44 (4))."""

NARRATIVE_FILTER = (
    "an archive row dated inside the window is included when its narrative is a string "
    "that is not blank after str.strip(); every other row is excluded and counted"
)
COUNT_PARAMS = (("size", "1"), ("no_aggs", "true"), ("no_highlight", "true"))
CSV_PARAMS = (("format", "csv"), ("no_aggs", "true"))

ARCHIVE_DIR = "archive"
API_DIR = "api"
SPILL_DIR = "spill"
"""Subdirectories Task 25 defines under D46 (B3); none is the page directory `cfpb/`."""

ONE_DAY = timedelta(days=1)
DAY_BITS = 21
"""An ordinal day fits in 21 bits until the year 5741; the index key is
``id << 22 | ordinal << 1 | included``."""
ID_SHIFT = DAY_BITS + 1
DAY_MASK = (1 << DAY_BITS) - 1
SPILL_FLUSH_CHARS = 32 * 1024 * 1024

Page = dict[str, Any]


class CFPBFetchError(AcquisitionError):
    """A CFPB acquisition cannot continue; the acquisition stays incomplete."""


class ContextMismatch(CFPBFetchError):
    """The fetcher was given, or called for, a source or window that is not its own."""


class ArchiveDiscoveryError(CFPBFetchError):
    """The reading room's list of exports cannot be read or selects nothing."""


class ArchiveContentError(CFPBFetchError):
    """An export's content is not what the verified archive holds (D44 (5))."""


class ArchivePinMismatch(CFPBFetchError):
    """A pinned export is missing or its bytes have changed; never re-pinned (D49 (3))."""


class MissingStartState(CFPBFetchError):
    """Days are journaled but no start state is, so the pins are unknown."""


class DuplicateComplaintId(CFPBFetchError):
    """A Complaint ID appears twice in the archive, or twice in one API day."""


class CountNotExact(CFPBFetchError):
    """`hits.total` is a lower bound, not a count."""


class CountDisagreement(CFPBFetchError):
    """A day's CSV rows differed from its count twice (D49 (2))."""


class RowOutsideDay(CFPBFetchError):
    """An API row's `date_received` is not on the requested UTC day."""


class UnusableApiRecord(CFPBFetchError):
    """An API record cannot be read: a `None` Complaint ID, an unreadable instant, or
    a by-ID answer with more than one hit or a null value where one is required."""


class MissingApiRecord(CFPBFetchError):
    """An included archive record has no API record: absent from its day, a by-ID 404,
    or a by-ID answer with zero hits (D43)."""


class ReturnedIdMismatch(CFPBFetchError):
    """A by-ID answer names another Complaint ID than the one requested (D43)."""


class FieldDisagreement(CFPBFetchError):
    """`product` or `timely` differs between the archive and the API (D43)."""


class DateMismatch(CFPBFetchError):
    """A record's archive date is not the UTC date of its API `date_received` (D43)."""


class PopulationMismatch(CFPBFetchError):
    """The days' matched records do not add up to the archive's included records."""


@dataclass(frozen=True)
class Export:
    """One export the reading room lists and the window selects."""

    number: int
    url: str
    first: date
    last: date
    role: str
    """``population`` or ``boundary``."""

    @property
    def name(self) -> str:
        return self.url.rsplit("/", 1)[1]


def named_months(label: str) -> tuple[date, date]:
    """The first and last day of the months an export's file name states."""
    match = NAMED_MONTHS.fullmatch(label)
    if match is None or any(m not in MONTHS for m in (match.group(1), match.group(3)) if m):
        raise ArchiveDiscoveryError(f"cannot read the months an export named {label!r} covers")
    first = date(int(match.group(2)), MONTHS.index(match.group(1)) + 1, 1)
    year = int(match.group(4) or match.group(2))
    month = MONTHS.index(match.group(3) or match.group(1)) + 1
    last = date(year, month, monthrange(year, month)[1])
    if last < first:
        raise ArchiveDiscoveryError(f"an export named {label!r} ends before it starts")
    return first, last


def _role(first: date, last: date, start: date, end: date) -> str | None:
    if first <= end and last >= start:
        return "population"
    if first <= start - ONE_DAY <= last or first <= end + ONE_DAY <= last:
        return "boundary"
    return None


def select_exports(page: str, start: date, end: date) -> list[Export]:
    """The exports the window needs, by number, from the reading room's HTML."""
    urls: dict[int, str] = {}
    for match in EXPORT_LINK.finditer(page):
        number = int(match.group(1))
        if urls.setdefault(number, match.group(0)) != match.group(0):
            raise ArchiveDiscoveryError(f"export #{number} is listed at two URLs")
    selected = []
    for number, url in sorted(urls.items()):
        first, last = named_months(EXPORT_LINK.fullmatch(url).group(2))  # type: ignore[union-attr]
        role = _role(first, last, start, end)
        if role is not None:
            selected.append(Export(number, url, first, last, role))
    if not any(export.role == "population" for export in selected):
        raise ArchiveDiscoveryError(f"no export listed in the reading room covers {start}..{end}")
    return selected


def _write_file(path: Path, data: bytes) -> None:
    """Written beside its destination and moved into place, so never seen half-written."""
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _canonical_id(value: str) -> bool:
    return value.isascii() and value.isdigit() and value == str(int(value))


def _date_only(value: str) -> date | None:
    if len(value) != 10 or value[4] != "-" or value[7] != "-":
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


# --- response bodies: a body that is not what its request returns is invalid, so retried --


def _json(body: bytes) -> Any:
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise InvalidResponse(f"the body is not JSON: {exc}") from exc


def _hits(value: Any) -> dict[str, Any]:
    hits = value.get("hits") if isinstance(value, dict) else None
    total = hits.get("total") if isinstance(hits, dict) else None
    if not (
        isinstance(total, dict)
        and isinstance(total.get("value"), int)
        and not isinstance(total.get("value"), bool)
        and isinstance(total.get("relation"), str)
        and isinstance(hits.get("hits"), list)  # type: ignore[union-attr]
    ):
        raise InvalidResponse("a search response needs hits.total {value, relation} and hits.hits")
    return value


def parse_count(body: bytes) -> dict[str, Any]:
    value = _hits(_json(body))
    if not isinstance(value.get("_meta"), dict):
        raise InvalidResponse("a count response carries _meta")
    return value


def parse_by_id(body: bytes) -> dict[str, Any]:
    return _hits(_json(body))


def parse_csv(body: bytes) -> list[list[str]]:
    """A day's CSV: the 15 served columns, every row that wide."""
    try:
        rows = list(csv.reader(io.StringIO(body.decode("utf-8"), newline="")))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise InvalidResponse(f"the CSV cannot be read: {exc}") from exc
    if not rows or tuple(rows[0]) != API_HEADER:
        raise InvalidResponse(f"the CSV header is {rows[0] if rows else None!r}")
    if any(len(row) != len(API_HEADER) for row in rows[1:]):
        raise InvalidResponse("a CSV row does not have the 15 columns")
    return rows[1:]


def parse_zip(response: Response) -> None:
    length = response.headers.get("content-length")
    if length is not None and length.isdigit() and int(length) != len(response.body):
        raise InvalidResponse(f"short body: {len(response.body)} of {length} bytes")
    if not zipfile.is_zipfile(io.BytesIO(response.body)):
        raise InvalidResponse("not a ZIP archive")


def parse_page(response: Response) -> None:
    if not response.body:
        raise InvalidResponse("empty reading-room page")


class _Spill:
    """Each wanted day's archive rows, on disk, so no narrative is held for long."""

    def __init__(self, folder: Path, days: set[str]) -> None:
        self.folder = folder
        self.days = days
        self.buffers: dict[str, list[str]] = {}
        self.size = 0
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)

    def add(self, day: str, record: list[str]) -> None:
        if day not in self.days:
            return
        line = json.dumps(record, ensure_ascii=False) + "\n"
        self.buffers.setdefault(day, []).append(line)
        self.size += len(line)
        if self.size >= SPILL_FLUSH_CHARS:
            self.flush()

    def flush(self) -> None:
        for day, lines in sorted(self.buffers.items()):
            with gzip.open(self.folder / f"{day}.jsonl.gz", "ab") as stream:
                stream.write("".join(lines).encode("utf-8"))
        self.buffers.clear()
        self.size = 0

    def read(self, day: str) -> tuple[dict[str, tuple[str, str, str]], set[str]]:
        """The day's included rows by Complaint ID, and its other archive IDs."""
        included: dict[str, tuple[str, str, str]] = {}
        excluded: set[str] = set()
        path = self.folder / f"{day}.jsonl.gz"
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8", newline="\n") as stream:
                for line in stream:
                    kind, cid, *rest = json.loads(line)
                    if kind == "i":
                        included[cid] = (rest[0], rest[1], rest[2])
                    else:
                        excluded.add(cid)
        return included, excluded


@dataclass
class _Archive:
    keys: array[int]
    included_total: int
    facts: list[dict[str, Any]]
    spill: _Spill

    def lookup(self, cid: str) -> tuple[int, bool] | None:
        """The archive date (as an ordinal) and whether the record is included."""
        if not _canonical_id(cid):
            return None
        number = int(cid)
        at = bisect.bisect_left(self.keys, number << ID_SHIFT)
        if at < len(self.keys) and self.keys[at] >> ID_SHIFT == number:
            key = self.keys[at]
            return (key >> 1) & DAY_MASK, bool(key & 1)
        return None


class _Run:
    """One run of the fetcher over its context's window."""

    def __init__(self, context: FetchContext) -> None:
        self.context = context
        self.start, self.end = context.start, context.end
        span = (self.end - self.start).days + 3
        self.days = [self.start - ONE_DAY + ONE_DAY * n for n in range(span)]
        self.archive_dir = Path(context.directory) / ARCHIVE_DIR
        self.api_dir = Path(context.directory) / API_DIR

    def in_window(self, day: date) -> bool:
        return self.start <= day <= self.end

    # --- the acquisition ------------------------------------------------------------------

    def pages(self) -> Iterator[Page]:
        context = self.context
        with Acquisition(
            context.directory,
            source=SOURCE,
            start=self.start,
            end=self.end,
            resolved_start=context.resolved_start,
            resolved_end=context.resolved_end,
            client=context.client,
            sentinel_commit=context.sentinel_commit,
            now=context.now,
        ) as acquisition:
            done = acquisition.completed_slices()
            start_state = acquisition.start_state
            wanted = {day.isoformat() for day in self.days} - set(done)
            if start_state is None:
                if done:
                    raise MissingStartState(
                        f"{context.directory}: days are journaled but there is no start state, "
                        "so the archive pins are unknown; no later one is substituted (D49)"
                    )
                reading_room, exports = self.discover()
                pins = [self.download(export) for export in exports]
                archive = self.read_archive(exports, pins, wanted)
                start_state = {
                    "api": self.snapshot(),
                    "archive": {"exports": pins, "reading_room": reading_room},
                }
                acquisition.record_start_state(start_state)
            else:
                pins = start_state["archive"]["exports"]
                exports = self.pinned(pins)
                archive = self.read_archive(exports, pins, wanted)
            for day in self.days:
                key = day.isoformat()
                if key in done:
                    continue
                page, requests, verification = self.fetch_day(day, archive)
                if page is not None:
                    yield page
                # Reached only after the caller has cached the page (D46).
                acquisition.record_slice(
                    key,
                    requests=requests,
                    pages=[] if page is None else [page_digest(page)],
                    verification=verification,
                )
            end = self.snapshot()
            slices = acquisition.completed_slices()
            join = self.reconcile(slices, archive)
            acquisition.complete(
                {
                    "acquisition_kind": ACQUISITION_KIND,
                    "api": {
                        "base_url": API_URL,
                        "days": {key: self.day_provenance(entry) for key, entry in slices.items()},
                        "endpoints": {"by_id": BY_ID, "search": API_URL},
                        "parameters": {
                            "count": [["date_received_min", "D"], ["date_received_max", "D"]]
                            + [list(pair) for pair in COUNT_PARAMS],
                            "csv": [["date_received_min", "D"], ["date_received_max", "D"]]
                            + [list(pair) for pair in CSV_PARAMS],
                        },
                    },
                    "archive": {
                        "exports": archive.facts,
                        "narrative_filter": NARRATIVE_FILTER,
                        "reading_room": start_state["archive"]["reading_room"],
                    },
                    "end": end,
                    "join": join,
                    "source_api_version": SOURCE_API_VERSION,
                    "start": start_state,
                }
            )

    # --- the archive ----------------------------------------------------------------------

    def discover(self) -> tuple[dict[str, Any], list[Export]]:
        fetched = self.context.http.get(READING_ROOM, validate=parse_page)
        body = fetched.response.body
        _write_file(self.archive_dir / f"reading-room-{fetched.record.response_sha256}.html", body)
        exports = select_exports(body.decode("utf-8", "replace"), self.start, self.end)
        room = {
            "retrieved_at": fetched.record.retrieved_at,
            "sha256": fetched.record.response_sha256,
            "url": READING_ROOM,
        }
        return room, exports

    def download(self, export: Export) -> dict[str, Any]:
        fetched = self.context.http.get(export.url, validate=parse_zip)
        body = fetched.response.body
        _write_file(self.archive_dir / export.name, body)
        headers = fetched.response.headers
        return {
            "bytes": len(body),
            "etag": headers.get("etag"),
            "last_modified": headers.get("last-modified"),
            "number": export.number,
            "sha256": sha256_hex(body),
            "url": export.url,
        }

    def pinned(self, pins: Sequence[dict[str, Any]]) -> list[Export]:
        """The pinned exports, each present with exactly its pinned bytes, or a refusal."""
        exports = []
        for pin in pins:
            match = EXPORT_LINK.fullmatch(pin["url"])
            if match is None:
                raise ArchivePinMismatch(f"a pinned export has an unreadable URL: {pin['url']!r}")
            first, last = named_months(match.group(2))
            role = _role(first, last, self.start, self.end)
            export = Export(pin["number"], pin["url"], first, last, role or "boundary")
            path = self.archive_dir / export.name
            if not path.is_file():
                raise ArchivePinMismatch(f"pinned export {export.name} is missing; not re-pinned")
            if sha256_hex(path.read_bytes()) != pin["sha256"]:
                raise ArchivePinMismatch(
                    f"pinned export {export.name} no longer has its pinned bytes; not re-pinned"
                )
            exports.append(export)
        return exports

    def read_archive(
        self, exports: Sequence[Export], pins: Sequence[dict[str, Any]], wanted: set[str]
    ) -> _Archive:
        spill = _Spill(Path(self.context.directory) / SPILL_DIR, wanted)
        parts: list[array[int]] = []
        facts = []
        covered: set[date] = set()
        included_total = 0
        for export, pin in zip(exports, pins, strict=True):
            keys, fact, included = self.read_export(export, spill, covered)
            parts.append(array("q", sorted(keys)))
            facts.append({**pin, **fact, "role": export.role})
            included_total += included
        spill.flush()
        missing = [
            day.isoformat() for day in self.days if self.in_window(day) and day not in covered
        ]
        if missing:
            raise ArchiveContentError(f"window days no export's content covers: {missing[:10]}")
        merged: array[int] = array("q")
        repeated: list[int] = []
        previous = -1
        for key in heapq.merge(*parts):
            number = key >> ID_SHIFT
            if number == previous:
                repeated.append(number)
            previous = number
            merged.append(key)
        if repeated:
            raise DuplicateComplaintId(
                f"Complaint IDs in the archive more than once: {[str(n) for n in repeated[:10]]}"
            )
        return _Archive(keys=merged, included_total=included_total, facts=facts, spill=spill)

    def read_export(
        self, export: Export, spill: _Spill, covered: set[date]
    ) -> tuple[array[int], dict[str, Any], int]:
        """Stream one export's member, check every row, index it and spill what is wanted."""
        path = self.archive_dir / export.name
        keys: array[int] = array("q")
        rows = included = 0
        first: date | None = None
        last: date | None = None
        at = {name: i for i, name in enumerate(ARCHIVE_HEADER)}
        try:
            with zipfile.ZipFile(path) as bundle:
                members = bundle.infolist()
                if len(members) != 1:
                    raise ArchiveContentError(
                        f"{export.name} holds {len(members)} members, not one CSV member"
                    )
                member = members[0]
                with bundle.open(member) as raw:
                    reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
                    header = next(reader, None)
                    if header is None or tuple(header) != ARCHIVE_HEADER:
                        raise ArchiveContentError(
                            f"{export.name}: the header {header!r} is not the verified 16 columns"
                        )
                    for row in reader:
                        rows += 1
                        if len(row) != len(ARCHIVE_HEADER):
                            raise ArchiveContentError(
                                f"{export.name}: row {rows} has {len(row)} columns, not 16"
                            )
                        cid = row[at["Complaint ID"]]
                        if not _canonical_id(cid):
                            raise ArchiveContentError(
                                f"{export.name}: row {rows}: Complaint ID {cid!r} is not a "
                                "plain decimal string"
                            )
                        day = _date_only(row[at["Date received"]])
                        if day is None:
                            raise ArchiveContentError(
                                f"{export.name}: row {rows}: Date received "
                                f"{row[at['Date received']]!r} is not YYYY-MM-DD"
                            )
                        inside = self.in_window(day)
                        if inside and export.role == "boundary":
                            raise ArchiveContentError(
                                f"{export.name} is a boundary export but holds a row dated {day}"
                            )
                        story = row[at["Consumer complaint narrative"]]
                        keep = inside and bool(story.strip())
                        if keep:
                            included += 1
                            spill.add(
                                day.isoformat(),
                                ["i", cid, row[at["Product"]], row[at["Timely response?"]], story],
                            )
                        else:
                            spill.add(day.isoformat(), ["x", cid])
                        if inside:
                            covered.add(day)
                        keys.append((int(cid) << ID_SHIFT) | (day.toordinal() << 1) | keep)
                        first = day if first is None or day < first else first
                        last = day if last is None or day > last else last
        except (zipfile.BadZipFile, UnicodeDecodeError, csv.Error) as exc:
            raise ArchiveContentError(f"{export.name} cannot be read: {exc}") from exc
        fact = {
            "date_max": last.isoformat() if last else None,
            "date_min": first.isoformat() if first else None,
            "header": list(ARCHIVE_HEADER),
            "member": member.filename,
            "member_bytes": member.file_size,
            "rows": rows,
        }
        return keys, fact, included

    # --- the API --------------------------------------------------------------------------

    def get(
        self, url: str, params: Sequence[tuple[str, str]], parse: Callable[[bytes], Any]
    ) -> tuple[Any, Fetched]:
        """GET under D47's policy, keeping the raw body; a body `parse` refuses is retried."""
        parsed = []

        def validate(response: Response) -> None:
            parsed.append(parse(response.body))

        fetched = self.context.http.get(url, params, validate=validate)
        suffix = "csv" if parse is parse_csv else "json"
        _write_file(
            self.api_dir / f"{fetched.record.response_sha256}.{suffix}.gz",
            gzip.compress(fetched.response.body, mtime=0),
        )
        return parsed[-1], fetched

    def count(self, first: date, last: date) -> tuple[dict[str, Any], Fetched]:
        bounds = (("date_received_min", first.isoformat()), ("date_received_max", last.isoformat()))
        data, fetched = self.get(API_URL, bounds + COUNT_PARAMS, parse_count)
        relation = data["hits"]["total"]["relation"]
        if relation != "eq":
            raise CountNotExact(f"{first}..{last}: hits.total is {relation!r}, not an exact count")
        return data, fetched

    def snapshot(self) -> dict[str, Any]:
        """The window's count and the API's metadata: provenance only (D49 (1))."""
        data, fetched = self.count(self.start, self.end)
        meta = data["_meta"]
        hits = data["hits"]["hits"]
        return {
            "_index": hits[0].get("_index") if hits and isinstance(hits[0], dict) else None,
            "hits_total": data["hits"]["total"]["value"],
            "last_indexed": meta.get("last_indexed"),
            "last_updated": meta.get("last_updated"),
            "request": fetched.record.as_record(),
            "total_record_count": meta.get("total_record_count"),
        }

    def by_id(self, cid: str) -> tuple[dict[str, Any], RequestRecord]:
        try:
            data, fetched = self.get(API_URL + cid, (), parse_by_id)
        except FetchFailed as exc:
            if exc.status == 404:
                raise MissingApiRecord(f"Complaint ID {cid}: GET by ID answered 404") from exc
            raise
        total = data["hits"]["total"]["value"]
        hits = data["hits"]["hits"]
        if total == 0 and not hits:
            raise MissingApiRecord(f"Complaint ID {cid}: GET by ID found no record")
        if total != 1 or len(hits) != 1 or not isinstance(hits[0].get("_source"), dict):
            raise UnusableApiRecord(f"Complaint ID {cid}: GET by ID found {total} records, not one")
        source = hits[0]["_source"]
        if source.get("complaint_id") != cid:
            raise ReturnedIdMismatch(
                f"GET by ID for {cid} returned Complaint ID {source.get('complaint_id')!r}"
            )
        return source, fetched.record

    def fetch_day(
        self, day: date, archive: _Archive
    ) -> tuple[Page | None, list[RequestRecord], dict[str, int]]:
        key = day.isoformat()
        disagreed = []
        # The count and the CSV, and at most one fresh pair of both (D49 (2)).
        for _ in range(2):
            counted, count_fetch = self.count(day, day)
            rows, csv_fetch = self.get(
                API_URL,
                (("date_received_min", key), ("date_received_max", key)) + CSV_PARAMS,
                parse_csv,
            )
            total = counted["hits"]["total"]["value"]
            if len(rows) == total:
                break
            disagreed.append({"count": total, "rows": len(rows)})
        else:
            raise CountDisagreement(
                f"{key}: the CSV's rows differed from the count twice, {disagreed}; the "
                "acquisition stays incomplete"
            )
        at = {name: i for i, name in enumerate(API_HEADER)}
        ids = [row[at["Complaint ID"]] for row in rows]
        if NONE in ids:
            raise UnusableApiRecord(
                f"{key}: a Complaint ID is the literal None, so the record can be neither "
                "joined nor classified; no identifier is invented (D49 (4))"
            )
        repeated = sorted(cid for cid, n in Counter(ids).items() if n > 1)
        if repeated:
            raise DuplicateComplaintId(f"{key}: Complaint IDs repeated in the day: {repeated[:10]}")
        for row in rows:
            if row[at["Date received"]] != NONE:
                self.check_day(row[at["Complaint ID"]], row[at["Date received"]], day)

        included, excluded = archive.spill.read(key)
        requests = [count_fetch.record, csv_fetch.record]
        counts = Counter[str]()
        matched: list[dict[str, Any]] = []
        for row in rows:
            cid = row[at["Complaint ID"]]
            if cid in included:
                product, timely, story = included[cid]
                values = self.values(row, at, cid, requests, counts)
                self.check_day(cid, values["date_received"], day)
                for field, archived in (("product", product), ("timely", timely)):
                    if values[field] != archived:
                        raise FieldDisagreement(
                            f"Complaint ID {cid}: {field} is {archived!r} in the archive and "
                            f"{values[field]!r} in the API"
                        )
                matched.append({"complaint_id": cid, "complaint_what_happened": story, **values})
                continue
            found = archive.lookup(cid)
            if found is None:
                counts["api_only"] += 1
            elif found[0] != day.toordinal():
                raise DateMismatch(
                    f"Complaint ID {cid} is on UTC day {key} in the API but dated "
                    f"{date.fromordinal(found[0])} in the archive"
                )
            else:
                counts["excluded"] += 1
        absent = sorted(set(included) - {row["complaint_id"] for row in matched})
        if absent:
            raise MissingApiRecord(
                f"{key}: included archive records with no API record: {absent[:10]}"
            )
        verification = {
            "api_count": total,
            "api_distinct_ids": len(set(ids)),
            "api_rows": len(rows),
            "archive_only": len(excluded - set(ids)),
            "excluded": counts["excluded"],
            "included": len(included),
            "matched": len(matched),
            "api_only": counts["api_only"],
            "resolved_by_id": counts["resolved_by_id"],
        }
        if not self.in_window(day):
            return None, requests, verification
        matched.sort(key=lambda row: (row["date_received"], row["complaint_id"]))
        return {"hits": {"hits": [{"_source": row} for row in matched]}}, requests, verification

    def values(
        self,
        row: list[str],
        at: dict[str, int],
        cid: str,
        requests: list[RequestRecord],
        counts: Counter[str],
    ) -> dict[str, Any]:
        """A joined row's API fields, each literal None resolved by Complaint ID."""
        values: dict[str, Any] = {field: row[at[column]] for column, field in FALLBACK.items()}
        if any(row[at[column]] == NONE for column in FALLBACK):
            source, record = self.by_id(cid)
            requests.append(record)
            counts["resolved_by_id"] += 1
            for column, field in FALLBACK.items():
                if row[at[column]] == NONE:
                    values[field] = source.get(field)
            for field in ("product", "timely", "date_received"):
                if not isinstance(values[field], str):
                    raise UnusableApiRecord(f"Complaint ID {cid}: {field} is {values[field]!r}")
            sent = values["date_sent_to_company"]
            if sent is not None and not isinstance(sent, str):
                raise UnusableApiRecord(f"Complaint ID {cid}: date_sent_to_company is {sent!r}")
        return values

    @staticmethod
    def check_day(cid: str, value: str, day: date) -> None:
        try:
            moment = datetime.fromisoformat(value)
        except ValueError as exc:
            raise UnusableApiRecord(f"Complaint ID {cid}: Date received {value!r}") from exc
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise UnusableApiRecord(f"Complaint ID {cid}: Date received {value!r} has no offset")
        if moment.astimezone(UTC).date() != day:
            raise RowOutsideDay(f"Complaint ID {cid} is dated {value}, not on UTC day {day}")

    # --- completion -----------------------------------------------------------------------

    def day_provenance(self, entry: dict[str, Any]) -> dict[str, Any]:
        """A day's hits_total, last_indexed and _index, from its retained count body."""
        digest = entry["requests"][0]["response_sha256"]
        path = self.api_dir / f"{digest}.json.gz"
        if not path.is_file():
            raise CFPBFetchError(f"{entry['key']}: the retained count response {digest} is missing")
        data = json.loads(gzip.decompress(path.read_bytes()))
        hits = data["hits"]["hits"]
        return {
            "_index": hits[0].get("_index") if hits else None,
            "hits_total": data["hits"]["total"]["value"],
            "last_indexed": data["_meta"].get("last_indexed"),
        }

    def reconcile(self, slices: dict[str, dict[str, Any]], archive: _Archive) -> dict[str, Any]:
        window = [entry["verification"] for key, entry in slices.items()
                  if self.in_window(date.fromisoformat(key))]  # fmt: skip
        totals = {
            field: sum(v[field] for v in window)
            for field in ("matched", "excluded", "archive_only", "api_only", "resolved_by_id")
        }
        if totals["matched"] != archive.included_total:
            raise PopulationMismatch(
                f"the days matched {totals['matched']} records but the archive includes "
                f"{archive.included_total}"
            )
        return {"key": "exact Complaint ID string equality", **totals}


def make_cfpb_fetcher(context: FetchContext) -> Callable[[str, date, date], Iterator[Page]]:
    """The CFPB `Fetcher` for `context`, which must be a CFPB context."""
    if context.source != SOURCE:
        raise ContextMismatch(f"the CFPB fetcher was given a {context.source!r} context")

    def fetch(source: str, start: date, end: date) -> Iterator[Page]:
        if (source, start, end) != (context.source, context.start, context.end):
            raise ContextMismatch(
                f"asked for {source} {start} .. {end}; this fetcher acquires only "
                f"{context.source} {context.start} .. {context.end}"
            )
        return _Run(context).pages()

    return fetch

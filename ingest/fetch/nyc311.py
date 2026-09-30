"""NYC 311's fetcher: day-sliced, count-verified SODA 2.0 pages (D44 (7)-(10), D48).

One slice per New York civil day D, and exactly one page per slice: the array the data
request returns, yielded unchanged. For each day the count comes first, then the data::

    count   $select=count(*) AS n
            $where=created_date >= 'DT00:00:00' AND created_date < 'D+1T00:00:00'
    data    $select=unique_key,created_date,closed_date,complaint_type,descriptor
            $where=<the same>   $order=unique_key   $limit=50000

The bounds are literal floating-timestamp strings, so a daylight-saving day needs no
conversion and loses no row. No app token is read, sent or logged (D44 (9)).

**A day is accepted only when its count, its rows and its distinct `unique_key`s
agree.** A count of 50,000 or more refuses before the data is requested, since a page
is never truncated. A `unique_key` repeated within a day refuses at once. Rows that
differ from the count are a disagreement at the source, not an HTTP failure: the count
and the data are both requested again, once, and a second disagreement refuses (D48
(3)). D47's retry policy applies beneath every one of those requests, and a body that
is not what its request returns is retried as D47 retries any invalid body.

**The source is checked again once every day is fetched (D48 (4)).** A snapshot --
the metadata's `rowsUpdatedAt` and the window's count -- that differs from the start
snapshot, or from the latest snapshot at which every completed day was verified, is
movement: every completed day's count is requested again, and the earliest day whose
count changed, in journal order, is the rewind point. The rewind records the changed
days, so a day found changed once is never rewound again (D48 (1)). A pass that finds
no change ends the checks; its window count must equal the sum of the day counts, and
no `unique_key` may appear in two days.

**Rows pass through untouched (D45).** A null or absent descriptor, a negative
duration, a daylight-saving-edge timestamp: none is filtered, repaired or converted.
Normalization decides what happens to them, and D45 is undecided.

The start snapshot is recorded once, before the first day, and a resumed acquisition
keeps it. Imports only the standard library and ``ingest.fetch`` (D44).
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

from ingest.fetch.acquisition import PAGE_SUFFIX, Acquisition, AcquisitionError, pages_dir
from ingest.fetch.canonical import page_digest
from ingest.fetch.http import Fetched, InvalidResponse, RequestRecord, Response

if TYPE_CHECKING:
    from ingest.fetch.registry import FetchContext

SOURCE = "nyc311"
"""The adapter's `SOURCE_SLUG`, restated: this package may not import the adapter."""

SOURCE_API_VERSION = "socrata-soda2"
"""The adapter's `SOURCE_API_VERSION`, restated for the same reason; a test holds both
equal to the adapter's."""

DATASET_ID = "erm2-nwe9"
ENDPOINT = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.json"
METADATA_ENDPOINT = f"https://data.cityofnewyork.us/api/views/{DATASET_ID}.json"
"""Where `rowsUpdatedAt` is read."""
SODA_VERSION = "2.0"

FIELDS = "unique_key,created_date,closed_date,complaint_type,descriptor"
"""Exactly the adapter's fields."""
LIMIT = 50000
"""SODA 2.0's documented maximum. A day holding this many rows or more refuses."""

FRESHNESS_HEADERS = {
    "x-soda2-truth-last-modified": "truth_last_modified",
    "x-soda2-data-out-of-date": "data_out_of_date",
}
"""Recorded exactly as sent, in a snapshot only; provenance, never a refusal (D48 (6))."""

ONE_DAY = timedelta(days=1)

Page = list[dict[str, Any]]


class NYC311FetchError(AcquisitionError):
    """An NYC 311 acquisition cannot continue; the acquisition stays incomplete."""


class ContextMismatch(NYC311FetchError):
    """The fetcher was given, or called for, a source or window that is not its own."""


class SliceTooLarge(NYC311FetchError):
    """A day's count is 50,000 or more; its page would be truncated (D44 (7))."""


class CountDisagreement(NYC311FetchError):
    """A day's rows differed from its count twice (D48 (3))."""


class DuplicateUniqueKey(NYC311FetchError):
    """A `unique_key` appears twice, within a day or in two days (D44 (10))."""


class WindowDoesNotReconcile(NYC311FetchError):
    """The day counts do not sum to the window's final count (D44 (10))."""


class MissingStartState(NYC311FetchError):
    """Days are journaled but no start snapshot is, so the start is unknown (D48 (2))."""


def where(first: date, last_exclusive: date) -> str:
    """The half-open day bounds, as literal floating timestamps."""
    return (
        f"created_date >= '{first.isoformat()}T00:00:00' "
        f"AND created_date < '{last_exclusive.isoformat()}T00:00:00'"
    )


def data_params(day: date) -> tuple[tuple[str, str], ...]:
    return (
        ("$select", FIELDS),
        ("$where", where(day, day + ONE_DAY)),
        ("$order", "unique_key"),
        ("$limit", str(LIMIT)),
    )


def count_params(first: date, last_exclusive: date) -> tuple[tuple[str, str], ...]:
    return (("$select", "count(*) AS n"), ("$where", where(first, last_exclusive)))


def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def _json(body: bytes) -> Any:
    try:
        return json.loads(body.decode("utf-8"), parse_constant=_no_constant)
    except (UnicodeDecodeError, ValueError) as exc:
        raise InvalidResponse(f"the body is not JSON: {exc}") from exc


def parse_rows(body: bytes) -> Page:
    """A data response: a bare array of objects, each with a text `unique_key`."""
    rows = _json(body)
    if not isinstance(rows, list):
        raise InvalidResponse(f"a data response is a JSON array, not {type(rows).__name__}")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("unique_key"), str):
            raise InvalidResponse(f"row {index} is not an object with a text unique_key")
    return rows


def parse_count(body: bytes) -> int:
    """A count response: exactly ``[{"n": "<ASCII decimal digits>"}]``."""
    value = _json(body)
    if not (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], dict)
        and set(value[0]) == {"n"}
    ):
        raise InvalidResponse(f'a count response is exactly [{{"n": "<digits>"}}], not {value!r}')
    text = value[0]["n"]
    if not (isinstance(text, str) and text.isascii() and text.isdigit()):
        raise InvalidResponse(f"the count {text!r} is not ASCII decimal digits")
    return int(text)


def parse_rows_updated_at(body: bytes) -> int:
    """The dataset metadata's `rowsUpdatedAt`, in integer seconds."""
    value = _json(body)
    stamp = value.get("rowsUpdatedAt") if isinstance(value, dict) else None
    if isinstance(stamp, bool) or not isinstance(stamp, int):
        raise InvalidResponse(f"the metadata's rowsUpdatedAt is {stamp!r}, not an integer")
    return stamp


def _state(snapshot: dict[str, Any]) -> tuple[Any, Any]:
    """What movement is judged by: never the headers, never the request records."""
    return snapshot["rows_updated_at"], snapshot["window_count"]


class _Run:
    """One run of the fetcher over its context's window."""

    def __init__(self, context: FetchContext) -> None:
        self.context = context
        span = (context.end - context.start).days + 1
        self.days = [context.start + ONE_DAY * n for n in range(span)]

    def pages(self) -> Iterator[Page]:
        context = self.context
        with Acquisition(
            context.directory,
            source=SOURCE,
            start=context.start,
            end=context.end,
            resolved_start=context.resolved_start,
            resolved_end=context.resolved_end,
            client=context.client,
            sentinel_commit=context.sentinel_commit,
            now=context.now,
        ) as acquisition:
            start = acquisition.start_state
            if start is None:
                if acquisition.completed_slices():
                    raise MissingStartState(
                        f"{context.directory}: days are journaled but there is no start "
                        "snapshot, and a later one is never substituted for it (D48)"
                    )
                start = self.snapshot()
                acquisition.record_start_state(start)
            # A resumed run begins from the start snapshot (D48 (4)).
            verified = start
            while True:
                yield from self.unfinished(acquisition)
                end = self.snapshot()
                if _state(end) != _state(start) or _state(end) != _state(verified):
                    changed = self.recount(acquisition)
                    # Every day a rewind keeps has now been verified at this snapshot.
                    verified = end
                    if changed:
                        acquisition.rewind(
                            next(iter(changed)),
                            reason={"changed": list(changed), "counts": changed, "snapshot": end},
                        )
                        continue
                break
            self.reconcile(acquisition, end)
            acquisition.complete(
                {
                    "dataset_id": DATASET_ID,
                    "end": end,
                    "endpoint": ENDPOINT,
                    "metadata_endpoint": METADATA_ENDPOINT,
                    "soda_version": SODA_VERSION,
                    "source_api_version": SOURCE_API_VERSION,
                    "start": start,
                }
            )

    def unfinished(self, acquisition: Acquisition) -> Iterator[Page]:
        done = acquisition.completed_slices()
        for day in self.days:
            key = day.isoformat()
            if key in done:
                continue
            rows, fetched, verification = self.fetch_day(day)
            yield rows
            # Reached only after the caller has cached the page: a day is never
            # journaled before its page is stored (D46).
            acquisition.record_slice(
                key, requests=fetched, pages=[page_digest(rows)], verification=verification
            )

    def fetch_day(self, day: date) -> tuple[Page, list[RequestRecord], dict[str, int]]:
        """The day's rows, the count and data requests they came from, and their counts."""
        disagreed = []
        # The count and the data, and at most one fresh pair of both (D48 (3)).
        for _ in range(2):
            count, counted = self.count(day, day + ONE_DAY)
            if count >= LIMIT:
                raise SliceTooLarge(f"{day}: count {count} is at or above {LIMIT}")
            rows, fetched = self.get(ENDPOINT, data_params(day), parse_rows)
            keys = Counter(row["unique_key"] for row in rows)
            repeated = sorted(key for key, n in keys.items() if n > 1)
            if repeated:
                # Not retried: a second request cannot remove what the source publishes.
                raise DuplicateUniqueKey(f"{day}: unique_key repeated in the day: {repeated[:10]}")
            if len(rows) == count:
                verification = {
                    "count": count,
                    "distinct_unique_keys": len(keys),
                    "rows": len(rows),
                }
                return rows, [counted.record, fetched.record], verification
            disagreed.append({"count": count, "rows": len(rows)})
        raise CountDisagreement(
            f"{day}: the rows differed from the count twice, {disagreed}; the acquisition "
            "stays incomplete"
        )

    def count(self, first: date, last_exclusive: date) -> tuple[int, Fetched]:
        return self.get(ENDPOINT, count_params(first, last_exclusive), parse_count)

    def get(
        self, url: str, params: Sequence[tuple[str, str]], parse: Callable[[bytes], Any]
    ) -> tuple[Any, Fetched]:
        """GET under D47's policy; a body `parse` refuses is invalid, and so retried."""
        parsed = []

        def validate(response: Response) -> None:
            parsed.append(parse(response.body))

        fetched = self.context.http.get(url, params, validate=validate)
        return parsed[-1], fetched

    def snapshot(self) -> dict[str, Any]:
        rows_updated_at, metadata = self.get(METADATA_ENDPOINT, (), parse_rows_updated_at)
        window_count, counted = self.count(self.days[0], self.days[-1] + ONE_DAY)
        snapshot: dict[str, Any] = {
            "requests": {
                "metadata": metadata.record.as_record(),
                "window_count": counted.record.as_record(),
            },
            "rows_updated_at": rows_updated_at,
            "window_count": window_count,
        }
        for header, name in FRESHNESS_HEADERS.items():
            value = counted.response.headers.get(header)
            if value is not None:
                snapshot[name] = value
        return snapshot

    def recount(self, acquisition: Acquisition) -> dict[str, dict[str, int]]:
        """Every completed day whose count changed, in journal order."""
        changed = {}
        for key, entry in acquisition.completed_slices().items():
            day = date.fromisoformat(key)
            recounted, _ = self.count(day, day + ONE_DAY)
            journaled = entry["verification"]["count"]
            if recounted != journaled:
                changed[key] = {"journaled": journaled, "recounted": recounted}
        return changed

    def reconcile(self, acquisition: Acquisition, end: dict[str, Any]) -> None:
        slices = acquisition.completed_slices()
        total = sum(entry["verification"]["count"] for entry in slices.values())
        if total != end["window_count"]:
            raise WindowDoesNotReconcile(
                f"the window's final count is {end['window_count']}, but its days sum to {total}"
            )
        # Read back from the cache, which the acquisition has verified, so a resumed run
        # checks the days an earlier run fetched as well as its own.
        folder = pages_dir(self.context.directory, SOURCE)
        seen: set[str] = set()
        repeated: set[str] = set()
        for entry in slices.values():
            for digest in entry["pages"]:
                path = folder / f"{digest}{PAGE_SUFFIX}"
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    for row in json.load(handle):
                        key = row["unique_key"]
                        (repeated if key in seen else seen).add(key)
        if repeated:
            raise DuplicateUniqueKey(
                f"unique_key in more than one day: {sorted(repeated)[:10]}; nothing is kept "
                "in favour of another"
            )


def make_nyc311_fetcher(context: FetchContext) -> Callable[[str, date, date], Iterator[Page]]:
    """The NYC 311 `Fetcher` for `context`, which must be an NYC 311 context."""
    if context.source != SOURCE:
        raise ContextMismatch(f"the NYC 311 fetcher was given a {context.source!r} context")

    def fetch(source: str, start: date, end: date) -> Iterator[Page]:
        if (source, start, end) != (context.source, context.start, context.end):
            raise ContextMismatch(
                f"asked for {source} {start} .. {end}; this fetcher acquires only "
                f"{context.source} {context.start} .. {context.end}"
            )
        return _Run(context).pages()

    return fetch

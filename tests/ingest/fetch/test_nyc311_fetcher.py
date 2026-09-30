"""The NYC 311 fetcher: day-sliced, count-verified SODA 2.0 pages (D44, D48).

Every response comes from `FakeSocrata`, an in-memory stand-in answering the four
requests the fetcher makes -- the dataset's metadata, the window's count, a day's count
and a day's rows -- from a dictionary of rows. Drift is simulated by editing that
dictionary after a chosen request has been answered. Pages are cached the way
``ingest.cli.cache_page`` caches them, without importing ``ingest.cli``, whose scipy
import would keep this module out of the application job. The directory's conftest
closes the network.
"""

import ast
import gzip
import hashlib
import json
import re
import socket
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from ingest.fetch import nyc311
from ingest.fetch.acquisition import (
    LOCK_NAME,
    QUARANTINE_DIR,
    REWINDS_NAME,
    START_NAME,
    AcquisitionError,
    journal_path,
    record_path,
    verify_acquisition,
)
from ingest.fetch.canonical import canonical_bytes, page_digest
from ingest.fetch.http import USER_AGENT, ClientIdentity, FetchFailed, HttpClient, Response
from ingest.fetch.nyc311 import (
    ContextMismatch,
    CountDisagreement,
    DuplicateUniqueKey,
    MissingStartState,
    NYC311FetchError,
    SliceTooLarge,
    WindowDoesNotReconcile,
    count_params,
    data_params,
    make_nyc311_fetcher,
    parse_count,
    parse_rows,
    parse_rows_updated_at,
)
from ingest.fetch.registry import FETCHERS, FetchContext
from ingest.sources import nyc311 as adapter

ROOT = Path(__file__).resolve().parents[3]
START, END = date(2024, 1, 1), date(2024, 1, 3)
D1, D2, D3 = "2024-01-01", "2024-01-02", "2024-01-03"
DAYS = (D1, D2, D3)
COMMIT = "a" * 40
CLIENT = ClientIdentity(user_agent="Sentinel-test/0", library="fake", library_version="0")
NOW = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
STAMP = "2026-09-28T12:00:00.000000+00:00"
UPDATED = 1790559478

DATA_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
METADATA_URL = "https://data.cityofnewyork.us/api/views/erm2-nwe9.json"
SELECT = "unique_key,created_date,closed_date,complaint_type,descriptor"
BOUNDS = re.compile(
    r"created_date >= '(\d{4}-\d{2}-\d{2})T00:00:00' "
    r"AND created_date < '(\d{4}-\d{2}-\d{2})T00:00:00'"
)


def following(day):
    return (date.fromisoformat(day) + timedelta(days=1)).isoformat()


def day_count(day):
    return ("count", day, following(day))


def day_data(day):
    return ("data", day, following(day))


METADATA = ("metadata",)
WINDOW = ("count", D1, "2024-01-04")


def row(day, n, **extra):
    return {
        "unique_key": f"{day.replace('-', '')}{n:03d}",
        "created_date": f"{day}T09:00:00.000",
        "complaint_type": "Noise",
        "descriptor": "Loud Music/Party",
        **extra,
    }


def dataset(days=DAYS, per_day=2):
    return {day: [row(day, n) for n in range(per_day)] for day in days}


class FakeSocrata:
    """Socrata as the fetcher sees it, answered from `rows`, a dict of day -> rows.

    `script` queues replacement responses for one request, oldest first; `after` runs a
    change once a request has been answered for the n-th time. `log` names each request
    in order and `sent` keeps exactly what the transport was given.
    """

    def __init__(self, rows=None, *, updated=UPDATED, headers=None):
        self.rows = dataset() if rows is None else rows
        self.updated = updated
        self.headers = dict(headers or {})
        self.log = []
        self.sent = []
        self.scripted = {}
        self.hooks = {}

    def script(self, request, *responses):
        self.scripted.setdefault(request, []).extend(responses)

    def after(self, request, change, occurrence=1):
        self.hooks[(request, occurrence)] = change

    def get(self, url, params, headers):
        assert len(self.sent) < 400, "the fetcher is looping"
        self.sent.append((url, list(params), dict(headers)))
        request = self.classify(url, dict(params))
        self.log.append(request)
        queued = self.scripted.get(request)
        response = queued.pop(0) if queued else self.answer(request)
        change = self.hooks.pop((request, self.log.count(request)), None)
        if change is not None:
            change(self)
        return response

    @staticmethod
    def classify(url, params):
        if url == METADATA_URL:
            assert params == {}, params
            return METADATA
        assert url == DATA_URL, url
        first, last = BOUNDS.fullmatch(params["$where"]).groups()
        return ("count" if params["$select"] == "count(*) AS n" else "data", first, last)

    def answer(self, request):
        if request == METADATA:
            body = {
                "id": "erm2-nwe9",
                "name": "311 Service Requests",
                "rowsUpdatedAt": self.updated,
            }
            return Response(200, {}, json.dumps(body).encode())
        kind, first, last = request
        rows = [r for day in sorted(self.rows) if first <= day < last for r in self.rows[day]]
        if kind == "count":
            return Response(200, dict(self.headers), json.dumps([{"n": str(len(rows))}]).encode())
        return Response(200, dict(self.headers), json.dumps(rows).encode())


class FakeClock:
    """Monotonic seconds that move only when something sleeps."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def a_context(directory, transport, *, source="nyc311", start=START, end=END, clock=None):
    clock = clock or FakeClock()
    return FetchContext(
        source=source,
        start=start,
        end=end,
        resolved_start=datetime(start.year, start.month, start.day, 5, tzinfo=UTC),
        resolved_end=datetime(end.year, end.month, end.day, 4, 59, tzinfo=UTC) + timedelta(days=1),
        directory=directory,
        http=HttpClient(transport, clock=clock, sleep=clock.sleep, now=lambda: NOW),
        client=CLIENT,
        sentinel_commit=COMMIT,
        now=lambda: NOW,
    )


def cache(directory, page):
    """Store a page as ``cache_page`` does: gzipped canonical JSON under its digest."""
    folder = directory / "nyc311"
    folder.mkdir(parents=True, exist_ok=True)
    with gzip.open(folder / f"{page_digest(page)}.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(page, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def run(context, *, source="nyc311", start=None, end=None):
    """Drive the fetcher as ``fetch_into_cache`` does: cache each page, then ask again."""
    pages = []
    fetcher = make_nyc311_fetcher(context)
    for page in fetcher(source, start or context.start, end or context.end):
        cache(context.directory, page)
        pages.append(page)
    return pages


def acquire(tmp_path, transport=None, *, name="acq", **overrides):
    transport = transport or FakeSocrata()
    context = a_context(tmp_path / name, transport, **overrides)
    pages = run(context)
    return context, transport, pages


def the_record(context):
    return json.loads(record_path(context.directory).read_bytes())


def journaled(context):
    path = journal_path(context.directory)
    if not path.exists():
        return []
    return [json.loads(line)["key"] for line in path.read_bytes().splitlines()]


def request_record(url, params, body):
    return {
        "params": [list(pair) for pair in params],
        "response_bytes": len(body),
        "response_sha256": hashlib.sha256(body).hexdigest(),
        "retrieved_at": STAMP,
        "status": 200,
        "url": url,
    }


def where(first, last):
    return f"created_date >= '{first}T00:00:00' AND created_date < '{last}T00:00:00'"


def rows_json(rows):
    return Response(200, {}, json.dumps(rows).encode())


def count_json(n):
    return Response(200, {}, json.dumps([{"n": str(n)}]).encode())


# --- A-D: exactly the frozen requests, and no token -------------------------------------


def test_each_day_asks_for_exactly_the_frozen_data_parameters(tmp_path):
    _, transport, _ = acquire(tmp_path)
    data = [
        (url, params) for url, params, _ in transport.sent if params and "$order" in dict(params)
    ]
    assert data == [
        (
            DATA_URL,
            [
                ("$select", SELECT),
                ("$where", where(day, following(day))),
                ("$order", "unique_key"),
                ("$limit", "50000"),
            ],
        )
        for day in DAYS
    ]


def test_each_count_asks_for_count_star_over_the_same_bounds(tmp_path):
    _, transport, _ = acquire(tmp_path)
    counts = [
        params for url, params, _ in transport.sent if params and params[0][1] == "count(*) AS n"
    ]
    window = [("$select", "count(*) AS n"), ("$where", where(D1, "2024-01-04"))]
    days = [[("$select", "count(*) AS n"), ("$where", where(d, following(d)))] for d in DAYS]
    assert counts == [window, *days, window]


def test_the_metadata_request_is_the_views_endpoint_with_no_parameters(tmp_path):
    _, transport, _ = acquire(tmp_path)
    metadata = [(url, params) for url, params, _ in transport.sent if not params]
    assert metadata == [(METADATA_URL, []), (METADATA_URL, [])]


def test_no_token_is_ever_read_or_sent(tmp_path):
    _, transport, _ = acquire(tmp_path)
    assert transport.sent
    for _, params, headers in transport.sent:
        assert headers == {"User-Agent": USER_AGENT}
        assert not [name for name, _ in params if "token" in name.lower()]
    source = Path(nyc311.__file__).read_text(encoding="utf-8").lower()
    assert "app-token" not in source and "app_token" not in source and "environ" not in source


def test_the_parameter_builders_are_the_frozen_ones():
    day = date(2024, 1, 2)
    assert data_params(day) == (
        ("$select", SELECT),
        ("$where", where(D2, D3)),
        ("$order", "unique_key"),
        ("$limit", "50000"),
    )
    assert count_params(day, date(2024, 2, 1)) == (
        ("$select", "count(*) AS n"),
        ("$where", where(D2, "2024-02-01")),
    )


# --- E: one slice and one page per day, fetched in order --------------------------------


def test_one_slice_and_one_page_per_day_in_day_order(tmp_path):
    context, transport, pages = acquire(tmp_path)
    assert transport.log == [
        METADATA,
        WINDOW,
        day_count(D1),
        day_data(D1),
        day_count(D2),
        day_data(D2),
        day_count(D3),
        day_data(D3),
        METADATA,
        WINDOW,
    ], "the start snapshot first, then count before data each day, then the end check"
    assert pages == [transport.rows[day] for day in DAYS]
    record = the_record(context)
    assert [entry["key"] for entry in record["slices"]] == list(DAYS)
    for entry, day in zip(record["slices"], DAYS):
        assert entry["pages"] == [page_digest(transport.rows[day])]
    verify_acquisition(context.directory, source="nyc311", start=START, end=END)


# --- F-H: the 50,000 bound and the count's representation -------------------------------


def test_a_count_of_49999_is_accepted(tmp_path):
    rows = [{"unique_key": str(n)} for n in range(49999)]
    transport = FakeSocrata({D1: rows})
    context, _, pages = acquire(tmp_path, transport, end=START)
    assert len(pages[0]) == 49999
    assert the_record(context)["slices"][0]["verification"]["count"] == 49999


def test_a_count_of_50000_refuses_before_the_data_is_requested(tmp_path):
    transport = FakeSocrata()
    transport.script(day_count(D2), count_json(50000))
    with pytest.raises(SliceTooLarge, match="50000") as caught:
        acquire(tmp_path, transport)
    assert isinstance(caught.value, NYC311FetchError) and isinstance(caught.value, AcquisitionError)
    assert day_data(D2) not in transport.log, "never truncated, never requested"
    assert transport.log[-1] == day_count(D2)
    assert journaled(a_context(tmp_path / "acq", transport)) == [D1]
    assert not record_path(tmp_path / "acq").exists()


def test_the_count_is_read_from_its_ascii_decimal_string():
    assert parse_count(b'[{"n":"16388"}]') == 16388
    assert parse_count(b'[{"n":"0"}]') == 0


@pytest.mark.parametrize(
    "body",
    [
        b'[{"n":16388}]',
        b'[{"n":"\xd9\xa3"}]',
        b'[{"n":"\xc2\xb2"}]',
        b'[{"n":"-1"}]',
        b'[{"n":"1.0"}]',
        b'[{"n":" 7"}]',
        b'[{"n":""}]',
        b'[{"n":"7","m":"1"}]',
        b'[{"count":"7"}]',
        b'[{"n":"7"},{"n":"8"}]',
        b"[]",
        b'{"n":"7"}',
        b'["7"]',
    ],
)
def test_a_count_in_any_other_form_is_invalid_retried_and_refused(tmp_path, body):
    transport = FakeSocrata()
    transport.script(day_count(D1), *[Response(200, {}, body)] * 6)
    with pytest.raises(FetchFailed, match="invalid body"):
        acquire(tmp_path, transport)
    assert transport.log.count(day_count(D1)) == 6
    assert day_data(D1) not in transport.log


# --- I-J: a count that disagrees with its rows ------------------------------------------


def test_a_count_that_disagrees_with_its_rows_is_requested_again_with_its_data_once(tmp_path):
    transport = FakeSocrata()
    transport.script(day_data(D2), rows_json(transport.rows[D2][:1]))
    context, _, _ = acquire(tmp_path, transport)
    day_two = [request for request in transport.log if request[1:2] == (D2,)]
    assert day_two == [day_count(D2), day_data(D2), day_count(D2), day_data(D2)]
    entry = the_record(context)["slices"][1]
    full = json.dumps(transport.rows[D2]).encode()
    assert entry["requests"][1]["response_sha256"] == hashlib.sha256(full).hexdigest()
    assert len(entry["requests"]) == 2, "the pair the slice used, and only it"
    assert entry["verification"] == {"count": 2, "distinct_unique_keys": 2, "rows": 2}


def test_a_stale_count_is_what_the_second_pair_corrects(tmp_path):
    transport = FakeSocrata()
    transport.script(day_count(D2), count_json(5))
    context, _, _ = acquire(tmp_path, transport)
    assert [r for r in transport.log if r[1:2] == (D2,)] == [
        day_count(D2),
        day_data(D2),
        day_count(D2),
        day_data(D2),
    ]
    assert the_record(context)["slices"][1]["verification"]["count"] == 2


def test_a_second_disagreement_refuses_and_keeps_the_acquisition_incomplete(tmp_path):
    transport = FakeSocrata()
    short = rows_json(transport.rows[D2][:1])
    transport.script(day_data(D2), short, short)
    with pytest.raises(CountDisagreement, match=D2):
        acquire(tmp_path, transport)
    assert transport.log.count(day_data(D2)) == 2 and transport.log[-1] == day_data(D2)
    directory = tmp_path / "acq"
    assert not record_path(directory).exists()
    assert not (directory / LOCK_NAME).exists()

    resumed = FakeSocrata()
    context, _, _ = acquire(tmp_path, resumed)
    assert resumed.log[0] == day_count(D2), "kept for resume: day one is not asked for again"
    assert journaled(context) == list(DAYS)


# --- K-L: a unique_key is never repeated -------------------------------------------------


def test_a_unique_key_repeated_within_a_day_refuses_at_once_without_a_retry(tmp_path):
    rows = dataset()
    rows[D2] = [row(D2, 0), row(D2, 0, descriptor="Other")]
    transport = FakeSocrata(rows)
    with pytest.raises(DuplicateUniqueKey, match="20240102000"):
        acquire(tmp_path, transport)
    assert [r for r in transport.log if r[1:2] == (D2,)] == [day_count(D2), day_data(D2)]
    assert journaled(a_context(tmp_path / "acq", transport)) == [D1]


def test_a_unique_key_in_two_days_refuses_before_the_record(tmp_path):
    rows = dataset()
    rows[D3] = [row(D3, 0), {**row(D1, 1), "created_date": f"{D3}T10:00:00.000"}]
    transport = FakeSocrata(rows)
    with pytest.raises(DuplicateUniqueKey, match="20240101001"):
        acquire(tmp_path, transport)
    context = a_context(tmp_path / "acq", transport)
    assert journaled(context) == list(DAYS), "each day was consistent on its own"
    assert not record_path(context.directory).exists()


# --- M-P: bodies that are not what they claim -------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        b'[{"unique_key":"1"',
        b"",
        b"<!DOCTYPE html><html><body>Service unavailable</body></html>",
        b'{"unique_key":"1"}',
        b"{}",
        b'{"data":[{"unique_key":"1"}]}',
        b'[{"unique_key":"1"},2]',
        b'[["unique_key","1"]]',
        b'[{"descriptor":"no key"}]',
        b'[{"unique_key":1}]',
        b'[{"unique_key":null}]',
        b'[{"unique_key":"1","x":NaN}]',
        b"\xff\xfe[]",
    ],
    ids=[
        "truncated",
        "empty",
        "html",
        "object",
        "empty object",
        "envelope",
        "non-object row",
        "array row",
        "no unique_key",
        "numeric unique_key",
        "null unique_key",
        "nan",
        "not utf-8",
    ],
)
def test_an_unusable_data_body_is_retried_then_refused(tmp_path, body):
    transport = FakeSocrata()
    transport.script(day_data(D1), *[Response(200, {}, body)] * 6)
    with pytest.raises(FetchFailed, match="invalid body"):
        acquire(tmp_path, transport)
    assert transport.log.count(day_data(D1)) == 6
    assert journaled(a_context(tmp_path / "acq", transport)) == []


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"<html></html>",
        b"[]",
        b'{"rowsUpdatedAt":"1790559478"}',
        b'{"rowsUpdatedAt":true}',
        b'{"rowsUpdatedAt":1790559478.5}',
        b'{"name":"no stamp"}',
    ],
)
def test_metadata_without_an_integer_rows_updated_at_is_retried_then_refused(tmp_path, body):
    transport = FakeSocrata()
    transport.script(METADATA, *[Response(200, {}, body)] * 6)
    with pytest.raises(FetchFailed, match="invalid body"):
        acquire(tmp_path, transport)
    assert transport.log == [METADATA] * 6
    assert not (tmp_path / "acq" / START_NAME).exists()


def test_the_parsers_accept_exactly_the_documented_shapes():
    rows = [{"unique_key": "1", "descriptor": None}, {"unique_key": "2"}]
    assert parse_rows(json.dumps(rows).encode()) == rows
    assert parse_rows(b"[]") == []
    assert parse_rows_updated_at(b'{"rowsUpdatedAt":1790559478,"id":"erm2-nwe9"}') == 1790559478


# --- Q, AB: daylight-saving days, and the rows D45 leaves undecided ----------------------


@pytest.mark.parametrize(
    "day, bounds",
    [
        ("2024-03-10", ("2024-03-10", "2024-03-11")),
        ("2024-11-03", ("2024-11-03", "2024-11-04")),
        ("2025-03-09", ("2025-03-09", "2025-03-10")),
        ("2025-11-02", ("2025-11-02", "2025-11-03")),
        ("2024-02-29", ("2024-02-29", "2024-03-01")),
        ("2024-12-31", ("2024-12-31", "2025-01-01")),
    ],
)
def test_day_bounds_are_plain_civil_date_literals(day, bounds):
    assert dict(data_params(date.fromisoformat(day)))["$where"] == where(*bounds)


@pytest.mark.parametrize("day", ["2024-03-10", "2024-11-03", "2025-03-09", "2025-11-02"])
def test_rows_the_adapter_would_refuse_reach_the_cache_exactly_as_served(tmp_path, day):
    served = [
        {"unique_key": "1", "created_date": f"{day}T01:30:00.000", "complaint_type": "Noise"},
        {
            "unique_key": "2",
            "created_date": f"{day}T02:30:00.000",
            "closed_date": f"{day}T01:15:00.000",
            "complaint_type": "Noise",
            "descriptor": None,
        },
        {
            "unique_key": "3",
            "created_date": f"{day}T23:59:59.999",
            "closed_date": f"{day}T01:59:59.000",
            "complaint_type": "",
            "descriptor": "",
        },
    ]
    transport = FakeSocrata({day: served})
    the_day = date.fromisoformat(day)
    context, _, pages = acquire(tmp_path, transport, start=the_day, end=the_day)
    assert pages == [served]
    digest = the_record(context)["slices"][0]["pages"][0]
    with gzip.open(context.directory / "nyc311" / f"{digest}.json.gz", "rt") as handle:
        assert json.load(handle) == served, "no filter, no repair, no conversion"
    assert day_data(day) in transport.log


# --- R: D47's retry policy applies beneath every request --------------------------------


def test_the_d47_retry_policy_applies_beneath_every_request(tmp_path):
    transport = FakeSocrata()
    transport.script(day_count(D2), Response(500, {}, b""))
    transport.script(day_data(D2), Response(429, {"retry-after": "7"}, b""))
    transport.script(METADATA, Response(503, {}, b""))
    clock = FakeClock()
    context, _, _ = acquire(tmp_path, transport, clock=clock)
    assert transport.log.count(day_count(D2)) == 2 and transport.log.count(day_data(D2)) == 2
    assert transport.log[:2] == [METADATA, METADATA]
    assert clock.sleeps.count(2) == 2 and 7 in clock.sleeps
    assert journaled(context) == list(DAYS)


def test_a_403_stops_at_once_and_leaves_the_acquisition_to_resume(tmp_path):
    transport = FakeSocrata()
    transport.script(day_data(D2), Response(403, {}, b"Forbidden"))
    with pytest.raises(FetchFailed, match="403"):
        acquire(tmp_path, transport)
    assert transport.log[-1] == day_data(D2) and transport.log.count(day_data(D2)) == 1
    directory = tmp_path / "acq"
    assert not record_path(directory).exists() and not (directory / LOCK_NAME).exists()
    assert journaled(a_context(directory, transport)) == [D1]


# --- S-T: the start snapshot, recorded once -----------------------------------------------


def test_the_start_snapshot_is_recorded_before_the_first_slice(tmp_path):
    transport = FakeSocrata()
    seen = []
    transport.after(day_count(D1), lambda t: seen.append((tmp_path / "acq" / START_NAME).exists()))
    context, _, _ = acquire(tmp_path, transport)
    assert seen == [True]
    metadata = transport.answer(METADATA).body
    window = transport.answer(WINDOW).body
    start = json.loads((context.directory / START_NAME).read_bytes())
    assert start == {
        "requests": {
            "metadata": request_record(METADATA_URL, [], metadata),
            "window_count": request_record(
                DATA_URL,
                [("$select", "count(*) AS n"), ("$where", where(D1, "2024-01-04"))],
                window,
            ),
        },
        "rows_updated_at": UPDATED,
        "window_count": 6,
    }
    assert the_record(context)["source_details"]["start"] == start


def test_a_resumed_acquisition_keeps_its_first_start_snapshot(tmp_path):
    first = FakeSocrata()
    first.script(day_data(D2), Response(403, {}, b""))
    with pytest.raises(FetchFailed):
        acquire(tmp_path, first)
    directory = tmp_path / "acq"
    start = (directory / START_NAME).read_bytes()

    later = FakeSocrata(updated=UPDATED + 86400)
    context, _, _ = acquire(tmp_path, later)
    assert later.log[:4] == [day_count(D2), day_data(D2), day_count(D3), day_data(D3)]
    assert (directory / START_NAME).read_bytes() == start
    details = the_record(context)["source_details"]
    assert details["start"]["rows_updated_at"] == UPDATED
    assert details["end"]["rows_updated_at"] == UPDATED + 86400


def test_journaled_slices_without_a_start_state_refuse_before_any_request(tmp_path):
    first = FakeSocrata()
    first.script(day_data(D2), Response(403, {}, b""))
    with pytest.raises(FetchFailed):
        acquire(tmp_path, first)
    (tmp_path / "acq" / START_NAME).unlink()

    later = FakeSocrata()
    with pytest.raises(MissingStartState, match="start"):
        acquire(tmp_path, later)
    assert later.sent == []
    assert not (tmp_path / "acq" / START_NAME).exists(), "no later start is invented"


# --- U-X: drift, re-verification and the at-most-once rewind -----------------------------


def bump(transport, seconds=60):
    transport.updated += seconds


def test_an_unmoved_source_is_not_counted_again(tmp_path):
    _, transport, _ = acquire(tmp_path)
    assert transport.log[-2:] == [METADATA, WINDOW]
    assert transport.log.count(day_count(D1)) == 1


def test_movement_triggers_a_recount_of_every_completed_slice(tmp_path):
    transport = FakeSocrata()
    transport.after(day_data(D3), bump)
    context, _, _ = acquire(tmp_path, transport)
    assert transport.log[8:] == [METADATA, WINDOW, day_count(D1), day_count(D2), day_count(D3)]
    record = the_record(context)
    assert record["rewinds"] == []
    assert record["source_details"]["end"]["rows_updated_at"] == UPDATED + 60
    assert record["source_details"]["start"]["rows_updated_at"] == UPDATED


def test_a_moved_window_count_alone_is_movement(tmp_path):
    transport = FakeSocrata()
    transport.script(WINDOW, count_json(5))
    context, _, _ = acquire(tmp_path, transport)
    assert transport.log[8:] == [METADATA, WINDOW, day_count(D1), day_count(D2), day_count(D3)]
    details = the_record(context)["source_details"]
    assert details["start"]["window_count"] == 5 and details["end"]["window_count"] == 6
    assert details["start"]["rows_updated_at"] == details["end"]["rows_updated_at"]


def change_days(*days):
    def change(transport):
        for day in days:
            rows = transport.rows[day]
            transport.rows[day] = [*rows, row(day, 100 + len(rows))]
        bump(transport)

    return change


def test_the_earliest_changed_slice_is_the_rewind_point(tmp_path):
    transport = FakeSocrata()
    transport.after(day_data(D3), change_days(D2, D3))
    context, _, _ = acquire(tmp_path, transport)
    assert transport.log[8:] == [
        METADATA,
        WINDOW,
        day_count(D1),
        day_count(D2),
        day_count(D3),
        day_count(D2),
        day_data(D2),
        day_count(D3),
        day_data(D3),
        METADATA,
        WINDOW,
        day_count(D1),
        day_count(D2),
        day_count(D3),
    ]
    (event,) = the_record(context)["rewinds"]
    assert event["from_key"] == D2 and event["removed"] == [D2, D3]
    assert event["reason"]["changed"] == [D2, D3]
    assert event["reason"]["counts"] == {
        D2: {"journaled": 2, "recounted": 3},
        D3: {"journaled": 2, "recounted": 3},
    }
    assert event["reason"]["snapshot"]["rows_updated_at"] == UPDATED + 60


def test_a_rewound_day_is_fetched_again_and_its_old_page_quarantined(tmp_path):
    transport = FakeSocrata()
    old = page_digest(transport.rows[D2])
    transport.after(day_data(D3), change_days(D2))
    context, _, pages = acquire(tmp_path, transport)
    assert pages[3:] == [transport.rows[D2], transport.rows[D3]], "day two, then the rest"
    record = the_record(context)
    assert [entry["key"] for entry in record["slices"]] == list(DAYS)
    assert record["slices"][1]["pages"] == [page_digest(transport.rows[D2])]
    assert record["slices"][1]["verification"] == {
        "count": 3,
        "distinct_unique_keys": 3,
        "rows": 3,
    }
    quarantined = sorted(p.name for p in (context.directory / QUARANTINE_DIR).rglob("*.json.gz"))
    unchanged = page_digest(transport.rows[D3])
    assert quarantined == sorted([f"{old}.json.gz", f"{unchanged}.json.gz"]), "moved, not deleted"
    assert (context.directory / "nyc311" / f"{unchanged}.json.gz").exists(), "cached again"
    assert json.loads((context.directory / REWINDS_NAME).read_bytes())["from_key"] == D2
    verify_acquisition(context.directory, source="nyc311", start=START, end=END)


def test_a_day_that_changes_again_after_its_refetch_refuses(tmp_path):
    transport = FakeSocrata()
    transport.after(day_data(D3), change_days(D2))
    transport.after(day_data(D2), change_days(D2), occurrence=2)
    with pytest.raises(AcquisitionError, match=D2):
        acquire(tmp_path, transport)
    directory = tmp_path / "acq"
    assert not record_path(directory).exists()
    assert len((directory / REWINDS_NAME).read_bytes().splitlines()) == 1, "no second rewind"
    assert transport.log[-3:] == [day_count(D1), day_count(D2), day_count(D3)]


def test_an_earlier_day_changing_after_a_later_refetch_refuses(tmp_path):
    transport = FakeSocrata()
    transport.after(day_data(D3), change_days(D3))
    transport.after(day_data(D3), change_days(D1), occurrence=2)
    with pytest.raises(AcquisitionError, match=D3):
        acquire(tmp_path, transport)
    assert not record_path(tmp_path / "acq").exists()
    events = (tmp_path / "acq" / REWINDS_NAME).read_bytes().splitlines()
    assert [json.loads(line)["from_key"] for line in events] == [D3]


def test_a_return_to_the_start_state_is_still_movement_after_a_rewind(tmp_path):
    def move_one_row(transport):
        moved = transport.rows[D3].pop()
        transport.rows[D2] = [*transport.rows[D2], {**moved, "unique_key": "moved"}]
        bump(transport)

    transport = FakeSocrata()
    transport.after(day_data(D3), move_one_row)
    transport.after(day_data(D3), lambda t: bump(t, -60), occurrence=2)
    context, _, _ = acquire(tmp_path, transport)
    assert transport.log[-5:] == [METADATA, WINDOW, day_count(D1), day_count(D2), day_count(D3)]
    details = the_record(context)["source_details"]
    assert details["end"]["rows_updated_at"] == details["start"]["rows_updated_at"]
    assert details["end"]["window_count"] == details["start"]["window_count"]


# --- Y-Z, AA: provenance, freshness headers and the final reconciliation ----------------


def test_source_details_hold_exactly_the_frozen_keys(tmp_path):
    context, transport, _ = acquire(tmp_path)
    record = the_record(context)
    details = record["source_details"]
    start = json.loads((context.directory / START_NAME).read_bytes())
    assert details == {
        "dataset_id": "erm2-nwe9",
        "end": start,
        "endpoint": DATA_URL,
        "metadata_endpoint": METADATA_URL,
        "soda_version": "2.0",
        "source_api_version": "socrata-soda2",
        "start": start,
    }
    for entry, day in zip(record["slices"], DAYS):
        body = json.dumps(transport.rows[day]).encode()
        count = json.dumps([{"n": "2"}]).encode()
        assert entry == {
            "key": day,
            "pages": [page_digest(transport.rows[day])],
            "requests": [
                request_record(DATA_URL, count_params(date.fromisoformat(day), _next(day)), count),
                request_record(DATA_URL, data_params(date.fromisoformat(day)), body),
            ],
            "verification": {"count": 2, "distinct_unique_keys": 2, "rows": 2},
        }
    assert record["record_version"] == 1


def _next(day):
    return date.fromisoformat(day) + timedelta(days=1)


def test_freshness_headers_are_recorded_exactly_as_sent_and_refuse_nothing(tmp_path):
    headers = {
        "x-soda2-truth-last-modified": "Mon, 28 Sep 2026 01:38:24 GMT",
        "x-soda2-data-out-of-date": "true",
    }
    context, _, _ = acquire(tmp_path, FakeSocrata(headers=headers))
    record = the_record(context)
    for name in ("start", "end"):
        snapshot = record["source_details"][name]
        assert snapshot["truth_last_modified"] == "Mon, 28 Sep 2026 01:38:24 GMT"
        assert snapshot["data_out_of_date"] == "true"
    for entry in record["slices"]:
        assert set(entry["verification"]) == {"count", "rows", "distinct_unique_keys"}


def test_absent_freshness_headers_are_absent_from_the_snapshot(tmp_path):
    context, _, _ = acquire(tmp_path)
    snapshot = the_record(context)["source_details"]["end"]
    assert set(snapshot) == {"requests", "rows_updated_at", "window_count"}


def test_the_slice_counts_must_reconcile_with_the_final_window_count(tmp_path):
    transport = FakeSocrata()
    transport.script(WINDOW, count_json(7), count_json(7))
    with pytest.raises(WindowDoesNotReconcile, match="7") as caught:
        acquire(tmp_path, transport)
    assert "6" in str(caught.value)
    context = a_context(tmp_path / "acq", transport)
    assert journaled(context) == list(DAYS)
    assert not record_path(context.directory).exists()


# --- AC-AD: cache before journal, and determinism ---------------------------------------


def test_a_slice_is_journaled_only_after_its_page_is_cached(tmp_path):
    transport = FakeSocrata()
    context = a_context(tmp_path / "acq", transport)
    at_yield = []
    for page in make_nyc311_fetcher(context)("nyc311", START, END):
        at_yield.append(journaled(context))
        cache(context.directory, page)
    assert at_yield == [[], [D1], [D1, D2]]


def test_a_page_the_caller_never_cached_is_never_journaled(tmp_path):
    transport = FakeSocrata()
    context = a_context(tmp_path / "acq", transport)
    pages = make_nyc311_fetcher(context)("nyc311", START, END)
    next(pages)
    with pytest.raises(AcquisitionError, match="not cached"):
        next(pages)
    assert journaled(context) == []


def test_identical_responses_give_identical_pages_and_record_bytes(tmp_path):
    first, _, first_pages = acquire(tmp_path, FakeSocrata(), name="a")
    second, _, second_pages = acquire(tmp_path, FakeSocrata(), name="b")
    assert [page_digest(p) for p in first_pages] == [page_digest(p) for p in second_pages]
    assert record_path(first.directory).read_bytes() == record_path(second.directory).read_bytes()
    data = record_path(first.directory).read_bytes()
    assert canonical_bytes(json.loads(data)) == data


# --- AE-AH: registration, the context, the network and a completed acquisition ----------


def test_nyc311_is_the_one_registered_fetcher():
    assert FETCHERS == {"nyc311": make_nyc311_fetcher}


@pytest.mark.parametrize(
    "first, then",
    [
        ("ingest.fetch.nyc311", "ingest.fetch.registry"),
        ("ingest.fetch.registry", "ingest.fetch.nyc311"),
    ],
)
def test_the_registry_and_the_fetcher_import_in_either_order_and_alone(first, then):
    code = (
        f"import sys, {first}, {then}\n"
        "from ingest.fetch.registry import FETCHERS\n"
        "from ingest.fetch.nyc311 import make_nyc311_fetcher\n"
        "assert FETCHERS == {'nyc311': make_nyc311_fetcher}\n"
        "for name in ('ingest.cli', 'ingest.sources', 'scipy', 'pyarrow', 'django'):\n"
        "    assert name not in sys.modules, name\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, timeout=120)


def test_the_fetchers_names_are_the_adapters():
    assert nyc311.SOURCE == adapter.SOURCE_SLUG
    assert nyc311.SOURCE_API_VERSION == adapter.SOURCE_API_VERSION


def test_the_cached_pages_are_what_the_adapter_reads(tmp_path):
    _, _, pages = acquire(tmp_path)
    for page in pages:
        records = [adapter.normalize(r)[0] for r in adapter.rows_from_page(page)]
        assert [r.external_id for r in records] == [r["unique_key"] for r in page]


def test_a_context_for_another_source_is_refused(tmp_path):
    context = a_context(tmp_path / "acq", FakeSocrata(), source="cfpb")
    with pytest.raises(ContextMismatch, match="cfpb"):
        make_nyc311_fetcher(context)


@pytest.mark.parametrize(
    "call",
    [
        ("cfpb", START, END),
        ("nyc311", date(2023, 12, 31), END),
        ("nyc311", START, date(2024, 1, 2)),
    ],
)
def test_a_call_for_another_source_or_window_is_refused_before_anything(tmp_path, call):
    transport = FakeSocrata()
    context = a_context(tmp_path / "acq", transport)
    with pytest.raises(ContextMismatch):
        make_nyc311_fetcher(context)(*call)
    assert transport.sent == [] and not context.directory.exists()


def test_the_fetcher_reaches_the_network_only_through_its_context():
    with pytest.raises(AssertionError, match="network"):
        socket.create_connection(("data.cityofnewyork.us", 443))
    tree = ast.parse(Path(nyc311.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    for module in imported:
        top = module.split(".")[0]
        assert module.startswith("ingest.fetch.") or top in sys.stdlib_module_names, module
    assert not {"socket", "urllib", "http", "requests"} & {m.split(".")[0] for m in imported}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "RequestsTransport" not in names


def test_a_completed_acquisition_is_never_reopened_and_makes_no_request(tmp_path):
    context, _, _ = acquire(tmp_path)
    record = record_path(context.directory).read_bytes()
    again = FakeSocrata()
    with pytest.raises(AcquisitionError, match="completed acquisition"):
        run(a_context(context.directory, again))
    assert again.sent == []
    assert record_path(context.directory).read_bytes() == record

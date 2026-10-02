"""The CFPB fetcher: the Narratives Archive joined to the API by Complaint ID (D43, D44, D49).

Every response comes from `FakeCFPB`, an in-memory stand-in for the reading room, the
archive's files host and the search API, built from a list of complaints. Its ZIP
exports are made here with `zipfile`, deterministically, so a pinned export hashes the
same on every request. Pages are cached the way ``ingest.cli.cache_page`` caches them,
without importing ``ingest.cli``, whose scipy import would keep this module out of the
application job. The directory's conftest closes the network.
"""

import ast
import csv
import gzip
import hashlib
import io
import json
import socket
import subprocess
import sys
import zipfile
from array import array
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from ingest.fetch import cfpb
from ingest.fetch.acquisition import (
    LOCK_NAME,
    START_NAME,
    AcquisitionError,
    journal_path,
    record_path,
    verify_acquisition,
)
from ingest.fetch.canonical import canonical_bytes, page_digest
from ingest.fetch.cfpb import (
    ArchiveContentError,
    ArchiveDiscoveryError,
    ArchivePinMismatch,
    CFPBFetchError,
    ContextMismatch,
    CountDisagreement,
    CountNotExact,
    DateMismatch,
    DuplicateComplaintId,
    FieldDisagreement,
    MissingApiRecord,
    MissingStartState,
    ReturnedIdMismatch,
    RowOutsideDay,
    UnusableApiRecord,
    make_cfpb_fetcher,
    select_exports,
)
from ingest.fetch.http import USER_AGENT, ClientIdentity, FetchFailed, HttpClient, Response
from ingest.fetch.nyc311 import make_nyc311_fetcher
from ingest.fetch.registry import FETCHERS, FetchContext
from ingest.sources import cfpb as adapter

ROOT = Path(__file__).resolve().parents[3]
ROOM = (
    "https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/"
    "cfpb-consumer-complaint-database-narratives-archive/"
)
API = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
FILES = "https://files.consumerfinance.gov/f/documents/"
START, END = date(2024, 1, 30), date(2024, 1, 31)
BEFORE, D30, D31, AFTER = "2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01"
DAYS = (BEFORE, D30, D31, AFTER)
COMMIT = "b" * 40
CLIENT = ClientIdentity(user_agent="Sentinel-test/0", library="fake", library_version="0")
NOW = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
STAMP = "2026-09-30T12:00:00.000000+00:00"
LAST_INDEXED = "2026-09-30T12:00:00-05:00"
NULL = object()
"""A JSON null in the API, which its CSV spells as the literal None."""

ARCHIVE_HEADER = [
    "Date received", "Product", "Sub-product", "Issue", "Sub-issue",
    "Consumer complaint narrative", "Company public response", "Company", "State",
    "ZIP code", "Tags", "Submitted via", "Date sent to company",
    "Company response to consumer", "Timely response?", "Complaint ID",
]  # fmt: skip
API_HEADER = [
    "Date received", "Product", "Sub-product", "Issue", "Sub-issue",
    "Company public response", "Company", "State", "ZIP code", "Tags", "Submitted via",
    "Date sent to company", "Company response to consumer", "Timely response?",
    "Complaint ID",
]  # fmt: skip
EXPORTS = {
    "2011-12": "CCDB_Export_1_December_2011_through_April_2018.zip",
    "2023-12": "CCDB_Export_4_December_2023.zip",
    "2024-01": "CCDB_Export_5_January_2024.zip",
    "2024-02": "CCDB_Export_6_February_2024.zip",
}
JANUARY, FEBRUARY = EXPORTS["2024-01"], EXPORTS["2024-02"]
ROOM_REQUEST = ("room",)
WINDOW_COUNT = ("count", D30, D31)


def complaint(
    cid,
    day,
    *,
    story="A story.",
    product="Credit card",
    timely="Yes",
    clock="09:15:00",
    in_api=True,
    in_archive=True,
    api_day=None,
    api_product=None,
    api_timely=None,
    sent=None,
    csv_none=(),
):
    received = f"{api_day or day}T{clock}.000Z"
    return {
        "id": cid,
        "day": day,
        "story": story,
        "product": product,
        "timely": timely,
        "in_api": in_api,
        "in_archive": in_archive,
        "api_day": api_day or day,
        "received": received,
        "api_product": product if api_product is None else api_product,
        "api_timely": timely if api_timely is None else api_timely,
        "sent": received if sent is None else sent,
        "csv_none": tuple(csv_none),
    }


def dataset():
    return [
        complaint("09999999", "2023-12-15", in_api=False),
        complaint("10000001", BEFORE),
        complaint("10000002", D30),
        complaint("10000003", D30, clock="08:00:00", story='First line, "quoted",\r\nsecond line'),
        complaint("10000004", D30, story=""),
        complaint("10000005", D30, story="   \n "),
        complaint("10000006", D31, clock="03:00:00"),
        complaint("10000007", D31, story="  keeps its spaces  "),
        complaint("10000008", D31, story="", in_api=False),
        complaint("10000009", D31, in_archive=False),
        complaint("10000010", AFTER),
    ]


def archive_row(c):
    return [
        c["day"], c["product"], "General-purpose credit card", "Problem", "", c["story"], "",
        "Bank", "NY", "10001", "", "Web", c["day"], "Closed with explanation", c["timely"],
        c["id"],
    ]  # fmt: skip


def api_row(c):
    values = {
        "Date received": c["received"],
        "Product": c["api_product"],
        "Sub-product": "General-purpose credit card",
        "Issue": "Problem",
        "Sub-issue": "None",
        "Company public response": "None",
        "Company": "Bank",
        "State": "NY",
        "ZIP code": "10001",
        "Tags": "None",
        "Submitted via": "Web",
        "Date sent to company": "None" if c["sent"] is NULL else c["sent"],
        "Company response to consumer": "Closed with explanation",
        "Timely response?": c["api_timely"],
        "Complaint ID": c["id"],
    }
    for column in c["csv_none"]:
        values[column] = "None"
    return [values[name] for name in API_HEADER]


def csv_text(header, rows):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


def zip_bytes(member, text, *, extra=()):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in ((member, text), *extra):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 13, 19, 52, 28))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data.encode("utf-8"))
    return buffer.getvalue()


class FakeCFPB:
    """The reading room, the files host and the API, answered from `complaints`.

    `script` queues replacement responses for one request, oldest first; `after` runs a
    change once a request has been answered for the n-th time; `zips` replaces an
    export's bytes; `by_id` replaces a by-ID answer. `log` names each request in order.
    """

    def __init__(self, complaints=None):
        self.complaints = dataset() if complaints is None else complaints
        self.log = []
        self.sent = []
        self.scripted = {}
        self.hooks = {}
        self.zips = {}
        self.by_id = {}
        self.relation = "eq"
        self.index = "complaint-public-v1"

    def script(self, request, *responses):
        self.scripted.setdefault(request, []).extend(responses)

    def after(self, request, change, occurrence=1):
        self.hooks[(request, occurrence)] = change

    def get(self, url, params, headers):
        assert len(self.sent) < 300, "the fetcher is looping"
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
        if url == ROOM:
            assert params == {}
            return ROOM_REQUEST
        if url.startswith(FILES):
            assert params == {}
            return ("zip", url[len(FILES) :])
        if url == API:
            if params.get("format") == "csv":
                assert params["date_received_min"] == params["date_received_max"]
                return ("csv", params["date_received_min"])
            return ("count", params["date_received_min"], params["date_received_max"])
        assert url.startswith(API) and params == {}, (url, params)
        return ("id", url[len(API) :])

    def export(self, name):
        if name in self.zips:
            return self.zips[name]
        month = next(m for m, n in EXPORTS.items() if n == name)
        rows = [archive_row(c) for c in self.complaints
                if c["in_archive"] and c["day"].startswith(month)]  # fmt: skip
        return zip_bytes(name.replace(".zip", ".csv"), csv_text(ARCHIVE_HEADER, rows))

    def api_rows(self, first, last):
        return [c for c in self.complaints if c["in_api"] and first <= c["api_day"] <= last]

    def source(self, c):
        return {
            "complaint_id": c["id"],
            "product": c["api_product"],
            "timely": c["api_timely"],
            "date_received": c["received"],
            "date_sent_to_company": None if c["sent"] is NULL else c["sent"],
            "company": "Bank",
        }

    def answer(self, request):
        kind = request[0]
        if kind == "room":
            links = "".join(f'<a href="{FILES}{name}">{name}</a>' for name in EXPORTS.values())
            body = f"<html><body>{links}<a href='{FILES}{JANUARY}'>again</a></body></html>"
            return Response(200, {}, body.encode())
        if kind == "zip":
            data = self.export(request[1])
            etag = f'"{hashlib.md5(data).hexdigest()}-5"'
            headers = {
                "content-length": str(len(data)),
                "last-modified": "Mon, 14 Sep 2026 14:47:22 GMT",
                "etag": etag,
            }
            return Response(200, headers, data)
        if kind == "count":
            rows = self.api_rows(request[1], request[2])
            hits = [{"_index": self.index, "_id": c["id"], "_source": self.source(c)} for c in rows]
            body = {
                "_meta": {
                    "last_indexed": LAST_INDEXED,
                    "last_updated": LAST_INDEXED,
                    "total_record_count": 18091520,
                    "is_data_stale": False,
                    "has_data_issue": False,
                    "license": "CC0",
                },
                "hits": {
                    "total": {"value": len(rows), "relation": self.relation},
                    "hits": hits[:1],
                },
                "timed_out": False,
            }
            return Response(200, {}, json.dumps(body).encode())
        if kind == "csv":
            rows = [api_row(c) for c in reversed(self.api_rows(request[1], request[1]))]
            return Response(200, {"content-type": "text/csv"}, csv_text(API_HEADER, rows).encode())
        cid = request[1]
        if cid in self.by_id:
            return self.by_id[cid]
        found = [c for c in self.complaints if c["in_api"] and c["id"] == cid]
        hits = [{"_index": self.index, "_id": cid, "_source": self.source(c)} for c in found]
        body = {"hits": {"total": {"value": len(hits), "relation": "eq"}, "hits": hits}}
        return Response(200, {}, json.dumps(body).encode())


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def a_context(directory, transport, *, source="cfpb", start=START, end=END, clock=None):
    clock = clock or FakeClock()
    return FetchContext(
        source=source,
        start=start,
        end=end,
        resolved_start=datetime(start.year, start.month, start.day, tzinfo=UTC),
        resolved_end=datetime(end.year, end.month, end.day, 23, 59, 59, 999999, tzinfo=UTC),
        directory=directory,
        http=HttpClient(transport, clock=clock, sleep=clock.sleep, now=lambda: NOW),
        client=CLIENT,
        sentinel_commit=COMMIT,
        now=lambda: NOW,
    )


def cache(directory, page):
    folder = directory / "cfpb"
    folder.mkdir(parents=True, exist_ok=True)
    with gzip.open(folder / f"{page_digest(page)}.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(page, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def run(context):
    pages = []
    for page in make_cfpb_fetcher(context)(context.source, context.start, context.end):
        cache(context.directory, page)
        pages.append(page)
    return pages


def acquire(tmp_path, transport=None, *, name="acq", **overrides):
    transport = transport or FakeCFPB()
    context = a_context(tmp_path / name, transport, **overrides)
    return context, transport, run(context)


def the_record(context):
    return json.loads(record_path(context.directory).read_bytes())


def journaled(directory):
    path = journal_path(directory)
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


def count_params(first, last):
    return [
        ("date_received_min", first),
        ("date_received_max", last),
        ("size", "1"),
        ("no_aggs", "true"),
        ("no_highlight", "true"),
    ]


def csv_params(day):
    return [
        ("date_received_min", day),
        ("date_received_max", day),
        ("format", "csv"),
        ("no_aggs", "true"),
    ]


def sources(page):
    return [hit["_source"] for hit in page["hits"]["hits"]]


CLEAN_LOG = [
    ROOM_REQUEST,
    ("zip", JANUARY),
    ("zip", FEBRUARY),
    WINDOW_COUNT,
    *[request for day in DAYS for request in (("count", day, day), ("csv", day))],
    WINDOW_COUNT,
]


# --- the archive: discovery, selection and content --------------------------------------


REAL_LINKS = [
    "CCDB_Export_1_December_2011_through_April_2018",
    "CCDB_Export_2_May_2018_through_April_2021",
    "CCDB_Export_3_May_2021_through_October_2022",
    "CCDB_Export_4_November_2022_through_August_2023",
    "CCDB_Export_5_September_2023_through_March_2024",
    "CCDB_Export_6_April_2024_through_July_2024",
    "CCDB_Export_7_August_2024_through_October_2024",
    "CCDB_Export_8_November_2024_through_December_2024",
    "CCDB_Export_9_January_2025_through_February_2025",
    "CCDB_Export_10_March_2025_through_April_2025",
    "CCDB_Export_11_May_2025_through_June_2025",
    "CCDB_Export_12_July_2025_through_August_2025",
    "CCDB_Export_13_September_2025_through_October_2025",
    "CCDB_Export_14_November_2025_through_December_2025",
    "CCDB_Export_15_January_2026_through_February_2026",
    "CCDB_Export_16_March_2026",
    "CCDB_Export_17_April_2026",
    "CCDB_Export_18_May_2026",
    "CCDB_Export_19_June_2026",
    "CCDB_Export_20_July_2026",
    "CCDB_Export_21_August_2026",
]


def test_the_real_reading_room_selects_5_to_14_for_the_population_and_15_as_boundary():
    page = "".join(f'<a href="{FILES}{name}.zip">x</a>' for name in reversed(REAL_LINKS))
    selected = select_exports(page, date(2024, 1, 1), date(2025, 12, 31))
    assert [(e.number, e.role) for e in selected] == [
        *[(n, "population") for n in range(5, 15)],
        (15, "boundary"),
    ]
    assert selected[0].url == f"{FILES}CCDB_Export_5_September_2023_through_March_2024.zip"
    assert (selected[0].first, selected[0].last) == (date(2023, 9, 1), date(2024, 3, 31))
    assert (selected[-1].first, selected[-1].last) == (date(2026, 1, 1), date(2026, 2, 28))


def test_a_margin_day_alone_makes_an_export_boundary_only():
    page = "".join(f'<a href="{FILES}{name}">x</a>' for name in EXPORTS.values())
    selected = select_exports(page, date(2024, 1, 1), date(2024, 1, 31))
    assert [(e.number, e.role) for e in selected] == [
        (4, "boundary"),
        (5, "population"),
        (6, "boundary"),
    ]
    assert [(e.number, e.role) for e in select_exports(page, START, END)] == [
        (5, "population"),
        (6, "boundary"),
    ]


@pytest.mark.parametrize(
    "links",
    [
        ["CCDB_Export_5_Janvier_2024"],
        ["CCDB_Export_5_January_2024_through_Smarch_2024"],
        ["CCDB_Export_5_March_2024_through_January_2024"],
        ["CCDB_Export_5_January_2024", "CCDB_Export_5_January_2024_through_March_2024"],
        ["CCDB_Export_4_December_2023"],
    ],
)
def test_an_unreadable_or_useless_reading_room_refuses(links):
    page = "".join(f'<a href="{FILES}{name}.zip">x</a>' for name in links)
    with pytest.raises(ArchiveDiscoveryError):
        select_exports(page, START, END)


def test_only_the_selected_exports_are_downloaded_and_they_are_pinned(tmp_path):
    context, transport, _ = acquire(tmp_path)
    assert transport.log == CLEAN_LOG
    start = json.loads((context.directory / START_NAME).read_bytes())
    pins = start["archive"]["exports"]
    for pin, name, number in zip(pins, (JANUARY, FEBRUARY), (5, 6)):
        data = transport.export(name)
        assert pin == {
            "bytes": len(data),
            "etag": f'"{hashlib.md5(data).hexdigest()}-5"',
            "last_modified": "Mon, 14 Sep 2026 14:47:22 GMT",
            "number": number,
            "sha256": hashlib.sha256(data).hexdigest(),
            "url": FILES + name,
        }
        assert (context.directory / "archive" / name).read_bytes() == data, "retained"
    room = transport.answer(ROOM_REQUEST).body
    assert start["archive"]["reading_room"] == {
        "retrieved_at": STAMP,
        "sha256": hashlib.sha256(room).hexdigest(),
        "url": ROOM,
    }


def refuses_before_any_day(tmp_path, transport, error, match):
    with pytest.raises(error, match=match):
        acquire(tmp_path, transport)
    directory = tmp_path / "acq"
    assert not (directory / START_NAME).exists()
    assert journaled(directory) == []
    assert not any(r[0] in ("count", "csv", "id") for r in transport.log)
    assert not (directory / LOCK_NAME).exists()


def january(rows, *, header=ARCHIVE_HEADER, extra=()):
    return zip_bytes(JANUARY.replace(".zip", ".csv"), csv_text(header, rows), extra=extra)


def january_rows(transport):
    return [
        archive_row(c)
        for c in transport.complaints
        if c["in_archive"] and c["day"].startswith("2024-01")
    ]


def test_archive_header_drift_refuses(tmp_path):
    transport = FakeCFPB()
    header = [*ARCHIVE_HEADER]
    header[5] = "Narrative"
    transport.zips[JANUARY] = january(january_rows(transport), header=header)
    refuses_before_any_day(tmp_path, transport, ArchiveContentError, "header")


@pytest.mark.parametrize(
    "value", ["2024-01-30T00:00:00", "01/30/2024", "2024-02-30", "", "20240130", "2024-W05-2"]
)
def test_a_date_that_is_not_yyyy_mm_dd_refuses(tmp_path, value):
    transport = FakeCFPB()
    rows = january_rows(transport)
    rows[1][0] = value
    transport.zips[JANUARY] = january(rows)
    refuses_before_any_day(tmp_path, transport, ArchiveContentError, "Date received")


def test_a_complaint_id_twice_in_one_export_refuses(tmp_path):
    transport = FakeCFPB()
    rows = january_rows(transport)
    transport.zips[JANUARY] = january([*rows, [*rows[2][:-1], rows[1][-1]]])
    refuses_before_any_day(tmp_path, transport, DuplicateComplaintId, "10000002")


def test_a_complaint_id_in_two_exports_refuses(tmp_path):
    transport = FakeCFPB([*dataset(), complaint("10000002", "2024-02-02", in_api=False)])
    refuses_before_any_day(tmp_path, transport, DuplicateComplaintId, "10000002")


@pytest.mark.parametrize("cid", ["010000002", "1000000A", "", " 10000002"])
def test_an_archive_complaint_id_that_is_not_a_plain_decimal_refuses(tmp_path, cid):
    transport = FakeCFPB()
    rows = january_rows(transport)
    rows[2][-1] = cid
    transport.zips[JANUARY] = january(rows)
    refuses_before_any_day(tmp_path, transport, ArchiveContentError, "Complaint ID")


@pytest.mark.parametrize("cid", [str(1 << 41), str(1 << 63), str(1 << 64)])
def test_an_archive_complaint_id_beyond_the_index_refuses(tmp_path, cid):
    """The index key is a non-negative signed 64-bit integer, so an ID must be below 2**41."""
    transport = FakeCFPB()
    rows = january_rows(transport)
    rows[2][-1] = cid
    transport.zips[JANUARY] = january(rows)
    refuses_before_any_day(tmp_path, transport, ArchiveContentError, "beyond the archive index")


@pytest.mark.parametrize("value", ["5742-10-22", "9999-12-31"])
def test_an_archive_date_beyond_the_index_refuses(tmp_path, value):
    """A later day's ordinal would not fit its 21 bits and would spill into the ID."""
    transport = FakeCFPB()
    rows = january_rows(transport)
    rows[1][0] = value
    transport.zips[JANUARY] = january(rows)
    refuses_before_any_day(tmp_path, transport, ArchiveContentError, "beyond the archive index")


def test_the_last_id_and_day_the_index_holds_are_accepted(tmp_path):
    transport = FakeCFPB()
    last = complaint(str(cfpb.ID_LIMIT - 1), "5742-10-21", in_api=False)
    transport.zips[JANUARY] = january([*january_rows(transport), archive_row(last)])
    context, _, _ = acquire(tmp_path, transport)
    exports = the_record(context)["source_details"]["archive"]["exports"]
    assert exports[0]["date_max"] == "5742-10-21"


def test_the_index_key_is_reversible_and_sorts_by_id_at_its_limits():
    top, last = cfpb.ID_LIMIT - 1, cfpb.LAST_INDEXED_DAY
    assert (cfpb.ID_LIMIT, last, last.toordinal()) == (1 << 41, date(5742, 10, 21), cfpb.DAY_MASK)
    assert cfpb._index_key(0, date.min, False) == 2 and date.min.toordinal() == 1
    assert cfpb._index_key(top, last, True) == (1 << 63) - 1
    corners = [
        (0, date.min, False),
        (0, last, True),
        (10000002, date(2024, 1, 30), True),
        (top, date.min, False),
        (top, last, True),
    ]
    for number, day, included in corners:
        keys = array("q", [cfpb._index_key(number, day, included)])
        index = cfpb._Archive(keys=keys, included_total=0, facts=[], spill=None)
        assert index.lookup(str(number)) == (day.toordinal(), included)
        assert index.lookup(str(number + 1)) is None
    for number in (0, 10000002, top - 1):
        assert cfpb._index_key(number, last, True) < cfpb._index_key(number + 1, date.min, False)


def test_a_ragged_archive_row_refuses(tmp_path):
    transport = FakeCFPB()
    rows = january_rows(transport)
    rows[2] = rows[2][:-2]
    transport.zips[JANUARY] = january(rows)
    refuses_before_any_day(tmp_path, transport, ArchiveContentError, "columns")


def test_an_export_that_is_not_one_csv_member_refuses(tmp_path):
    transport = FakeCFPB()
    transport.zips[JANUARY] = january(january_rows(transport), extra=[("readme.txt", "hello")])
    refuses_before_any_day(tmp_path, transport, ArchiveContentError, "member")


def test_a_boundary_export_holding_a_window_row_refuses(tmp_path):
    transport = FakeCFPB()
    rows = [archive_row(c) for c in transport.complaints if c["day"].startswith("2024-02")]
    stray = archive_row(complaint("10000099", D31, in_api=False))
    transport.zips[FEBRUARY] = zip_bytes(
        FEBRUARY.replace(".zip", ".csv"), csv_text(ARCHIVE_HEADER, [*rows, stray])
    )
    refuses_before_any_day(tmp_path, transport, ArchiveContentError, "boundary")


def test_a_window_day_the_archive_content_does_not_cover_refuses(tmp_path):
    transport = FakeCFPB([c for c in dataset() if c["day"] != D31])
    refuses_before_any_day(tmp_path, transport, ArchiveContentError, D31)


def test_a_body_that_is_not_a_zip_is_retried_then_refused(tmp_path):
    transport = FakeCFPB()
    transport.script(("zip", JANUARY), *[Response(200, {}, b"<html>not a zip</html>")] * 6)
    with pytest.raises(FetchFailed, match="invalid body"):
        acquire(tmp_path, transport)
    assert transport.log.count(("zip", JANUARY)) == 6


def test_a_short_zip_body_is_retried(tmp_path):
    transport = FakeCFPB()
    data = transport.export(JANUARY)
    short = Response(200, {"content-length": str(len(data) + 10)}, data)
    transport.script(("zip", JANUARY), short)
    context, _, _ = acquire(tmp_path, transport)
    assert transport.log.count(("zip", JANUARY)) == 2
    assert journaled(context.directory) == list(DAYS)


# --- pages: one per window day, six fields, exact strings, deterministic order ------------


def test_one_page_per_window_day_and_none_for_the_margins(tmp_path):
    context, _, pages = acquire(tmp_path)
    record = the_record(context)
    assert [entry["key"] for entry in record["slices"]] == list(DAYS)
    by_key = {entry["key"]: entry["pages"] for entry in record["slices"]}
    assert by_key[BEFORE] == [] and by_key[AFTER] == []
    assert [by_key[D30], by_key[D31]] == [[page_digest(page)] for page in pages]
    assert len(pages) == 2
    verify_acquisition(context.directory, source="cfpb", start=START, end=END)


def test_pages_hold_the_six_fields_exactly_as_supplied_in_a_fixed_order(tmp_path):
    _, transport, pages = acquire(tmp_path)
    by_id = {c["id"]: c for c in transport.complaints}
    assert [[r["complaint_id"] for r in sources(page)] for page in pages] == [
        ["10000003", "10000002"],
        ["10000006", "10000007"],
    ]
    for page in pages:
        rows = sources(page)
        assert rows == sorted(rows, key=lambda r: (r["date_received"], r["complaint_id"]))
        for row in rows:
            c = by_id[row["complaint_id"]]
            assert row == {
                "complaint_id": c["id"],
                "complaint_what_happened": c["story"],
                "product": c["api_product"],
                "timely": c["api_timely"],
                "date_received": c["received"],
                "date_sent_to_company": c["sent"],
            }
    assert sources(pages[0])[0]["complaint_what_happened"] == 'First line, "quoted",\r\nsecond line'
    assert sources(pages[1])[1]["complaint_what_happened"] == "  keeps its spaces  "


def test_the_pages_are_what_the_adapter_reads(tmp_path):
    _, _, pages = acquire(tmp_path)
    records = [adapter.normalize(row)[0] for page in pages for row in adapter.rows_from_page(page)]
    assert [r.external_id for r in records] == ["10000003", "10000002", "10000006", "10000007"]
    assert records[2].submitted_at == datetime(2024, 1, 31, 3, 0, tzinfo=UTC), "UTC, not New York"


def test_a_window_day_with_no_narrative_still_gets_its_page(tmp_path):
    complaints = [c for c in dataset() if c["id"] not in ("10000006", "10000007")]
    context, _, pages = acquire(tmp_path, FakeCFPB(complaints))
    assert pages[1] == {"hits": {"hits": []}}
    assert the_record(context)["slices"][2]["pages"] == [page_digest(pages[1])]


# --- the API: exact requests, exact counts, whole days ----------------------------------


def test_each_day_asks_for_exactly_the_frozen_count_and_csv(tmp_path):
    _, transport, _ = acquire(tmp_path)
    api = [(url, params) for url, params, _ in transport.sent if url == API]
    assert api == [
        (API, count_params(D30, D31)),
        *[pair for day in DAYS for pair in ((API, count_params(day, day)), (API, csv_params(day)))],
        (API, count_params(D30, D31)),
    ]


def test_no_token_or_other_header_is_ever_sent(tmp_path):
    _, transport, _ = acquire(tmp_path)
    for _, params, headers in transport.sent:
        assert headers == {"User-Agent": USER_AGENT}
        assert not [name for name, _ in params if "token" in name.lower()]


def test_a_count_that_is_only_a_lower_bound_refuses(tmp_path):
    transport = FakeCFPB()
    transport.after(WINDOW_COUNT, lambda t: setattr(t, "relation", "gte"))
    with pytest.raises(CountNotExact, match="gte"):
        acquire(tmp_path, transport)
    assert transport.log[-1] == ("count", BEFORE, BEFORE)


def test_a_start_snapshot_count_that_is_only_a_lower_bound_refuses(tmp_path):
    transport = FakeCFPB()
    transport.relation = "gte"
    with pytest.raises(CountNotExact):
        acquire(tmp_path, transport)
    assert not (tmp_path / "acq" / START_NAME).exists()


def count_response(n):
    body = {
        "_meta": {"last_indexed": LAST_INDEXED},
        "hits": {"total": {"value": n, "relation": "eq"}, "hits": []},
    }
    return Response(200, {}, json.dumps(body).encode())


def test_a_count_that_disagrees_with_its_csv_is_requested_again_with_the_csv_once(tmp_path):
    transport = FakeCFPB()
    transport.script(("count", D30, D30), count_response(9))
    context, _, _ = acquire(tmp_path, transport)
    on_day = [r for r in transport.log if r in (("count", D30, D30), ("csv", D30))]
    assert on_day == [("count", D30, D30), ("csv", D30), ("count", D30, D30), ("csv", D30)]
    entry = the_record(context)["slices"][1]
    assert entry["verification"]["api_count"] == 4 and len(entry["requests"]) == 2


def test_a_second_disagreement_refuses_and_keeps_the_acquisition_incomplete(tmp_path):
    transport = FakeCFPB()
    transport.script(("count", D30, D30), count_response(9), count_response(9))
    with pytest.raises(CountDisagreement, match=D30):
        acquire(tmp_path, transport)
    assert transport.log.count(("csv", D30)) == 2
    assert journaled(tmp_path / "acq") == [BEFORE]
    assert not record_path(tmp_path / "acq").exists()


def csv_response(rows):
    return Response(200, {}, csv_text(API_HEADER, rows).encode())


def test_a_complaint_id_twice_in_a_day_refuses_at_once(tmp_path):
    transport = FakeCFPB()
    rows = [api_row(c) for c in transport.api_rows(D30, D30)]
    transport.script(("count", D30, D30), count_response(5))
    transport.script(("csv", D30), csv_response([*rows, rows[0]]))
    with pytest.raises(DuplicateComplaintId, match="10000002"):
        acquire(tmp_path, transport)
    assert transport.log.count(("csv", D30)) == 1


def test_a_row_outside_the_requested_utc_day_refuses(tmp_path):
    transport = FakeCFPB()
    rows = [api_row(c) for c in transport.api_rows(D30, D30)]
    assert rows[2][-1] == "10000004", "an excluded row: every row is checked, not the join's only"
    rows[2][0] = "2024-01-31T00:00:01.000Z"
    transport.script(("csv", D30), csv_response(rows))
    with pytest.raises(RowOutsideDay, match=D30):
        acquire(tmp_path, transport)


@pytest.mark.parametrize("value", ["2024-01-30T09:15:00.000", "yesterday"])
def test_a_date_received_that_cannot_be_read_as_an_instant_refuses(tmp_path, value):
    transport = FakeCFPB()
    rows = [api_row(c) for c in transport.api_rows(D30, D30)]
    rows[2][0] = value
    transport.script(("csv", D30), csv_response(rows))
    with pytest.raises(UnusableApiRecord, match="Date received"):
        acquire(tmp_path, transport)


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"<!DOCTYPE html><html><body>Service unavailable</body></html>",
        b"Date received,Product\r\n2024-01-30T09:15:00.000Z,Credit card\r\n",
        None,
        b"\xff\xfe",
    ],
    ids=["empty", "html", "header drift", "truncated", "not utf-8"],
)
def test_an_unusable_csv_body_is_retried_then_refused(tmp_path, body):
    transport = FakeCFPB()
    if body is None:
        whole = transport.answer(("csv", D30)).body
        body = whole[: len(whole) - 40]
    transport.script(("csv", D30), *[Response(200, {}, body)] * 6)
    with pytest.raises(FetchFailed, match="invalid body"):
        acquire(tmp_path, transport)
    assert transport.log.count(("csv", D30)) == 6


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"<html></html>",
        b"[]",
        b'{"hits":{"total":9,"hits":[]},"_meta":{}}',
        b'{"hits":{"total":{"value":"9","relation":"eq"},"hits":[]},"_meta":{}}',
        b'{"hits":{"total":{"value":9,"relation":"eq"},"hits":[]}}',
    ],
)
def test_an_unusable_count_body_is_retried_then_refused(tmp_path, body):
    transport = FakeCFPB()
    transport.script(("count", BEFORE, BEFORE), *[Response(200, {}, body)] * 6)
    with pytest.raises(FetchFailed, match="invalid body"):
        acquire(tmp_path, transport)


def test_the_d47_retry_policy_applies_beneath_every_request(tmp_path):
    transport = FakeCFPB()
    transport.script(ROOM_REQUEST, Response(503, {}, b""))
    transport.script(("csv", D30), Response(500, {}, b""))
    transport.script(("count", D31, D31), Response(429, {"retry-after": "7"}, b""))
    clock = FakeClock()
    context, _, _ = acquire(tmp_path, transport, clock=clock)
    assert transport.log[:2] == [ROOM_REQUEST, ROOM_REQUEST]
    assert transport.log.count(("csv", D30)) == 2
    assert clock.sleeps.count(2) == 2 and 7 in clock.sleeps
    assert journaled(context.directory) == list(DAYS)


def test_a_403_stops_at_once(tmp_path):
    transport = FakeCFPB()
    transport.script(("csv", D31), Response(403, {}, b"Forbidden"))
    with pytest.raises(FetchFailed, match="403"):
        acquire(tmp_path, transport)
    assert transport.log[-1] == ("csv", D31) and transport.log.count(("csv", D31)) == 1
    assert journaled(tmp_path / "acq") == [BEFORE, D30]


# --- the join: exact IDs, agreeing fields, one date --------------------------------------


def with_changed(cid, **changes):
    return [complaint(c["id"], c["day"], **changes) if c["id"] == cid else c for c in dataset()]


def test_a_product_that_disagrees_refuses(tmp_path):
    transport = FakeCFPB(with_changed("10000002", api_product="Mortgage"))
    with pytest.raises(FieldDisagreement, match="product"):
        acquire(tmp_path, transport)
    assert journaled(tmp_path / "acq") == [BEFORE]


def test_a_timely_value_that_disagrees_refuses(tmp_path):
    transport = FakeCFPB(with_changed("10000007", api_timely="No"))
    with pytest.raises(FieldDisagreement, match="timely"):
        acquire(tmp_path, transport)
    assert journaled(tmp_path / "acq") == [BEFORE, D30]


def test_an_included_record_on_another_api_day_refuses(tmp_path):
    transport = FakeCFPB(with_changed("10000006", api_day=D30))
    with pytest.raises(DateMismatch, match="10000006"):
        acquire(tmp_path, transport)
    assert journaled(tmp_path / "acq") == [BEFORE]


def test_an_excluded_record_on_another_api_day_refuses_as_a_date_mismatch(tmp_path):
    transport = FakeCFPB(with_changed("10000004", story="", api_day=D31))
    with pytest.raises(DateMismatch, match="10000004"):
        acquire(tmp_path, transport)


def test_an_included_record_missing_from_its_api_day_refuses(tmp_path):
    transport = FakeCFPB(with_changed("10000003", story="Story", in_api=False))
    with pytest.raises(MissingApiRecord, match="10000003"):
        acquire(tmp_path, transport)
    assert journaled(tmp_path / "acq") == [BEFORE]


def test_complaint_ids_join_as_exact_strings(tmp_path):
    transport = FakeCFPB()
    rows = [api_row(c) for c in transport.api_rows(D30, D30)]
    rows = [[*row[:-1], "0" + row[-1]] if row[-1] == "10000002" else row for row in rows]
    transport.script(("csv", D30), csv_response(rows))
    with pytest.raises(MissingApiRecord, match="10000002"):
        acquire(tmp_path, transport)


def test_populations_are_classified_and_counted(tmp_path):
    context, _, _ = acquire(tmp_path)
    slices = {entry["key"]: entry["verification"] for entry in the_record(context)["slices"]}
    zero = {"included": 0, "matched": 0, "archive_only": 0, "api_only": 0, "resolved_by_id": 0}
    assert slices == {
        BEFORE: {"api_count": 1, "api_rows": 1, "api_distinct_ids": 1, **zero, "excluded": 1},
        D30: {
            "api_count": 4,
            "api_rows": 4,
            "api_distinct_ids": 4,
            "included": 2,
            "matched": 2,
            "excluded": 2,
            "archive_only": 0,
            "api_only": 0,
            "resolved_by_id": 0,
        },
        D31: {
            "api_count": 3,
            "api_rows": 3,
            "api_distinct_ids": 3,
            "included": 2,
            "matched": 2,
            "excluded": 0,
            "archive_only": 1,
            "api_only": 1,
            "resolved_by_id": 0,
        },
        AFTER: {"api_count": 1, "api_rows": 1, "api_distinct_ids": 1, **zero, "excluded": 1},
    }


# --- None: resolved by ID for the join's rows only ---------------------------------------


def test_a_none_in_an_included_row_is_resolved_by_id_and_the_json_value_used(tmp_path):
    transport = FakeCFPB(with_changed("10000002", csv_none=("Product", "Timely response?")))
    context, _, pages = acquire(tmp_path, transport)
    assert ("id", "10000002") in transport.log
    assert transport.log.index(("id", "10000002")) == transport.log.index(("csv", D30)) + 1
    row = sources(pages[0])[0]
    assert row["product"] == "Credit card" and row["timely"] == "Yes"
    entry = the_record(context)["slices"][1]
    assert entry["verification"]["resolved_by_id"] == 1
    by_id = transport.answer(("id", "10000002")).body
    assert entry["requests"][2] == request_record(API + "10000002", [], by_id)


def test_a_date_received_resolved_by_id_must_still_fall_on_the_day(tmp_path):
    transport = FakeCFPB(with_changed("10000007", csv_none=("Date received",)))
    source = {**transport.source(transport.complaints[7]), "date_received": f"{D30}T10:00:00.000Z"}
    transport.by_id["10000007"] = by_id_body([{"_source": source}])
    with pytest.raises(RowOutsideDay, match="10000007"):
        acquire(tmp_path, transport)


def test_only_the_none_fields_take_the_json_value(tmp_path):
    transport = FakeCFPB(with_changed("10000002", csv_none=("Product",)))
    source = {**transport.source(transport.complaints[2]), "timely": "No"}
    transport.by_id["10000002"] = by_id_body([{"_source": source}])
    _, _, pages = acquire(tmp_path, transport)
    row = next(r for r in sources(pages[0]) if r["complaint_id"] == "10000002")
    assert (row["product"], row["timely"]) == ("Credit card", "Yes"), "timely stays the CSV's"


def test_api_ids_match_archive_ids_only_as_exact_strings(tmp_path):
    transport = FakeCFPB()
    rows = [api_row(c) for c in transport.api_rows(D30, D30)]
    rows = [[*row[:-1], "0" + row[-1]] if row[-1] == "10000004" else row for row in rows]
    transport.script(("csv", D30), csv_response(rows))
    context, _, _ = acquire(tmp_path, transport)
    day = the_record(context)["slices"][1]["verification"]
    assert (day["excluded"], day["api_only"], day["archive_only"]) == (1, 1, 1)


def test_a_null_date_sent_to_company_resolves_to_null(tmp_path):
    transport = FakeCFPB(with_changed("10000006", clock="03:00:00", sent=NULL))
    _, _, pages = acquire(tmp_path, transport)
    assert sources(pages[1])[0]["date_sent_to_company"] is None
    assert ("id", "10000006") in transport.log


def test_a_none_date_received_is_resolved_then_checked(tmp_path):
    transport = FakeCFPB(with_changed("10000007", csv_none=("Date received",)))
    _, _, pages = acquire(tmp_path, transport)
    assert sources(pages[1])[1]["date_received"] == f"{D31}T09:15:00.000Z"


def by_id_body(hits, total=None):
    total = len(hits) if total is None else total
    body = {"hits": {"total": {"value": total, "relation": "eq"}, "hits": hits}}
    return Response(200, {}, json.dumps(body).encode())


def none_case(tmp_path, answer, error, match):
    transport = FakeCFPB(with_changed("10000002", csv_none=("Product",)))
    transport.by_id["10000002"] = answer
    with pytest.raises(error, match=match):
        acquire(tmp_path, transport)
    assert transport.log.count(("id", "10000002")) == 1
    assert journaled(tmp_path / "acq") == [BEFORE]


def test_zero_hits_by_id_is_the_missing_record_refusal(tmp_path):
    none_case(tmp_path, by_id_body([]), MissingApiRecord, "10000002")


def test_a_404_by_id_is_the_missing_record_refusal(tmp_path):
    none_case(tmp_path, Response(404, {}, b"Not found"), MissingApiRecord, "10000002")


def test_a_different_returned_id_refuses(tmp_path):
    hit = {"_source": {"complaint_id": "10000003", "product": "Credit card", "timely": "Yes"}}
    none_case(tmp_path, by_id_body([hit]), ReturnedIdMismatch, "10000003")


def test_a_numeric_returned_id_is_not_the_requested_string(tmp_path):
    hit = {"_source": {"complaint_id": 10000002, "product": "Credit card", "timely": "Yes"}}
    none_case(tmp_path, by_id_body([hit]), ReturnedIdMismatch, "10000002")


def test_more_than_one_hit_by_id_refuses(tmp_path):
    hit = {"_source": {"complaint_id": "10000002", "product": "Credit card", "timely": "Yes"}}
    none_case(tmp_path, by_id_body([hit, hit]), UnusableApiRecord, "10000002")


def test_a_null_product_by_id_refuses(tmp_path):
    hit = {"_source": {"complaint_id": "10000002", "product": None, "timely": "Yes"}}
    none_case(tmp_path, by_id_body([hit]), UnusableApiRecord, "product")


def test_a_none_complaint_id_refuses_without_any_lookup(tmp_path):
    transport = FakeCFPB()
    rows = [api_row(c) for c in transport.api_rows(D30, D30)]
    rows[0][-1] = "None"
    transport.script(("csv", D30), csv_response(rows))
    with pytest.raises(UnusableApiRecord, match="Complaint ID"):
        acquire(tmp_path, transport)
    assert not [r for r in transport.log if r[0] == "id"]


def test_a_none_in_an_api_only_or_excluded_row_triggers_no_lookup(tmp_path):
    complaints = [
        complaint(c["id"], c["day"], story=c["story"], in_api=c["in_api"],
                  in_archive=c["in_archive"], csv_none=("Product", "Date received"))
        if c["id"] in ("10000004", "10000009") else c
        for c in dataset()
    ]  # fmt: skip
    context, transport, _ = acquire(tmp_path, FakeCFPB(complaints))
    assert not [r for r in transport.log if r[0] == "id"]
    assert journaled(context.directory) == list(DAYS)


# --- provenance: source_details, start and end -------------------------------------------


def test_the_start_state_is_recorded_before_the_first_day(tmp_path):
    transport = FakeCFPB()
    seen = []
    start = tmp_path / "acq" / START_NAME
    transport.after(("count", BEFORE, BEFORE), lambda t: seen.append(start.exists()))
    acquire(tmp_path, transport)
    assert seen == [True]


def snapshot(transport):
    body = transport.answer(WINDOW_COUNT).body
    return {
        "_index": "complaint-public-v1",
        "hits_total": 7,
        "last_indexed": LAST_INDEXED,
        "last_updated": LAST_INDEXED,
        "request": request_record(API, count_params(D30, D31), body),
        "total_record_count": 18091520,
    }


def test_source_details_hold_exactly_d49s_keys(tmp_path):
    context, transport, _ = acquire(tmp_path)
    details = the_record(context)["source_details"]
    start = json.loads((context.directory / START_NAME).read_bytes())
    assert set(details) == {
        "acquisition_kind", "source_api_version", "archive", "api", "join", "start", "end"
    }  # fmt: skip
    assert details["acquisition_kind"] == "cfpb-archive-api-reconstruction-v1"
    assert details["source_api_version"] == "cfpb-ccdb-v1"
    assert details["start"] == start
    assert start["api"] == snapshot(transport)
    assert details["end"] == snapshot(transport)
    assert details["join"] == {
        "key": "exact Complaint ID string equality",
        "matched": 4,
        "excluded": 2,
        "archive_only": 1,
        "api_only": 1,
        "resolved_by_id": 0,
    }
    exports = details["archive"]["exports"]
    assert [(e["number"], e["role"]) for e in exports] == [(5, "population"), (6, "boundary")]
    january_bytes = transport.export(JANUARY)
    member = zipfile.ZipFile(io.BytesIO(january_bytes)).infolist()[0]
    assert exports[0] == {
        **start["archive"]["exports"][0],
        "role": "population",
        "member": "CCDB_Export_5_January_2024.csv",
        "member_bytes": member.file_size,
        "header": ARCHIVE_HEADER,
        "rows": 8,
        "date_min": BEFORE,
        "date_max": D31,
    }
    assert (exports[1]["rows"], exports[1]["date_min"], exports[1]["date_max"]) == (1, AFTER, AFTER)
    assert details["archive"]["reading_room"] == start["archive"]["reading_room"]
    assert "strip()" in details["archive"]["narrative_filter"]
    api = details["api"]
    assert api["base_url"] == API
    assert api["endpoints"] == {"by_id": API + "{complaintId}", "search": API}
    assert api["parameters"] == {
        "count": [
            ["date_received_min", "D"],
            ["date_received_max", "D"],
            ["size", "1"],
            ["no_aggs", "true"],
            ["no_highlight", "true"],
        ],
        "csv": [
            ["date_received_min", "D"],
            ["date_received_max", "D"],
            ["format", "csv"],
            ["no_aggs", "true"],
        ],
    }
    assert api["days"] == {
        day: {"_index": "complaint-public-v1", "hits_total": n, "last_indexed": LAST_INDEXED}
        for day, n in ((BEFORE, 1), (D30, 4), (D31, 3), (AFTER, 1))
    }


def test_each_day_keeps_the_requests_it_used_and_their_raw_bodies(tmp_path):
    context, transport, _ = acquire(tmp_path)
    for entry in the_record(context)["slices"]:
        day = entry["key"]
        count = transport.answer(("count", day, day)).body
        body = transport.answer(("csv", day)).body
        assert entry["requests"] == [
            request_record(API, count_params(day, day), count),
            request_record(API, csv_params(day), body),
        ]
        for data in (count, body):
            digest = hashlib.sha256(data).hexdigest()
            kept = list((context.directory / "api").glob(f"{digest}.*.gz"))
            assert len(kept) == 1 and gzip.decompress(kept[0].read_bytes()) == data


def test_the_served_index_is_provenance_that_may_change_between_days(tmp_path):
    transport = FakeCFPB()
    transport.after(("csv", D30), lambda t: setattr(t, "index", "complaint-public-v2"))
    context, _, _ = acquire(tmp_path, transport)
    details = the_record(context)["source_details"]
    assert details["api"]["days"][D30]["_index"] == "complaint-public-v1"
    assert details["api"]["days"][D31]["_index"] == "complaint-public-v2"
    assert details["end"]["_index"] == "complaint-public-v2"
    assert transport.log[-1] == WINDOW_COUNT, "no recount or rewind for API drift"
    assert the_record(context)["rewinds"] == []


def test_identical_responses_give_identical_pages_and_record_bytes(tmp_path):
    first, _, first_pages = acquire(tmp_path, name="a")
    second, _, second_pages = acquire(tmp_path, name="b")
    assert [page_digest(p) for p in first_pages] == [page_digest(p) for p in second_pages]
    assert record_path(first.directory).read_bytes() == record_path(second.directory).read_bytes()
    data = record_path(first.directory).read_bytes()
    assert canonical_bytes(json.loads(data)) == data


# --- resume: pinned archives, journaled days, a completed acquisition ----------------------


def interrupted(tmp_path):
    transport = FakeCFPB()
    transport.script(("csv", D31), Response(403, {}, b""))
    with pytest.raises(FetchFailed):
        acquire(tmp_path, transport)
    return tmp_path / "acq"


def test_a_resumed_acquisition_uses_its_pinned_archive_and_skips_journaled_days(tmp_path):
    clean, _, _ = acquire(tmp_path, name="clean")
    directory = interrupted(tmp_path)
    start = (directory / START_NAME).read_bytes()

    later = FakeCFPB()
    context, _, _ = acquire(tmp_path, later)
    assert later.log == [
        ("count", D31, D31),
        ("csv", D31),
        ("count", AFTER, AFTER),
        ("csv", AFTER),
        WINDOW_COUNT,
    ], "no reading room, no export and no finished day is requested again"
    assert (directory / START_NAME).read_bytes() == start
    assert record_path(directory).read_bytes() == record_path(clean.directory).read_bytes()


def test_a_changed_pinned_export_refuses_the_resume_before_any_request(tmp_path):
    directory = interrupted(tmp_path)
    path = directory / "archive" / JANUARY
    path.write_bytes(path.read_bytes() + b"\x00")
    later = FakeCFPB()
    with pytest.raises(ArchivePinMismatch, match=JANUARY):
        acquire(tmp_path, later)
    assert later.sent == []


def test_a_missing_pinned_export_refuses_the_resume_before_any_request(tmp_path):
    directory = interrupted(tmp_path)
    (directory / "archive" / FEBRUARY).unlink()
    later = FakeCFPB()
    with pytest.raises(ArchivePinMismatch, match=FEBRUARY):
        acquire(tmp_path, later)
    assert later.sent == []


def test_journaled_days_without_a_start_state_refuse_before_any_request(tmp_path):
    directory = interrupted(tmp_path)
    (directory / START_NAME).unlink()
    later = FakeCFPB()
    with pytest.raises(MissingStartState, match="start"):
        acquire(tmp_path, later)
    assert later.sent == []


def test_a_completed_acquisition_is_never_reopened_and_makes_no_request(tmp_path):
    context, _, _ = acquire(tmp_path)
    record = record_path(context.directory).read_bytes()
    again = FakeCFPB()
    with pytest.raises(AcquisitionError, match="completed acquisition"):
        run(a_context(context.directory, again))
    assert again.sent == []
    assert record_path(context.directory).read_bytes() == record


def test_a_slice_is_journaled_only_after_its_page_is_cached(tmp_path):
    transport = FakeCFPB()
    context = a_context(tmp_path / "acq", transport)
    at_yield = []
    for page in make_cfpb_fetcher(context)("cfpb", START, END):
        at_yield.append(journaled(context.directory))
        cache(context.directory, page)
    assert at_yield == [[BEFORE], [BEFORE, D30]]


# --- registry, context, constants and the network -----------------------------------------


def test_cfpb_and_nyc311_are_the_registered_fetchers():
    assert FETCHERS == {"cfpb": make_cfpb_fetcher, "nyc311": make_nyc311_fetcher}


@pytest.mark.parametrize(
    "first, then",
    [
        ("ingest.fetch.cfpb", "ingest.fetch.registry"),
        ("ingest.fetch.registry", "ingest.fetch.cfpb"),
    ],
)
def test_the_registry_and_the_fetcher_import_in_either_order_and_alone(first, then):
    code = (
        f"import sys, {first}, {then}\n"
        "from ingest.fetch.registry import FETCHERS\n"
        "from ingest.fetch.cfpb import make_cfpb_fetcher\n"
        "assert FETCHERS['cfpb'] is make_cfpb_fetcher\n"
        "for name in ('ingest.cli', 'ingest.sources', 'scipy', 'pyarrow', 'django'):\n"
        "    assert name not in sys.modules, name\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, timeout=120)


def test_the_fetchers_names_are_the_adapters():
    assert cfpb.SOURCE == adapter.SOURCE_SLUG
    assert cfpb.SOURCE_API_VERSION == adapter.SOURCE_API_VERSION


def test_a_context_for_another_source_is_refused(tmp_path):
    with pytest.raises(ContextMismatch, match="nyc311"):
        make_cfpb_fetcher(a_context(tmp_path / "acq", FakeCFPB(), source="nyc311"))


@pytest.mark.parametrize(
    "call",
    [("nyc311", START, END), ("cfpb", date(2024, 1, 29), END), ("cfpb", START, date(2024, 2, 1))],
)
def test_a_call_for_another_source_or_window_is_refused_before_anything(tmp_path, call):
    transport = FakeCFPB()
    context = a_context(tmp_path / "acq", transport)
    with pytest.raises(ContextMismatch):
        make_cfpb_fetcher(context)(*call)
    assert transport.sent == [] and not context.directory.exists()


def test_every_refusal_is_an_acquisition_error():
    for error in (
        ArchiveContentError, ArchiveDiscoveryError, ArchivePinMismatch, ContextMismatch,
        CountDisagreement, CountNotExact, DateMismatch, DuplicateComplaintId,
        FieldDisagreement, MissingApiRecord, MissingStartState, ReturnedIdMismatch,
        RowOutsideDay, UnusableApiRecord,
    ):  # fmt: skip
        assert issubclass(error, CFPBFetchError) and issubclass(error, AcquisitionError)


def test_the_fetcher_reaches_the_network_only_through_its_context():
    with pytest.raises(AssertionError, match="network"):
        socket.create_connection(("www.consumerfinance.gov", 443))
    tree = ast.parse(Path(cfpb.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    for module in imported:
        assert module.startswith("ingest.fetch.") or module.split(".")[0] in sys.stdlib_module_names
    assert not {"socket", "urllib", "http", "requests"} & {m.split(".")[0] for m in imported}
    assert "RequestsTransport" not in {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def test_no_narrative_is_held_for_the_whole_archive(tmp_path):
    """Narratives go to per-day spill files; the index keeps integers only."""
    context, _, _ = acquire(tmp_path)
    spill = sorted(p.name for p in (context.directory / "spill").glob("*.jsonl.gz"))
    assert spill == [f"{day}.jsonl.gz" for day in DAYS]

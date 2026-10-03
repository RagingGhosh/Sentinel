"""NYC 311 normalization: floating timestamps, elapsed resolution, purity.

The fixture is a **top-level JSON array**, which is what Socrata's SODA API
actually returns -- not an object with a `hits` envelope. That is the whole
reason `SourcePage` is a union, and this adapter exercises its sequence branch.

Addendum §2.4 governs the timestamps. `created_date` and `closed_date` are
Floating Timestamps carrying no offset; Phase 2 reads them as `America/New_York`
civil time and converts to UTC. Both DST edge cases are rejected rather than
resolved. These tests use fixed dates around the 2024 transitions so they do not
depend on the machine's clock or its local zone.
"""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from ingest.schema import CFPBOutcome, CorpusRecord, NYC311Outcome
from ingest.sources.base import SourceAdapter
from ingest.sources.nyc311 import (
    EXCLUSION_KINDS,
    SOURCE_API_VERSION,
    SOURCE_SLUG,
    SOURCE_TIMEZONE,
    AmbiguousLocalTime,
    Exclusion,
    MissingDescriptor,
    MissingField,
    NegativeResolutionTime,
    NonexistentLocalTime,
    NYC311Adapter,
    classify_row,
    normalize,
    rows_from_page,
    to_source_local,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "nyc311_page.json"
NY = ZoneInfo("America/New_York")


def page() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def rows() -> dict[str, dict]:
    return {r["unique_key"]: r for r in rows_from_page(page())}


def row(unique_key: str) -> dict:
    return rows()[unique_key]


# --- page shape --------------------------------------------------------------


def test_the_fixture_is_a_top_level_array_not_an_envelope():
    """Structural fidelity: Socrata returns a bare array. Wrapping it in a
    CFPB-style object would make the sequence branch of SourcePage untested."""
    parsed = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(parsed, list)
    assert all(isinstance(entry, dict) for entry in parsed)


def test_rows_are_yielded_in_published_order():
    assert [r["unique_key"] for r in rows_from_page(page())] == [
        "60000001",
        "60000002",
        "60000003",
        "60000004",
        "60000005",
        "60000006",
    ]


def test_a_sequence_page_needs_no_unwrapping_key():
    """The rows are the page; there is no nesting to descend."""
    extracted = list(rows_from_page([{"unique_key": "1"}]))
    assert extracted == [{"unique_key": "1"}]


def test_a_mapping_page_is_refused_by_this_adapter():
    with pytest.raises(TypeError, match="sequence"):
        list(rows_from_page({"hits": {"hits": []}}))


def test_the_fixture_is_small_and_carries_no_personal_data():
    parsed = page()
    assert len(parsed) <= 6
    for entry in parsed:
        assert entry["incident_zip"] == "1XXXX", "no real ZIP"
        assert "incident_address" not in entry
        assert "street_name" not in entry
        assert "latitude" not in entry and "longitude" not in entry


# --- the exact mapping -------------------------------------------------------


def test_a_fixture_row_normalizes_to_exact_expected_values():
    record, outcome = normalize(row("60000001"))

    assert record == CorpusRecord(
        source="nyc311",
        external_id="60000001",
        text="ENTIRE BUILDING",
        label="HEAT/HOT WATER",
        # 09:30 EST (UTC-5) -> 14:30 UTC
        submitted_at=datetime(2024, 1, 15, 14, 30, tzinfo=UTC),
    )
    assert outcome == NYC311Outcome(
        external_id="60000001",
        # 19:18 EST (UTC-5) -> 00:18 UTC the following day
        closed_at=datetime(2024, 1, 16, 0, 18, tzinfo=UTC),
        resolution_hours=9.8,
    )


def test_source_is_always_the_nyc311_slug():
    for key in ("60000001", "60000002", "60000003", "60000004"):
        record, _ = normalize(row(key))
        assert record.source == "nyc311" == SOURCE_SLUG


def test_unique_key_becomes_the_external_id_on_both_objects():
    record, outcome = normalize(row("60000003"))
    assert record.external_id == "60000003"
    assert outcome.external_id == record.external_id


def test_complaint_type_becomes_the_label_verbatim():
    assert normalize(row("60000001"))[0].label == "HEAT/HOT WATER"
    assert normalize(row("60000004"))[0].label == "Noise - Residential"


def test_descriptor_becomes_the_text_verbatim():
    assert normalize(row("60000003"))[0].text == "Recycling"


def test_descriptor_is_not_reshaped():
    """311 descriptors have a median of 15 characters and feed `text_length`."""
    original = "  Loud Music/Party  "
    record, _ = normalize(dict(row("60000004"), descriptor=original))
    assert record.text == original


def test_a_null_descriptor_raises():
    with pytest.raises(MissingDescriptor) as exc:
        normalize(row("60000006"))
    assert "60000006" in str(exc.value)


def test_an_absent_descriptor_key_raises():
    source = {k: v for k, v in row("60000001").items() if k != "descriptor"}
    with pytest.raises(MissingDescriptor):
        normalize(source)


def test_an_empty_descriptor_is_preserved_not_refused():
    """Task 6 specifies only that a *null* descriptor raises. Task 5's CFPB rule
    named "null or empty" for the narrative; this one does not, and the
    difference is the specification's, not an oversight to be smoothed over."""
    record, _ = normalize(dict(row("60000001"), descriptor=""))
    assert record.text == ""


@pytest.mark.parametrize("value", [None, 0, 1, True, 2.5, [], {}])
def test_a_non_string_complaint_type_raises(value):
    with pytest.raises(MissingField, match="complaint_type"):
        normalize(dict(row("60000001"), complaint_type=value))


@pytest.mark.parametrize("value", [None, 60000001, True, 1.5, [], {}])
def test_a_non_string_unique_key_raises(value):
    """`CorpusRecord.external_id` is typed `str`. Unlike CFPB -- where the source
    is known to publish the id both as text and as a number -- nothing
    establishes that for Socrata, so no numeric form is invented here."""
    with pytest.raises(MissingField, match="unique_key"):
        normalize(dict(row("60000001"), unique_key=value))


def test_an_absent_unique_key_raises():
    source = {k: v for k, v in row("60000001").items() if k != "unique_key"}
    with pytest.raises(MissingField, match="unique_key"):
        normalize(source)


# --- timestamps: addendum §2.4 -----------------------------------------------


def test_a_winter_timestamp_is_read_as_est_and_stored_as_utc():
    record, _ = normalize(row("60000001"))
    assert record.submitted_at == datetime(2024, 1, 15, 14, 30, tzinfo=UTC)
    assert record.submitted_at.utcoffset() == timedelta(0)


def test_a_summer_timestamp_is_read_as_edt_and_stored_as_utc():
    record, _ = normalize(row("60000002"))
    assert record.submitted_at == datetime(2024, 7, 15, 13, 30, tzinfo=UTC)


def test_the_two_seasons_use_different_offsets():
    """Proof the zone is applied, not a fixed offset: EST is -5, EDT is -4."""
    winter, _ = normalize(row("60000001"))
    summer, _ = normalize(row("60000002"))
    assert to_source_local(winter.submitted_at).utcoffset() == timedelta(hours=-5)
    assert to_source_local(summer.submitted_at).utcoffset() == timedelta(hours=-4)


def test_the_local_hour_survives_the_round_trip():
    """Addendum §2.4: `submitted_hour` must come from the local representation.
    Building that feature is a later task; what Task 6 owes it is a stored
    instant from which the original wall clock is exactly recoverable."""
    for key, expected_hour, expected_weekday in (
        ("60000001", 9, 0),  # Monday 15 Jan 2024, 09:30 EST
        ("60000002", 9, 0),  # Monday 15 Jul 2024, 09:30 EDT
        ("60000003", 8, 3),  # Thursday 4 Jul 2024, 08:00 EDT
    ):
        record, _ = normalize(row(key))
        local = to_source_local(record.submitted_at)
        assert local.hour == expected_hour
        assert local.weekday() == expected_weekday


def test_deriving_the_hour_from_utc_would_differ_from_the_local_hour():
    """The reason §2.4 insists on the local representation, made concrete."""
    record, _ = normalize(row("60000002"))
    assert record.submitted_at.hour == 13
    assert to_source_local(record.submitted_at).hour == 9


def test_an_ambiguous_autumn_fold_timestamp_is_rejected():
    """01:30 on 3 Nov 2024 happens twice in New York. Nothing in a floating
    timestamp says which, so neither is chosen."""
    with pytest.raises(AmbiguousLocalTime, match="created_date"):
        normalize(dict(row("60000001"), created_date="2024-11-03T01:30:00.000"))


def test_a_nonexistent_spring_gap_timestamp_is_rejected():
    """02:30 on 10 Mar 2024 never occurs in New York."""
    with pytest.raises(NonexistentLocalTime, match="created_date"):
        normalize(dict(row("60000001"), created_date="2024-03-10T02:30:00.000"))


def test_dst_rejection_applies_to_the_closed_date_too():
    with pytest.raises(AmbiguousLocalTime, match="closed_date"):
        normalize(dict(row("60000001"), closed_date="2024-11-03T01:30:00.000"))
    with pytest.raises(NonexistentLocalTime, match="closed_date"):
        normalize(dict(row("60000001"), closed_date="2024-03-10T02:30:00.000"))


@pytest.mark.parametrize(
    "value",
    [
        "2024-11-03T00:59:59.000",  # one second before the fold
        "2024-11-03T03:00:00.000",  # after it
        "2024-03-10T01:59:59.000",  # one second before the gap
        "2024-03-10T03:00:00.000",  # after it
    ],
)
def test_timestamps_either_side_of_a_transition_are_accepted(value):
    """The rejection is narrow: only the genuinely ambiguous or nonexistent
    instants, not the hours around them."""
    source = {k: v for k, v in row("60000001").items() if k != "closed_date"}
    record, outcome = normalize(dict(source, created_date=value))
    assert record.submitted_at.tzinfo is not None
    assert outcome.resolution_hours is None


def test_a_timestamp_carrying_an_offset_is_rejected():
    """Socrata publishes Floating Timestamps. An offset means the source changed
    shape, and §2.4's interpretation would no longer be the right one to apply."""
    with pytest.raises(MissingField, match="offset"):
        normalize(dict(row("60000001"), created_date="2024-01-15T09:30:00-05:00"))


def test_an_unparseable_timestamp_raises():
    with pytest.raises(MissingField, match="created_date"):
        normalize(dict(row("60000001"), created_date="the fifteenth of January"))


def test_an_absent_created_date_raises():
    source = {k: v for k, v in row("60000001").items() if k != "created_date"}
    with pytest.raises(MissingField, match="created_date"):
        normalize(source)


def test_normalization_does_not_depend_on_the_machine_timezone(monkeypatch):
    """A fixed input must give a fixed answer wherever the suite runs."""
    import time

    before, _ = normalize(row("60000001"))
    monkeypatch.setenv("TZ", "Asia/Kolkata")
    if hasattr(time, "tzset"):  # absent on Windows; the assertion still holds
        time.tzset()
    after, _ = normalize(row("60000001"))
    assert after == before
    assert after.submitted_at == datetime(2024, 1, 15, 14, 30, tzinfo=UTC)


def test_the_source_timezone_is_the_one_the_addendum_names():
    assert SOURCE_TIMEZONE == NY


# --- resolution hours --------------------------------------------------------


def test_resolution_hours_for_a_closed_request():
    _, outcome = normalize(row("60000001"))
    assert outcome.resolution_hours == pytest.approx(9.8, abs=1 / 3600)


def test_resolution_hours_is_exact_for_a_whole_day():
    _, outcome = normalize(row("60000003"))
    assert outcome.resolution_hours == 24.0
    assert outcome.closed_at == datetime(2024, 7, 5, 12, 0, tzinfo=UTC)


def test_resolution_hours_is_computed_from_instants_across_a_dst_boundary():
    """Opened 01:00 EST before the spring-forward, closed 04:00 EDT after it.
    The local clock shows three hours; only two actually elapsed."""
    source = dict(
        row("60000001"),
        created_date="2024-03-10T01:00:00.000",
        closed_date="2024-03-10T04:00:00.000",
    )
    _, outcome = normalize(source)
    assert outcome.resolution_hours == 2.0, "wall-clock subtraction would say 3.0"


def test_an_open_request_yields_none_for_both_outcome_fields():
    """Absent key. Not zero -- an open request has no resolution time."""
    assert "closed_date" not in row("60000002")
    _, outcome = normalize(row("60000002"))
    assert outcome.closed_at is None
    assert outcome.resolution_hours is None


def test_a_null_closed_date_yields_none_for_both_outcome_fields():
    assert row("60000004")["closed_date"] is None
    _, outcome = normalize(row("60000004"))
    assert outcome.closed_at is None
    assert outcome.resolution_hours is None


def test_an_open_request_is_never_zero_hours():
    for key in ("60000002", "60000004"):
        _, outcome = normalize(row(key))
        assert outcome.resolution_hours is not None or outcome.resolution_hours != 0
        assert outcome.resolution_hours is None


def test_a_zero_duration_is_allowed():
    """Closed in the same second is odd but not impossible, and not negative."""
    source = dict(
        row("60000001"),
        created_date="2024-01-15T09:30:00.000",
        closed_date="2024-01-15T09:30:00.000",
    )
    _, outcome = normalize(source)
    assert outcome.resolution_hours == 0.0


def test_a_negative_duration_raises():
    with pytest.raises(NegativeResolutionTime) as exc:
        normalize(row("60000005"))
    assert "60000005" in str(exc.value)


def test_a_negative_duration_is_not_clamped_absolute_or_nulled():
    """The three silent repairs the plan forbids, each named."""
    with pytest.raises(NegativeResolutionTime):
        normalize(row("60000005"))

    source = dict(
        row("60000001"),
        created_date="2024-01-15T12:00:00.000",
        closed_date="2024-01-15T11:00:00.000",
    )
    with pytest.raises(NegativeResolutionTime) as exc:
        normalize(source)
    message = str(exc.value)
    assert "-1.0" in message or "-1" in message, f"the error should state it: {message}"


# --- outcome separation from CFPB --------------------------------------------


def test_nyc311_outcome_has_no_cfpb_fields():
    import dataclasses

    names = {f.name for f in dataclasses.fields(NYC311Outcome)}
    assert names == {"external_id", "closed_at", "resolution_hours"}
    assert not names & {"timely_response", "sent_to_company_at", "timely", "sla_met"}


def test_cfpb_outcome_is_unchanged_by_this_adapter():
    import dataclasses

    names = {f.name for f in dataclasses.fields(CFPBOutcome)}
    assert names == {"external_id", "timely_response", "sent_to_company_at"}
    assert not names & {"closed_at", "resolution_hours", "sla_met"}


def test_the_adapter_module_never_mentions_cfpb_outcome_semantics():
    source = Path("ingest/sources/nyc311.py").read_text(encoding="utf-8")
    for forbidden in ("sla_met", "timely_response", "sent_to_company_at", "CFPBOutcome"):
        assert forbidden not in source, f"{forbidden} must not appear in the 311 adapter"


def test_the_outcome_is_frozen():
    import dataclasses

    _, outcome = normalize(row("60000001"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.resolution_hours = 0.0  # type: ignore[misc]


# --- what this adapter deliberately does not do ------------------------------


def test_the_adapter_does_not_validate_a_complaint_type_roster():
    """311 has 276 distinct complaint types and no locked roster in the spec."""
    record, _ = normalize(dict(row("60000001"), complaint_type="A Type That Does Not Exist"))
    assert record.label == "A Type That Does Not Exist"


def test_the_adapter_never_remaps_a_label_to_other():
    for unexpected in ("Other", "", "   ", "UNKNOWN"):
        record, _ = normalize(dict(row("60000001"), complaint_type=unexpected))
        assert record.label == unexpected


def test_no_module_level_label_roster_exists():
    import ingest.sources.nyc311 as adapter

    for name in dir(adapter):
        value = getattr(adapter, name)
        if isinstance(value, (frozenset, set)) and value:
            pytest.fail(f"{name} looks like a hardcoded roster")


def test_the_adapter_does_not_filter_by_window():
    record, _ = normalize(dict(row("60000001"), created_date="2019-06-01T00:00:00.000"))
    assert record.submitted_at.year == 2019


# --- purity and boundaries ---------------------------------------------------


def test_normalize_is_deterministic():
    assert normalize(row("60000001")) == normalize(row("60000001"))


def test_normalize_does_not_mutate_its_input():
    source = row("60000001")
    before = json.dumps(source, sort_keys=True)
    normalize(source)
    assert json.dumps(source, sort_keys=True) == before


def test_normalize_touches_no_network_and_no_filesystem(monkeypatch):
    import socket

    def boom(*args, **kwargs):
        raise AssertionError("normalize must be pure")

    source = row("60000001")
    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(Path, "open", boom)

    record, outcome = normalize(source)
    assert record.external_id == "60000001"
    assert outcome.resolution_hours == pytest.approx(9.8, abs=1 / 3600)


def test_normalize_reads_no_clock():
    import ast

    tree = ast.parse(Path("ingest/sources/nyc311.py").read_text(encoding="utf-8"))
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not attributes & {"now", "today", "utcnow"}, "normalize must not read the clock"


def test_the_adapter_module_imports_no_django_no_ml_and_no_training_deps():
    import ast

    tree = ast.parse(Path("ingest/sources/nyc311.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    for module in imported:
        root = module.split(".")[0]
        assert root != "django", f"ingest must not import Django ({module})"
        assert root not in {"pandas", "pyarrow"}, f"Task 6 needs no training tier ({module})"
        assert not module.startswith("ml."), f"ingest must not import ml ({module})"


def test_the_adapter_satisfies_the_source_adapter_protocol_at_runtime():
    adapter: SourceAdapter[NYC311Outcome] = NYC311Adapter()
    assert adapter.source_slug == "nyc311"
    assert adapter.source_api_version == SOURCE_API_VERSION
    assert isinstance(adapter.source_api_version, str) and adapter.source_api_version

    extracted = list(adapter.rows_from_page(page()))
    assert extracted[0]["unique_key"] == "60000001"

    record, outcome = adapter.normalize(row("60000001"))
    assert isinstance(record, CorpusRecord) and isinstance(outcome, NYC311Outcome)


def test_normalizing_the_whole_fixture_yields_the_expected_split():
    ok, refused = [], []
    for source in rows_from_page(page()):
        try:
            ok.append(normalize(source))
        except (
            MissingField,
            MissingDescriptor,
            NegativeResolutionTime,
            AmbiguousLocalTime,
            NonexistentLocalTime,
        ) as exc:
            refused.append((source["unique_key"], type(exc).__name__))

    assert [r.external_id for r, _ in ok] == ["60000001", "60000002", "60000003", "60000004"]
    assert refused == [
        ("60000005", "NegativeResolutionTime"),
        ("60000006", "MissingDescriptor"),
    ]


# --- D50: classify_row, the typed exclusion of D45's three classes (T1-T10) -----------
#
# `normalize` is unchanged and still raises for every row below; `classify_row` is the
# seam the command line uses (addendum D50 (3)). Stage one applies every check outside
# the D45 classes and raises exactly what `normalize` raises; stage two returns one
# `Exclusion`, the first D45 condition that applies, in D50 (2)'s order.

ABSENT = object()
"""Marks a key to remove from a row, as opposed to setting it to null."""


def variant(key: str = "60000001", **changes) -> dict:
    """A fixture row with fields replaced, or removed when given `ABSENT`."""
    source = dict(row(key))
    for field, value in changes.items():
        if value is ABSENT:
            source.pop(field, None)
        else:
            source[field] = value
    return source


def excluded(source: dict, kind: str, civil: date) -> Exclusion:
    return Exclusion(external_id=source["unique_key"], kind=kind, created_civil_date=civil)


def test_the_six_exclusion_kinds_in_precedence_order():
    assert EXCLUSION_KINDS == (
        "MissingDescriptor",
        "AmbiguousLocalTime:created_date",
        "NonexistentLocalTime:created_date",
        "AmbiguousLocalTime:closed_date",
        "NonexistentLocalTime:closed_date",
        "NegativeResolutionTime",
    )


def test_an_exclusion_is_frozen():
    import dataclasses

    exclusion = Exclusion(
        external_id="1", kind="MissingDescriptor", created_civil_date=date(2024, 1, 15)
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        exclusion.kind = "NegativeResolutionTime"  # type: ignore[misc]


# T1: every row normalize accepts classifies to the identical pair.
ACCEPTED = {
    "a closed request": lambda: row("60000001"),
    "an open request without closed_date": lambda: row("60000002"),
    "a closed request in summer": lambda: row("60000003"),
    "an open request with a null closed_date": lambda: row("60000004"),
    "an empty descriptor": lambda: variant(descriptor=""),
    "a whitespace-only descriptor": lambda: variant(descriptor="   "),
    "a zero duration": lambda: variant(
        created_date="2024-01-15T09:30:00.000", closed_date="2024-01-15T09:30:00.000"
    ),
    "an interval spanning the autumn transition": lambda: variant(
        created_date="2024-11-02T12:00:00.000", closed_date="2024-11-03T12:00:00.000"
    ),
    "an interval spanning the spring transition": lambda: variant(
        created_date="2024-03-09T12:00:00.000", closed_date="2024-03-10T12:00:00.000"
    ),
}


@pytest.mark.parametrize("case", sorted(ACCEPTED))
def test_t1_an_accepted_row_classifies_to_exactly_what_normalize_returns(case):
    source = ACCEPTED[case]()
    assert classify_row(source) == normalize(source)


# T2: the whole MissingDescriptor class, and nothing beyond it.
@pytest.mark.parametrize(
    "descriptor",
    [None, ABSENT, 42, 1.5, True, ["Loud Music"], {"text": "Loud Music"}],
    ids=["null", "absent", "int", "float", "bool", "list", "object"],
)
def test_t2_a_null_absent_or_non_text_descriptor_is_missing_descriptor(descriptor):
    source = variant(descriptor=descriptor)
    assert classify_row(source) == excluded(source, "MissingDescriptor", date(2024, 1, 15))


def test_t2_the_fixture_s_null_descriptor_row_is_missing_descriptor():
    source = row("60000006")
    assert classify_row(source) == excluded(source, "MissingDescriptor", date(2024, 5, 20))


@pytest.mark.parametrize("descriptor", ["", " ", "   ", "\t\n"])
def test_t2_an_empty_or_whitespace_only_descriptor_is_kept_verbatim(descriptor):
    record, _ = classify_row(variant(descriptor=descriptor))
    assert record.text == descriptor


# T3: created_date edges, at and around their boundaries.
CREATED_EDGES = [
    (f"{day}T{clock}", kind)
    for day, kind in (
        ("2024-11-03", "AmbiguousLocalTime:created_date"),
        ("2025-11-02", "AmbiguousLocalTime:created_date"),
    )
    for clock in ("01:00:00.000", "01:30:00.000", "01:59:59.999")
] + [
    (f"{day}T{clock}", kind)
    for day, kind in (
        ("2024-03-10", "NonexistentLocalTime:created_date"),
        ("2025-03-09", "NonexistentLocalTime:created_date"),
    )
    for clock in ("02:00:00.000", "02:30:00.000", "02:59:59.999")
]
EDGE_NEIGHBOURS = [
    f"{day}T{clock}"
    for day in ("2024-11-03", "2025-11-02")
    for clock in ("00:59:59.999", "02:00:00.000")
] + [
    f"{day}T{clock}"
    for day in ("2024-03-10", "2025-03-09")
    for clock in ("01:59:59.999", "03:00:00.000")
]


@pytest.mark.parametrize("value, kind", CREATED_EDGES)
def test_t3_a_created_date_edge_is_its_own_kind(value, kind):
    source = variant(created_date=value, closed_date=ABSENT)
    assert classify_row(source) == excluded(source, kind, date.fromisoformat(value[:10]))


@pytest.mark.parametrize("value", EDGE_NEIGHBOURS)
def test_t3_a_created_date_beside_an_edge_is_kept(value):
    source = variant(created_date=value, closed_date=ABSENT)
    assert classify_row(source) == normalize(source)


# T4: closed_date edges, counted under created_date's civil date.
CLOSED_EDGES = (
    [
        ("2024-11-02T12:00:00.000", f"2024-11-03T{clock}", "AmbiguousLocalTime:closed_date")
        for clock in ("01:00:00.000", "01:30:00.000", "01:59:59.999")
    ]
    + [
        ("2025-11-01T12:00:00.000", "2025-11-02T01:15:00.000", "AmbiguousLocalTime:closed_date"),
    ]
    + [
        ("2024-03-09T12:00:00.000", f"2024-03-10T{clock}", "NonexistentLocalTime:closed_date")
        for clock in ("02:00:00.000", "02:30:00.000", "02:59:59.999")
    ]
    + [
        ("2025-03-08T12:00:00.000", "2025-03-09T02:30:00.000", "NonexistentLocalTime:closed_date"),
        ("2025-12-31T12:00:00.000", "2026-03-08T02:30:00.000", "NonexistentLocalTime:closed_date"),
    ]
)


@pytest.mark.parametrize("created, closed, kind", CLOSED_EDGES)
def test_t4_a_closed_date_edge_counts_under_created_date(created, closed, kind):
    source = variant(created_date=created, closed_date=closed)
    assert classify_row(source) == excluded(source, kind, date.fromisoformat(created[:10]))


@pytest.mark.parametrize("value", EDGE_NEIGHBOURS)
def test_t4_a_closed_date_beside_an_edge_is_kept(value):
    source = variant(created_date="2024-03-01T00:00:00.000", closed_date=value)
    assert classify_row(source) == normalize(source)


# T5: a negative duration, measured between instants.
def test_t5_a_close_one_second_before_the_open_is_negative_resolution_time():
    source = variant(created_date="2024-02-01T12:00:00.000", closed_date="2024-02-01T11:59:59.000")
    assert classify_row(source) == excluded(source, "NegativeResolutionTime", date(2024, 2, 1))


def test_t5_the_fixture_s_negative_row_is_negative_resolution_time():
    source = row("60000005")
    assert classify_row(source) == excluded(source, "NegativeResolutionTime", date(2024, 2, 1))


def test_t5_a_zero_duration_is_kept():
    _, outcome = classify_row(
        variant(created_date="2024-01-15T09:30:00.000", closed_date="2024-01-15T09:30:00.000")
    )
    assert outcome.resolution_hours == 0.0


def test_t5_the_duration_is_measured_between_instants():
    """00:50 EDT to 02:00 EST is 2h10m between instants, 1h10m by the wall clock."""
    _, outcome = classify_row(
        variant(created_date="2024-11-03T00:50:00.000", closed_date="2024-11-03T02:00:00.000")
    )
    assert outcome.resolution_hours == pytest.approx(2 + 10 / 60)


# T6: one row with several D45 conditions takes exactly one kind, the first that applies.
OVERLAPS = {
    "null descriptor and a created fold": (
        dict(descriptor=None, created_date="2024-11-03T01:30:00.000", closed_date=ABSENT),
        "MissingDescriptor",
    ),
    "null descriptor and a closed gap": (
        dict(descriptor=None, closed_date="2024-03-10T02:30:00.000"),
        "MissingDescriptor",
    ),
    "null descriptor and a negative duration": (
        dict(descriptor=None, closed_date="2024-01-15T09:00:00.000"),
        "MissingDescriptor",
    ),
    "a created fold and a closed gap": (
        dict(created_date="2024-11-03T01:30:00.000", closed_date="2025-03-09T02:30:00.000"),
        "AmbiguousLocalTime:created_date",
    ),
    "a created gap and a closed fold": (
        dict(created_date="2024-03-10T02:30:00.000", closed_date="2024-11-03T01:30:00.000"),
        "NonexistentLocalTime:created_date",
    ),
    "a created fold whose wall clock follows the close": (
        dict(created_date="2024-11-03T01:30:00.000", closed_date="2024-11-03T00:30:00.000"),
        "AmbiguousLocalTime:created_date",
    ),
    "a created gap whose wall clock follows the close": (
        dict(created_date="2024-03-10T02:30:00.000", closed_date="2024-03-10T01:00:00.000"),
        "NonexistentLocalTime:created_date",
    ),
    "a closed fold whose wall clock precedes the open": (
        dict(created_date="2024-11-03T02:30:00.000", closed_date="2024-11-03T01:30:00.000"),
        "AmbiguousLocalTime:closed_date",
    ),
    "a closed gap whose wall clock precedes the open": (
        dict(created_date="2024-03-10T03:30:00.000", closed_date="2024-03-10T02:30:00.000"),
        "NonexistentLocalTime:closed_date",
    ),
}


@pytest.mark.parametrize("case", sorted(OVERLAPS))
def test_t6_overlapping_conditions_take_the_first_applicable_kind(case):
    changes, kind = OVERLAPS[case]
    source = variant(**changes)
    civil = date.fromisoformat(source["created_date"][:10])
    assert classify_row(source) == excluded(source, kind, civil)


# T7: every check outside the D45 classes still refuses, whatever D45 condition is present.
NON_D45 = {
    "unique_key absent": dict(unique_key=ABSENT),
    "unique_key blank": dict(unique_key="   "),
    "unique_key not text": dict(unique_key=60000001),
    "complaint_type absent": dict(complaint_type=ABSENT),
    "complaint_type not text": dict(complaint_type=7),
    "created_date absent": dict(created_date=ABSENT),
    "created_date blank": dict(created_date="  "),
    "created_date not text": dict(created_date=20240115),
    "created_date not ISO-8601": dict(created_date="the fifteenth of January"),
    "created_date with an offset": dict(created_date="2024-01-15T09:30:00-05:00"),
    "closed_date blank": dict(closed_date=""),
    "closed_date not text": dict(closed_date=5),
    "closed_date not ISO-8601": dict(closed_date="soon"),
    "closed_date with an offset": dict(closed_date="2024-01-15T19:18:00+00:00"),
}
D45_CONDITIONS = {
    "MissingDescriptor": dict(descriptor=None),
    "AmbiguousLocalTime:created_date": dict(created_date="2024-11-03T01:30:00.000"),
    "NonexistentLocalTime:created_date": dict(created_date="2024-03-10T02:30:00.000"),
    "AmbiguousLocalTime:closed_date": dict(closed_date="2024-11-03T01:30:00.000"),
    "NonexistentLocalTime:closed_date": dict(closed_date="2024-03-10T02:30:00.000"),
    "NegativeResolutionTime": dict(closed_date="2024-01-15T09:00:00.000"),
}
NON_D45_WITH_D45 = [
    (problem, condition)
    for problem in sorted(NON_D45)
    for condition in D45_CONDITIONS
    if not set(NON_D45[problem]) & set(D45_CONDITIONS[condition])
]


def failure(function, source) -> tuple[type, str]:
    try:
        function(source)
    except Exception as exc:  # the comparison is the point: type and message
        return type(exc), str(exc)
    raise AssertionError(f"{function.__name__} accepted {source!r}")


@pytest.mark.parametrize("problem, condition", NON_D45_WITH_D45)
def test_t7_a_non_d45_problem_refuses_as_it_does_alone(problem, condition):
    alone = failure(normalize, variant(**NON_D45[problem]))
    assert alone[0] is MissingField
    assert failure(classify_row, variant(**NON_D45[problem], **D45_CONDITIONS[condition])) == alone


@pytest.mark.parametrize("problem", sorted(NON_D45))
def test_t7_a_non_d45_problem_alone_refuses_exactly_as_normalize_does(problem):
    source = variant(**NON_D45[problem])
    assert failure(classify_row, source) == failure(normalize, source)


def test_t7_the_cross_product_covers_every_kind_and_every_problem():
    assert {condition for _, condition in NON_D45_WITH_D45} == set(EXCLUSION_KINDS)
    assert {problem for problem, _ in NON_D45_WITH_D45} == set(NON_D45)


@pytest.mark.parametrize("source", [[], "a row", 5, None], ids=["list", "str", "int", "none"])
def test_t7_a_row_that_is_not_a_mapping_fails_as_it_does_under_normalize(source):
    assert failure(classify_row, source) == failure(normalize, source)


# T8: the year is created_date's New York civil year, never a UTC instant's.
def test_t8_a_row_created_late_on_new_year_s_eve_counts_under_its_civil_year():
    source = variant("60000006", created_date="2024-12-31T23:30:00.000", closed_date=ABSENT)
    instant = datetime(2024, 12, 31, 23, 30, tzinfo=NY).astimezone(UTC)
    assert instant.year == 2025, "the UTC instant falls in the next year"
    exclusion = classify_row(source)
    assert exclusion == excluded(source, "MissingDescriptor", date(2024, 12, 31))
    assert exclusion.created_civil_date.year == 2024


def test_t8_an_edge_row_takes_its_literal_s_civil_date_though_it_has_no_instant():
    source = variant(created_date="2024-03-10T02:30:00.000", closed_date=ABSENT)
    assert classify_row(source).created_civil_date == date(2024, 3, 10)


# T9: normalize still raises the original typed error for every D45 row above.
ERROR_OF_KIND = {
    "MissingDescriptor": MissingDescriptor,
    "AmbiguousLocalTime:created_date": AmbiguousLocalTime,
    "NonexistentLocalTime:created_date": NonexistentLocalTime,
    "AmbiguousLocalTime:closed_date": AmbiguousLocalTime,
    "NonexistentLocalTime:closed_date": NonexistentLocalTime,
    "NegativeResolutionTime": NegativeResolutionTime,
}
D45_ROWS = (
    [(f"descriptor {i}", lambda d=d: variant(descriptor=d), "MissingDescriptor")
     for i, d in enumerate([None, ABSENT, 42])]
    + [(f"created {v}", lambda v=v: variant(created_date=v, closed_date=ABSENT), k)
       for v, k in CREATED_EDGES]
    + [(f"closed {c}", lambda o=o, c=c: variant(created_date=o, closed_date=c), k)
       for o, c, k in CLOSED_EDGES]
    + [("negative", lambda: row("60000005"), "NegativeResolutionTime")]
    + [(f"overlap {case}", lambda case=case: variant(**OVERLAPS[case][0]), OVERLAPS[case][1])
       for case in sorted(OVERLAPS)]
)  # fmt: skip


@pytest.mark.parametrize("case, make, kind", D45_ROWS, ids=[case for case, _, _ in D45_ROWS])
def test_t9_normalize_still_raises_the_d45_error_classify_row_names(case, make, kind):
    source = make()
    assert classify_row(source).kind == kind
    with pytest.raises(ERROR_OF_KIND[kind]):
        normalize(source)


# T10: pure, deterministic, window-blind, and importing nothing new.
def test_t10_classify_row_is_deterministic():
    for source in (row("60000001"), row("60000005"), row("60000006")):
        assert classify_row(source) == classify_row(source)


def test_t10_classify_row_does_not_mutate_its_input():
    for key in ("60000001", "60000005", "60000006"):
        source = row(key)
        before = json.dumps(source, sort_keys=True)
        classify_row(source)
        assert json.dumps(source, sort_keys=True) == before


def test_t10_classify_row_touches_no_network_and_no_filesystem(monkeypatch):
    import socket

    def boom(*args, **kwargs):
        raise AssertionError("classify_row must be pure")

    sources = [row("60000001"), row("60000005"), row("60000006")]
    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(Path, "open", boom)
    results = [classify_row(source) for source in sources]
    assert isinstance(results[0], tuple)
    assert [r.kind for r in results[1:]] == ["NegativeResolutionTime", "MissingDescriptor"]


def test_t10_classify_row_does_not_filter_by_window():
    record, _ = classify_row(variant(created_date="2019-06-01T00:00:00.000", closed_date=ABSENT))
    assert record.submitted_at.year == 2019
    source = variant("60000006", created_date="2019-06-01T00:00:00.000", closed_date=ABSENT)
    assert classify_row(source).created_civil_date == date(2019, 6, 1)


def test_t10_classify_row_imports_nothing_and_the_module_only_the_standard_library():
    import ast
    import sys

    tree = ast.parse(Path("ingest/sources/nyc311.py").read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "classify_row"
    )
    assert not [n for n in ast.walk(function) if isinstance(n, ast.Import | ast.ImportFrom)]
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    for module in imported:
        assert module in {"ingest.schema", "ingest.sources.base"} or (
            module.split(".")[0] in sys.stdlib_module_names
        ), module

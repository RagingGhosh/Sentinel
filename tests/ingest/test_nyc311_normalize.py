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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from ingest.schema import CFPBOutcome, CorpusRecord, NYC311Outcome
from ingest.sources.base import SourceAdapter
from ingest.sources.nyc311 import (
    SOURCE_API_VERSION,
    SOURCE_SLUG,
    SOURCE_TIMEZONE,
    AmbiguousLocalTime,
    MissingDescriptor,
    MissingField,
    NegativeResolutionTime,
    NonexistentLocalTime,
    NYC311Adapter,
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

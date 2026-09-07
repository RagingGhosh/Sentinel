"""CFPB normalization: field mapping, typed failures, and purity.

Scope note. This adapter maps one source row to one `CorpusRecord` plus one
`CFPBOutcome`, and nothing else. It does **not** validate the label roster
(plan Task 7, `ingest/roster.py`, which derives the roster from data and whose
acceptance forbids any spec label appearing as a literal) and it does **not**
enforce the 2024-2025 window (plan Task 8, the CLI's `--start`/`--end`). Tests
below assert those responsibilities stay out of here, so a later task cannot
quietly duplicate them.
"""

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from ingest.schema import CFPBOutcome, CorpusRecord
from ingest.sources.base import SourceAdapter
from ingest.sources.cfpb import (
    SOURCE_API_VERSION,
    SOURCE_SLUG,
    CFPBAdapter,
    InvalidTimelyValue,
    MissingField,
    MissingNarrative,
    NaiveTimestamp,
    normalize,
    rows_from_page,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "cfpb_page.json"


def page() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def rows() -> dict[str, dict]:
    return {r["complaint_id"]: r for r in rows_from_page(page())}


def row(complaint_id: str) -> dict:
    return rows()[complaint_id]


# --- fixture hygiene ---------------------------------------------------------


def test_fixture_is_small_and_has_no_personal_data():
    parsed = page()
    assert len(parsed["hits"]["hits"]) <= 5, "plan caps the fixture at 5 rows"
    blob = FIXTURE.read_text(encoding="utf-8")
    # Real CFPB narratives are redacted with XXXX runs and the API returns real
    # company names. Neither belongs in a committed fixture.
    assert "XXXX" not in blob
    for entry in parsed["hits"]["hits"]:
        assert entry["_source"]["company"].startswith("EXAMPLE ")
        assert entry["_source"]["zip_code"].endswith("XX"), "no full ZIP"


def test_page_unwrapping_yields_every_source_row():
    extracted = list(rows_from_page(page()))
    assert len(extracted) == 5
    assert [r["complaint_id"] for r in extracted] == [
        "9000001",
        "9000002",
        "9000003",
        "9000004",
        "9000005",
    ]


# --- the exact mapping -------------------------------------------------------


def test_a_fixture_row_normalizes_to_exact_expected_values():
    record, outcome = normalize(row("9000001"))

    assert record == CorpusRecord(
        source="cfpb",
        external_id="9000001",
        text=(
            "An account I do not recognise appears on my report and the dispute "
            "was closed without explanation."
        ),
        label="Credit reporting or other personal consumer reports",
        submitted_at=datetime(2024, 3, 15, 5, 0, 0, tzinfo=UTC),
    )
    assert outcome == CFPBOutcome(
        external_id="9000001",
        timely_response=True,
        sent_to_company_at=datetime(2024, 3, 18, 5, 0, 0, tzinfo=UTC),
    )


def test_source_is_always_the_cfpb_slug():
    for identifier in ("9000001", "9000002", "9000003"):
        record, _ = normalize(row(identifier))
        assert record.source == "cfpb" == SOURCE_SLUG


def test_external_id_is_preserved_exactly_and_shared_by_both_objects():
    record, outcome = normalize(row("9000002"))
    assert record.external_id == "9000002"
    assert outcome.external_id == record.external_id


def test_an_integer_complaint_id_becomes_its_exact_string():
    source = dict(row("9000001"), complaint_id=9000001)
    record, _ = normalize(source)
    assert record.external_id == "9000001", "no padding, no reformatting"


def test_a_float_complaint_id_is_rejected_rather_than_coerced():
    """A float id has already lost precision; stringifying it invents an answer."""
    with pytest.raises(MissingField, match="complaint_id"):
        normalize(dict(row("9000001"), complaint_id=9000001.0))


def test_product_becomes_the_label_verbatim():
    assert normalize(row("9000002"))[0].label == "Debt collection"
    assert normalize(row("9000003"))[0].label == "Checking or savings account"


# --- text --------------------------------------------------------------------


def test_narrative_is_preserved_verbatim_without_cleaning():
    original = "  Leading and trailing spaces, a\ttab, and a\nnewline stay put.  "
    record, _ = normalize(dict(row("9000001"), complaint_what_happened=original))
    assert record.text == original, "text_length is a measured feature; do not reshape it"


def test_a_null_narrative_raises_missing_narrative():
    with pytest.raises(MissingNarrative) as exc:
        normalize(row("9000004"))
    assert "9000004" in str(exc.value), "the error must name the record it refused"


def test_an_empty_narrative_raises_missing_narrative():
    with pytest.raises(MissingNarrative):
        normalize(dict(row("9000001"), complaint_what_happened=""))


def test_a_whitespace_only_narrative_raises_missing_narrative():
    with pytest.raises(MissingNarrative):
        normalize(dict(row("9000001"), complaint_what_happened="   \n\t "))


def test_an_absent_narrative_key_raises_missing_narrative():
    source = {k: v for k, v in row("9000001").items() if k != "complaint_what_happened"}
    with pytest.raises(MissingNarrative):
        normalize(source)


def test_a_non_string_narrative_never_becomes_the_text_nan():
    """float('nan') stringifies to 'nan'. That must never reach the corpus."""
    with pytest.raises(MissingNarrative):
        normalize(dict(row("9000001"), complaint_what_happened=float("nan")))


# --- timestamps --------------------------------------------------------------


def test_submitted_at_is_converted_to_utc():
    record, _ = normalize(row("9000003"))
    assert record.submitted_at == datetime(2024, 7, 22, 4, 0, 0, tzinfo=UTC)
    assert record.submitted_at.utcoffset() == timedelta(0)


def test_an_offset_timestamp_keeps_its_instant():
    source = dict(row("9000001"), date_received="2024-03-15T09:30:00+05:30")
    record, _ = normalize(source)
    assert record.submitted_at == datetime(2024, 3, 15, 4, 0, 0, tzinfo=UTC)


def test_a_naive_date_received_is_rejected():
    with pytest.raises(NaiveTimestamp, match="date_received"):
        normalize(row("9000005"))


def test_a_naive_date_sent_to_company_is_rejected():
    source = dict(row("9000001"), date_sent_to_company="2024-03-18T00:00:00")
    with pytest.raises(NaiveTimestamp, match="date_sent_to_company"):
        normalize(source)


def test_an_unparseable_timestamp_raises():
    with pytest.raises(MissingField, match="date_received"):
        normalize(dict(row("9000001"), date_received="the fifteenth of March"))


def test_an_absent_date_received_raises():
    source = {k: v for k, v in row("9000001").items() if k != "date_received"}
    with pytest.raises(MissingField, match="date_received"):
        normalize(source)


def test_a_utc_designator_is_accepted():
    source = dict(row("9000001"), date_received="2024-03-15T12:00:00Z")
    record, _ = normalize(source)
    assert record.submitted_at == datetime(2024, 3, 15, 12, 0, 0, tzinfo=UTC)


def test_a_non_utc_offset_is_not_merely_relabelled():
    """Guards the classic bug: attaching UTC to a local wall-clock time."""
    source = dict(row("9000001"), date_received="2024-03-15T00:00:00-05:00")
    record, _ = normalize(source)
    naive_relabelled = datetime(2024, 3, 15, 0, 0, 0, tzinfo=UTC)
    assert record.submitted_at != naive_relabelled
    assert record.submitted_at == naive_relabelled + timedelta(hours=5)


def test_a_zero_offset_that_is_not_named_utc_still_normalizes():
    source = dict(row("9000001"), date_received="2024-03-15T05:00:00+00:00")
    record, _ = normalize(source)
    assert record.submitted_at == datetime(2024, 3, 15, 5, tzinfo=timezone(timedelta(0)))


# --- outcome -----------------------------------------------------------------


def test_timely_yes_maps_to_true_and_no_maps_to_false():
    assert normalize(row("9000001"))[1].timely_response is True
    assert normalize(row("9000002"))[1].timely_response is False


@pytest.mark.parametrize("value", ["yes", "YES", "true", "Y", "", None, True, 1])
def test_any_other_timely_value_raises(value):
    """Case matters: 'yes' is not 'Yes', and guessing would silently relabel."""
    with pytest.raises(InvalidTimelyValue):
        normalize(dict(row("9000001"), timely=value))


def test_an_absent_timely_key_raises():
    source = {k: v for k, v in row("9000001").items() if k != "timely"}
    with pytest.raises(InvalidTimelyValue):
        normalize(source)


def test_sent_to_company_at_is_parsed_when_present():
    _, outcome = normalize(row("9000001"))
    assert outcome.sent_to_company_at == datetime(2024, 3, 18, 5, 0, 0, tzinfo=UTC)


def test_sent_to_company_at_is_none_when_the_key_is_absent():
    assert "date_sent_to_company" not in row("9000002")
    _, outcome = normalize(row("9000002"))
    assert outcome.sent_to_company_at is None


def test_sent_to_company_at_is_none_when_the_value_is_null():
    assert row("9000003")["date_sent_to_company"] is None
    _, outcome = normalize(row("9000003"))
    assert outcome.sent_to_company_at is None


def test_outcome_carries_no_nyc311_or_sla_fields():
    """CFPB timely-response is not an SLA duration and must not acquire one."""
    import dataclasses

    names = {f.name for f in dataclasses.fields(CFPBOutcome)}
    assert not names & {"sla_met", "resolution_hours", "closed_at", "breached"}
    _, outcome = normalize(row("9000001"))
    assert not hasattr(outcome, "sla_met")


def test_outcome_is_frozen():
    import dataclasses

    _, outcome = normalize(row("9000001"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.timely_response = False  # type: ignore[misc]


# --- what this adapter deliberately does not do ------------------------------


def test_the_adapter_does_not_validate_the_label_roster():
    """Roster membership is plan Task 7 (ingest/roster.py), asserted over the
    whole window before any record is processed. Enforcing it per-record here
    would both duplicate that check and hardcode a vocabulary the addendum says
    must be derived from data."""
    record, _ = normalize(dict(row("9000001"), product="A Product That Does Not Exist"))
    assert record.label == "A Product That Does Not Exist", "passed through untouched"


def test_the_adapter_never_remaps_a_label_to_other():
    for unexpected in ("Prepaid card", "Payday loan", "Other"):
        record, _ = normalize(dict(row("9000001"), product=unexpected))
        assert record.label == unexpected


@pytest.mark.parametrize("value", [None, 0, 1, True, False, 3.5, [], {}, ("Mortgage",)])
def test_a_non_string_product_raises_rather_than_being_stringified(value):
    """`CorpusRecord.label` is typed `str`; there is no record to build. This is
    a schema constraint from Task 3, not a decision about the value's meaning."""
    with pytest.raises(MissingField, match="product"):
        normalize(dict(row("9000001"), product=value))


@pytest.mark.parametrize("value", ["", " ", "   ", "\t", "\n", "  \t\n "])
def test_an_empty_or_whitespace_product_is_preserved_exactly(value):
    """Addendum §1.1 gives every product-value decision to the roster gate --
    "Fail. Report the unexpected label and its record count." An empty product
    is a value outside the locked roster, so Task 7 must be able to see it and
    count it. Refusing it here would delete that record before it could be
    reported, and the plan's Task 5 specifies no product rejection at all."""
    record, _ = normalize(dict(row("9000001"), product=value))
    assert record.label == value
    assert record.label is not None
    assert record.label != "Other", "no remapping"


def test_the_adapter_does_not_filter_by_window():
    """The 2024-2025 window is the CLI's argument (plan Task 8). An adapter that
    silently dropped a 2023 row would make --start/--end untestable."""
    source = dict(row("9000001"), date_received="2023-06-01T00:00:00-04:00")
    record, _ = normalize(source)
    assert record.submitted_at.year == 2023


def test_no_module_level_label_roster_exists():
    import ingest.sources.cfpb as adapter

    for name in dir(adapter):
        value = getattr(adapter, name)
        if isinstance(value, (frozenset, set)) and value:
            pytest.fail(f"{name} looks like a hardcoded roster; Task 7 owns the roster")


# --- purity and boundaries ---------------------------------------------------


def test_normalize_is_deterministic_for_equivalent_inputs():
    assert normalize(row("9000001")) == normalize(row("9000001"))


def test_equivalent_timestamp_spellings_normalize_identically():
    utc = dict(row("9000001"), date_received="2024-03-15T05:00:00+00:00")
    offset = dict(row("9000001"), date_received="2024-03-15T00:00:00-05:00")
    assert normalize(utc)[0] == normalize(offset)[0]


def test_normalize_does_not_mutate_the_row_it_is_given():
    source = row("9000001")
    before = json.dumps(source, sort_keys=True)
    normalize(source)
    assert json.dumps(source, sort_keys=True) == before


def test_normalize_touches_no_network_and_no_filesystem(monkeypatch):
    """Acceptance: normalize is pure."""
    import socket

    def no_network(*args, **kwargs):
        raise AssertionError("normalize must not open a socket")

    def no_files(*args, **kwargs):
        raise AssertionError("normalize must not touch the filesystem")

    source = row("9000001")
    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(Path, "open", no_files)

    record, outcome = normalize(source)
    assert record.external_id == "9000001"
    assert outcome.timely_response is True


def test_normalize_reads_no_clock():
    """A pure mapping cannot depend on when it runs."""
    import ast

    tree = ast.parse(Path("ingest/sources/cfpb.py").read_text(encoding="utf-8"))
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not attributes & {"now", "today", "utcnow"}, "normalize must not read the clock"


def test_the_adapter_satisfies_the_source_adapter_protocol():
    adapter: SourceAdapter[CFPBOutcome] = CFPBAdapter()
    assert adapter.source_slug == "cfpb"
    assert isinstance(adapter.source_api_version, str) and adapter.source_api_version
    assert adapter.source_api_version == SOURCE_API_VERSION
    record, outcome = adapter.normalize(row("9000001"))
    assert isinstance(record, CorpusRecord) and isinstance(outcome, CFPBOutcome)
    assert [r["complaint_id"] for r in adapter.rows_from_page(page())][0] == "9000001"


def test_the_adapter_module_imports_no_django_and_no_ml_code():
    import ast

    tree = ast.parse(Path("ingest/sources/cfpb.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    for module in imported:
        assert module.split(".")[0] != "django", f"ingest must not import Django ({module})"
        assert not module.startswith("ml."), f"ingest must not import ml ({module})"


def test_normalizing_the_whole_fixture_yields_the_expected_split():
    ok, refused = [], []
    for source in rows_from_page(page()):
        try:
            ok.append(normalize(source))
        except (MissingNarrative, NaiveTimestamp, MissingField, InvalidTimelyValue) as exc:
            refused.append((source["complaint_id"], type(exc).__name__))

    assert [r.external_id for r, _ in ok] == ["9000001", "9000002", "9000003"]
    assert refused == [("9000004", "MissingNarrative"), ("9000005", "NaiveTimestamp")]

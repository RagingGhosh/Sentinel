"""Corpus records are value objects, and the two sources' outcomes stay apart.

CFPB publishes a regulatory responsiveness flag; NYC 311 publishes a resolution
time. The addendum defines those as non-equivalent constructs, so they live in
separate types rather than a unified outcome with nullable halves.
"""

import ast
import dataclasses
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ingest.schema import SCHEMA_VERSION, CFPBOutcome, CorpusRecord, NYC311Outcome

SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "ingest" / "schema.py"


def a_record(**overrides) -> CorpusRecord:
    defaults = dict(
        source="cfpb",
        external_id="12345",
        text="My servicer applied last month's payment to the wrong account.",
        label="Mortgage",
        submitted_at=datetime(2024, 9, 3, 22, 42, 53, tzinfo=UTC),
    )
    return CorpusRecord(**{**defaults, **overrides})


def test_schema_version_is_one():
    assert SCHEMA_VERSION == 1


def test_corpus_record_carries_the_five_shared_fields():
    record = a_record()
    assert record.source == "cfpb"
    assert record.external_id == "12345"
    assert record.text.startswith("My servicer")
    assert record.label == "Mortgage"
    assert record.submitted_at == datetime(2024, 9, 3, 22, 42, 53, tzinfo=UTC)


def test_corpus_record_rejects_mutation():
    record = a_record()
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.label = "Credit card"  # type: ignore[misc]


def test_corpus_record_field_names_are_exactly_the_shared_five():
    """A field added here would be a source-specific concept leaking into the
    shared record, which is what the per-source outcome types exist to prevent."""
    names = tuple(f.name for f in dataclasses.fields(CorpusRecord))
    assert names == ("source", "external_id", "text", "label", "submitted_at")


def test_cfpb_outcome_carries_sent_to_company_at():
    outcome = CFPBOutcome(
        external_id="12345",
        timely_response=True,
        sent_to_company_at=datetime(2024, 9, 3, 22, 42, 56, tzinfo=UTC),
    )
    assert outcome.timely_response is True
    assert outcome.sent_to_company_at == datetime(2024, 9, 3, 22, 42, 56, tzinfo=UTC)


def test_cfpb_outcome_accepts_none_for_sent_to_company_at():
    outcome = CFPBOutcome(external_id="12345", timely_response=False, sent_to_company_at=None)
    assert outcome.sent_to_company_at is None
    assert outcome.timely_response is False


def test_cfpb_outcome_rejects_mutation():
    outcome = CFPBOutcome(external_id="1", timely_response=True, sent_to_company_at=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.timely_response = False  # type: ignore[misc]


def test_nyc311_outcome_carries_closed_at_and_resolution_hours():
    outcome = NYC311Outcome(
        external_id="98765",
        closed_at=datetime(2024, 6, 2, 8, 0, 0, tzinfo=UTC),
        resolution_hours=9.8,
    )
    assert outcome.closed_at == datetime(2024, 6, 2, 8, 0, 0, tzinfo=UTC)
    assert outcome.resolution_hours == 9.8


def test_nyc311_outcome_accepts_none_for_an_unclosed_request():
    """2.4% of measured 311 records have no closed_date; both fields go None
    together rather than resolution_hours defaulting to zero."""
    outcome = NYC311Outcome(external_id="98765", closed_at=None, resolution_hours=None)
    assert outcome.closed_at is None
    assert outcome.resolution_hours is None


def test_nyc311_outcome_rejects_mutation():
    outcome = NYC311Outcome(external_id="1", closed_at=None, resolution_hours=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.resolution_hours = 1.0  # type: ignore[misc]


def test_the_two_outcome_types_share_no_target_field():
    """CFPB timeliness and 311 SLA breach are non-equivalent constructs. If the
    two types ever grew a common outcome field, they would invite exactly the
    unification the addendum prohibits."""
    cfpb = {f.name for f in dataclasses.fields(CFPBOutcome)}
    nyc = {f.name for f in dataclasses.fields(NYC311Outcome)}
    assert cfpb & nyc == {"external_id"}, (
        f"outcome types may share only the identifier, but share {sorted(cfpb & nyc)}"
    )


def test_no_generic_unified_outcome_type_exists():
    """`sla_met` was withdrawn precisely because it unified two constructs."""
    import ingest.schema as schema

    assert not hasattr(schema, "RawComplaint")
    for name in dir(schema):
        assert "sla_met" not in name


def test_no_field_is_an_integer_identifier():
    """Acceptance criterion: corpus records must never carry an integer that
    could be mistaken for a Complaint primary key."""
    tree = ast.parse(SCHEMA_PATH.read_text(encoding="utf-8"))
    int_fields = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    rendered = ast.unparse(stmt.annotation)
                    if rendered == "int" or rendered.startswith("int "):
                        int_fields.append(f"{node.name}.{stmt.target.id}")
    assert not int_fields, f"integer identifiers found: {int_fields}"

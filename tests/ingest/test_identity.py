"""`(source, external_id)` is the only identity a corpus record has.

No synthetic integer is ever minted for one, so a corpus record can never be
confused with a `Complaint` primary key — the distinction the addendum turns on.
"""

import dataclasses
from datetime import UTC, datetime

import pytest

from ingest.identity import RecordRef, make_ref, parse_ref
from ingest.schema import CorpusRecord


def a_record(**overrides) -> CorpusRecord:
    defaults = dict(
        source="cfpb",
        external_id="12345",
        text="Billed twice in March.",
        label="Credit card",
        submitted_at=datetime(2025, 3, 1, 9, 0, 0, tzinfo=UTC),
    )
    return CorpusRecord(**{**defaults, **overrides})


def test_ref_renders_as_source_colon_external_id():
    assert str(RecordRef(source="cfpb", external_id="12345")) == "cfpb:12345"


def test_parse_ref_round_trips():
    ref = RecordRef(source="nyc311", external_id="98765")
    assert parse_ref(str(ref)) == ref


def test_external_id_containing_a_colon_round_trips():
    """Split on the first colon only — a source's own identifier may contain one."""
    ref = RecordRef(source="cfpb", external_id="2024:0001:abc")
    assert str(ref) == "cfpb:2024:0001:abc"
    assert parse_ref(str(ref)) == ref
    assert parse_ref(str(ref)).external_id == "2024:0001:abc"


def test_same_external_id_in_different_sources_is_a_different_record():
    cfpb = RecordRef(source="cfpb", external_id="1")
    nyc = RecordRef(source="nyc311", external_id="1")
    assert cfpb != nyc
    assert hash(cfpb) != hash(nyc)


def test_ref_is_hashable_and_usable_as_a_dict_key():
    a = RecordRef(source="cfpb", external_id="1")
    b = RecordRef(source="cfpb", external_id="1")
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1
    assert {a: "value"}[b] == "value"


def test_ref_rejects_mutation():
    ref = RecordRef(source="cfpb", external_id="1")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.source = "nyc311"  # type: ignore[misc]


def test_make_ref_takes_identity_from_the_record():
    record = a_record(source="nyc311", external_id="98765")
    assert make_ref(record) == RecordRef(source="nyc311", external_id="98765")


def test_make_ref_round_trips_through_the_string_form():
    record = a_record()
    assert parse_ref(str(make_ref(record))) == make_ref(record)


def test_parse_ref_rejects_a_string_with_no_separator():
    """Without a colon the input names no source, so there is no ref to return."""
    with pytest.raises(ValueError):
        parse_ref("12345")


def test_ref_fields_are_exactly_source_and_external_id():
    names = tuple(f.name for f in dataclasses.fields(RecordRef))
    assert names == ("source", "external_id")


def test_ref_carries_no_integer_identifier():
    ref = RecordRef(source="cfpb", external_id="12345")
    assert isinstance(ref.external_id, str)
    for field in dataclasses.fields(RecordRef):
        assert not isinstance(getattr(ref, field.name), int), (
            f"{field.name} is an int; corpus refs must never look like a Complaint pk"
        )

"""CFPB Consumer Complaint Database → `CorpusRecord` + `CFPBOutcome`.

The mapping, and nothing beyond it::

    complaint_id            -> CorpusRecord.external_id
    complaint_what_happened -> CorpusRecord.text
    product                 -> CorpusRecord.label
    date_received           -> CorpusRecord.submitted_at      (aware, UTC)
    timely                  -> CFPBOutcome.timely_response    ("Yes"/"No" only)
    date_sent_to_company    -> CFPBOutcome.sent_to_company_at (aware UTC or None)

`sent_to_company_at` is carried **only** so the field-delta provenance
measurement (addendum §2.3, plan Task 8) is computable from the corpus rather
than only from the raw cache. It is never a feature and never a target: it is
downstream of intake and does not exist for a live complaint.

**Deliberately not done here.**

*No roster validation.* Label membership is plan Task 7 (`ingest/roster.py`),
which derives the roster from the data as the intersection across years and
asserts it in both directions before any record is processed. The addendum is
explicit that no label list is transcribed into code from the document, so
`product` is passed through verbatim and a per-record check would be both a
duplicate and a violation.

*No window filtering.* The 2024-2025 window is the CLI's `--start`/`--end`
(plan Task 8). An adapter that silently dropped an out-of-window row would make
that argument untestable and would hide a fetch bug as a normalization result.

Every refusal is a typed error naming the record and the field. Nothing is
dropped, defaulted, remapped or coerced — a silently repaired record changes the
population underneath a published benchmark.

Django-independent, and pure: no network, no filesystem, no clock.
"""

from collections.abc import Iterator, Mapping
from datetime import UTC, datetime

from ingest.schema import CFPBOutcome, CorpusRecord
from ingest.sources.base import SourcePage, SourceRow

SOURCE_SLUG = "cfpb"

SOURCE_API_VERSION = "cfpb-ccdb-v1"
"""The CFPB Consumer Complaint Database search API, Elasticsearch-shaped
response (`hits.hits[]._source`). Bump this if the response shape changes."""

TIMELY_VALUES = {"Yes": True, "No": False}
"""The only two accepted spellings, matched case-sensitively. CFPB publishes
exactly these; accepting "yes" or "true" would mean guessing at a value the
source did not send, and this field is a published outcome."""


class CFPBNormalizationError(Exception):
    """A CFPB row could not be normalized. Never raised for a valid record."""


class MissingField(CFPBNormalizationError):
    """A required field is absent, null, empty, or of an unusable type."""


class MissingNarrative(CFPBNormalizationError):
    """`complaint_what_happened` is absent, null, or blank.

    Its own class rather than a `MissingField`: only about a fifth of CFPB
    complaints carry a published narrative, so this is the expected refusal and
    a caller filtering the corpus needs to distinguish it from a malformed row.
    """


class NaiveTimestamp(CFPBNormalizationError):
    """A timestamp parsed but carries no offset.

    Rejected rather than assumed to be UTC. `submitted_hour` and
    `submitted_weekday` are model features; inventing an offset would shift
    every derived hour by an unknown amount.
    """


class InvalidTimelyValue(CFPBNormalizationError):
    """`timely` is absent or is neither "Yes" nor "No"."""


def rows_from_page(page: SourcePage) -> Iterator[SourceRow]:
    """Unwrap a CFPB API page into its `_source` rows, in published order.

    `SourcePage` is a union because Socrata publishes a bare array; CFPB's
    branch is the mapping one, so this narrows to it. The check is a type
    guard, not normalization: a page of the wrong shape is a caller error, and
    it raises `TypeError` rather than any `CFPBNormalizationError`, which stay
    reserved for a row the source published that we refuse.
    """
    if not isinstance(page, Mapping):
        raise TypeError(f"a CFPB page is a mapping with a 'hits' key, not {type(page).__name__}")
    for hit in page.get("hits", {}).get("hits", []):
        yield hit["_source"]


def _external_id(row: SourceRow) -> str:
    """`complaint_id` as text, exactly as published.

    An int is accepted and stringified — CFPB sends the id both ways and the
    decimal expansion is lossless. A float is refused: it has already lost
    precision, so any string built from it would be invented rather than read.
    """
    value = row.get("complaint_id")
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise MissingField(f"complaint_id is missing or not an id: {value!r}")
    text = str(value).strip()
    if not text:
        raise MissingField("complaint_id is empty")
    return text


def _product(row: SourceRow, external_id: str) -> str:
    """`product`, preserved exactly — including an empty or whitespace-only one.

    The type check is the only check, and it is inherited rather than chosen:
    `CorpusRecord.label` is typed `str`, so a non-string leaves no record to
    build. Emptiness is deliberately *not* checked. Addendum §1.1 gives every
    product-value decision to the roster gate — "Fail. Report the unexpected
    label and its record count" — and an empty product is a value outside the
    locked roster. Refusing it here would delete the record before Task 7 could
    count it, and the plan's Task 5 specifies no product rejection at all.
    """
    value = row.get("product")
    if not isinstance(value, str):
        raise MissingField(
            f"product is missing or not text on complaint_id {external_id}: {value!r}"
        )
    return value


def _narrative(row: SourceRow, external_id: str) -> str:
    """The consumer narrative, preserved byte for byte.

    Only emptiness is checked, and the check strips while the stored value does
    not: `text_length` is a measured feature and reshaping the text here would
    move a published number. The `isinstance` guard is what keeps a float NaN
    from arriving in the corpus as the four characters "nan".
    """
    value = row.get("complaint_what_happened")
    if not isinstance(value, str) or not value.strip():
        raise MissingNarrative(f"complaint_id {external_id} has no published narrative: {value!r}")
    return value


def _timestamp(row: SourceRow, field: str, external_id: str) -> datetime:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise MissingField(f"{field} is missing on complaint_id {external_id}: {value!r}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise MissingField(
            f"{field} on complaint_id {external_id} is not ISO-8601: {value!r}"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise NaiveTimestamp(f"{field} on complaint_id {external_id} carries no offset: {value!r}")
    return parsed.astimezone(UTC)


def _timely(row: SourceRow, external_id: str) -> bool:
    value = row.get("timely")
    if isinstance(value, bool) or not isinstance(value, str) or value not in TIMELY_VALUES:
        raise InvalidTimelyValue(
            f"timely on complaint_id {external_id} is not 'Yes' or 'No': {value!r}"
        )
    return TIMELY_VALUES[value]


def normalize(row: SourceRow) -> tuple[CorpusRecord, CFPBOutcome]:
    """Map one CFPB row. Pure, and never partially applied: every field is
    validated before either object is built, so a refused row leaves nothing."""
    external_id = _external_id(row)
    text = _narrative(row, external_id)
    label = _product(row, external_id)
    submitted_at = _timestamp(row, "date_received", external_id)

    timely_response = _timely(row, external_id)
    sent_raw = row.get("date_sent_to_company")
    sent_to_company_at = (
        None if sent_raw is None else _timestamp(row, "date_sent_to_company", external_id)
    )

    return (
        CorpusRecord(
            source=SOURCE_SLUG,
            external_id=external_id,
            text=text,
            label=label,
            submitted_at=submitted_at,
        ),
        CFPBOutcome(
            external_id=external_id,
            timely_response=timely_response,
            sent_to_company_at=sent_to_company_at,
        ),
    )


class CFPBAdapter:
    """The `SourceAdapter` implementation. Holds no state and no configuration:
    normalization must not vary between two runs over the same page."""

    @property
    def source_slug(self) -> str:
        return SOURCE_SLUG

    @property
    def source_api_version(self) -> str:
        return SOURCE_API_VERSION

    def rows_from_page(self, page: SourcePage) -> Iterator[SourceRow]:
        return rows_from_page(page)

    def normalize(self, row: SourceRow) -> tuple[CorpusRecord, CFPBOutcome]:
        return normalize(row)

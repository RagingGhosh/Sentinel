"""NYC 311 Service Requests (Socrata) → `CorpusRecord` + `NYC311Outcome`.

The mapping, and nothing beyond it::

    unique_key     -> CorpusRecord.external_id
    descriptor     -> CorpusRecord.text
    complaint_type -> CorpusRecord.label
    created_date   -> CorpusRecord.submitted_at            (aware, UTC)
    closed_date    -> NYC311Outcome.closed_at              (aware UTC, or None)
                   -> NYC311Outcome.resolution_hours       (float, or None)

**The page is a bare array.** Socrata's SODA API returns a top-level JSON list
of row objects, with no envelope to descend. That is the sequence branch of
`SourcePage`; CFPB's object branch is refused here.

**Timestamps follow addendum §2.4.** `created_date` and `closed_date` are
Floating Timestamps: they carry no offset. Phase 2 reads them as
`America/New_York` civil time and converts to UTC for storage, because
`write_partition` refuses a naive `submitted_at` and a floating timestamp cannot
reach the corpus uninterpreted. That interpretation is the project's reading of
an unlabelled field, not something the source schema provides.

`normalize` refuses both daylight-saving edge cases rather than resolving them.
In the autumn fold a wall clock occurs twice and nothing in the row says which;
choosing one would assign a wrong instant to about half the affected records,
invisibly. In the spring gap the wall clock never occurs, so a value there means
the source's clock handling is broken or §2.4's interpretation is wrong.

**`normalize` never repairs a row; the run excludes and counts (addendum D50).**
A null, absent or non-text descriptor, a daylight-saving edge on either timestamp
and a negative duration are D45's three classes. `normalize` still raises its
typed error for each. `classify_row` is the seam the command line uses instead: it
applies every other check first, raising exactly what `normalize` raises, and
then returns the one `Exclusion` the first applicable D45 condition earns, which
the run counts by kind and civil year rather than refusing on.

`to_source_local` converts a stored instant back to New York civil time. It
exists because §2.4 requires `submitted_hour` and `submitted_weekday` to derive
from the local representation rather than from the UTC one — a UTC-derived hour
would shift the diurnal pattern, and shift it by different amounts either side
of a transition. Building those features is a later task; what this module owes
them is an instant the original wall clock is exactly recoverable from.

`resolution_hours` is computed from the two UTC instants, never from wall-clock
readings: a request spanning a transition would otherwise gain or lose an hour
that nobody worked.

Django-independent, and pure: no network, no filesystem, no clock, no database.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from ingest.schema import CorpusRecord, NYC311Outcome
from ingest.sources.base import SourcePage, SourceRow

SOURCE_SLUG = "nyc311"

SOURCE_API_VERSION = "socrata-soda2"
"""The Socrata SODA 2.0 JSON endpoint, which returns a bare array of row
objects. Bump this if the response shape changes."""

SOURCE_TIMEZONE = ZoneInfo("America/New_York")
"""Addendum §2.4. A project interpretation of an unlabelled Floating Timestamp,
not an offset the source publishes."""

SECONDS_PER_HOUR = 3600.0


class NYC311NormalizationError(Exception):
    """A 311 row could not be normalized. Never raised for a valid record."""


class MissingField(NYC311NormalizationError):
    """A required field is absent, null, or of an unusable type."""


class MissingDescriptor(NYC311NormalizationError):
    """`descriptor` is absent, null, or not text.

    Its own class because the descriptor is this source's only text: a row
    without one carries nothing for a text model to learn from, which is a
    different situation from a malformed field.
    """


class AmbiguousLocalTime(NYC311NormalizationError):
    """A wall clock that occurs twice, in the autumn daylight-saving fold."""


class NonexistentLocalTime(NYC311NormalizationError):
    """A wall clock that never occurs, in the spring daylight-saving gap."""


class NegativeResolutionTime(NYC311NormalizationError):
    """`closed_date` precedes `created_date`.

    Plan Task 6 was written when the reconnaissance had measured zero negatives;
    addendum D50 records 1,904 measured on 2026-09-28. `normalize` never clamps,
    absolute-values or nulls the duration, since each would turn a broken
    assumption into a plausible-looking number, and never drops the row: the run
    excludes it as a typed, counted exclusion under D50.
    """


EXCLUSION_KINDS = (
    "MissingDescriptor",
    "AmbiguousLocalTime:created_date",
    "NonexistentLocalTime:created_date",
    "AmbiguousLocalTime:closed_date",
    "NonexistentLocalTime:closed_date",
    "NegativeResolutionTime",
)
"""Addendum D50's six exclusion kinds, in its order of precedence.

Each is the complete typed adapter error of one of D45's three classes, and the two
daylight-saving errors are counted separately for each timestamp. A row with several
D45 conditions takes the first kind that applies, and only that one."""


@dataclass(frozen=True)
class Exclusion:
    """One otherwise-valid row the run excludes under addendum D50, and why.

    `created_civil_date` is the calendar date of the row's published `created_date`
    literal, read as New York civil time (§2.4); its year is the year the exclusion
    is counted under. It needs no UTC instant, which a `created_date` edge does not
    have, and none is invented.
    """

    external_id: str
    kind: str
    created_civil_date: date


def rows_from_page(page: SourcePage) -> Iterator[SourceRow]:
    """Yield the rows of a Socrata page, which *is* the array of rows."""
    if isinstance(page, Mapping) or not isinstance(page, Sequence):
        raise TypeError(f"a 311 page is a sequence of rows, not {type(page).__name__}")
    yield from page


def to_source_local(instant: datetime) -> datetime:
    """A stored UTC instant, back in `America/New_York` civil time.

    The conversion is lossless in the direction that matters: every instant this
    module produces came from a New York wall clock, so converting back recovers
    exactly that wall clock. `submitted_hour` and `submitted_weekday` are read
    from here, never from the UTC value (§2.4).
    """
    return instant.astimezone(SOURCE_TIMEZONE)


def _external_id(row: SourceRow) -> str:
    """`unique_key`, exactly as published.

    Required to be text. CFPB's adapter also accepts an integer id because that
    source is known to publish it both ways; nothing establishes that for
    Socrata, so no numeric form is invented here.
    """
    value = row.get("unique_key")
    if not isinstance(value, str) or not value.strip():
        raise MissingField(f"unique_key is missing or not text: {value!r}")
    return value


def _complaint_type(row: SourceRow, external_id: str) -> str:
    """`complaint_type`, preserved exactly, including an empty one.

    Only the type is checked, and that is inherited: `CorpusRecord.label` is
    typed `str`. 311 has 276 distinct complaint types and no locked roster, so
    there is nothing here to validate a value against — and remapping an
    unexpected one would change the population silently.
    """
    value = row.get("complaint_type")
    if not isinstance(value, str):
        raise MissingField(
            f"complaint_type is missing or not text on unique_key {external_id}: {value!r}"
        )
    return value


def _descriptor(row: SourceRow, external_id: str) -> str:
    """`descriptor`, preserved byte for byte.

    A null descriptor is refused, which the plan requires. An *empty* one is
    not: Task 5's CFPB rule names "null or empty" for the narrative, Task 6
    names only null, and that difference is the specification's rather than an
    inconsistency to smooth over. `text_length` is a measured feature — 311
    descriptors run to a median of 15 characters — so the text is never
    reshaped, only type-checked.
    """
    value = row.get("descriptor")
    if not isinstance(value, str):
        raise MissingDescriptor(f"unique_key {external_id} has no descriptor: {value!r}")
    return value


def _local_to_utc(naive: datetime, field: str, external_id: str) -> datetime:
    """Read a floating wall clock as New York civil time; refuse both DST edges.

    A wall clock in the fold has two valid offsets, and one in the gap has none.
    `zoneinfo` answers both questions through `fold`: when the two folds
    disagree the time is one or the other, and a round trip through UTC tells
    them apart — an existing time comes back unchanged, a nonexistent one does
    not.
    """
    local = naive.replace(tzinfo=SOURCE_TIMEZONE)

    if local.replace(fold=0).utcoffset() != local.replace(fold=1).utcoffset():
        returned = local.astimezone(UTC).astimezone(SOURCE_TIMEZONE).replace(tzinfo=None)
        if returned != naive:
            raise NonexistentLocalTime(
                f"{field} on unique_key {external_id} is {naive.isoformat()}, a wall "
                f"clock that does not occur in {SOURCE_TIMEZONE.key} (spring gap)"
            )
        raise AmbiguousLocalTime(
            f"{field} on unique_key {external_id} is {naive.isoformat()}, a wall "
            f"clock that occurs twice in {SOURCE_TIMEZONE.key} (autumn fold); "
            "the row does not say which"
        )

    return local.astimezone(UTC)


def _wall_clock(row: SourceRow, field: str, external_id: str) -> datetime:
    """A Floating Timestamp's naive wall clock, checked for form only.

    Text, ISO-8601 and no offset; a failure is `MissingField`, never a D45 class.
    Where the wall clock falls in New York civil time is `_local_to_utc`'s question.
    """
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise MissingField(f"{field} is missing on unique_key {external_id}: {value!r}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise MissingField(
            f"{field} on unique_key {external_id} is not ISO-8601: {value!r}"
        ) from exc
    if parsed.tzinfo is not None:
        raise MissingField(
            f"{field} on unique_key {external_id} carries an offset ({value!r}); "
            "311 publishes Floating Timestamps, so the source shape has changed "
            "and the interpretation in addendum §2.4 may no longer apply"
        )
    return parsed


def _timestamp(row: SourceRow, field: str, external_id: str) -> datetime:
    return _local_to_utc(_wall_clock(row, field, external_id), field, external_id)


def _resolution_hours(
    submitted_at: datetime, closed_at: datetime | None, external_id: str
) -> float | None:
    """Hours between the two instants, `None` for an open request; never negative."""
    if closed_at is None:
        return None
    elapsed = (closed_at - submitted_at).total_seconds() / SECONDS_PER_HOUR
    if elapsed < 0:
        raise NegativeResolutionTime(
            f"unique_key {external_id} closed {elapsed} hours after it opened: "
            f"created {submitted_at.isoformat()}, closed {closed_at.isoformat()}"
        )
    return elapsed


def _pair(
    external_id: str,
    text: str,
    label: str,
    submitted_at: datetime,
    closed_at: datetime | None,
    resolution_hours: float | None,
) -> tuple[CorpusRecord, NYC311Outcome]:
    return (
        CorpusRecord(
            source=SOURCE_SLUG,
            external_id=external_id,
            text=text,
            label=label,
            submitted_at=submitted_at,
        ),
        NYC311Outcome(
            external_id=external_id,
            closed_at=closed_at,
            resolution_hours=resolution_hours,
        ),
    )


def normalize(row: SourceRow) -> tuple[CorpusRecord, NYC311Outcome]:
    """Map one 311 row. Pure, and never partially applied."""
    external_id = _external_id(row)
    text = _descriptor(row, external_id)
    label = _complaint_type(row, external_id)
    submitted_at = _timestamp(row, "created_date", external_id)

    closed_raw = row.get("closed_date")
    closed_at = None if closed_raw is None else _timestamp(row, "closed_date", external_id)

    resolution_hours = _resolution_hours(submitted_at, closed_at, external_id)
    return _pair(external_id, text, label, submitted_at, closed_at, resolution_hours)


def classify_row(row: SourceRow) -> tuple[CorpusRecord, NYC311Outcome] | Exclusion:
    """`normalize`'s pair for a valid row, or the one `Exclusion` addendum D50 assigns.

    **Stage one** applies every check outside D45's classes, in `normalize`'s relative
    order -- `unique_key`, `complaint_type`, then the form of `created_date` and of a
    present `closed_date` -- and raises exactly the error `normalize` raises, so a row
    with any such problem is refused whatever D45 condition it also has. **Stage two**
    applies the D45 conditions in `normalize`'s relative order -- the descriptor, a
    `created_date` edge, a `closed_date` edge, a negative duration -- and the first that
    applies is the row's only kind. Pure, like `normalize`, and window-blind: which
    exclusions a run counts is the command line's to decide.
    """
    external_id = _external_id(row)
    label = _complaint_type(row, external_id)
    created = _wall_clock(row, "created_date", external_id)
    closed_raw = row.get("closed_date")
    closed = None if closed_raw is None else _wall_clock(row, "closed_date", external_id)

    def excluded(kind: str) -> Exclusion:
        return Exclusion(external_id=external_id, kind=kind, created_civil_date=created.date())

    try:
        text = _descriptor(row, external_id)
    except MissingDescriptor:
        return excluded("MissingDescriptor")
    try:
        submitted_at = _local_to_utc(created, "created_date", external_id)
    except AmbiguousLocalTime:
        return excluded("AmbiguousLocalTime:created_date")
    except NonexistentLocalTime:
        return excluded("NonexistentLocalTime:created_date")
    closed_at = None
    if closed is not None:
        try:
            closed_at = _local_to_utc(closed, "closed_date", external_id)
        except AmbiguousLocalTime:
            return excluded("AmbiguousLocalTime:closed_date")
        except NonexistentLocalTime:
            return excluded("NonexistentLocalTime:closed_date")
    try:
        resolution_hours = _resolution_hours(submitted_at, closed_at, external_id)
    except NegativeResolutionTime:
        return excluded("NegativeResolutionTime")
    return _pair(external_id, text, label, submitted_at, closed_at, resolution_hours)


class NYC311Adapter:
    """The `SourceAdapter` implementation. Stateless, like the CFPB one."""

    @property
    def source_slug(self) -> str:
        return SOURCE_SLUG

    @property
    def source_api_version(self) -> str:
        return SOURCE_API_VERSION

    def rows_from_page(self, page: SourcePage) -> Iterator[SourceRow]:
        return rows_from_page(page)

    def normalize(self, row: SourceRow) -> tuple[CorpusRecord, NYC311Outcome]:
        return normalize(row)

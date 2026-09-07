"""The `SourceAdapter` protocol.

An adapter does exactly two things: split a fetched page into raw source rows,
and map one raw row to a `CorpusRecord` plus that source's outcome type. It does
not fetch, does not write, does not filter by window, and does not validate the
label roster — those belong to the CLI (plan Task 8) and to roster derivation
(plan Task 7) respectively, which is what keeps `normalize` pure and testable
against a fixture with no network and no clock.

The outcome type is a parameter because the two sources publish genuinely
different outcomes: CFPB reports whether a company replied inside a regulatory
window, NYC 311 reports how long resolution took. `ingest.schema` explains at
length why those must not be unified under a shared name.
"""

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Protocol, TypeVar

from ingest.schema import CorpusRecord

OutcomeT = TypeVar("OutcomeT", covariant=True)

SourceRow = Mapping[str, Any]
"""One raw record exactly as the source published it, before any mapping."""

SourcePage = Mapping[str, Any] | Sequence[SourceRow]
"""One fetched page, in whichever envelope its source publishes.

Both branches are real. CFPB's search API wraps its rows in an
Elasticsearch-shaped object (`hits.hits[]._source`); Socrata returns a bare
top-level array of row objects. An adapter narrows to its own branch.

This union replaced a bare `Mapping[str, Any]`, which was fitted to the only
source that existed when the protocol was written. Parameters are
contravariant, so a sequence-shaped adapter could not satisfy the narrower
type -- the second source was unrepresentable. `Any` would also have silenced
that error, but it would have stopped saying anything about what a page is; the
union keeps the two admissible shapes checkable and documents why there are two.
"""


class SourceAdapter(Protocol[OutcomeT]):
    @property
    def source_slug(self) -> str:
        """Domain pack slug written into every `CorpusRecord.source`."""
        ...

    @property
    def source_api_version(self) -> str:
        """Recorded in the corpus manifest, so a corpus names the API shape it
        was read through. A source that changes its response format gets a new
        value here rather than silently producing different records."""
        ...

    def rows_from_page(self, page: SourcePage) -> Iterator[SourceRow]:
        """Unwrap one fetched page into its raw rows, in the order published.

        Takes the union rather than one branch: an implementation may not
        narrow a parameter and still satisfy the protocol.
        """
        ...

    def normalize(self, row: SourceRow) -> tuple[CorpusRecord, OutcomeT]:
        """Map one raw row. Pure: no network, no filesystem, no clock.

        Raises a typed error rather than dropping, guessing or defaulting when a
        required field is missing or malformed. The caller decides what to do
        with a refused record; an adapter that skipped silently would change the
        experimental population without anyone deciding to.
        """
        ...

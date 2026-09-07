"""Negative control. Must FAIL type-checking.

An adapter that narrows `page` back to `Mapping` no longer satisfies the
protocol -- parameters are contravariant. This is exactly the state the
protocol was in before `SourcePage` existed, and it is what made a
sequence-shaped source impossible to describe. If this file ever type-checks
clean, `SourcePage` has stopped doing its job.
"""

from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from typing import Any

from ingest.schema import CFPBOutcome, CorpusRecord
from ingest.sources.base import SourceAdapter, SourceRow

_RECORD = CorpusRecord(
    source="x", external_id="1", text="t", label="l", submitted_at=datetime(2024, 1, 1, tzinfo=UTC)
)
_OUTCOME = CFPBOutcome(external_id="1", timely_response=True, sent_to_company_at=None)


class NarrowedToMappingAdapter:
    @property
    def source_slug(self) -> str:
        return "too-narrow"

    @property
    def source_api_version(self) -> str:
        return "v1"

    def rows_from_page(self, page: Mapping[str, Any]) -> Iterator[SourceRow]:
        yield from ()

    def normalize(self, row: SourceRow) -> tuple[CorpusRecord, CFPBOutcome]:
        return (_RECORD, _OUTCOME)


too_narrow: SourceAdapter[CFPBOutcome] = NarrowedToMappingAdapter()

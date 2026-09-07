"""Both page shapes satisfy `SourceAdapter`. Must type-check clean.

CFPB's API returns an Elasticsearch-shaped object; Socrata returns a top-level
array. A protocol that admits only one of those cannot describe both sources.
"""

from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime

from ingest.schema import CFPBOutcome, CorpusRecord
from ingest.sources.base import SourceAdapter, SourcePage, SourceRow

_RECORD = CorpusRecord(
    source="x", external_id="1", text="t", label="l", submitted_at=datetime(2024, 1, 1, tzinfo=UTC)
)
_OUTCOME = CFPBOutcome(external_id="1", timely_response=True, sent_to_company_at=None)


class MappingShapedAdapter:
    """A page shaped like CFPB's: {"hits": {"hits": [{"_source": {...}}]}}."""

    @property
    def source_slug(self) -> str:
        return "mapping-shaped"

    @property
    def source_api_version(self) -> str:
        return "v1"

    def rows_from_page(self, page: SourcePage) -> Iterator[SourceRow]:
        if not isinstance(page, Mapping):
            raise TypeError("expected a mapping page")
        for hit in page.get("hits", {}).get("hits", []):
            yield hit["_source"]

    def normalize(self, row: SourceRow) -> tuple[CorpusRecord, CFPBOutcome]:
        return (_RECORD, _OUTCOME)


class SequenceShapedAdapter:
    """A page shaped like Socrata's: a bare top-level array of row objects."""

    @property
    def source_slug(self) -> str:
        return "sequence-shaped"

    @property
    def source_api_version(self) -> str:
        return "v1"

    def rows_from_page(self, page: SourcePage) -> Iterator[SourceRow]:
        if isinstance(page, Mapping):
            raise TypeError("expected a sequence page")
        yield from page

    def normalize(self, row: SourceRow) -> tuple[CorpusRecord, CFPBOutcome]:
        return (_RECORD, _OUTCOME)


mapping_adapter: SourceAdapter[CFPBOutcome] = MappingShapedAdapter()
sequence_adapter: SourceAdapter[CFPBOutcome] = SequenceShapedAdapter()

# The real adapter must satisfy the same protocol.
from ingest.sources.cfpb import CFPBAdapter  # noqa: E402

cfpb_adapter: SourceAdapter[CFPBOutcome] = CFPBAdapter()

# Both shapes are callable arguments.
_mapping_page: Mapping[str, object] = {"hits": {"hits": []}}
_sequence_page: Sequence[SourceRow] = [{"unique_key": "1"}]
_a = list(mapping_adapter.rows_from_page(_mapping_page))
_b = list(sequence_adapter.rows_from_page(_sequence_page))

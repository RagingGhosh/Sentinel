"""A sequence-shaped NYC 311 adapter satisfies `SourceAdapter`. Must check clean.

This is the case that was impossible before `SourcePage` became a union: Socrata
publishes a bare array, and an adapter reading one could not satisfy a protocol
whose page parameter was `Mapping[str, Any]`.
"""

from collections.abc import Sequence

from ingest.schema import NYC311Outcome
from ingest.sources.base import SourceAdapter, SourceRow
from ingest.sources.nyc311 import NYC311Adapter

adapter: SourceAdapter[NYC311Outcome] = NYC311Adapter()

# A Socrata page is a sequence of row objects, and passes as one.
socrata_page: Sequence[SourceRow] = [
    {"unique_key": "1", "created_date": "2024-01-15T09:30:00.000"},
]
_rows = list(adapter.rows_from_page(socrata_page))

# The CFPB adapter continues to satisfy the same protocol with an object page.
from ingest.schema import CFPBOutcome  # noqa: E402
from ingest.sources.cfpb import CFPBAdapter  # noqa: E402

cfpb_adapter: SourceAdapter[CFPBOutcome] = CFPBAdapter()
_cfpb_rows = list(cfpb_adapter.rows_from_page({"hits": {"hits": []}}))

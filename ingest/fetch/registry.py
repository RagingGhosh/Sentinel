"""Where a source registers its fetcher (D46, D48 (8)). NYC 311 is registered (Task 24).

A factory takes the `FetchContext` ``ingest.cli.main`` builds for a ``--fetch`` run
and returns a plain ``Fetcher``: a callable of ``(source, start, end)`` yielding
pages in the source's existing page shape. Nothing here fetches anything. For a
source not in `FETCHERS`, ``--fetch`` raises ``FetcherUnavailable`` before any
transport is built, any request is made, any directory is created and anything is
written.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ingest.fetch.http import ClientIdentity, HttpClient
from ingest.fetch.nyc311 import make_nyc311_fetcher


@dataclass(frozen=True)
class FetchContext:
    """Everything a source's fetcher is given; it reaches nothing else."""

    source: str
    start: date
    end: date
    resolved_start: datetime
    resolved_end: datetime
    directory: Path
    """The acquisition directory, ``data/acquisitions/<source>/<start>_<end>``."""
    http: HttpClient
    client: ClientIdentity
    sentinel_commit: str | None
    now: Callable[[], datetime]


FetcherFactory = Callable[[FetchContext], Callable[[str, date, date], Iterable[Any]]]

FETCHERS: dict[str, FetcherFactory] = {"nyc311": make_nyc311_fetcher}
"""Source slug -> fetcher factory. Task 24 registers NYC 311; Task 25 adds CFPB."""

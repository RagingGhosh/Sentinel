"""Where a source registers its fetcher. Task 23 registers none (D46).

A factory takes the `FetchContext` ``ingest.cli.main`` builds for a ``--fetch`` run
and returns a plain ``Fetcher``: a callable of ``(source, start, end)`` yielding
pages in the source's existing page shape. Nothing here fetches anything. Until a
source appears in `FETCHERS`, ``--fetch`` for it raises ``FetcherUnavailable`` before
any transport is built, any request is made, any directory is created and anything
is written.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ingest.fetch.http import ClientIdentity, HttpClient


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

FETCHERS: dict[str, FetcherFactory] = {}
"""Source slug -> fetcher factory. Empty in Task 23; Tasks 24 and 25 each add one."""

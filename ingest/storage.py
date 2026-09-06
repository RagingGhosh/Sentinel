"""Partitioned Parquet storage for the external corpus.

Layout, per the Phase 2 addendum's corpus representation::

    <root>/<source>/v<SCHEMA_VERSION>/year=<YYYY>/part-<NNNN>.parquet

The version segment is part of the path rather than a column so that a schema
change is a new tree beside the old one, not an in-place rewrite: an artifact
that cites a `corpus_id` must still be able to find the bytes it trained on.

Two properties matter more than throughput here.

*Deterministic order.* `read_corpus` yields in `(submitted_at, external_id)`
order regardless of how many part files exist or what order the filesystem
reports them in. Temporal splits and out-of-fold encodings are computed off this
sequence, so a read order that varied between runs would make a measured metric
unreproducible.

*Bounded memory.* Each part file is streamed in batches and the streams are
merged lazily, so peak memory is one batch per part file rather than one whole
file -- let alone one whole corpus. The CFPB window alone is on the order of
millions of narratives.

Parquet output is **not** claimed to be byte-for-byte reproducible; pyarrow makes
no such guarantee across versions or compression codecs. Corpus identity is
therefore taken from the bytes actually on disk (see `ingest.manifest`), which is
the honest construction: it detects a changed file without pretending a rewrite
of unchanged records would reproduce the same bytes.

Django-independent by design. This module is invoked as `python -m`, never as a
management command.
"""

from __future__ import annotations

import heapq
import re
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ingest.schema import SCHEMA_VERSION, CorpusRecord

CORPUS_ROOT = Path("data") / "corpus"
"""Default location. Gitignored: no corpus record is ever committed."""

READ_BATCH_SIZE = 8192
"""Rows pulled from a part file at a time. Bounds per-file read memory."""

PART_DIGITS = 4
_PART_NAME = re.compile(rf"^part-\d{{{PART_DIGITS}}}\.parquet$")
_YEAR_DIR = re.compile(r"^year=(\d{4})$")

ARROW_SCHEMA = pa.schema(
    [
        pa.field("source", pa.string(), nullable=False),
        pa.field("external_id", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("label", pa.string(), nullable=False),
        # Microseconds, because that is exactly datetime's resolution -- storing
        # nanoseconds would imply a precision the source never had.
        pa.field("submitted_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)


def _sort_key(record: CorpusRecord) -> tuple[datetime, str]:
    return (record.submitted_at, record.external_id)


def source_root(root: Path, source: str) -> Path:
    """The versioned tree for one source."""
    return Path(root) / source / f"v{SCHEMA_VERSION}"


def partition_path(root: Path, source: str, year: int, part_index: int) -> Path:
    """Where one part file lives. Pure: no filesystem access, no side effects."""
    return source_root(root, source) / f"year={year}" / f"part-{part_index:0{PART_DIGITS}d}.parquet"


def write_partition(
    records: Sequence[CorpusRecord],
    source: str,
    year: int,
    part_index: int,
    root: Path = CORPUS_ROOT,
) -> Path:
    """Write one part file, sorted by `(submitted_at, external_id)`.

    Sorting on write is what lets `read_corpus` merge streams instead of loading
    and sorting everything. Records are validated against the partition they are
    being written into: a record filed under the wrong source or year would be
    invisible to a `years=` filtered read, which is a silent wrong answer rather
    than a loud failure.
    """
    for record in records:
        if record.source != source:
            raise ValueError(
                f"record {record.external_id!r} has source {record.source!r}, "
                f"but is being written into the {source!r} partition"
            )
        if record.submitted_at.tzinfo is None:
            raise ValueError(
                f"record {record.external_id!r} has a naive submitted_at; "
                "corpus timestamps must be timezone-aware"
            )
        if record.submitted_at.year != year:
            raise ValueError(
                f"record {record.external_id!r} was submitted in "
                f"{record.submitted_at.year}, but is being written into the "
                f"year={year} partition"
            )

    ordered = sorted(records, key=_sort_key)
    table = pa.Table.from_pydict(
        {
            "source": [r.source for r in ordered],
            "external_id": [r.external_id for r in ordered],
            "text": [r.text for r in ordered],
            "label": [r.label for r in ordered],
            "submitted_at": [r.submitted_at for r in ordered],
        },
        schema=ARROW_SCHEMA,
    )

    path = partition_path(root, source, year, part_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return path


def iter_part_files(
    source: str,
    years: Iterable[int] | None = None,
    root: Path = CORPUS_ROOT,
) -> list[Path]:
    """Every part file for a source, in deterministic `(year, part)` path order.

    Returns a list rather than a generator: callers checksum it, count it and
    merge it, and a directory listing is small even when the corpus is not.
    """
    wanted = None if years is None else set(years)
    base = source_root(root, source)
    if not base.is_dir():
        return []

    found: list[Path] = []
    for year_dir in sorted(base.iterdir()):
        match = _YEAR_DIR.match(year_dir.name)
        if not (year_dir.is_dir() and match):
            continue
        if wanted is not None and int(match.group(1)) not in wanted:
            continue
        found.extend(
            sorted(p for p in year_dir.iterdir() if p.is_file() and _PART_NAME.match(p.name))
        )
    return found


def _stream_part(path: Path) -> Iterator[CorpusRecord]:
    """Yield one part file's records in stored order, a batch at a time."""
    parquet_file = pq.ParquetFile(path)
    try:
        batches = parquet_file.iter_batches(batch_size=READ_BATCH_SIZE, columns=ARROW_SCHEMA.names)
        for batch in batches:
            for row in batch.to_pylist():
                yield CorpusRecord(
                    source=row["source"],
                    external_id=row["external_id"],
                    text=row["text"],
                    label=row["label"],
                    submitted_at=row["submitted_at"],
                )
    finally:
        parquet_file.close()


def read_corpus(
    source: str,
    years: Iterable[int] | None = None,
    root: Path = CORPUS_ROOT,
) -> Iterator[CorpusRecord]:
    """Stream a source's corpus in `(submitted_at, external_id)` order.

    Each part file is already sorted, so a lazy k-way merge produces globally
    sorted output while holding at most one batch per part file. Nothing here
    ever calls `read_table`: that would materialise a whole part file, and a
    test asserts it is never reached.
    """
    streams = [_stream_part(path) for path in iter_part_files(source, years, root)]
    return heapq.merge(*streams, key=_sort_key)

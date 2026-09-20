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

*Partitions are not a corpus.* This module writes, lists, removes and merges part
files. Whether a tree of them is a valid corpus is decided by its manifest, and
reading one as a corpus goes through `ingest.manifest.load_corpus` (D27).

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
import shutil
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ingest.schema import SCHEMA_VERSION, CorpusRecord, NYC311Outcome

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

OUTCOMES_DIR = "outcomes"
"""The sidecar's subtree, beside the year partitions rather than inside them.

Record partitions do not move (D37): `iter_part_files` matches `year=YYYY`
directories only, so this sibling is invisible to it and every existing manifest
keeps describing exactly the files it always did.
"""

OUTCOME_ARROW_SCHEMA = pa.schema(
    [
        pa.field("external_id", pa.string(), nullable=False),
        pa.field("resolution_hours", pa.float64(), nullable=True),
        # The source's own normalised close instant, never
        # `submitted_at + resolution_hours`: a derived value would quietly become
        # authoritative the moment the two disagreed (D37).
        pa.field("closed_at", pa.timestamp("us", tz="UTC"), nullable=True),
    ]
)
"""NYC 311 outcomes only. A future source's sidecar defines its own schema in the
task that first consumes it, so nothing here is generalised across sources."""


def _sort_key(record: CorpusRecord) -> tuple[datetime, str]:
    return (record.submitted_at, record.external_id)


def source_root(root: Path, source: str) -> Path:
    """The versioned tree for one source."""
    return Path(root) / source / f"v{SCHEMA_VERSION}"


def partition_path(root: Path, source: str, year: int, part_index: int) -> Path:
    """Where one part file lives. Pure: no filesystem access, no side effects."""
    return source_root(root, source) / f"year={year}" / f"part-{part_index:0{PART_DIGITS}d}.parquet"


def outcomes_root(root: Path, source: str) -> Path:
    """The sidecar subtree, inside the same versioned root as the records (D37)."""
    return source_root(root, source) / OUTCOMES_DIR


def outcome_partition_path(root: Path, source: str, year: int, part_index: int) -> Path:
    """Where one outcome part file lives. Pure, like `partition_path`."""
    return (
        outcomes_root(root, source) / f"year={year}" / f"part-{part_index:0{PART_DIGITS}d}.parquet"
    )


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


def write_outcome_partition(
    outcomes: Sequence[NYC311Outcome],
    source: str,
    year: int,
    part_index: int,
    root: Path = CORPUS_ROOT,
) -> Path:
    """Write one outcome part file, sorted by `external_id`.

    An outcome carries no timestamp of its own, so identity is the only stable
    order; sorting on write is what makes a reread deterministic. The D37 pairing
    is enforced here rather than trusted: a resolved request has both a close time
    and a resolution time, an open one has neither, and a row carrying exactly one
    of them describes a request that was and was not closed.
    """
    for outcome in outcomes:
        if not isinstance(outcome, NYC311Outcome):
            raise ValueError(f"outcome {outcome!r} is {type(outcome).__name__}, not NYC311Outcome")
        resolved = outcome.resolution_hours is not None
        closed = outcome.closed_at is not None
        if resolved != closed:
            raise ValueError(
                f"outcome {outcome.external_id!r} has resolution_hours="
                f"{outcome.resolution_hours!r} and closed_at={outcome.closed_at!r}; "
                "a resolved request carries both and an open request neither (D37)"
            )
        if outcome.closed_at is not None and outcome.closed_at.tzinfo is None:
            raise ValueError(
                f"outcome {outcome.external_id!r} has a naive closed_at; "
                "corpus timestamps are timezone-aware"
            )

    ordered = sorted(outcomes, key=lambda outcome: outcome.external_id)
    table = pa.Table.from_pydict(
        {
            "external_id": [o.external_id for o in ordered],
            "resolution_hours": [o.resolution_hours for o in ordered],
            "closed_at": [o.closed_at for o in ordered],
        },
        schema=OUTCOME_ARROW_SCHEMA,
    )

    path = outcome_partition_path(root, source, year, part_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return path


def remove_source_tree(source: str, root: Path = CORPUS_ROOT) -> None:
    """Delete one source's versioned tree, and nothing else (D27).

    Only `<root>/<source>/v<SCHEMA_VERSION>/` goes. Other sources, and other
    schema versions of this one, stay where they are, so an artifact citing them
    can still find its bytes. An absent tree is not an error.

    Deleting the manifest *first* is `ingest.manifest.clear_corpus`'s job: that
    ordering is the validity boundary, and this module does not know the
    manifest's name.
    """
    tree = source_root(root, source)
    if tree.exists():
        shutil.rmtree(tree)


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


def iter_outcome_part_files(
    source: str,
    years: Iterable[int] | None = None,
    root: Path = CORPUS_ROOT,
) -> list[Path]:
    """Every outcome part file for a source, in deterministic `(year, part)` order.

    Deliberately a separate scanner from `iter_part_files`: two subtrees, two
    listings, so neither can ever pick up the other's files (D37).
    """
    wanted = None if years is None else set(years)
    base = outcomes_root(Path(root), source)
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


def _stream_outcome_part(path: Path) -> Iterator[NYC311Outcome]:
    """Yield one outcome part's rows in stored order, a batch at a time.

    The file's own schema is checked against `OUTCOME_ARROW_SCHEMA` first, so a
    string where a timestamp belongs, a naive timestamp, or a missing column is a
    loud failure rather than a silently mistyped outcome.
    """
    parquet_file = pq.ParquetFile(path)
    try:
        stored = parquet_file.schema_arrow
        for field in OUTCOME_ARROW_SCHEMA:
            if field.name not in stored.names:
                raise ValueError(f"{path} has no {field.name!r} column")
            actual = stored.field(field.name).type
            if actual != field.type:
                raise ValueError(f"{path} stores {field.name!r} as {actual}, not {field.type}")

        batches = parquet_file.iter_batches(
            batch_size=READ_BATCH_SIZE, columns=OUTCOME_ARROW_SCHEMA.names
        )
        for batch in batches:
            for row in batch.to_pylist():
                yield NYC311Outcome(
                    external_id=row["external_id"],
                    closed_at=row["closed_at"],
                    resolution_hours=row["resolution_hours"],
                )
    finally:
        parquet_file.close()


def read_outcome_parts(paths: Iterable[Path]) -> Iterator[NYC311Outcome]:
    """Merge the given outcome part files in `external_id` order.

    Each part is already identity-sorted, so a lazy k-way merge produces globally
    sorted output while holding at most one batch per file -- the same bound
    `read_parts` gives the records.
    """
    return heapq.merge(
        *(_stream_outcome_part(path) for path in paths),
        key=lambda outcome: outcome.external_id,
    )


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


def read_parts(paths: Iterable[Path]) -> Iterator[CorpusRecord]:
    """Merge the given part files in `(submitted_at, external_id)` order.

    Each part file is already sorted, so a lazy k-way merge produces globally
    sorted output while holding at most one batch per part file. Nothing here
    ever calls `read_table`: that would materialise a whole part file, and a
    test asserts it is never reached. Callers choose the files — `read_corpus`
    passes whatever exists, `load_corpus` only what a manifest lists.
    """
    return heapq.merge(*(_stream_part(path) for path in paths), key=_sort_key)


def read_corpus(
    source: str,
    years: Iterable[int] | None = None,
    root: Path = CORPUS_ROOT,
) -> Iterator[CorpusRecord]:
    """Stream whatever part files exist for a source — a partition reader (D27).

    **Not a corpus reader.** It asserts nothing about validity: it reads the
    Parquet files present whether or not a manifest describes them. That is
    exactly what `build_manifest` needs, since it reads the partitions a run has
    just written before any manifest exists. Everything else reads a corpus
    through `ingest.manifest.load_corpus`, and a test enforces that no other
    production module imports this function.
    """
    return read_parts(iter_part_files(source, years, root))

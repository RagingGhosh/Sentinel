"""Corpus manifest: what was ingested, and proof the bytes have not moved since.

The manifest is the corpus's provenance record. A trained artifact cites a
`corpus_id`; that citation is only worth something if the id is derived from the
part files' actual contents, so that editing, truncating or replacing a part file
produces a different id.

`corpus_id` is a SHA256 over the sorted `(relative path, sha256)` pairs. Two
consequences follow deliberately from that definition: the id is stable when a
file's mtime changes but its bytes do not, and it changes when a part is added
or removed even if every surviving part is untouched.

**The manifest is the validity boundary (D27).** A source/version tree without
a valid manifest is not a corpus. A run deletes the manifest before anything
else (`clear_corpus`) and writes it last, atomically (`write_manifest`), so a
manifest on disk always describes a completed run. `load_corpus` is the only
corpus reader; `read_corpus` is the partition reader `build_manifest` uses. This
is a validity boundary, not atomic directory replacement: a failed run loses the
previous corpus, which is rebuilt by rerunning ingest from the raw cache.

Django-independent, like the rest of `ingest/`.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ingest.schema import SCHEMA_VERSION, CFPBOutcome, CorpusRecord, NYC311Outcome

# The slug rather than a second literal: one source of truth for the name, and
# the adapter is pure, so this adds no weight and no cycle.
from ingest.sources.cfpb import SOURCE_SLUG as CFPB_SOURCE
from ingest.storage import (
    CORPUS_ROOT,
    iter_outcome_part_files,
    iter_part_files,
    read_cfpb_outcome_parts,
    read_corpus,
    read_outcome_parts,
    read_parts,
    remove_source_tree,
    source_root,
)

MANIFEST_NAME = "manifest.json"
_CHECKSUM_CHUNK = 1 << 20

MANIFEST_VERSION = 2
"""Version 2 adds `outcome_part_files` (D37.16).

Only the manifest document changed, so `schema_version` stays 1 and every
record partition stays exactly where it is: plan section G separates the two
numbers for precisely this case. A v1 manifest still reads, its absent
`outcome_part_files` becoming an empty mapping with nothing else
reinterpreted. That compatibility is read-only; every new write emits 2.
"""
"""The manifest document's own shape (plan §G).

Deliberately **not** `SCHEMA_VERSION`. That one versions `CorpusRecord` and is
the `v<N>` segment of the storage path, so incrementing it for a manifest change
would leave every record field identical while every written partition became
unreachable under a new tree. Adding, removing or retyping a manifest field
increments this instead and leaves the corpus tree exactly where it is.
"""


class CorpusIntegrityError(Exception):
    """The corpus on disk does not match what the manifest says it is."""


class ManifestNotFound(CorpusIntegrityError):
    """No manifest exists for this source."""


class ChecksumMismatch(CorpusIntegrityError):
    """A part file is missing, or its bytes differ from the recorded digest."""


class OutcomeSidecarNotFound(CorpusIntegrityError):
    """The corpus is valid, but it declares no outcome sidecar (D37.16).

    Raised only for absence. A tampered, missing or unlisted outcome part is
    an integrity failure and keeps its own error, because "there is no
    sidecar" and "the sidecar is damaged" call for different responses.
    """


class UnlistedPartFile(CorpusIntegrityError):
    """A part file exists on disk that the manifest does not list (D27).

    A completed run leaves exactly the files it lists, so an extra one means the
    tree was changed by something other than that run.
    """


@dataclass(frozen=True)
class CorpusManifest:
    """Everything needed to identify a corpus without reading it.

    Note what is absent: no complaint id, no primary key, no operational field.
    A corpus record never becomes a `Complaint` row, so nothing here may look
    like a handle to one.
    """

    manifest_version: int
    """This document's shape. See `MANIFEST_VERSION`."""
    schema_version: int
    """The `CorpusRecord` schema, which is also the storage path's `v<N>`."""
    source_slug: str
    window_start: datetime
    window_end: datetime
    ingested_at: datetime
    record_count: int
    per_year_counts: dict[int, int]
    label_roster: dict[str, int]
    """Observed labels with their counts -- the roster as ingested, which is what
    a later run compares against to detect that the source changed its taxonomy."""
    part_files: dict[str, str]
    """Corpus-root-relative POSIX path -> SHA256 of the file's bytes."""
    source_api_version: str
    corpus_id: str

    limit: int | None
    """The `--limit` the ingestion ran under, or `None` for an unbounded run.

    Recorded so a truncated development corpus can never be mistaken for a full
    one: an artifact citing a `corpus_id` can see whether the corpus behind it
    was complete."""

    timestamp_diagnostic: dict[str, Any]
    """The §2.3 provenance diagnostic, stored verbatim.

    This module records it and never computes it — no verdict rule, no delta
    arithmetic, no `hour_concentration` lives here. Its structure is documented
    in plan §G, and it is held as a plain mapping so that adding a metric to the
    diagnostic does not require a change in this file."""

    outcome_part_files: dict[str, str] = field(default_factory=dict)
    """Corpus-root-relative POSIX path -> SHA256 for the outcome sidecar (D37.17).

    Kept apart from `part_files` so a reader can tell which bytes are records
    and which are outcomes without parsing a path, and last in the field order
    because it is the only field carrying a default, which is what lets a v1
    manifest written before the sidecar existed still be read."""


def sha256_file(path: Path) -> str:
    """Digest a file's bytes, read in chunks so a large part file is not loaded."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHECKSUM_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def compute_corpus_id(part_checksums: dict[str, str]) -> str:
    """SHA256 over the sorted `(path, sha256)` pairs.

    Sorting is what makes the id independent of directory listing order, and the
    NUL separator keeps a path ending in a digest-like suffix from colliding with
    a different path/digest split.
    """
    digest = hashlib.sha256()
    for path in sorted(part_checksums):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(part_checksums[path].encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_manifest(
    source: str,
    window_start: datetime,
    window_end: datetime,
    source_api_version: str,
    limit: int | None,
    timestamp_diagnostic: dict[str, Any],
    root: Path = CORPUS_ROOT,
    ingested_at: datetime | None = None,
) -> CorpusManifest:
    """Describe the corpus currently on disk for one source.

    Counts come from a streaming read, so building a manifest costs one pass and
    no more memory than reading does.
    """
    root = Path(root)
    part_checksums = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in iter_part_files(source, root=root)
    }
    outcome_checksums = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in iter_outcome_part_files(source, root=root)
    }

    per_year: Counter[int] = Counter()
    labels: Counter[str] = Counter()
    record_count = 0
    for record in read_corpus(source, root=root):
        record_count += 1
        per_year[record.submitted_at.year] += 1
        labels[record.label] += 1

    return CorpusManifest(
        manifest_version=MANIFEST_VERSION,
        schema_version=SCHEMA_VERSION,
        source_slug=source,
        window_start=window_start,
        window_end=window_end,
        ingested_at=ingested_at or datetime.now(UTC),
        record_count=record_count,
        per_year_counts=dict(sorted(per_year.items())),
        label_roster=dict(sorted(labels.items())),
        part_files=dict(sorted(part_checksums.items())),
        outcome_part_files=dict(sorted(outcome_checksums.items())),
        source_api_version=source_api_version,
        # Merged at the call site; `compute_corpus_id` itself is unchanged. An
        # empty outcome set contributes nothing, so a record-only corpus keeps
        # the identity it already published, while a corpus holding sidecar
        # bytes gets one that covers them (D37.17).
        corpus_id=compute_corpus_id({**part_checksums, **outcome_checksums}),
        limit=limit,
        timestamp_diagnostic=timestamp_diagnostic,
    )


def manifest_path(source: str, root: Path = CORPUS_ROOT) -> Path:
    """Beside the year partitions, inside the versioned tree it describes."""
    return source_root(Path(root), source) / MANIFEST_NAME


def write_manifest(manifest: CorpusManifest, root: Path = CORPUS_ROOT) -> Path:
    """Serialise the manifest deterministically: equal manifests, equal bytes."""
    payload = asdict(manifest)
    payload["window_start"] = manifest.window_start.isoformat()
    payload["window_end"] = manifest.window_end.isoformat()
    payload["ingested_at"] = manifest.ingested_at.isoformat()
    payload["per_year_counts"] = {str(year): n for year, n in manifest.per_year_counts.items()}

    path = manifest_path(manifest.source_slug, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    # Written beside its destination and moved into place, so a reader finds the
    # previous manifest or the new one and never half of either. Only this file
    # is replaced atomically; the tree around it is not (D27).
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".manifest-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


def read_manifest(source: str, root: Path = CORPUS_ROOT) -> CorpusManifest:
    path = manifest_path(source, root)
    if not path.is_file():
        raise ManifestNotFound(f"no manifest for source {source!r} at {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    return CorpusManifest(
        manifest_version=payload["manifest_version"],
        schema_version=payload["schema_version"],
        source_slug=payload["source_slug"],
        window_start=datetime.fromisoformat(payload["window_start"]),
        window_end=datetime.fromisoformat(payload["window_end"]),
        ingested_at=datetime.fromisoformat(payload["ingested_at"]),
        record_count=payload["record_count"],
        per_year_counts={int(year): n for year, n in payload["per_year_counts"].items()},
        label_roster=dict(payload["label_roster"]),
        part_files=dict(payload["part_files"]),
        # The one field with a silent default, granted to it by D37.16 so a v1
        # manifest stays readable. Every other key keeps its strict lookup.
        outcome_part_files=dict(payload.get("outcome_part_files", {})),
        source_api_version=payload["source_api_version"],
        corpus_id=payload["corpus_id"],
        limit=payload["limit"],
        timestamp_diagnostic=dict(payload["timestamp_diagnostic"]),
    )


def verify_manifest(manifest: CorpusManifest, root: Path = CORPUS_ROOT) -> None:
    """Raise unless every recorded part file is present with the recorded bytes.

    Verification is per file so the error names the offending path: "the corpus
    changed" is not actionable, "this part file changed" is.
    """
    root = Path(root)
    # Both declared sets: the record partitions and the outcome sidecar (D37.16).
    declared = {**manifest.part_files, **manifest.outcome_part_files}
    for relative, expected in sorted(declared.items()):
        path = root / relative
        if not path.is_file():
            raise ChecksumMismatch(f"part file recorded in the manifest is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ChecksumMismatch(
                f"part file {relative} does not match its recorded checksum "
                f"(expected {expected}, found {actual})"
            )


# --- D27: the validity boundary ----------------------------------------------


def clear_corpus(source: str, root: Path = CORPUS_ROOT) -> None:
    """Delete a source's corpus: its manifest first, then its versioned tree (D27).

    The order is the validity boundary. Once the manifest is gone the tree is no
    longer a corpus, so if removing the partitions then fails part-way, what is
    left cannot be read as one. Other sources and schema versions are untouched.
    """
    manifest_path(source, root).unlink(missing_ok=True)
    remove_source_tree(source, root=root)


def load_corpus(
    source: str,
    years: Iterable[int] | None = None,
    root: Path = CORPUS_ROOT,
) -> tuple[CorpusManifest, Iterator[CorpusRecord]]:
    """The only corpus reader: a source's corpus, validated against its manifest.

    A tree without a manifest is not a corpus, whatever Parquet it holds, so a
    missing manifest raises `ManifestNotFound`. The part files on disk must be
    exactly the ones the manifest lists: an extra one raises `UnlistedPartFile`,
    and a missing or altered one raises `ChecksumMismatch`. All of that is
    checked here, before a single record is yielded, so a caller never trains on
    part of a corpus that later fails to verify.

    Byte verification is not optional: `corpus_id` ties an artifact to exact
    input bytes only if the bytes are checked when the corpus is loaded. Only the
    listed files are then streamed, with `read_corpus`'s ordering and memory
    guarantees — `read_corpus` itself stays the partition reader `build_manifest`
    needs before any manifest exists (D27).
    """
    root = Path(root)
    manifest = read_manifest(source, root=root)

    on_disk = {path.relative_to(root).as_posix() for path in iter_part_files(source, root=root)}
    unlisted = sorted(on_disk - set(manifest.part_files))
    if unlisted:
        raise UnlistedPartFile(
            f"{source}: part files on disk that the manifest does not list: {', '.join(unlisted)}"
        )

    verify_manifest(manifest, root=root)

    wanted = set(iter_part_files(source, years, root=root))
    listed = [root / relative for relative in sorted(manifest.part_files)]
    return manifest, read_parts(path for path in listed if path in wanted)


def load_outcomes(
    source: str,
    years: Iterable[int] | None = None,
    root: Path = CORPUS_ROOT,
) -> tuple[CorpusManifest, Iterator[NYC311Outcome]]:
    """The outcome sidecar, validated against its manifest (D37).

    The mirror of `load_corpus` for the stream the 311 risk model is built on,
    and gated the same way: the manifest decides which files exist, every
    listed file's bytes are verified before a single outcome is yielded, and
    the raw cache is never consulted. Real `NYC311Outcome` instances come back,
    carrying the source's own `closed_at` rather than one derived from
    `submitted_at + resolution_hours`, which is what lets Task 11 and Task 13
    accept them with no adapter.

    Raises `OutcomeSidecarNotFound` when the manifest declares no sidecar: an
    absence, never an empty iterator, since a caller looping over nothing would
    read "no outcomes" as "no breaches". A damaged sidecar keeps its own error,
    `UnlistedPartFile` for a file the manifest does not list and
    `ChecksumMismatch` for one whose bytes moved.
    """
    root = Path(root)
    manifest = read_manifest(source, root=root)
    if not manifest.outcome_part_files:
        raise OutcomeSidecarNotFound(
            f"{source}: the manifest declares no outcome sidecar. The corpus is "
            "valid for record-only consumers, but an outcome-dependent one "
            "cannot proceed; re-ingest the source to persist its outcome stream."
        )

    on_disk = {
        path.relative_to(root).as_posix() for path in iter_outcome_part_files(source, root=root)
    }
    unlisted = sorted(on_disk - set(manifest.outcome_part_files))
    if unlisted:
        raise UnlistedPartFile(
            f"{source}: outcome part files on disk that the manifest does not "
            f"list: {', '.join(unlisted)}"
        )

    verify_manifest(manifest, root=root)

    wanted = set(iter_outcome_part_files(source, years, root=root))
    listed = [root / relative for relative in sorted(manifest.outcome_part_files)]
    return manifest, read_outcome_parts(path for path in listed if path in wanted)


def load_cfpb_outcomes(
    years: Iterable[int] | None = None,
    root: Path = CORPUS_ROOT,
) -> tuple[CorpusManifest, Iterator[CFPBOutcome]]:
    """The CFPB outcome sidecar, validated against its manifest (Task 19, O1).

    The mirror of `load_outcomes` for the stream Task 19's evaluation target
    derives from, gated identically: the manifest decides which files exist,
    every listed file's bytes are verified before a single outcome is yielded,
    and the raw cache is never consulted. Real `CFPBOutcome` instances come back,
    carrying `sent_to_company_at` from the persisted `date_sent_to_company`.

    Source-specific rather than generic over outcome types, so NYC 311's loader
    keeps its own name, signature and return type (O1). The integrity machinery
    is the shared one: this adds no second implementation of it.

    Raises `OutcomeSidecarNotFound` when the manifest declares no sidecar -- an
    absence, never an empty iterator, since a caller looping over nothing would
    read "no outcomes" as "every company replied in time". A damaged sidecar
    keeps its own error, `UnlistedPartFile` for a file the manifest does not list
    and `ChecksumMismatch` for one whose bytes moved.
    """
    root = Path(root)
    source = CFPB_SOURCE
    manifest = read_manifest(source, root=root)
    if not manifest.outcome_part_files:
        raise OutcomeSidecarNotFound(
            f"{source}: the manifest declares no outcome sidecar. The corpus is "
            "valid for record-only consumers, but an outcome-dependent one "
            "cannot proceed; re-ingest the source to persist its outcome stream."
        )

    on_disk = {
        path.relative_to(root).as_posix() for path in iter_outcome_part_files(source, root=root)
    }
    unlisted = sorted(on_disk - set(manifest.outcome_part_files))
    if unlisted:
        raise UnlistedPartFile(
            f"{source}: outcome part files on disk that the manifest does not "
            f"list: {', '.join(unlisted)}"
        )

    verify_manifest(manifest, root=root)

    wanted = set(iter_outcome_part_files(source, years, root=root))
    listed = [root / relative for relative in sorted(manifest.outcome_part_files)]
    return manifest, read_cfpb_outcome_parts(path for path in listed if path in wanted)

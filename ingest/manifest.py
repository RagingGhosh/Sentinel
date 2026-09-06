"""Corpus manifest: what was ingested, and proof the bytes have not moved since.

The manifest is the corpus's provenance record. A trained artifact cites a
`corpus_id`; that citation is only worth something if the id is derived from the
part files' actual contents, so that editing, truncating or replacing a part file
produces a different id.

`corpus_id` is a SHA256 over the sorted `(relative path, sha256)` pairs. Two
consequences follow deliberately from that definition: the id is stable when a
file's mtime changes but its bytes do not, and it changes when a part is added
or removed even if every surviving part is untouched.

Django-independent, like the rest of `ingest/`.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from ingest.schema import SCHEMA_VERSION
from ingest.storage import CORPUS_ROOT, iter_part_files, read_corpus, source_root

MANIFEST_NAME = "manifest.json"
_CHECKSUM_CHUNK = 1 << 20


class CorpusIntegrityError(Exception):
    """The corpus on disk does not match what the manifest says it is."""


class ManifestNotFound(CorpusIntegrityError):
    """No manifest exists for this source."""


class ChecksumMismatch(CorpusIntegrityError):
    """A part file is missing, or its bytes differ from the recorded digest."""


@dataclass(frozen=True)
class CorpusManifest:
    """Everything needed to identify a corpus without reading it.

    Note what is absent: no complaint id, no primary key, no operational field.
    A corpus record never becomes a `Complaint` row, so nothing here may look
    like a handle to one.
    """

    schema_version: int
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

    per_year: Counter[int] = Counter()
    labels: Counter[str] = Counter()
    record_count = 0
    for record in read_corpus(source, root=root):
        record_count += 1
        per_year[record.submitted_at.year] += 1
        labels[record.label] += 1

    return CorpusManifest(
        schema_version=SCHEMA_VERSION,
        source_slug=source,
        window_start=window_start,
        window_end=window_end,
        ingested_at=ingested_at or datetime.now(UTC),
        record_count=record_count,
        per_year_counts=dict(sorted(per_year.items())),
        label_roster=dict(sorted(labels.items())),
        part_files=dict(sorted(part_checksums.items())),
        source_api_version=source_api_version,
        corpus_id=compute_corpus_id(part_checksums),
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
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def read_manifest(source: str, root: Path = CORPUS_ROOT) -> CorpusManifest:
    path = manifest_path(source, root)
    if not path.is_file():
        raise ManifestNotFound(f"no manifest for source {source!r} at {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    return CorpusManifest(
        schema_version=payload["schema_version"],
        source_slug=payload["source_slug"],
        window_start=datetime.fromisoformat(payload["window_start"]),
        window_end=datetime.fromisoformat(payload["window_end"]),
        ingested_at=datetime.fromisoformat(payload["ingested_at"]),
        record_count=payload["record_count"],
        per_year_counts={int(year): n for year, n in payload["per_year_counts"].items()},
        label_roster=dict(payload["label_roster"]),
        part_files=dict(payload["part_files"]),
        source_api_version=payload["source_api_version"],
        corpus_id=payload["corpus_id"],
    )


def verify_manifest(manifest: CorpusManifest, root: Path = CORPUS_ROOT) -> None:
    """Raise unless every recorded part file is present with the recorded bytes.

    Verification is per file so the error names the offending path: "the corpus
    changed" is not actionable, "this part file changed" is.
    """
    root = Path(root)
    for relative, expected in sorted(manifest.part_files.items()):
        path = root / relative
        if not path.is_file():
            raise ChecksumMismatch(f"part file recorded in the manifest is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ChecksumMismatch(
                f"part file {relative} does not match its recorded checksum "
                f"(expected {expected}, found {actual})"
            )

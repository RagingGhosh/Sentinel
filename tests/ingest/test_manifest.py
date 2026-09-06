"""Manifest: checksums, corpus identity, and tamper detection.

`corpus_id` is provenance — an artifact cites it to name exactly which corpus it
trained on. If it did not change when a part file changed, that citation would
be a lie.
"""

from datetime import UTC, datetime

import pytest

pytest.importorskip("pyarrow.parquet", reason="pyarrow lives in requirements/train.txt")

from ingest.manifest import (  # noqa: E402
    ChecksumMismatch,
    CorpusManifest,
    ManifestNotFound,
    build_manifest,
    compute_corpus_id,
    read_manifest,
    sha256_file,
    verify_manifest,
    write_manifest,
)
from ingest.schema import SCHEMA_VERSION, CorpusRecord  # noqa: E402
from ingest.storage import write_partition  # noqa: E402


def record(external_id: str, day: int, label: str = "Mortgage") -> CorpusRecord:
    return CorpusRecord(
        source="cfpb",
        external_id=external_id,
        text=f"text {external_id}",
        label=label,
        submitted_at=datetime(2024, 1, day, tzinfo=UTC),
    )


def a_corpus(root, labels=("Mortgage", "Credit card")):
    write_partition(
        [record("1", 1, labels[0]), record("2", 2, labels[1])], "cfpb", 2024, 0, root=root
    )
    return build_manifest(
        source="cfpb",
        window_start=datetime(2024, 1, 1, tzinfo=UTC),
        window_end=datetime(2024, 12, 31, tzinfo=UTC),
        source_api_version="v1",
        root=root,
    )


# --- checksums ---------------------------------------------------------------


def test_sha256_of_a_file_is_stable_and_content_derived(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    first = sha256_file(p)
    assert first == sha256_file(p)
    assert len(first) == 64 and all(c in "0123456789abcdef" for c in first)
    p.write_bytes(b"hellp")
    assert sha256_file(p) != first


def test_corpus_id_is_sha256_over_sorted_path_checksum_pairs():
    a = compute_corpus_id({"y=2024/part-0000.parquet": "aa", "y=2025/part-0000.parquet": "bb"})
    reordered = compute_corpus_id(
        {"y=2025/part-0000.parquet": "bb", "y=2024/part-0000.parquet": "aa"}
    )
    assert a == reordered, "insertion order must not affect corpus_id"
    assert len(a) == 64


def test_corpus_id_changes_when_a_checksum_changes():
    before = compute_corpus_id({"p": "aa"})
    after = compute_corpus_id({"p": "ab"})
    assert before != after


def test_corpus_id_changes_when_a_part_is_added():
    assert compute_corpus_id({"p": "aa"}) != compute_corpus_id({"p": "aa", "q": "bb"})


def test_corpus_id_changes_when_a_part_file_bytes_change(tmp_path):
    """Byte-level, deliberately: the digest is taken over the file as stored, so
    a change no Parquet reader would even survive still moves the id."""
    manifest = a_corpus(tmp_path)
    relative = next(iter(manifest.part_files))
    part = tmp_path / relative

    part.write_bytes(part.read_bytes() + b"\x00")
    assert compute_corpus_id({relative: sha256_file(part)}) != manifest.corpus_id


def test_corpus_id_changes_when_the_corpus_gains_a_part(tmp_path):
    manifest = a_corpus(tmp_path)
    write_partition([record("3", 3)], "cfpb", 2024, 1, root=tmp_path)
    rebuilt = build_manifest(
        source="cfpb",
        window_start=manifest.window_start,
        window_end=manifest.window_end,
        source_api_version="v1",
        root=tmp_path,
    )
    assert rebuilt.corpus_id != manifest.corpus_id
    assert rebuilt.record_count == 3


def test_corpus_id_is_stable_when_only_mtime_changes(tmp_path):
    import os

    manifest = a_corpus(tmp_path)
    part = tmp_path / next(iter(manifest.part_files))
    os.utime(part, (0, 0))
    rebuilt = build_manifest(
        source="cfpb",
        window_start=manifest.window_start,
        window_end=manifest.window_end,
        source_api_version="v1",
        root=tmp_path,
    )
    assert rebuilt.corpus_id == manifest.corpus_id, "identity is content, not filesystem metadata"


# --- manifest contents -------------------------------------------------------


def test_manifest_records_the_specified_fields(tmp_path):
    m = a_corpus(tmp_path)
    assert m.schema_version == SCHEMA_VERSION
    assert m.source_slug == "cfpb"
    assert m.window_start == datetime(2024, 1, 1, tzinfo=UTC)
    assert m.window_end == datetime(2024, 12, 31, tzinfo=UTC)
    assert isinstance(m.ingested_at, datetime)
    assert m.record_count == 2
    assert m.per_year_counts == {2024: 2}
    assert m.label_roster == {"Credit card": 1, "Mortgage": 1}
    assert list(m.part_files) == [f"cfpb/v{SCHEMA_VERSION}/year=2024/part-0000.parquet"]
    assert m.source_api_version == "v1"
    assert len(m.corpus_id) == 64


def test_label_roster_counts_each_label(tmp_path):
    m = a_corpus(tmp_path, labels=("Mortgage", "Mortgage"))
    assert m.label_roster == {"Mortgage": 2}


def test_manifest_round_trips_through_json(tmp_path):
    m = a_corpus(tmp_path)
    write_manifest(m, root=tmp_path)
    assert read_manifest("cfpb", root=tmp_path) == m


def test_manifest_is_written_beside_the_version_tree(tmp_path):
    m = a_corpus(tmp_path)
    write_manifest(m, root=tmp_path)
    assert (tmp_path / "cfpb" / f"v{SCHEMA_VERSION}" / "manifest.json").is_file()


def test_manifest_json_is_deterministic_for_equal_content(tmp_path):
    m = a_corpus(tmp_path)
    write_manifest(m, root=tmp_path)
    first = (tmp_path / "cfpb" / f"v{SCHEMA_VERSION}" / "manifest.json").read_bytes()
    write_manifest(m, root=tmp_path)
    second = (tmp_path / "cfpb" / f"v{SCHEMA_VERSION}" / "manifest.json").read_bytes()
    assert first == second


def test_reading_an_absent_manifest_raises(tmp_path):
    with pytest.raises(ManifestNotFound):
        read_manifest("cfpb", root=tmp_path)


# --- verification ------------------------------------------------------------


def test_verification_succeeds_on_an_untouched_corpus(tmp_path):
    m = a_corpus(tmp_path)
    verify_manifest(m, root=tmp_path)


def test_verification_fails_on_a_mutated_part_and_succeeds_again_once_restored(tmp_path):
    m = a_corpus(tmp_path)
    part = tmp_path / next(iter(m.part_files))
    original = part.read_bytes()

    part.write_bytes(original + b"\x00")
    with pytest.raises(ChecksumMismatch) as exc:
        verify_manifest(m, root=tmp_path)
    assert next(iter(m.part_files)) in str(exc.value)

    part.write_bytes(original)
    verify_manifest(m, root=tmp_path)


def test_verification_fails_when_a_part_file_is_missing(tmp_path):
    m = a_corpus(tmp_path)
    (tmp_path / next(iter(m.part_files))).unlink()
    with pytest.raises(ChecksumMismatch):
        verify_manifest(m, root=tmp_path)


def test_manifest_is_immutable(tmp_path):
    import dataclasses

    m = a_corpus(tmp_path)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.corpus_id = "0" * 64  # type: ignore[misc]


def test_manifest_carries_no_operational_complaint_data(tmp_path):
    """The manifest describes files, not Django rows. No pk-shaped field."""
    import dataclasses

    names = {f.name for f in dataclasses.fields(CorpusManifest)}
    assert not names & {"complaint_id", "complaint_ids", "pk", "id"}

"""Manifest: checksums, corpus identity, and tamper detection.

`corpus_id` is provenance — an artifact cites it to name exactly which corpus it
trained on. If it did not change when a part file changed, that citation would
be a lie.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

pq = pytest.importorskip("pyarrow.parquet", reason="pyarrow lives in requirements/train.txt")

from ingest.manifest import (  # noqa: E402
    MANIFEST_VERSION,
    ChecksumMismatch,
    CorpusIntegrityError,
    CorpusManifest,
    ManifestNotFound,
    UnlistedPartFile,
    build_manifest,
    clear_corpus,
    compute_corpus_id,
    load_corpus,
    read_manifest,
    sha256_file,
    verify_manifest,
    write_manifest,
)
from ingest.schema import SCHEMA_VERSION, CorpusRecord  # noqa: E402
from ingest.storage import iter_part_files, write_partition  # noqa: E402


def record(external_id: str, day: int, label: str = "Mortgage") -> CorpusRecord:
    return CorpusRecord(
        source="cfpb",
        external_id=external_id,
        text=f"text {external_id}",
        label=label,
        submitted_at=datetime(2024, 1, day, tzinfo=UTC),
    )


# A structurally complete diagnostic, exactly as plan §G documents it. The
# manifest stores it opaquely -- computing one is Task 8, not this module -- so
# this is test data whose only job is to prove the structure survives a round
# trip byte for byte.
DIAGNOSTIC = {
    "verdict": "supported_plausible_event_time",
    "verdict_branch": "supported_plausible_event_time",
    "not_directly_testable": False,
    "verdict_rule": {
        "strongly_suspicious_median_delta_seconds_max": 60,
        "strongly_suspicious_frac_delta_le_1min_min": 0.5,
        "strongly_suspicious_frac_identical_timestamps_min": 0.2,
        "supported_median_delta_seconds_min": 3600,
        "supported_frac_delta_le_1min_max": 0.05,
        "supported_count_delta_negative_max": 0,
        "insufficient_pair_coverage_below": 0.5,
        "hour_concentration_downgrade_at": 0.5,
    },
    "primary_evidence": {
        "evidence_class": "field_delta",
        "available": True,
        "pair_coverage": 0.97,
        "median_delta_seconds": 259200.0,
        "delta_percentiles_seconds": {
            "p5": 3600.0,
            "p25": 86400.0,
            "p50": 259200.0,
            "p75": 432000.0,
            "p95": 864000.0,
            "p99": 1728000.0,
        },
        "frac_delta_le_1min": 0.0,
        "frac_delta_le_10min": 0.0,
        "frac_delta_le_1h": 0.01,
        "count_delta_negative": 0,
        "count_delta_zero": 0,
        "frac_identical_timestamps": 0.0,
    },
    "secondary_evidence": {
        "evidence_class": "distributional_anomaly",
        "hour_counts": [10] * 24,
        "weekday_counts": [30] * 7,
        "chi_square": {"statistic": 0.0, "p_value": 1.0, "degrees_of_freedom": 23},
        "hour_concentration": 0.0417,
        "downgraded_verdict": False,
    },
}


def a_corpus(root, labels=("Mortgage", "Credit card"), limit=None, diagnostic=None):
    write_partition(
        [record("1", 1, labels[0]), record("2", 2, labels[1])], "cfpb", 2024, 0, root=root
    )
    return build_manifest(
        source="cfpb",
        window_start=datetime(2024, 1, 1, tzinfo=UTC),
        window_end=datetime(2024, 12, 31, tzinfo=UTC),
        source_api_version="v1",
        limit=limit,
        timestamp_diagnostic=DIAGNOSTIC if diagnostic is None else diagnostic,
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
        limit=None,
        timestamp_diagnostic=DIAGNOSTIC,
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
        limit=None,
        timestamp_diagnostic=DIAGNOSTIC,
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


# --- the Task 8 manifest contract (plan section G) ---------------------------
#
# `manifest_version`, `limit` and `timestamp_diagnostic` are required fields.
# This module stores the diagnostic; it does not compute one -- no verdict
# logic, no hour_concentration, no delta arithmetic lives here.


def test_the_manifest_carries_the_three_contract_fields(tmp_path):
    m = a_corpus(tmp_path, limit=500)
    assert m.manifest_version == MANIFEST_VERSION == 1
    assert m.limit == 500
    assert m.timestamp_diagnostic == DIAGNOSTIC


def test_manifest_version_is_distinct_from_schema_version(tmp_path):
    """They version different things: `schema_version` is the `CorpusRecord`
    schema and the `v<N>` path segment; `manifest_version` is this document's
    own shape. A single number could not say that."""
    import dataclasses

    m = a_corpus(tmp_path)
    names = {f.name for f in dataclasses.fields(CorpusManifest)}
    assert {"manifest_version", "schema_version"} <= names

    assert m.schema_version == SCHEMA_VERSION
    assert m.manifest_version == MANIFEST_VERSION
    # The storage tree is keyed on the record schema, never on this field.
    assert f"v{SCHEMA_VERSION}" in next(iter(m.part_files))


def test_schema_version_was_not_incremented(tmp_path):
    """Bumping it for a manifest change would orphan every written partition."""
    assert SCHEMA_VERSION == 1
    assert a_corpus(tmp_path).schema_version == 1


def test_all_eleven_original_fields_survive(tmp_path):
    m = a_corpus(tmp_path)
    for name in (
        "schema_version",
        "source_slug",
        "window_start",
        "window_end",
        "ingested_at",
        "record_count",
        "per_year_counts",
        "label_roster",
        "part_files",
        "source_api_version",
        "corpus_id",
    ):
        assert hasattr(m, name), name
    assert m.source_slug == "cfpb"
    assert m.record_count == 2
    assert m.per_year_counts == {2024: 2}
    assert len(m.corpus_id) == 64


# --- serialisation round trip ------------------------------------------------


def test_write_manifest_emits_the_three_new_keys(tmp_path):
    m = a_corpus(tmp_path, limit=250)
    write_manifest(m, root=tmp_path)
    raw = (tmp_path / "cfpb" / f"v{SCHEMA_VERSION}" / "manifest.json").read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert payload["manifest_version"] == 1
    assert payload["limit"] == 250
    assert payload["timestamp_diagnostic"] == DIAGNOSTIC


def test_read_manifest_reconstructs_the_new_fields_exactly(tmp_path):
    m = a_corpus(tmp_path, limit=7)
    write_manifest(m, root=tmp_path)
    back = read_manifest("cfpb", root=tmp_path)
    assert back == m
    assert back.manifest_version == 1
    assert back.limit == 7
    assert back.timestamp_diagnostic == DIAGNOSTIC


def test_a_none_limit_serialises_as_json_null(tmp_path):
    m = a_corpus(tmp_path, limit=None)
    path = write_manifest(m, root=tmp_path)
    raw = path.read_text(encoding="utf-8")
    assert '"limit": null' in raw
    assert json.loads(raw)["limit"] is None
    assert read_manifest("cfpb", root=tmp_path).limit is None


def test_the_diagnostic_structure_survives_the_round_trip_completely(tmp_path):
    m = a_corpus(tmp_path)
    write_manifest(m, root=tmp_path)
    back = read_manifest("cfpb", root=tmp_path).timestamp_diagnostic

    assert back == DIAGNOSTIC
    assert back["primary_evidence"]["evidence_class"] == "field_delta"
    assert back["secondary_evidence"]["evidence_class"] == "distributional_anomaly"
    assert back["not_directly_testable"] is False
    assert len(back["secondary_evidence"]["hour_counts"]) == 24
    assert len(back["secondary_evidence"]["weekday_counts"]) == 7
    assert back["verdict_rule"]["hour_concentration_downgrade_at"] == 0.5
    assert back["primary_evidence"]["delta_percentiles_seconds"]["p50"] == 259200.0


def test_a_not_directly_testable_diagnostic_round_trips(tmp_path):
    """The 311 shape: secondary evidence only, no field-delta pair."""
    diagnostic = {
        "verdict": "suspicious_insufficient_evidence",
        "verdict_branch": "no_testable_pair",
        "not_directly_testable": True,
        "verdict_rule": {"insufficient_pair_coverage_below": 0.5},
        "primary_evidence": {
            "evidence_class": "field_delta",
            "available": False,
            "reason": "the created-to-closed interval is the target variable",
        },
        "secondary_evidence": {
            "evidence_class": "distributional_anomaly",
            "hour_counts": [1] * 24,
            "weekday_counts": [1] * 7,
            "chi_square": {"statistic": 0.0, "p_value": 1.0, "degrees_of_freedom": 23},
            "hour_concentration": 0.0417,
            "downgraded_verdict": False,
        },
    }
    m = a_corpus(tmp_path, diagnostic=diagnostic)
    write_manifest(m, root=tmp_path)
    back = read_manifest("cfpb", root=tmp_path)
    assert back.timestamp_diagnostic == diagnostic
    assert back.timestamp_diagnostic["not_directly_testable"] is True
    assert back.timestamp_diagnostic["primary_evidence"]["available"] is False


def test_serialisation_stays_deterministic_with_the_new_fields(tmp_path):
    m = a_corpus(tmp_path, limit=3)
    first = write_manifest(m, root=tmp_path).read_bytes()
    second = write_manifest(m, root=tmp_path).read_bytes()
    assert first == second


# --- required, not defaulted -------------------------------------------------


@pytest.mark.parametrize("key", ["manifest_version", "limit", "timestamp_diagnostic"])
def test_a_manifest_missing_a_required_new_key_is_rejected(tmp_path, key):
    """No silent default. The same behaviour the other ten fields already have:
    a manifest that does not carry the field cannot be read as though it did."""
    m = a_corpus(tmp_path, limit=9)
    path = write_manifest(m, root=tmp_path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload[key]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(KeyError, match=key):
        read_manifest("cfpb", root=tmp_path)


def test_build_manifest_requires_both_new_arguments(tmp_path):
    """`limit` in particular must not default: a forgotten argument would record
    a truncated corpus as unbounded, which is the confusion the field exists to
    prevent."""
    write_partition([record("1", 1)], "cfpb", 2024, 0, root=tmp_path)
    common = {
        "source": "cfpb",
        "window_start": datetime(2024, 1, 1, tzinfo=UTC),
        "window_end": datetime(2024, 12, 31, tzinfo=UTC),
        "source_api_version": "v1",
        "root": tmp_path,
    }
    with pytest.raises(TypeError):
        build_manifest(**common, timestamp_diagnostic=DIAGNOSTIC)
    with pytest.raises(TypeError):
        build_manifest(**common, limit=None)


# --- the new fields must not disturb corpus identity -------------------------


def test_corpus_id_ignores_the_limit_and_the_diagnostic(tmp_path):
    """`corpus_id` hashes part-file checksums only. If metadata entered it, two
    identical corpora described differently would claim to be different data."""
    baseline = a_corpus(tmp_path, limit=None)
    with_limit = a_corpus(tmp_path, limit=10)
    other_diagnostic = a_corpus(
        tmp_path, diagnostic={"verdict": "suspicious_insufficient_evidence"}
    )

    assert with_limit.corpus_id == baseline.corpus_id
    assert other_diagnostic.corpus_id == baseline.corpus_id


# --- D27: the manifest is written last, atomically ----------------------------

MANIFEST = Path("cfpb") / f"v{SCHEMA_VERSION}" / "manifest.json"


def record_at(external_id: str, when: datetime, label: str = "Mortgage") -> CorpusRecord:
    return CorpusRecord(
        source="cfpb",
        external_id=external_id,
        text=f"text {external_id}",
        label=label,
        submitted_at=when,
    )


def a_written_corpus(root, records=None):
    """Partitions plus a manifest on disk: a complete corpus under D27."""
    records = records or [record("1", 1), record("2", 2)]
    by_year: dict[int, list[CorpusRecord]] = {}
    for r in records:
        by_year.setdefault(r.submitted_at.year, []).append(r)
    for year, partition in by_year.items():
        write_partition(partition, "cfpb", year, 0, root=root)
    manifest = build_manifest(
        source="cfpb",
        window_start=datetime(2024, 1, 1, tzinfo=UTC),
        window_end=datetime(2025, 12, 31, tzinfo=UTC),
        source_api_version="v1",
        limit=None,
        timestamp_diagnostic=DIAGNOSTIC,
        root=root,
    )
    write_manifest(manifest, root=root)
    return manifest


def refuse_replace(src, dst):
    raise OSError("the disk filled at the last moment")


def test_write_manifest_moves_a_temporary_file_from_the_same_directory(tmp_path, monkeypatch):
    m = a_corpus(tmp_path)
    calls: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def spy(src, dst):
        calls.append((Path(src), Path(dst)))
        real_replace(src, dst)

    monkeypatch.setattr("ingest.manifest.os.replace", spy)
    path = write_manifest(m, root=tmp_path)

    assert len(calls) == 1, "exactly one atomic replacement"
    src, dst = calls[0]
    assert dst == path
    assert src.parent == path.parent, "a rename across directories is not atomic"
    assert src != path
    assert not src.exists(), "the temporary file becomes the manifest"
    assert read_manifest("cfpb", root=tmp_path) == m


def test_a_failed_manifest_write_leaves_the_previous_manifest_intact(tmp_path, monkeypatch):
    path = write_manifest(a_corpus(tmp_path, limit=None), root=tmp_path)
    before = path.read_bytes()

    monkeypatch.setattr("ingest.manifest.os.replace", refuse_replace)
    with pytest.raises(OSError):
        write_manifest(a_corpus(tmp_path, limit=5), root=tmp_path)
    monkeypatch.undo()

    assert path.read_bytes() == before, "never a half-written manifest"
    files = sorted(p.name for p in path.parent.iterdir() if p.is_file())
    assert files == ["manifest.json"], "no temporary file may be left behind"


def test_a_failed_first_manifest_write_leaves_no_manifest(tmp_path, monkeypatch):
    m = a_corpus(tmp_path)
    monkeypatch.setattr("ingest.manifest.os.replace", refuse_replace)
    with pytest.raises(OSError):
        write_manifest(m, root=tmp_path)
    monkeypatch.undo()

    assert not (tmp_path / MANIFEST).exists()
    assert [p for p in (tmp_path / MANIFEST).parent.iterdir() if p.is_file()] == []


# --- D27: clearing a corpus, manifest first ----------------------------------


def test_clearing_deletes_the_manifest_and_every_partition(tmp_path):
    a_written_corpus(tmp_path)
    clear_corpus("cfpb", root=tmp_path)
    assert not (tmp_path / "cfpb" / f"v{SCHEMA_VERSION}").exists()


def test_clearing_deletes_the_manifest_before_anything_else(tmp_path, monkeypatch):
    """If removing the partitions fails part-way, what is left must not be a corpus."""
    a_written_corpus(tmp_path)

    def locked(*args, **kwargs):
        raise OSError("a part file is locked")

    monkeypatch.setattr("ingest.manifest.remove_source_tree", locked)
    with pytest.raises(OSError):
        clear_corpus("cfpb", root=tmp_path)

    assert not (tmp_path / MANIFEST).exists(), "the manifest goes first"
    assert iter_part_files("cfpb", root=tmp_path), "the partitions were never reached"
    with pytest.raises(ManifestNotFound):
        load_corpus("cfpb", root=tmp_path)


def test_clearing_an_absent_corpus_is_a_no_op(tmp_path):
    clear_corpus("cfpb", root=tmp_path)
    assert list(tmp_path.iterdir()) == []


# --- D27: load_corpus, the only corpus reader --------------------------------


def test_load_corpus_requires_a_manifest(tmp_path):
    """Parquet files without a manifest are not a corpus."""
    write_partition([record("1", 1)], "cfpb", 2024, 0, root=tmp_path)
    with pytest.raises(ManifestNotFound):
        load_corpus("cfpb", root=tmp_path)


def test_load_corpus_rejects_a_part_file_the_manifest_does_not_list(tmp_path):
    a_written_corpus(tmp_path)
    write_partition([record("9", 9)], "cfpb", 2024, 1, root=tmp_path)

    with pytest.raises(UnlistedPartFile) as exc:
        load_corpus("cfpb", root=tmp_path)
    assert isinstance(exc.value, CorpusIntegrityError)
    assert "part-0001.parquet" in str(exc.value)


def test_load_corpus_rejects_a_missing_listed_part(tmp_path):
    m = a_written_corpus(tmp_path)
    (tmp_path / next(iter(m.part_files))).unlink()
    with pytest.raises(ChecksumMismatch):
        load_corpus("cfpb", root=tmp_path)


def test_load_corpus_rejects_altered_bytes_before_yielding_anything(tmp_path):
    """Verification happens when the corpus is loaded, not lazily mid-iteration."""
    m = a_written_corpus(tmp_path)
    part = tmp_path / next(iter(m.part_files))
    part.write_bytes(part.read_bytes() + b"\x00")

    with pytest.raises(ChecksumMismatch):
        load_corpus("cfpb", root=tmp_path)


def two_years_of_records():
    return [
        record_at("late", datetime(2025, 3, 1, tzinfo=UTC)),
        record_at("b", datetime(2024, 1, 2, tzinfo=UTC)),
        record_at("a", datetime(2024, 1, 2, tzinfo=UTC)),
        record_at("first", datetime(2024, 1, 1, tzinfo=UTC)),
    ]


def test_load_corpus_streams_the_listed_files_in_merge_order(tmp_path):
    written = a_written_corpus(tmp_path, two_years_of_records())
    manifest, stream = load_corpus("cfpb", root=tmp_path)

    assert manifest == written == read_manifest("cfpb", root=tmp_path)
    assert [r.external_id for r in stream] == ["first", "a", "b", "late"]


def test_load_corpus_applies_a_years_filter(tmp_path):
    a_written_corpus(tmp_path, two_years_of_records())
    _, stream = load_corpus("cfpb", years=[2025], root=tmp_path)
    assert [r.external_id for r in stream] == ["late"]


def test_load_corpus_never_materialises_a_whole_part_file(tmp_path, monkeypatch):
    a_written_corpus(tmp_path, two_years_of_records())

    def explode(*args, **kwargs):
        raise AssertionError("load_corpus must stream batches, not read whole tables")

    monkeypatch.setattr(pq, "read_table", explode)
    _, stream = load_corpus("cfpb", root=tmp_path)
    assert [r.external_id for r in stream] == ["first", "a", "b", "late"]


def test_the_manifest_module_computes_no_verdict():
    """Contract alignment only: the diagnostic is stored, never derived here.

    Checked against executable code rather than raw text, so a docstring may
    explain what this module deliberately does *not* do without tripping the
    guard. Identifiers and non-docstring literals are what would betray real
    verdict logic.
    """
    import ast

    tree = ast.parse(Path("ingest/manifest.py").read_text(encoding="utf-8"))

    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                docstrings.add(doc)
        # A bare string expression is a field docstring in this module.
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                docstrings.add(node.value.value)

    code_tokens: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            code_tokens.add(node.id)
        elif isinstance(node, ast.Attribute):
            code_tokens.add(node.attr)
        elif isinstance(node, ast.arg):
            code_tokens.add(node.arg)
        elif isinstance(node, ast.FunctionDef | ast.ClassDef):
            code_tokens.add(node.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value not in docstrings:
                code_tokens.add(node.value)

    haystack = " ".join(code_tokens)
    for forbidden in (
        "hour_concentration",
        "strongly_suspicious",
        "supported_plausible",
        "chi_square",
        "median_delta",
    ):
        assert forbidden not in haystack, f"{forbidden} is Task 8's, not the manifest's"

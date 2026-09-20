"""Corpus storage: partitioned Parquet, deterministic read order, bounded memory.

Marked to skip rather than fail when pyarrow is absent: the corpus lives in the
training dependency tier, and the application CI job deliberately installs
neither pandas nor pyarrow.
"""

import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

pq = pytest.importorskip("pyarrow.parquet", reason="pyarrow lives in requirements/train.txt")

from ingest.schema import SCHEMA_VERSION, CorpusRecord  # noqa: E402
from ingest.storage import (  # noqa: E402
    READ_BATCH_SIZE,
    iter_part_files,
    partition_path,
    read_corpus,
    remove_source_tree,
    write_partition,
)


def record(external_id: str, when: datetime, label: str = "Mortgage") -> CorpusRecord:
    return CorpusRecord(
        source="cfpb",
        external_id=external_id,
        text=f"complaint text {external_id}",
        label=label,
        submitted_at=when,
    )


def t(day: int, hour: int = 0) -> datetime:
    return datetime(2024, 1, day, hour, 0, 0, tzinfo=UTC)


def test_partition_path_is_source_version_year_part(tmp_path):
    p = partition_path(tmp_path, "cfpb", 2024, 0)
    expected = f"cfpb/v{SCHEMA_VERSION}/year=2024/part-0000.parquet"
    assert p.relative_to(tmp_path).as_posix() == expected


def test_partition_path_is_deterministic(tmp_path):
    assert partition_path(tmp_path, "cfpb", 2024, 7) == partition_path(tmp_path, "cfpb", 2024, 7)
    assert partition_path(tmp_path, "cfpb", 2024, 7).name == "part-0007.parquet"


def test_write_then_read_round_trips_records_exactly(tmp_path):
    written = [record("3", t(3)), record("1", t(1)), record("2", t(2))]
    write_partition(written, "cfpb", 2024, 0, root=tmp_path)

    read_back = list(read_corpus("cfpb", root=tmp_path))
    assert len(read_back) == 3
    assert {r.external_id for r in read_back} == {"1", "2", "3"}
    for r in read_back:
        original = next(w for w in written if w.external_id == r.external_id)
        assert r == original, "round trip must preserve every field exactly"


def test_read_order_is_deterministic_and_sorted(tmp_path):
    write_partition([record("b", t(2)), record("a", t(1))], "cfpb", 2024, 0, root=tmp_path)
    write_partition([record("d", t(4)), record("c", t(3))], "cfpb", 2024, 1, root=tmp_path)

    first = [r.external_id for r in read_corpus("cfpb", root=tmp_path)]
    second = [r.external_id for r in read_corpus("cfpb", root=tmp_path)]
    assert first == second == ["a", "b", "c", "d"]


def test_read_order_ignores_file_order_on_disk(tmp_path):
    """Two records with the same timestamp are ordered by external_id, and the
    part file they happen to live in must not change the result."""
    write_partition([record("zz", t(1))], "cfpb", 2024, 0, root=tmp_path)
    write_partition([record("aa", t(1))], "cfpb", 2024, 1, root=tmp_path)
    assert [r.external_id for r in read_corpus("cfpb", root=tmp_path)] == ["aa", "zz"]


def test_read_orders_across_year_partitions(tmp_path):
    write_partition(
        [record("later", datetime(2025, 1, 1, tzinfo=UTC))], "cfpb", 2025, 0, root=tmp_path
    )
    write_partition([record("earlier", t(1))], "cfpb", 2024, 0, root=tmp_path)
    assert [r.external_id for r in read_corpus("cfpb", root=tmp_path)] == ["earlier", "later"]


def test_years_filter_selects_only_those_partitions(tmp_path):
    write_partition([record("a", t(1))], "cfpb", 2024, 0, root=tmp_path)
    write_partition([record("b", datetime(2025, 6, 1, tzinfo=UTC))], "cfpb", 2025, 0, root=tmp_path)
    assert [r.external_id for r in read_corpus("cfpb", years=[2024], root=tmp_path)] == ["a"]
    assert [r.external_id for r in read_corpus("cfpb", years=[2025], root=tmp_path)] == ["b"]


def test_sources_are_isolated_from_each_other(tmp_path):
    write_partition([record("1", t(1))], "cfpb", 2024, 0, root=tmp_path)
    other = CorpusRecord(
        source="nyc311", external_id="1", text="noise", label="Noise", submitted_at=t(1)
    )
    write_partition([other], "nyc311", 2024, 0, root=tmp_path)
    assert [r.source for r in read_corpus("cfpb", root=tmp_path)] == ["cfpb"]
    assert [r.source for r in read_corpus("nyc311", root=tmp_path)] == ["nyc311"]


def test_reading_an_absent_source_yields_nothing(tmp_path):
    assert list(read_corpus("cfpb", root=tmp_path)) == []


def test_timezone_aware_timestamps_survive_the_round_trip(tmp_path):
    when = datetime(2024, 9, 3, 22, 42, 53, tzinfo=UTC)
    write_partition([record("1", when)], "cfpb", 2024, 0, root=tmp_path)
    got = next(iter(read_corpus("cfpb", root=tmp_path)))
    assert got.submitted_at == when
    assert got.submitted_at.tzinfo is not None, "must not degrade to a naive datetime"


def test_read_never_materialises_a_whole_part_file(tmp_path, monkeypatch):
    """Acceptance: read_corpus streams. If it called read_table it would hold an
    entire part file in memory, which a multi-million-row corpus cannot afford."""
    write_partition([record("a", t(1))], "cfpb", 2024, 0, root=tmp_path)
    write_partition([record("b", t(2))], "cfpb", 2024, 1, root=tmp_path)

    def explode(*args, **kwargs):
        raise AssertionError("read_corpus must stream batches, not read whole tables")

    monkeypatch.setattr(pq, "read_table", explode)
    assert [r.external_id for r in read_corpus("cfpb", root=tmp_path)] == ["a", "b"]


def test_read_batch_size_is_bounded(tmp_path):
    assert isinstance(READ_BATCH_SIZE, int)
    assert 0 < READ_BATCH_SIZE <= 100_000, "batch size must bound per-file memory"


# --- partition validation ----------------------------------------------------
#
# A misfiled record is a silent wrong answer: it would simply be absent from a
# `years=`-filtered read. These raise instead.


def test_writing_a_foreign_source_into_a_partition_raises(tmp_path):
    with pytest.raises(ValueError, match="nyc311"):
        write_partition(
            [CorpusRecord("nyc311", "1", "x", "Noise", t(1))], "cfpb", 2024, 0, root=tmp_path
        )


def test_writing_a_record_from_another_year_raises(tmp_path):
    with pytest.raises(ValueError, match="2024"):
        write_partition([record("1", t(1))], "cfpb", 2025, 0, root=tmp_path)


def test_writing_a_naive_timestamp_raises(tmp_path):
    naive = datetime(2024, 1, 1, 0, 0, 0)
    with pytest.raises(ValueError, match="naive"):
        write_partition([record("1", naive)], "cfpb", 2024, 0, root=tmp_path)


# --- D27: the partition reader, and source-tree removal ----------------------


def test_read_corpus_reads_partitions_for_which_no_manifest_exists_yet(tmp_path):
    """`build_manifest` reads what a run has just written before any manifest
    describes it, so the partition reader must not require one (D27)."""
    write_partition([record("1", t(1))], "cfpb", 2024, 0, root=tmp_path)
    assert not (tmp_path / "cfpb" / f"v{SCHEMA_VERSION}" / "manifest.json").exists()
    assert [r.external_id for r in read_corpus("cfpb", root=tmp_path)] == ["1"]


def test_removing_a_source_tree_deletes_every_partition_of_that_version(tmp_path):
    write_partition([record("a", t(1))], "cfpb", 2024, 0, root=tmp_path)
    write_partition([record("b", datetime(2025, 6, 1, tzinfo=UTC))], "cfpb", 2025, 0, root=tmp_path)
    assert len(iter_part_files("cfpb", root=tmp_path)) == 2

    remove_source_tree("cfpb", root=tmp_path)

    assert iter_part_files("cfpb", root=tmp_path) == []
    assert not (tmp_path / "cfpb" / f"v{SCHEMA_VERSION}").exists()


def test_removing_a_source_tree_touches_no_other_source_or_schema_version(tmp_path):
    """An older `v<N>` stays findable so an artifact citing it can locate its bytes."""
    write_partition([record("a", t(1))], "cfpb", 2024, 0, root=tmp_path)
    other = CorpusRecord(
        source="nyc311", external_id="1", text="noise", label="Noise", submitted_at=t(1)
    )
    write_partition([other], "nyc311", 2024, 0, root=tmp_path)
    another_version = (
        tmp_path / "cfpb" / f"v{SCHEMA_VERSION + 1}" / "year=2024" / "part-0000.parquet"
    )
    another_version.parent.mkdir(parents=True)
    another_version.write_bytes(b"another schema version's bytes")
    (tmp_path / "unrelated.txt").write_text("keep me", encoding="utf-8")

    target = f"cfpb/v{SCHEMA_VERSION}/"
    survivors = {
        p: p.read_bytes()
        for p in tmp_path.rglob("*")
        if p.is_file() and not p.relative_to(tmp_path).as_posix().startswith(target)
    }
    assert len(survivors) == 3, "nyc311, the other version, and the stray file"

    remove_source_tree("cfpb", root=tmp_path)

    for path, content in survivors.items():
        assert path.is_file(), f"{path} must survive"
        assert path.read_bytes() == content


def test_removing_an_absent_source_tree_is_a_no_op(tmp_path):
    remove_source_tree("cfpb", root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_only_the_manifest_module_imports_the_partition_reader():
    """D27: `load_corpus` is the only corpus reader.

    `read_corpus` asserts nothing about validity, so a production module that
    imported it would be reading a tree that may not be a corpus. Tests are
    exempt — they exercise the partition reader directly.
    """
    allowed = {Path("ingest/storage.py"), Path("ingest/manifest.py")}
    importers: set[Path] = set()
    for package in ("ingest", "ml"):
        for path in Path(package).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and any(
                    alias.name == "read_corpus" for alias in node.names
                ):
                    importers.add(path)
                elif isinstance(node, ast.Attribute) and node.attr == "read_corpus":
                    importers.add(path)

    assert Path("ingest/manifest.py") in importers, "the guard must see the legitimate import"
    assert importers - allowed == set(), sorted(str(p) for p in importers - allowed)


# --- D37.1 / D37.17: the outcome sidecar ---------------------------------------------
#
# RED until the sidecar exists. Imported late so this module still collects and the
# record-storage tests above keep running.


def nyc311_record(external_id: str, when: datetime, label: str = "Noise") -> CorpusRecord:
    """A 311 record: the sidecar is NYC 311-specific, so its tests must be too."""
    return CorpusRecord(
        source="nyc311",
        external_id=external_id,
        text=f"descriptor {external_id}",
        label=label,
        submitted_at=when,
    )


def outcome(external_id: str, hours: float | None, closed: datetime | None = None):
    """A 311 outcome. Resolved rows carry a real close timestamp (D37 correction)."""
    from ingest.schema import NYC311Outcome

    if hours is not None and closed is None:
        closed = datetime(2024, 5, 1, tzinfo=UTC) + timedelta(hours=hours)
    return NYC311Outcome(external_id=external_id, closed_at=closed, resolution_hours=hours)


def storage():
    from ingest import storage as module

    return module


def test_an_outcome_partition_is_written_under_the_outcomes_subtree(tmp_path):
    """D37.17: `outcomes/year=YYYY/`, beside the record tree and never inside it."""
    path = storage().write_outcome_partition(
        [outcome("1", 12.0), outcome("2", None)], "nyc311", 2024, 0, root=tmp_path
    )
    relative = path.relative_to(tmp_path).as_posix()
    assert relative == f"nyc311/v{SCHEMA_VERSION}/outcomes/year=2024/part-0000.parquet"
    assert path.is_file()


def test_record_partitions_keep_their_existing_location(tmp_path):
    """D37.17: records do not move. Mutation: relocate them under `records/`."""
    record_path = write_partition(
        [nyc311_record("1", datetime(2024, 1, 1, tzinfo=UTC))], "nyc311", 2024, 0, root=tmp_path
    )
    storage().write_outcome_partition([outcome("1", 1.0)], "nyc311", 2024, 0, root=tmp_path)
    assert record_path.relative_to(tmp_path).as_posix() == (
        f"nyc311/v{SCHEMA_VERSION}/year=2024/part-0000.parquet"
    )
    assert record_path == partition_path(tmp_path, "nyc311", 2024, 0)


def test_the_outcome_schema_carries_all_three_fields(tmp_path):
    """D37 as corrected: `external_id`, `resolution_hours` and `closed_at`.

    Two columns would have forced the loader to invent the third, and the only
    value available -- `None` -- means "still open" in §2.1, which contradicts a
    resolved row.
    """
    schema = storage().OUTCOME_ARROW_SCHEMA
    assert schema.names == ["external_id", "resolution_hours", "closed_at"]
    assert not schema.field("external_id").nullable
    assert schema.field("resolution_hours").nullable
    assert schema.field("closed_at").nullable

    closed = datetime(2024, 5, 2, 3, tzinfo=UTC)
    path = storage().write_outcome_partition(
        [outcome("1", 12.0, closed), outcome("2", None)], "nyc311", 2024, 0, root=tmp_path
    )
    table = pq.read_table(path)
    assert table.schema.names == ["external_id", "resolution_hours", "closed_at"]
    assert table.column("resolution_hours").to_pylist() == [12.0, None]
    assert table.column("closed_at").to_pylist() == [closed, None]


def test_a_resolved_outcome_round_trips_with_every_field_populated(tmp_path):
    """D37: a resolved row carries all three, and the loader returns a real instance."""
    from ingest.schema import NYC311Outcome

    closed = datetime(2024, 5, 3, 7, 30, tzinfo=UTC)
    path = storage().write_outcome_partition(
        [outcome("1", 30.5, closed)], "nyc311", 2024, 0, root=tmp_path
    )
    (loaded,) = list(storage().read_outcome_parts([path]))
    assert isinstance(loaded, NYC311Outcome)
    assert loaded.external_id == "1"
    assert loaded.resolution_hours == pytest.approx(30.5)
    assert loaded.closed_at == closed


def test_an_open_outcome_round_trips_with_both_nullable_fields_none(tmp_path):
    """D37: an open request has no close time and no resolution time."""
    path = storage().write_outcome_partition([outcome("2", None)], "nyc311", 2024, 0, root=tmp_path)
    (loaded,) = list(storage().read_outcome_parts([path]))
    assert loaded.external_id == "2"
    assert loaded.resolution_hours is None
    assert loaded.closed_at is None


def test_closed_at_is_preserved_and_never_reconstructed(tmp_path):
    """D37: the source's own close timestamp survives.

    The stored `closed_at` here deliberately disagrees with
    `submitted_at + resolution_hours`, so a loader deriving it would return a
    different instant. Mutation: compute `closed_at` instead of reading it.
    """
    from ingest.schema import NYC311Outcome

    stored = NYC311Outcome(
        external_id="1",
        closed_at=datetime(2029, 1, 1, tzinfo=UTC),
        resolution_hours=1.0,
    )
    path = storage().write_outcome_partition([stored], "nyc311", 2024, 0, root=tmp_path)
    (loaded,) = list(storage().read_outcome_parts([path]))
    assert loaded.closed_at == datetime(2029, 1, 1, tzinfo=UTC)
    assert loaded.resolution_hours == pytest.approx(1.0)


def test_a_malformed_closed_at_fails_loudly(tmp_path):
    """A close timestamp that is not a timestamp is refused, never coerced."""
    import pyarrow as pa

    path = storage().outcome_partition_path(tmp_path, "nyc311", 2024, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "external_id": ["1"],
                "resolution_hours": [1.0],
                "closed_at": ["not a timestamp"],
            }
        ),
        path,
    )
    with pytest.raises(Exception):
        list(storage().read_outcome_parts([path]))


def test_a_naive_closed_at_is_refused(tmp_path):
    """§2.1 keeps corpus timestamps aware; a naive close time is a defect."""
    import pyarrow as pa

    path = storage().outcome_partition_path(tmp_path, "nyc311", 2024, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "external_id": ["1"],
                "resolution_hours": [1.0],
                "closed_at": pa.array([datetime(2024, 5, 1)], type=pa.timestamp("us")),
            }
        ),
        path,
    )
    with pytest.raises(Exception):
        list(storage().read_outcome_parts([path]))


def test_an_outcome_partition_round_trips_exactly(tmp_path):
    """Identity and the nullable hour survive the write. Mutation: coerce None to 0.0."""
    written = [outcome("b", None), outcome("a", 3.5)]
    path = storage().write_outcome_partition(written, "nyc311", 2024, 0, root=tmp_path)
    loaded = list(storage().read_outcome_parts([path]))
    assert [(o.external_id, o.resolution_hours) for o in loaded] == [("a", 3.5), ("b", None)]


def test_outcome_read_order_is_deterministic_and_identity_sorted(tmp_path):
    """An outcome carries no timestamp, so identity is the only stable order."""
    path = storage().write_outcome_partition(
        [outcome("10", 1.0), outcome("2", 2.0), outcome("1", 3.0)],
        "nyc311",
        2024,
        0,
        root=tmp_path,
    )
    first = [o.external_id for o in storage().read_outcome_parts([path])]
    second = [o.external_id for o in storage().read_outcome_parts([path])]
    assert first == second == ["1", "10", "2"]


def test_outcome_part_discovery_is_separate_from_record_part_discovery(tmp_path):
    """D37.17: two subtrees, two scanners; neither sees the other's files."""
    write_partition(
        [nyc311_record("1", datetime(2024, 1, 1, tzinfo=UTC))], "nyc311", 2024, 0, root=tmp_path
    )
    storage().write_outcome_partition([outcome("1", 1.0)], "nyc311", 2024, 0, root=tmp_path)

    records_found = iter_part_files("nyc311", root=tmp_path)
    outcomes_found = storage().iter_outcome_part_files("nyc311", root=tmp_path)
    assert len(records_found) == len(outcomes_found) == 1
    assert set(records_found) & set(outcomes_found) == set()
    assert "outcomes" not in records_found[0].relative_to(tmp_path).parts
    assert "outcomes" in outcomes_found[0].relative_to(tmp_path).parts


def test_outcome_part_discovery_filters_by_year(tmp_path):
    storage().write_outcome_partition([outcome("1", 1.0)], "nyc311", 2024, 0, root=tmp_path)
    storage().write_outcome_partition([outcome("2", 2.0)], "nyc311", 2025, 0, root=tmp_path)
    assert len(storage().iter_outcome_part_files("nyc311", root=tmp_path)) == 2
    only = storage().iter_outcome_part_files("nyc311", [2025], root=tmp_path)
    assert len(only) == 1
    assert "year=2025" in only[0].relative_to(tmp_path).parts


def test_a_malformed_outcome_part_fails_loudly(tmp_path):
    """D37.1: no silent skip. Mutation: swallow the read error and yield nothing."""
    path = storage().write_outcome_partition([outcome("1", 1.0)], "nyc311", 2024, 0, root=tmp_path)
    path.write_bytes(b"not a parquet file")
    with pytest.raises(Exception) as caught:
        list(storage().read_outcome_parts([path]))
    assert not isinstance(caught.value, StopIteration)


def test_an_outcome_part_missing_a_column_fails_loudly(tmp_path):
    """A structurally wrong sidecar is refused rather than partially read."""
    import pyarrow as pa

    path = storage().outcome_partition_path(tmp_path, "nyc311", 2024, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"external_id": ["1"]}), path)
    with pytest.raises(Exception):
        list(storage().read_outcome_parts([path]))


def test_removing_a_source_tree_removes_records_and_outcomes_together(tmp_path):
    """D37.17: both subtrees live under the versioned root, so one removal covers them."""
    write_partition(
        [nyc311_record("1", datetime(2024, 1, 1, tzinfo=UTC))], "nyc311", 2024, 0, root=tmp_path
    )
    outcome_path = storage().write_outcome_partition(
        [outcome("1", 1.0)], "nyc311", 2024, 0, root=tmp_path
    )
    assert outcome_path.is_file()
    remove_source_tree("nyc311", root=tmp_path)
    assert not outcome_path.exists()
    assert not (tmp_path / "nyc311" / f"v{SCHEMA_VERSION}").exists()


def test_removing_one_source_leaves_another_sources_outcomes_alone(tmp_path):
    storage().write_outcome_partition([outcome("1", 1.0)], "nyc311", 2024, 0, root=tmp_path)
    kept = storage().write_outcome_partition([outcome("9", 9.0)], "cfpb", 2024, 0, root=tmp_path)
    remove_source_tree("nyc311", root=tmp_path)
    assert kept.is_file()

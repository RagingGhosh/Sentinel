"""Corpus storage: partitioned Parquet, deterministic read order, bounded memory.

Marked to skip rather than fail when pyarrow is absent: the corpus lives in the
training dependency tier, and the application CI job deliberately installs
neither pandas nor pyarrow.
"""

from datetime import UTC, datetime

import pytest

pq = pytest.importorskip("pyarrow.parquet", reason="pyarrow lives in requirements/train.txt")

from ingest.schema import SCHEMA_VERSION, CorpusRecord  # noqa: E402
from ingest.storage import (  # noqa: E402
    READ_BATCH_SIZE,
    partition_path,
    read_corpus,
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

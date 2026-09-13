"""Ingestion CLI: resumable fetch, roster-before-write, timestamp diagnostic.

No test here touches the network. Fetching is an injected callable, and the
stubs below are the only thing that ever produces a page — which is also why
this module can assert that a second run performs zero fetches.

The diagnostic's verdict rule is exercised twice over: directly against
`build_diagnostic`, where each threshold can be driven to its exact boundary,
and end to end through `ingest`, where the object has to survive into the
manifest. §2.3 fixes the rule and D22 fixes the one distributional threshold;
neither is re-derived here.
"""

import ast
import gzip
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

pytest.importorskip("pyarrow.parquet", reason="pyarrow lives in requirements/train.txt")

from ingest.cli import (  # noqa: E402
    HOUR_CONCENTRATION_DOWNGRADE,
    INSUFFICIENT,
    RAW_ROOT,
    STRONGLY_SUSPICIOUS,
    SUPPORTED,
    VERDICT_RULE,
    AuthoritativeCorpusExists,
    EmptyWindow,
    IngestError,
    InvalidDateRange,
    authoritative_roster,
    build_diagnostic,
    decide_verdict,
    ingest,
    main,
    page_checksum,
    resolve_window,
)
from ingest.manifest import read_manifest  # noqa: E402
from ingest.roster import RosterMismatch  # noqa: E402
from ingest.schema import SCHEMA_VERSION  # noqa: E402
from ingest.storage import CORPUS_ROOT, iter_part_files, read_corpus  # noqa: E402

# --- building synthetic source pages -----------------------------------------


def cfpb_row(external_id, *, product="Alpha", received="2024-03-15T09:00:00-04:00", sent=None):
    row = {
        "complaint_id": external_id,
        "date_received": received,
        "product": product,
        "complaint_what_happened": f"narrative {external_id}",
        "timely": "Yes",
    }
    if sent is not None:
        row["date_sent_to_company"] = sent
    return row


def cfpb_page(rows):
    return {"hits": {"hits": [{"_source": r} for r in rows]}}


def nyc311_row(external_id, *, created="2024-03-15T09:00:00.000", complaint_type="Noise"):
    return {
        "unique_key": external_id,
        "created_date": created,
        "complaint_type": complaint_type,
        "descriptor": f"descriptor {external_id}",
    }


class StubFetcher:
    """Records every call so a resumed run can prove it fetched nothing."""

    def __init__(self, pages, fail_after=None):
        self.pages = list(pages)
        self.fail_after = fail_after
        self.calls = 0
        self.pages_yielded = 0

    def __call__(self, source, start, end):
        self.calls += 1
        for index, page in enumerate(self.pages):
            if self.fail_after is not None and index >= self.fail_after:
                raise RuntimeError("network died mid-run")
            self.pages_yielded += 1
            yield page


def a_cfpb_run(tmp_path, rows_per_page, **kwargs):
    """Ingest CFPB pages built from label groups; returns (manifest, fetcher)."""
    fetcher = StubFetcher([cfpb_page(rows) for rows in rows_per_page])
    manifest = ingest(
        source="cfpb",
        start=date(2024, 1, 1),
        end=date(2025, 12, 31),
        limit=kwargs.pop("limit", None),
        fetcher=fetcher,
        corpus_root=tmp_path / "corpus",
        raw_root=tmp_path / "raw",
        **kwargs,
    )
    return manifest, fetcher


# --- CLI arguments -----------------------------------------------------------


def test_the_documented_arguments_are_accepted(tmp_path, monkeypatch):
    seen = {}

    def fake_ingest(**kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr("ingest.cli.ingest", fake_ingest)
    code = main(
        [
            "--source",
            "cfpb",
            "--start",
            "2024-01-01",
            "--end",
            "2025-12-31",
            "--limit",
            "500",
        ]
    )
    assert code == 0
    assert seen["source"] == "cfpb"
    assert seen["start"] == date(2024, 1, 1)
    assert seen["end"] == date(2025, 12, 31)
    assert seen["limit"] == 500


def test_corpus_root_reaches_ingest_from_the_command_line(tmp_path, monkeypatch):
    """D26: the remedy for a refused limited run is a different root, so the
    command line has to be able to name one."""
    seen = {}
    monkeypatch.setattr("ingest.cli.ingest", lambda **kw: seen.update(kw))
    dev_root = tmp_path / "dev-corpus"
    main(
        [
            "--source",
            "cfpb",
            "--start",
            "2024-01-01",
            "--end",
            "2025-12-31",
            "--limit",
            "10",
            "--corpus-root",
            str(dev_root),
        ]
    )
    assert seen["corpus_root"] == dev_root
    assert isinstance(seen["corpus_root"], Path)


def test_corpus_root_defaults_to_the_existing_corpus_root(monkeypatch):
    seen = {}
    monkeypatch.setattr("ingest.cli.ingest", lambda **kw: seen.update(kw))
    main(["--source", "cfpb", "--start", "2024-01-01", "--end", "2025-12-31"])
    assert seen["corpus_root"] == CORPUS_ROOT


def test_limit_is_optional_and_defaults_to_none(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr("ingest.cli.ingest", lambda **kw: seen.update(kw))
    main(["--source", "nyc311", "--start", "2024-01-01", "--end", "2024-12-31"])
    assert seen["limit"] is None


@pytest.mark.parametrize("source", ["cfpb", "nyc311"])
def test_both_sources_are_selectable(source, monkeypatch):
    seen = {}
    monkeypatch.setattr("ingest.cli.ingest", lambda **kw: seen.update(kw))
    main(["--source", source, "--start", "2024-01-01", "--end", "2024-12-31"])
    assert seen["source"] == source


def test_an_unknown_source_is_rejected():
    with pytest.raises(SystemExit):
        main(["--source", "twitter", "--start", "2024-01-01", "--end", "2024-12-31"])


def test_an_end_before_start_is_rejected_and_never_swapped():
    with pytest.raises(InvalidDateRange) as exc:
        main(["--source", "cfpb", "--start", "2025-01-01", "--end", "2024-01-01"])
    message = str(exc.value)
    assert "2025-01-01" in message and "2024-01-01" in message


def test_a_malformed_date_is_rejected():
    with pytest.raises(SystemExit):
        main(["--source", "cfpb", "--start", "01-01-2024", "--end", "2024-12-31"])


def test_a_single_day_window_is_allowed(monkeypatch):
    seen = {}
    monkeypatch.setattr("ingest.cli.ingest", lambda **kw: seen.update(kw))
    main(["--source", "cfpb", "--start", "2024-06-01", "--end", "2024-06-01"])
    assert seen["start"] == seen["end"] == date(2024, 6, 1)


# --- raw cache and resumability ----------------------------------------------


def test_a_first_run_writes_gzipped_pages_under_the_raw_root(tmp_path):
    a_cfpb_run(tmp_path, [[cfpb_row("1")], [cfpb_row("2")]])
    cached = sorted((tmp_path / "raw" / "cfpb").glob("*.json.gz"))
    assert len(cached) == 2
    for path in cached:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            assert "hits" in json.load(handle)


def test_a_second_run_with_the_cache_present_performs_zero_fetches(tmp_path):
    rows = [[cfpb_row("1")], [cfpb_row("2")]]
    first, _ = a_cfpb_run(tmp_path, rows)

    second_fetcher = StubFetcher([cfpb_page(r) for r in rows])
    second = ingest(
        source="cfpb",
        start=date(2024, 1, 1),
        end=date(2025, 12, 31),
        limit=None,
        fetcher=second_fetcher,
        corpus_root=tmp_path / "corpus",
        raw_root=tmp_path / "raw",
    )
    assert second_fetcher.pages_yielded == 2, "pages are offered"
    assert second.record_count == first.record_count
    assert second.corpus_id == first.corpus_id, "identical corpus from the cache"


def test_a_page_whose_checksum_matches_an_existing_file_is_skipped(tmp_path):
    """The skip is the returned flag, not the file count.

    Content addressing alone keeps the count at one -- rewriting the same page
    lands on the same name -- so a count assertion would pass against an
    implementation that never skipped anything. What must be true is that the
    second store reports it wrote nothing.
    """
    from ingest.cli import cache_page

    page = cfpb_page([cfpb_row("1")])
    first_path, first_written = cache_page(page, "cfpb", tmp_path / "raw")
    second_path, second_written = cache_page(page, "cfpb", tmp_path / "raw")

    assert first_written is True
    assert second_written is False, "an already-cached page must not be rewritten"
    assert first_path == second_path
    assert len(list((tmp_path / "raw" / "cfpb").glob("*.json.gz"))) == 1


def test_a_cached_page_is_not_rewritten_on_disk(tmp_path, monkeypatch):
    """Counted at the write itself, so a no-op rewrite cannot pass as a skip."""
    import ingest.cli as cli

    writes = []
    real_open = cli.gzip.open

    def counting_open(path, mode="rb", *args, **kwargs):
        if "w" in mode:
            writes.append(path)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(cli.gzip, "open", counting_open)

    page = cfpb_page([cfpb_row("1")])
    cli.cache_page(page, "cfpb", tmp_path / "raw")
    assert len(writes) == 1
    cli.cache_page(page, "cfpb", tmp_path / "raw")
    assert len(writes) == 1, "the second store must not touch the file"


def test_a_second_run_writes_no_new_pages(tmp_path, monkeypatch):
    rows = [[cfpb_row("1")], [cfpb_row("2")]]
    a_cfpb_run(tmp_path, rows)

    import ingest.cli as cli

    writes = []
    real_open = cli.gzip.open

    def counting_open(path, mode="rb", *args, **kwargs):
        if "w" in mode:
            writes.append(path)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(cli.gzip, "open", counting_open)
    ingest(
        source="cfpb",
        start=date(2024, 1, 1),
        end=date(2025, 12, 31),
        limit=None,
        fetcher=StubFetcher([cfpb_page(r) for r in rows]),
        corpus_root=tmp_path / "corpus",
        raw_root=tmp_path / "raw",
    )
    assert writes == [], "every page was already cached"


def test_the_page_checksum_is_content_derived_and_order_independent():
    a = page_checksum({"hits": {"hits": [{"_source": {"b": 1, "a": 2}}]}})
    b = page_checksum({"hits": {"hits": [{"_source": {"a": 2, "b": 1}}]}})
    assert a == b, "key order must not change a page's identity"
    assert a != page_checksum({"hits": {"hits": []}})
    assert len(a) == 64


def test_an_interrupted_run_resumes_without_duplicating_records(tmp_path):
    pages = [cfpb_page([cfpb_row(str(i))]) for i in range(1, 5)]

    broken = StubFetcher(pages, fail_after=2)
    with pytest.raises(RuntimeError):
        ingest(
            source="cfpb",
            start=date(2024, 1, 1),
            end=date(2025, 12, 31),
            limit=None,
            fetcher=broken,
            corpus_root=tmp_path / "corpus",
            raw_root=tmp_path / "raw",
        )
    assert len(list((tmp_path / "raw" / "cfpb").glob("*.json.gz"))) == 2

    resumed = StubFetcher(pages)
    manifest = ingest(
        source="cfpb",
        start=date(2024, 1, 1),
        end=date(2025, 12, 31),
        limit=None,
        fetcher=resumed,
        corpus_root=tmp_path / "corpus",
        raw_root=tmp_path / "raw",
    )
    assert manifest.record_count == 4
    ids = [r.external_id for r in read_corpus("cfpb", root=tmp_path / "corpus")]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids)) == 4


def test_a_resumed_corpus_is_identical_to_a_clean_one(tmp_path):
    pages = [cfpb_page([cfpb_row(str(i))]) for i in range(1, 5)]

    clean = ingest(
        source="cfpb",
        start=date(2024, 1, 1),
        end=date(2025, 12, 31),
        limit=None,
        fetcher=StubFetcher(pages),
        corpus_root=tmp_path / "clean",
        raw_root=tmp_path / "clean-raw",
    )

    with pytest.raises(RuntimeError):
        ingest(
            source="cfpb",
            start=date(2024, 1, 1),
            end=date(2025, 12, 31),
            limit=None,
            fetcher=StubFetcher(pages, fail_after=1),
            corpus_root=tmp_path / "resumed",
            raw_root=tmp_path / "resumed-raw",
        )
    resumed = ingest(
        source="cfpb",
        start=date(2024, 1, 1),
        end=date(2025, 12, 31),
        limit=None,
        fetcher=StubFetcher(pages),
        corpus_root=tmp_path / "resumed",
        raw_root=tmp_path / "resumed-raw",
    )

    assert resumed.corpus_id == clean.corpus_id
    assert resumed.record_count == clean.record_count
    assert resumed.label_roster == clean.label_roster


def test_running_with_no_fetcher_uses_the_cache_alone(tmp_path):
    a_cfpb_run(tmp_path, [[cfpb_row("1")], [cfpb_row("2")]])
    manifest = ingest(
        source="cfpb",
        start=date(2024, 1, 1),
        end=date(2025, 12, 31),
        limit=None,
        fetcher=None,
        corpus_root=tmp_path / "corpus",
        raw_root=tmp_path / "raw",
    )
    assert manifest.record_count == 2


# --- limit -------------------------------------------------------------------


def test_an_unbounded_run_records_a_null_limit(tmp_path):
    manifest, _ = a_cfpb_run(tmp_path, [[cfpb_row("1"), cfpb_row("2")]])
    assert manifest.limit is None
    assert manifest.record_count == 2


def test_a_bounded_run_records_its_limit_and_honours_it(tmp_path):
    rows = [[cfpb_row(str(i)) for i in range(1, 6)]]
    manifest, _ = a_cfpb_run(tmp_path, rows, limit=3)
    assert manifest.limit == 3
    assert manifest.record_count == 3
    assert len(list(read_corpus("cfpb", root=tmp_path / "corpus"))) == 3


def test_the_limit_reaches_the_written_manifest_on_disk(tmp_path):
    a_cfpb_run(tmp_path, [[cfpb_row(str(i)) for i in range(1, 6)]], limit=2)
    assert read_manifest("cfpb", root=tmp_path / "corpus").limit == 2


# --- roster validation happens before any Parquet write ----------------------


def test_the_first_ingest_derives_and_locks_the_roster(tmp_path):
    manifest, _ = a_cfpb_run(
        tmp_path,
        [[cfpb_row("1", product="Alpha"), cfpb_row("2", product="Beta")]],
    )
    assert set(manifest.label_roster) == {"Alpha", "Beta"}
    assert manifest.label_roster == {"Alpha": 1, "Beta": 1}


def test_an_unexpected_label_on_a_later_run_raises(tmp_path):
    a_cfpb_run(tmp_path, [[cfpb_row("1", product="Alpha")]])

    with pytest.raises(RosterMismatch) as exc:
        ingest(
            source="cfpb",
            start=date(2024, 1, 1),
            end=date(2025, 12, 31),
            limit=None,
            fetcher=StubFetcher([cfpb_page([cfpb_row("2", product="Intruder")])]),
            corpus_root=tmp_path / "corpus",
            raw_root=tmp_path / "raw",
        )
    assert "Intruder" in str(exc.value)


def test_no_part_file_is_written_after_a_roster_mismatch(tmp_path):
    """The plan's explicit ordering test: validation precedes the first write."""
    corpus = tmp_path / "corpus"
    a_cfpb_run(tmp_path, [[cfpb_row("1", product="Alpha")]])
    before = {p: p.read_bytes() for p in iter_part_files("cfpb", root=corpus)}
    assert before, "the first run wrote something to compare against"

    with pytest.raises(RosterMismatch):
        ingest(
            source="cfpb",
            start=date(2024, 1, 1),
            end=date(2025, 12, 31),
            limit=None,
            fetcher=StubFetcher(
                [cfpb_page([cfpb_row("1", product="Alpha"), cfpb_row("9", product="New")])]
            ),
            corpus_root=corpus,
            raw_root=tmp_path / "raw",
        )

    after = {p: p.read_bytes() for p in iter_part_files("cfpb", root=corpus)}
    assert after == before, "no partition may be written or rewritten on a mismatch"


def test_a_roster_mismatch_leaves_no_partition_at_all_on_a_first_run(tmp_path):
    corpus = tmp_path / "corpus"
    with pytest.raises(RosterMismatch):
        ingest(
            source="cfpb",
            start=date(2024, 1, 1),
            end=date(2025, 12, 31),
            limit=None,
            fetcher=StubFetcher(
                [
                    cfpb_page(
                        [
                            cfpb_row("1", product="Alpha", received="2024-03-15T09:00:00-04:00"),
                            cfpb_row(
                                "2",
                                product="OnlyIn2025",
                                received="2025-03-15T09:00:00-04:00",
                            ),
                        ]
                    )
                ]
            ),
            corpus_root=corpus,
            raw_root=tmp_path / "raw",
        )
    assert iter_part_files("cfpb", root=corpus) == []
    assert not (corpus / "cfpb" / f"v{SCHEMA_VERSION}" / "manifest.json").exists()


def test_a_missing_locked_label_raises(tmp_path):
    """A later run whose data no longer contains a locked label.

    The second run reads a *different* raw cache: the first cache is cumulative
    and still holds the Beta record, so a vanished label only appears when the
    pages a run reads no longer carry it. The corpus root is shared, which is
    what makes the first run's manifest the locked roster.
    """
    a_cfpb_run(
        tmp_path,
        [[cfpb_row("1", product="Alpha"), cfpb_row("2", product="Beta")]],
    )
    with pytest.raises(RosterMismatch) as exc:
        ingest(
            source="cfpb",
            start=date(2024, 1, 1),
            end=date(2025, 12, 31),
            limit=None,
            fetcher=StubFetcher([cfpb_page([cfpb_row("3", product="Alpha")])]),
            corpus_root=tmp_path / "corpus",
            raw_root=tmp_path / "later-raw",
        )
    assert exc.value.missing == frozenset({"Beta"})


def test_nyc311_has_no_roster_assertion(tmp_path):
    """311 has 276 complaint types and no locked roster in the spec."""
    fetcher = StubFetcher(
        [[nyc311_row("1", complaint_type="Noise"), nyc311_row("2", complaint_type="Anything")]]
    )
    manifest = ingest(
        source="nyc311",
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        limit=None,
        fetcher=fetcher,
        corpus_root=tmp_path / "corpus",
        raw_root=tmp_path / "raw",
    )
    assert set(manifest.label_roster) == {"Noise", "Anything"}


# --- the diagnostic, driven directly -----------------------------------------


def local_times(hours):
    """One naive local datetime per record, on a fixed Monday."""
    return [datetime(2024, 3, 4, hour, 30) for hour in hours]


def spread_hours(n):
    """Hours spread evenly enough that concentration stays well under D22."""
    return [i % 24 for i in range(n)]


def diagnostic(deltas, *, total=None, hours=None, source="cfpb"):
    paired = 0 if deltas is None else len(deltas)
    total = paired if total is None else total
    hours = spread_hours(total) if hours is None else hours
    return build_diagnostic(
        source=source,
        submitted_local=local_times(hours),
        deltas_seconds=deltas,
        paired_count=paired,
        total_count=total,
    )


def test_the_diagnostic_carries_both_evidence_classes_and_the_rule():
    d = diagnostic([7200.0] * 100)
    assert d["primary_evidence"]["evidence_class"] == "field_delta"
    assert d["secondary_evidence"]["evidence_class"] == "distributional_anomaly"
    assert d["verdict"] in {STRONGLY_SUSPICIOUS, SUPPORTED, INSUFFICIENT}
    assert d["verdict_rule"] == VERDICT_RULE
    assert isinstance(d["verdict_branch"], str) and d["verdict_branch"]
    assert d["not_directly_testable"] is False


def test_a_three_second_median_is_strongly_suspicious():
    d = diagnostic([3.0] * 100)
    assert d["verdict"] == STRONGLY_SUSPICIOUS
    assert d["primary_evidence"]["median_delta_seconds"] == 3.0


def test_a_multi_hour_median_with_no_sub_minute_mass_is_supported():
    d = diagnostic([86400.0] * 100)
    assert d["verdict"] == SUPPORTED


def test_forty_percent_coverage_is_insufficient_regardless_of_deltas():
    d = diagnostic([86400.0] * 40, total=100)
    assert d["primary_evidence"]["pair_coverage"] == pytest.approx(0.40)
    assert d["verdict"] == INSUFFICIENT
    assert "coverage" in d["verdict_branch"]


def test_coverage_exactly_at_the_threshold_is_not_forced():
    d = diagnostic([86400.0] * 50, total=100)
    assert d["primary_evidence"]["pair_coverage"] == pytest.approx(0.50)
    assert d["verdict"] == SUPPORTED


def test_half_the_deltas_under_a_minute_is_strongly_suspicious():
    d = diagnostic([30.0] * 50 + [86400.0] * 50)
    assert d["primary_evidence"]["frac_delta_le_1min"] == pytest.approx(0.50)
    assert d["verdict"] == STRONGLY_SUSPICIOUS


def test_a_fifth_identical_timestamps_is_strongly_suspicious():
    d = diagnostic([0.0] * 20 + [86400.0] * 80)
    assert d["primary_evidence"]["frac_identical_timestamps"] == pytest.approx(0.20)
    assert d["verdict"] == STRONGLY_SUSPICIOUS


def test_a_median_of_exactly_sixty_seconds_is_strongly_suspicious():
    d = diagnostic([60.0] * 100)
    assert d["verdict"] == STRONGLY_SUSPICIOUS


def test_a_median_of_exactly_one_hour_can_be_supported():
    d = diagnostic([3600.0] * 100)
    assert d["primary_evidence"]["median_delta_seconds"] == 3600.0
    assert d["verdict"] == SUPPORTED


def test_a_negative_delta_prevents_the_supported_verdict():
    d = diagnostic([-10.0] + [86400.0] * 99)
    assert d["primary_evidence"]["count_delta_negative"] == 1
    assert d["verdict"] == INSUFFICIENT


def test_the_middle_case_falls_through_to_insufficient():
    """Median above a minute but below an hour: neither branch fires."""
    d = diagnostic([600.0] * 100)
    assert d["verdict"] == INSUFFICIENT


def test_the_recorded_metrics_are_the_ones_the_addendum_names():
    primary = diagnostic([100.0, 200.0, 300.0, 400.0] * 25)["primary_evidence"]
    for key in (
        "pair_coverage",
        "median_delta_seconds",
        "delta_percentiles_seconds",
        "frac_delta_le_1min",
        "frac_delta_le_10min",
        "frac_delta_le_1h",
        "count_delta_negative",
        "count_delta_zero",
        "frac_identical_timestamps",
    ):
        assert key in primary, key
    assert set(primary["delta_percentiles_seconds"]) == {"p5", "p25", "p50", "p75", "p95", "p99"}
    assert primary["median_delta_seconds"] == primary["delta_percentiles_seconds"]["p50"]


def test_the_secondary_evidence_records_the_shape_metrics():
    secondary = diagnostic([86400.0] * 48)["secondary_evidence"]
    assert len(secondary["hour_counts"]) == 24
    assert len(secondary["weekday_counts"]) == 7
    assert sum(secondary["hour_counts"]) == 48
    assert set(secondary["chi_square"]) == {"statistic", "p_value", "degrees_of_freedom"}
    assert secondary["chi_square"]["degrees_of_freedom"] == 23
    assert 0.0 <= secondary["hour_concentration"] <= 1.0


# --- D22: the one distributional threshold, downgrade only -------------------


def test_the_downgrade_threshold_is_the_committed_value():
    assert HOUR_CONCENTRATION_DOWNGRADE == 0.50
    assert VERDICT_RULE["hour_concentration_downgrade_at"] == 0.50


def test_extreme_concentration_downgrades_a_supported_verdict():
    hours = [9] * 60 + spread_hours(40)
    d = diagnostic([86400.0] * 100, hours=hours)
    assert d["secondary_evidence"]["hour_concentration"] >= 0.50
    assert d["verdict"] == INSUFFICIENT
    assert d["secondary_evidence"]["downgraded_verdict"] is True
    assert d["verdict_branch"] == SUPPORTED, "the primary branch is still recorded"


def test_extreme_concentration_never_produces_the_strong_verdict():
    """A uniform-delta corpus that is merely concentrated stays at doubt, not
    at an assertion about provenance."""
    hours = [9] * 90 + spread_hours(10)
    d = diagnostic([86400.0] * 100, hours=hours)
    assert d["secondary_evidence"]["hour_concentration"] >= 0.90
    assert d["verdict"] == INSUFFICIENT
    assert d["verdict"] != STRONGLY_SUSPICIOUS


def test_a_uniform_histogram_with_healthy_deltas_is_not_strongly_suspicious():
    """Distribution shape alone cannot establish artifact status."""
    d = diagnostic([86400.0] * 96, hours=[i % 24 for i in range(96)])
    concentration = d["secondary_evidence"]["hour_concentration"]
    assert concentration == pytest.approx(1 / 24, abs=1e-9)
    assert d["verdict"] == SUPPORTED
    assert d["verdict"] != STRONGLY_SUSPICIOUS


def test_concentration_below_the_threshold_has_no_effect():
    # Filler hours deliberately avoid 9, so the busiest hour is exactly the 40.
    hours = [9] * 40 + [(i % 14) + 10 for i in range(60)]
    d = diagnostic([86400.0] * 100, hours=hours)
    assert d["secondary_evidence"]["hour_concentration"] < 0.50
    assert d["verdict"] == SUPPORTED
    assert d["secondary_evidence"]["downgraded_verdict"] is False


def test_concentration_cannot_upgrade_a_strongly_suspicious_verdict():
    d = diagnostic([3.0] * 100, hours=[9] * 100)
    assert d["secondary_evidence"]["hour_concentration"] == 1.0
    assert d["verdict"] == STRONGLY_SUSPICIOUS
    assert d["secondary_evidence"]["downgraded_verdict"] is False


def test_concentration_does_not_change_an_already_insufficient_verdict():
    d = diagnostic([600.0] * 100, hours=[9] * 100)
    assert d["verdict"] == INSUFFICIENT
    assert d["secondary_evidence"]["downgraded_verdict"] is False


def test_hour_concentration_is_the_largest_count_over_the_total():
    d = diagnostic([86400.0] * 10, hours=[3] * 4 + [5] * 3 + [7] * 3)
    assert d["secondary_evidence"]["hour_concentration"] == pytest.approx(0.4)


# --- sources without a testable pair -----------------------------------------


def test_a_source_with_no_testable_pair_is_insufficient_by_construction():
    d = diagnostic(None, total=100, source="nyc311")
    assert d["verdict"] == INSUFFICIENT
    assert d["not_directly_testable"] is True
    assert d["primary_evidence"]["available"] is False
    assert d["primary_evidence"]["evidence_class"] == "field_delta"
    assert "reason" in d["primary_evidence"]


def test_a_source_with_no_pair_still_carries_secondary_evidence():
    d = diagnostic(None, total=48, source="nyc311")
    assert d["secondary_evidence"]["evidence_class"] == "distributional_anomaly"
    assert sum(d["secondary_evidence"]["hour_counts"]) == 48


def test_no_pair_is_never_upgraded_by_a_clean_histogram():
    d = diagnostic(None, total=96, source="nyc311")
    assert d["verdict"] == INSUFFICIENT


def test_the_diagnostic_never_claims_shape_proves_provenance():
    d = diagnostic([86400.0] * 100)
    blob = json.dumps(d).lower()
    assert "proof" not in blob and "proves" not in blob
    assert "fraud" not in blob
    assert d["secondary_evidence"]["evidence_class"] == "distributional_anomaly"


# --- the diagnostic reaches the manifest -------------------------------------


def test_every_manifest_carries_a_diagnostic(tmp_path):
    manifest, _ = a_cfpb_run(tmp_path, [[cfpb_row("1"), cfpb_row("2")]])
    d = manifest.timestamp_diagnostic
    assert d["primary_evidence"]["evidence_class"] == "field_delta"
    assert d["secondary_evidence"]["evidence_class"] == "distributional_anomaly"
    assert d["verdict_rule"] == VERDICT_RULE


def test_the_diagnostic_survives_into_the_manifest_on_disk(tmp_path):
    manifest, _ = a_cfpb_run(tmp_path, [[cfpb_row("1"), cfpb_row("2")]])
    assert read_manifest("cfpb", root=tmp_path / "corpus").timestamp_diagnostic == (
        manifest.timestamp_diagnostic
    )


def test_a_cfpb_corpus_with_seconds_apart_timestamps_is_flagged(tmp_path):
    rows = [
        cfpb_row(
            str(i),
            received=f"2024-03-{(i % 28) + 1:02d}T{i % 24:02d}:00:00-04:00",
            sent=f"2024-03-{(i % 28) + 1:02d}T{i % 24:02d}:00:03-04:00",
        )
        for i in range(1, 25)
    ]
    manifest, _ = a_cfpb_run(tmp_path, [rows])
    d = manifest.timestamp_diagnostic
    assert d["primary_evidence"]["median_delta_seconds"] == 3.0
    assert d["verdict"] == STRONGLY_SUSPICIOUS


def test_a_nyc311_manifest_is_not_directly_testable(tmp_path):
    fetcher = StubFetcher([[nyc311_row(str(i)) for i in range(1, 4)]])
    manifest = ingest(
        source="nyc311",
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        limit=None,
        fetcher=fetcher,
        corpus_root=tmp_path / "corpus",
        raw_root=tmp_path / "raw",
    )
    d = manifest.timestamp_diagnostic
    assert d["not_directly_testable"] is True
    assert d["verdict"] == INSUFFICIENT
    assert d["primary_evidence"]["available"] is False


def test_the_311_histogram_uses_new_york_local_hours(tmp_path):
    """§2.4 binds submitted_hour to the local representation for this source."""
    fetcher = StubFetcher(
        [[nyc311_row(str(i), created="2024-03-15T09:00:00.000") for i in range(1, 4)]]
    )
    manifest = ingest(
        source="nyc311",
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        limit=None,
        fetcher=fetcher,
        corpus_root=tmp_path / "corpus",
        raw_root=tmp_path / "raw",
    )
    counts = manifest.timestamp_diagnostic["secondary_evidence"]["hour_counts"]
    assert counts[9] == 3, "local 09:00, not the 13:00 UTC instant"
    assert counts[13] == 0


# --- boundaries --------------------------------------------------------------


def test_the_cli_module_imports_without_django():
    import ast

    tree = ast.parse(Path("ingest/cli.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    for module in imported:
        assert module.split(".")[0] != "django", module
        assert not module.startswith("ml."), module


def test_ingest_opens_no_socket(tmp_path, monkeypatch):
    import socket

    def boom(*args, **kwargs):
        raise AssertionError("ingestion must reach the network only through the fetcher")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    manifest, _ = a_cfpb_run(tmp_path, [[cfpb_row("1")]])
    assert manifest.record_count == 1


def test_the_default_raw_root_is_under_the_gitignored_data_tree():
    assert RAW_ROOT.parts[0] == "data"


def test_the_cli_computes_no_corpus_id_of_its_own():
    """Corpus identity stays Task 4's; this module must not invent another.

    Inspects code, not text. D26's refusal message *reads* an existing
    manifest's `corpus_id`, which is not computing one; what stays forbidden is
    binding, passing or deriving a `corpus_id`, touching `compute_corpus_id`, or
    hashing anywhere except the raw-page checksum.
    """
    tree = ast.parse(Path("ingest/cli.py").read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            assert node.id not in {"corpus_id", "compute_corpus_id"}, node.id
        elif isinstance(node, ast.alias):
            assert node.name != "compute_corpus_id"
        elif isinstance(node, ast.keyword):
            assert node.arg != "corpus_id", "the CLI must not supply a corpus_id"
        elif isinstance(node, ast.Attribute):
            assert node.attr != "compute_corpus_id"
            if node.attr == "corpus_id":
                assert isinstance(node.ctx, ast.Load), "a corpus_id may be read, never set"

    def sha256_calls(scope):
        return [
            n
            for n in ast.walk(scope)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "sha256"
        ]

    in_page_checksum = [
        call
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef) and fn.name == "page_checksum"
        for call in sha256_calls(fn)
    ]
    assert in_page_checksum, "the guard must find the one legitimate hash"
    assert len(sha256_calls(tree)) == len(in_page_checksum), "hashing outside page_checksum"


def test_records_outside_the_window_are_excluded(tmp_path):
    """The 2024-2025 scope is enforced here, never in the adapter."""
    rows = [
        cfpb_row("in", received="2024-06-01T09:00:00-04:00"),
        cfpb_row("old", received="2023-06-01T09:00:00-04:00"),
        cfpb_row("new", received="2026-06-01T09:00:00-04:00"),
    ]
    manifest, _ = a_cfpb_run(tmp_path, [rows])
    ids = [r.external_id for r in read_corpus("cfpb", root=tmp_path / "corpus")]
    assert ids == ["in"]
    assert manifest.record_count == 1


def test_the_window_comes_from_the_arguments_not_a_constant(tmp_path):
    rows = [
        cfpb_row("a", received="2024-06-01T09:00:00-04:00"),
        cfpb_row("b", received="2025-06-01T09:00:00-04:00"),
    ]
    manifest = ingest(
        source="cfpb",
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        limit=None,
        fetcher=StubFetcher([cfpb_page(rows)]),
        corpus_root=tmp_path / "corpus",
        raw_root=tmp_path / "raw",
    )
    assert manifest.record_count == 1
    # D25: CFPB resolves in UTC, and --end includes its whole day.
    assert manifest.window_start == datetime(2024, 1, 1, tzinfo=UTC)
    assert manifest.window_end == datetime(2024, 12, 31, 23, 59, 59, 999999, tzinfo=UTC)


# --- each threshold isolated at the rule's own seam ---------------------------
#
# On real delta data `median <= 60` always implies `frac_delta_le_1min >= 0.50`,
# so a corpus-shaped test can never show which of the two branches fired. These
# drive `decide_verdict` with hand-built metrics -- combinations the arithmetic
# would not produce together -- so each threshold in VERDICT_RULE is pinned on
# its own and a changed number cannot hide behind a neighbouring branch.


def metrics(**overrides):
    base = {
        "evidence_class": "field_delta",
        "available": True,
        "pair_coverage": 1.0,
        "median_delta_seconds": 7200.0,
        "frac_delta_le_1min": 0.0,
        "frac_delta_le_10min": 0.0,
        "frac_delta_le_1h": 0.0,
        "count_delta_negative": 0,
        "count_delta_zero": 0,
        "frac_identical_timestamps": 0.0,
    }
    base.update(overrides)
    return base


def test_the_strong_median_threshold_is_sixty_seconds_exactly():
    assert decide_verdict(metrics(median_delta_seconds=60.0))[0] == STRONGLY_SUSPICIOUS
    assert decide_verdict(metrics(median_delta_seconds=61.0))[0] != STRONGLY_SUSPICIOUS


def test_the_strong_sub_minute_fraction_threshold_is_one_half_exactly():
    assert decide_verdict(metrics(frac_delta_le_1min=0.50))[0] == STRONGLY_SUSPICIOUS
    assert decide_verdict(metrics(frac_delta_le_1min=0.49))[0] != STRONGLY_SUSPICIOUS


def test_the_identical_timestamp_threshold_is_one_fifth_exactly():
    assert decide_verdict(metrics(frac_identical_timestamps=0.20))[0] == STRONGLY_SUSPICIOUS
    assert decide_verdict(metrics(frac_identical_timestamps=0.19))[0] != STRONGLY_SUSPICIOUS


def test_the_supported_median_threshold_is_one_hour_exactly():
    assert decide_verdict(metrics(median_delta_seconds=3600.0))[0] == SUPPORTED
    assert decide_verdict(metrics(median_delta_seconds=3599.0))[0] == INSUFFICIENT


def test_the_supported_sub_minute_ceiling_is_five_percent_exactly():
    assert decide_verdict(metrics(frac_delta_le_1min=0.049))[0] == SUPPORTED
    assert decide_verdict(metrics(frac_delta_le_1min=0.05))[0] == INSUFFICIENT


def test_a_single_negative_delta_blocks_the_supported_branch():
    assert decide_verdict(metrics(count_delta_negative=0))[0] == SUPPORTED
    assert decide_verdict(metrics(count_delta_negative=1))[0] == INSUFFICIENT


def test_the_coverage_floor_is_one_half_exactly():
    assert decide_verdict(metrics(pair_coverage=0.50))[0] == SUPPORTED
    verdict, branch = decide_verdict(metrics(pair_coverage=0.49))
    assert verdict == INSUFFICIENT
    assert branch == "pair_coverage_below_threshold"


def test_low_coverage_overrides_even_a_strongly_suspicious_delta_set():
    """§2.3: below half coverage the verdict is insufficient, whatever the
    deltas say. It is an override, not one branch among several."""
    verdict, branch = decide_verdict(metrics(pair_coverage=0.10, median_delta_seconds=3.0))
    assert verdict == INSUFFICIENT
    assert branch == "pair_coverage_below_threshold"


def test_the_branch_is_recorded_so_a_reader_sees_which_fired():
    assert decide_verdict(metrics(median_delta_seconds=3.0))[1] == STRONGLY_SUSPICIOUS
    assert decide_verdict(metrics())[1] == SUPPORTED
    assert decide_verdict(metrics(median_delta_seconds=600.0))[1] == "no_branch_matched"
    assert decide_verdict({"available": False})[1] == "no_testable_pair"


# =============================================================================
# D23, D24, D25 -- the three contract corrections
# =============================================================================


# --- D23: an empty window is a typed failure ---------------------------------


def test_an_empty_cache_raises_empty_window(tmp_path):
    with pytest.raises(EmptyWindow) as exc:
        ingest(
            source="cfpb",
            start=date(2024, 1, 1),
            end=date(2025, 12, 31),
            limit=None,
            fetcher=None,
            corpus_root=tmp_path / "corpus",
            raw_root=tmp_path / "raw",
        )
    message = str(exc.value)
    assert "cfpb" in message
    assert "0 cached pages" in message or "0 page" in message
    assert "zero records" in message


def test_a_cache_whose_records_all_fall_outside_the_window_raises(tmp_path):
    """Distinguishable from an empty cache: pages were read, none qualified."""
    with pytest.raises(EmptyWindow) as exc:
        ingest(
            source="cfpb",
            start=date(2024, 1, 1),
            end=date(2024, 12, 31),
            limit=None,
            fetcher=StubFetcher([cfpb_page([cfpb_row("1", received="2019-06-01T09:00:00-04:00")])]),
            corpus_root=tmp_path / "corpus",
            raw_root=tmp_path / "raw",
        )
    message = str(exc.value)
    assert "1 cached page" in message, f"the page count separates the two cases: {message}"
    assert "2024-01-01" in message and "2024-12-31" in message


def test_empty_window_names_the_resolved_bounds_not_the_supplied_dates(tmp_path):
    with pytest.raises(EmptyWindow) as exc:
        ingest(
            source="nyc311",
            start=date(2024, 6, 1),
            end=date(2024, 6, 1),
            limit=None,
            fetcher=None,
            corpus_root=tmp_path / "corpus",
            raw_root=tmp_path / "raw",
        )
    # June is EDT, so a New York day resolves to 04:00Z .. 03:59:59Z.
    assert "04:00" in str(exc.value)


def test_empty_window_writes_neither_partition_nor_manifest(tmp_path):
    corpus = tmp_path / "corpus"
    outside = cfpb_page([cfpb_row("1", received="2019-06-01T09:00:00-04:00")])
    for fetcher in (None, StubFetcher([outside])):
        with pytest.raises(EmptyWindow):
            ingest(
                source="cfpb",
                start=date(2024, 1, 1),
                end=date(2024, 12, 31),
                limit=None,
                fetcher=fetcher,
                corpus_root=corpus,
                raw_root=tmp_path / "raw",
            )
        assert iter_part_files("cfpb", root=corpus) == []
        assert not (corpus / "cfpb" / f"v{SCHEMA_VERSION}" / "manifest.json").exists()


def test_empty_window_is_an_ingest_error_not_a_bare_value_error(tmp_path):
    """D23: Task 7's undefined-intersection guard must not be the operator's
    error. `derive_roster` is never reached with no years."""
    assert issubclass(EmptyWindow, IngestError)
    with pytest.raises(IngestError):
        ingest(
            source="cfpb",
            start=date(2024, 1, 1),
            end=date(2024, 12, 31),
            limit=None,
            fetcher=None,
            corpus_root=tmp_path / "corpus",
            raw_root=tmp_path / "raw",
        )


# --- D24: --limit bounds persistence only ------------------------------------


def test_roster_validation_sees_the_whole_window_not_the_limited_subset(tmp_path):
    """D24's invariant, proven on a run D26 permits: a limited run into a fresh root.

    The page holds Alpha in both years and `OnlyIn2025` in 2025 alone. `--limit 1`
    keeps just the first Alpha. Validated against that kept subset the run would
    pass — one label, one year, nothing unexpected. Validated against the
    complete window, `OnlyIn2025` is outside the derived intersection and must
    fail, before anything is written.
    """
    corpus = tmp_path / "corpus"
    assert not (corpus / "cfpb" / f"v{SCHEMA_VERSION}" / "manifest.json").exists()

    with pytest.raises(RosterMismatch) as exc:
        ingest(
            source="cfpb",
            start=date(2024, 1, 1),
            end=date(2025, 12, 31),
            limit=1,
            fetcher=StubFetcher(
                [
                    cfpb_page(
                        [
                            cfpb_row("1", product="Alpha", received="2024-03-15T09:00:00-04:00"),
                            cfpb_row("2", product="Alpha", received="2025-03-15T09:00:00-04:00"),
                            cfpb_row(
                                "3", product="OnlyIn2025", received="2025-04-15T09:00:00-04:00"
                            ),
                        ]
                    )
                ]
            ),
            corpus_root=corpus,
            raw_root=tmp_path / "raw",
        )
    assert exc.value.unexpected == {"OnlyIn2025": 1}
    assert iter_part_files("cfpb", root=corpus) == []


def test_a_limited_manifest_never_becomes_the_authoritative_roster(tmp_path):
    """D24: the lock comes only from an unbounded corpus."""
    corpus = tmp_path / "corpus"
    pages = [cfpb_page([cfpb_row("1", product="Alpha"), cfpb_row("2", product="Beta")])]

    limited = ingest(
        source="cfpb",
        start=date(2024, 1, 1),
        end=date(2025, 12, 31),
        limit=1,
        fetcher=StubFetcher(pages),
        corpus_root=corpus,
        raw_root=tmp_path / "raw",
    )
    assert limited.limit == 1
    assert set(limited.label_roster) == {"Alpha"}, "the persisted subset, as §G says"

    # A later unbounded run must not be judged against that truncated view.
    full = ingest(
        source="cfpb",
        start=date(2024, 1, 1),
        end=date(2025, 12, 31),
        limit=None,
        fetcher=None,
        corpus_root=corpus,
        raw_root=tmp_path / "raw",
    )
    assert full.limit is None
    assert set(full.label_roster) == {"Alpha", "Beta"}


def test_an_unbounded_corpus_stays_authoritative_when_a_limited_run_is_separate(tmp_path):
    """A limited run against its own corpus root leaves the authoritative one
    untouched and still authoritative.

    This is the route §2.6 prescribes for a development corpus. The same-root
    case is refused outright (D26) and is tested in the D26 section below.
    """
    authoritative = tmp_path / "corpus"
    a_cfpb_run(tmp_path, [[cfpb_row("1", product="Alpha"), cfpb_row("2", product="Beta")]])
    assert authoritative_roster("cfpb", authoritative) == frozenset({"Alpha", "Beta"})

    ingest(
        source="cfpb",
        start=date(2024, 1, 1),
        end=date(2025, 12, 31),
        limit=1,
        fetcher=None,
        corpus_root=tmp_path / "dev-corpus",
        raw_root=tmp_path / "raw",
    )

    assert authoritative_roster("cfpb", authoritative) == frozenset({"Alpha", "Beta"})
    with pytest.raises(RosterMismatch) as exc:
        ingest(
            source="cfpb",
            start=date(2024, 1, 1),
            end=date(2025, 12, 31),
            limit=None,
            fetcher=StubFetcher([cfpb_page([cfpb_row("9", product="Intruder")])]),
            corpus_root=authoritative,
            raw_root=tmp_path / "later-raw",
        )
    assert exc.value.unexpected == {"Intruder": 1}


def test_authoritative_roster_selection_ignores_a_limited_manifest(tmp_path):
    corpus = tmp_path / "corpus"
    ingest(
        source="cfpb",
        start=date(2024, 1, 1),
        end=date(2025, 12, 31),
        limit=1,
        fetcher=StubFetcher([cfpb_page([cfpb_row("1", product="Alpha")])]),
        corpus_root=corpus,
        raw_root=tmp_path / "raw",
    )
    assert read_manifest("cfpb", root=corpus).limit == 1
    assert authoritative_roster("cfpb", corpus) is None, "a truncated manifest is not a lock"


def test_an_unbounded_manifest_is_the_authoritative_source(tmp_path):
    corpus = tmp_path / "corpus"
    a_cfpb_run(tmp_path, [[cfpb_row("1", product="Alpha"), cfpb_row("2", product="Beta")]])
    assert authoritative_roster("cfpb", corpus) == frozenset({"Alpha", "Beta"})


def test_the_limit_still_reaches_the_manifest_after_the_reordering(tmp_path):
    manifest, _ = a_cfpb_run(tmp_path, [[cfpb_row(str(i)) for i in range(1, 6)]], limit=2)
    assert manifest.limit == 2
    assert manifest.record_count == 2
    assert read_manifest("cfpb", root=tmp_path / "corpus").limit == 2


# --- D26: a limited run may not replace an authoritative corpus --------------


def an_authoritative_corpus(tmp_path):
    """An unbounded CFPB corpus under `tmp_path / "corpus"`; returns (root, manifest)."""
    manifest, _ = a_cfpb_run(
        tmp_path, [[cfpb_row("1", product="Alpha"), cfpb_row("2", product="Beta")]]
    )
    assert manifest.limit is None, "the fixture must be authoritative"
    return tmp_path / "corpus", manifest


def every_file_under(root):
    """Byte snapshot of every file in a corpus root: partitions and manifest."""
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_a_limited_run_into_an_authoritative_root_is_refused(tmp_path):
    """The exact failure that motivated D26: without the refusal, this run
    replaced a two-label corpus with a one-record one."""
    corpus, existing = an_authoritative_corpus(tmp_path)
    before = every_file_under(corpus)
    assert iter_part_files("cfpb", root=corpus), "there is something to destroy"

    with pytest.raises(AuthoritativeCorpusExists) as exc:
        ingest(
            source="cfpb",
            start=date(2024, 1, 1),
            end=date(2025, 12, 31),
            limit=1,
            fetcher=None,
            corpus_root=corpus,
            raw_root=tmp_path / "raw",
        )
    assert isinstance(exc.value, IngestError)

    assert every_file_under(corpus) == before, "partitions and manifest byte-identical"
    after = read_manifest("cfpb", root=corpus)
    assert after.corpus_id == existing.corpus_id
    assert after.limit is None
    assert after.record_count == 2
    assert authoritative_roster("cfpb", corpus) == frozenset({"Alpha", "Beta"})


def test_the_refusal_happens_before_any_fetch_or_write(tmp_path):
    corpus, _ = an_authoritative_corpus(tmp_path)
    later_raw = tmp_path / "later-raw"
    fetcher = StubFetcher([cfpb_page([cfpb_row("3", product="Alpha")])])

    with pytest.raises(AuthoritativeCorpusExists):
        ingest(
            source="cfpb",
            start=date(2024, 1, 1),
            end=date(2025, 12, 31),
            limit=1,
            fetcher=fetcher,
            corpus_root=corpus,
            raw_root=later_raw,
        )
    assert fetcher.calls == 0, "fetch_into_cache must not have run"
    assert fetcher.pages_yielded == 0
    assert not later_raw.exists(), "not even the raw cache may be written"


def test_the_refusal_names_the_corpus_it_protects_and_the_remedy(tmp_path):
    corpus, existing = an_authoritative_corpus(tmp_path)

    with pytest.raises(AuthoritativeCorpusExists) as exc:
        ingest(
            source="cfpb",
            start=date(2024, 1, 1),
            end=date(2025, 12, 31),
            limit=1,
            fetcher=None,
            corpus_root=corpus,
            raw_root=tmp_path / "raw",
        )
    message = str(exc.value)
    assert "cfpb" in message
    assert str(corpus) in message
    assert existing.corpus_id in message
    assert f"{existing.record_count} records" in message
    assert "must target a different root" in message


def test_a_limited_run_into_a_fresh_root_succeeds_and_records_its_limit(tmp_path):
    corpus = tmp_path / "corpus"
    assert not corpus.exists()
    manifest, _ = a_cfpb_run(tmp_path, [[cfpb_row(str(i)) for i in range(1, 4)]], limit=2)
    assert manifest.limit == 2
    assert manifest.record_count == 2
    assert read_manifest("cfpb", root=corpus).limit == 2


def test_a_limited_run_into_a_truncated_root_succeeds(tmp_path):
    """Nothing authoritative is at risk: one development corpus replaces another."""
    corpus = tmp_path / "corpus"
    first, _ = a_cfpb_run(tmp_path, [[cfpb_row(str(i)) for i in range(1, 4)]], limit=1)
    assert first.limit == 1

    second = ingest(
        source="cfpb",
        start=date(2024, 1, 1),
        end=date(2025, 12, 31),
        limit=2,
        fetcher=None,
        corpus_root=corpus,
        raw_root=tmp_path / "raw",
    )
    assert second.limit == 2
    assert second.record_count == 2
    assert read_manifest("cfpb", root=corpus).limit == 2


def test_an_unbounded_rerun_over_an_authoritative_corpus_is_still_idempotent(tmp_path):
    """The narrowness guard: the resume path must not be caught by the refusal."""
    corpus, first = an_authoritative_corpus(tmp_path)
    before = every_file_under(corpus)

    second = ingest(
        source="cfpb",
        start=date(2024, 1, 1),
        end=date(2025, 12, 31),
        limit=None,
        fetcher=None,
        corpus_root=corpus,
        raw_root=tmp_path / "raw",
    )
    assert second.limit is None
    assert second.corpus_id == first.corpus_id
    assert {k: v for k, v in every_file_under(corpus).items() if k.suffix == ".parquet"} == {
        k: v for k, v in before.items() if k.suffix == ".parquet"
    }


def test_the_command_line_refuses_a_limited_run_into_an_authoritative_root(tmp_path, monkeypatch):
    """End to end through `main`: `--corpus-root` must actually reach the guard.

    The working directory is moved to `tmp_path`, so the default raw cache is
    empty there. Without the guard this run would reach an empty window instead.
    """
    corpus, existing = an_authoritative_corpus(tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(AuthoritativeCorpusExists):
        main(
            [
                "--source",
                "cfpb",
                "--start",
                "2024-01-01",
                "--end",
                "2025-12-31",
                "--limit",
                "1",
                "--corpus-root",
                str(corpus),
            ]
        )
    assert read_manifest("cfpb", root=corpus).corpus_id == existing.corpus_id


# --- D25: per-source civil-time window resolution ----------------------------


def test_cfpb_bounds_resolve_in_utc():
    start, end = resolve_window("cfpb", date(2024, 6, 1), date(2024, 6, 30))
    assert start == datetime(2024, 6, 1, 0, 0, 0, tzinfo=UTC)
    assert end == datetime(2024, 6, 30, 23, 59, 59, 999999, tzinfo=UTC)


def test_nyc311_bounds_resolve_in_new_york_civil_time():
    start, end = resolve_window("nyc311", date(2024, 6, 1), date(2024, 6, 30))
    # June is EDT (UTC-4): local midnight is 04:00Z.
    assert start == datetime(2024, 6, 1, 4, 0, 0, tzinfo=UTC)
    assert end == datetime(2024, 7, 1, 3, 59, 59, 999999, tzinfo=UTC)


def test_nyc311_bounds_follow_the_standard_time_offset_in_winter():
    start, _ = resolve_window("nyc311", date(2024, 1, 15), date(2024, 1, 15))
    # January is EST (UTC-5): local midnight is 05:00Z.
    assert start == datetime(2024, 1, 15, 5, 0, 0, tzinfo=UTC)


def test_a_dst_sensitive_window_uses_each_days_own_offset():
    """10 March 2024 is the spring transition: the day starts EST and ends EDT,
    so a UTC-sliced day would be an hour wrong at one end."""
    start, end = resolve_window("nyc311", date(2024, 3, 10), date(2024, 3, 10))
    assert start == datetime(2024, 3, 10, 5, 0, 0, tzinfo=UTC)
    assert end == datetime(2024, 3, 11, 3, 59, 59, 999999, tzinfo=UTC)
    assert (end - start).total_seconds() < 24 * 3600, "the day is 23 hours long"


def test_the_autumn_transition_day_is_twenty_five_hours():
    start, end = resolve_window("nyc311", date(2024, 11, 3), date(2024, 11, 3))
    assert (end - start).total_seconds() > 24 * 3600


def test_both_bounds_are_inclusive_whole_days():
    start, end = resolve_window("cfpb", date(2024, 6, 1), date(2024, 6, 1))
    assert start.hour == 0 and start.minute == 0 and start.second == 0
    assert (end.hour, end.minute, end.second) == (23, 59, 59)


def test_a_nyc311_record_at_local_midnight_is_inside_its_own_day(tmp_path):
    manifest = ingest(
        source="nyc311",
        start=date(2024, 6, 1),
        end=date(2024, 6, 1),
        limit=None,
        fetcher=StubFetcher([[nyc311_row("1", created="2024-06-01T00:00:00.000")]]),
        corpus_root=tmp_path / "corpus",
        raw_root=tmp_path / "raw",
    )
    assert manifest.record_count == 1


def test_a_nyc311_record_late_on_the_last_local_day_is_inside_the_window(tmp_path):
    """20:00 New York on 30 June is 00:00Z on 1 July. Under UTC bounds it would
    fall outside; under §2.5 it is inside its own civil day."""
    manifest = ingest(
        source="nyc311",
        start=date(2024, 6, 1),
        end=date(2024, 6, 30),
        limit=None,
        fetcher=StubFetcher([[nyc311_row("1", created="2024-06-30T20:00:00.000")]]),
        corpus_root=tmp_path / "corpus",
        raw_root=tmp_path / "raw",
    )
    assert manifest.record_count == 1
    assert manifest.window_end == datetime(2024, 7, 1, 3, 59, 59, 999999, tzinfo=UTC)


def test_a_nyc311_record_just_past_the_local_day_is_outside(tmp_path):
    with pytest.raises(EmptyWindow):
        ingest(
            source="nyc311",
            start=date(2024, 6, 1),
            end=date(2024, 6, 30),
            limit=None,
            fetcher=StubFetcher([[nyc311_row("1", created="2024-07-01T00:00:01.000")]]),
            corpus_root=tmp_path / "corpus",
            raw_root=tmp_path / "raw",
        )


def test_a_cfpb_record_at_the_utc_boundary_is_inside(tmp_path):
    manifest = ingest(
        source="cfpb",
        start=date(2024, 6, 1),
        end=date(2024, 6, 30),
        limit=None,
        fetcher=StubFetcher([cfpb_page([cfpb_row("1", received="2024-06-30T23:59:59+00:00")])]),
        corpus_root=tmp_path / "corpus",
        raw_root=tmp_path / "raw",
    )
    assert manifest.record_count == 1


def test_a_cfpb_record_just_past_the_utc_boundary_is_outside(tmp_path):
    with pytest.raises(EmptyWindow):
        ingest(
            source="cfpb",
            start=date(2024, 6, 1),
            end=date(2024, 6, 30),
            limit=None,
            fetcher=StubFetcher([cfpb_page([cfpb_row("1", received="2024-07-01T00:00:01+00:00")])]),
            corpus_root=tmp_path / "corpus",
            raw_root=tmp_path / "raw",
        )


def test_the_manifest_records_the_resolved_instants(tmp_path):
    manifest = ingest(
        source="nyc311",
        start=date(2024, 6, 1),
        end=date(2024, 6, 30),
        limit=None,
        fetcher=StubFetcher([[nyc311_row("1", created="2024-06-15T09:00:00.000")]]),
        corpus_root=tmp_path / "corpus",
        raw_root=tmp_path / "raw",
    )
    assert manifest.window_start == datetime(2024, 6, 1, 4, 0, 0, tzinfo=UTC)
    assert manifest.window_end == datetime(2024, 7, 1, 3, 59, 59, 999999, tzinfo=UTC)


def test_the_311_day_is_not_a_utc_slice(tmp_path):
    """The decisive comparison: a UTC-sliced 30 June would exclude this record,
    and a New York 30 June includes it."""
    utc_start, utc_end = resolve_window("cfpb", date(2024, 6, 30), date(2024, 6, 30))
    ny_start, ny_end = resolve_window("nyc311", date(2024, 6, 30), date(2024, 6, 30))
    late = datetime(2024, 7, 1, 0, 30, tzinfo=UTC)  # 20:30 New York on 30 June

    assert not (utc_start <= late <= utc_end)
    assert ny_start <= late <= ny_end

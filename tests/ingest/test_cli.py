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
import csv
import gzip
import hashlib
import io
import json
import re
import zipfile
from datetime import UTC, date, datetime, timedelta
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
    DuplicateExternalId,
    EmptyWindow,
    FetcherUnavailable,
    IngestError,
    InvalidDateRange,
    InvalidLimit,
    authoritative_manifest,
    authoritative_roster,
    build_diagnostic,
    build_parser,
    cache_page,
    decide_verdict,
    ingest,
    main,
    page_checksum,
    resolve_window,
)
from ingest.fetch.acquisition import (  # noqa: E402
    Acquisition,
    AcquisitionIncomplete,
    AcquisitionIntegrityError,
    record_path,
)
from ingest.fetch.canonical import canonical_bytes, page_digest  # noqa: E402
from ingest.fetch.http import ClientIdentity, FetchFailed, HttpClient, Response  # noqa: E402
from ingest.manifest import (  # noqa: E402
    ManifestNotFound,
    load_corpus,
    manifest_path,
    read_manifest,
)
from ingest.roster import RosterMismatch  # noqa: E402
from ingest.schema import SCHEMA_VERSION  # noqa: E402
from ingest.sources.cfpb import MissingNarrative  # noqa: E402
from ingest.sources.nyc311 import EXCLUSION_KINDS, MissingField  # noqa: E402
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


# --- D27: a successful run replaces the source's tree ------------------------


def assert_disk_is_the_manifest(corpus, manifest):
    """D27's post-success invariant, checked on disk and through the gated reader."""
    on_disk = {
        p.relative_to(corpus).as_posix() for p in iter_part_files(manifest.source_slug, root=corpus)
    }
    assert on_disk == set(manifest.part_files), "disk must be exactly what the manifest lists"

    loaded, stream = load_corpus(manifest.source_slug, root=corpus)
    records = list(stream)
    assert loaded == manifest
    assert len(records) == manifest.record_count
    assert all(manifest.window_start <= r.submitted_at <= manifest.window_end for r in records)


def two_years(product="Alpha"):
    return [
        cfpb_row("a", product=product, received="2024-03-15T09:00:00-04:00"),
        cfpb_row("b", product=product, received="2025-03-15T09:00:00-04:00"),
    ]


def test_a_successful_run_leaves_disk_exactly_equal_to_its_manifest(tmp_path):
    manifest, _ = a_cfpb_run(tmp_path, [two_years()])
    assert manifest.per_year_counts == {2024: 1, 2025: 1}
    assert_disk_is_the_manifest(tmp_path / "corpus", manifest)


def test_a_limited_run_into_a_truncated_root_leaves_no_stale_partition(tmp_path):
    """The observed D27 defect: `limit: 1` once recorded `record_count: 2`."""
    corpus = tmp_path / "corpus"
    first, _ = a_cfpb_run(tmp_path, [two_years()], limit=2)
    assert first.per_year_counts == {2024: 1, 2025: 1}

    second = ingest(
        source="cfpb",
        start=date(2024, 1, 1),
        end=date(2025, 12, 31),
        limit=1,
        fetcher=None,
        corpus_root=corpus,
        raw_root=tmp_path / "raw",
    )

    assert second.limit == 1
    assert second.record_count == 1 <= second.limit
    assert second.per_year_counts == {2024: 1}
    assert not any("year=2025" in part for part in second.part_files)
    assert_disk_is_the_manifest(corpus, second)


def test_a_narrower_unbounded_window_replaces_an_authoritative_corpus_completely(tmp_path):
    """The other observed defect: a 2024-only manifest kept its 2025 partition."""
    corpus = tmp_path / "corpus"
    full, _ = a_cfpb_run(tmp_path, [two_years()])
    assert full.limit is None and full.record_count == 2

    narrow = ingest(
        source="cfpb",
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        limit=None,
        fetcher=None,
        corpus_root=corpus,
        raw_root=tmp_path / "raw",
    )

    assert narrow.limit is None, "still an authoritative corpus, just a narrower one"
    assert narrow.record_count == 1
    assert narrow.per_year_counts == {2024: 1}
    assert narrow.window_end == datetime(2024, 12, 31, 23, 59, 59, 999999, tzinfo=UTC)
    assert narrow.corpus_id != full.corpus_id
    assert not (corpus / "cfpb" / f"v{SCHEMA_VERSION}" / "year=2025").exists()
    assert_disk_is_the_manifest(corpus, narrow)


def test_a_rerun_replaces_only_its_own_source_and_schema_version(tmp_path):
    corpus = tmp_path / "corpus"
    ingest(
        source="nyc311",
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        limit=None,
        fetcher=StubFetcher([[nyc311_row("1")]]),
        corpus_root=corpus,
        raw_root=tmp_path / "raw",
    )
    another_version = corpus / "cfpb" / f"v{SCHEMA_VERSION + 1}" / "year=2024" / "part-0000.parquet"
    another_version.parent.mkdir(parents=True)
    another_version.write_bytes(b"another schema version's bytes")

    target = f"cfpb/v{SCHEMA_VERSION}/"
    survivors = {
        p: p.read_bytes()
        for p in corpus.rglob("*")
        if p.is_file() and not p.relative_to(corpus).as_posix().startswith(target)
    }

    a_cfpb_run(tmp_path, [two_years()])
    ingest(
        source="cfpb",
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        limit=None,
        fetcher=None,
        corpus_root=corpus,
        raw_root=tmp_path / "raw",
    )

    for path, content in survivors.items():
        assert path.is_file() and path.read_bytes() == content, f"{path} must be untouched"


def test_the_manifest_and_old_partitions_are_gone_before_the_first_write(tmp_path, monkeypatch):
    import ingest.cli as cli_module

    corpus, _ = an_authoritative_corpus(tmp_path)
    real_write_partition = cli_module.write_partition
    observed = []

    def spy(records, source, year, part_index, root):
        observed.append(
            (manifest_path(source, root).exists(), list(iter_part_files(source, root=root)))
        )
        return real_write_partition(records, source, year, part_index, root=root)

    monkeypatch.setattr("ingest.cli.write_partition", spy)
    ingest(
        source="cfpb",
        start=date(2024, 1, 1),
        end=date(2025, 12, 31),
        limit=None,
        fetcher=None,
        corpus_root=corpus,
        raw_root=tmp_path / "raw",
    )

    assert observed, "the run wrote partitions"
    manifest_existed, parts_present = observed[0]
    assert not manifest_existed, "the manifest must be deleted before any partition write"
    assert parts_present == [], "the previous tree must be cleared before any partition write"


def test_a_failed_partition_write_leaves_no_manifest_and_a_recoverable_root(tmp_path, monkeypatch):
    import ingest.cli as cli_module

    corpus = tmp_path / "corpus"
    first, _ = a_cfpb_run(tmp_path, [two_years()])
    real_write_partition = cli_module.write_partition
    calls = []

    def fail_on_second(records, source, year, part_index, root):
        calls.append(year)
        if len(calls) == 2:
            raise OSError("disk full while writing the second partition")
        return real_write_partition(records, source, year, part_index, root=root)

    monkeypatch.setattr("ingest.cli.write_partition", fail_on_second)
    with pytest.raises(OSError):
        ingest(
            source="cfpb",
            start=date(2024, 1, 1),
            end=date(2025, 12, 31),
            limit=None,
            fetcher=None,
            corpus_root=corpus,
            raw_root=tmp_path / "raw",
        )
    monkeypatch.undo()

    assert not manifest_path("cfpb", corpus).exists(), "a failed load leaves no manifest"
    with pytest.raises(ManifestNotFound):
        load_corpus("cfpb", root=corpus)
    assert authoritative_manifest("cfpb", corpus) is None, "the root is not a corpus"

    recovered = ingest(
        source="cfpb",
        start=date(2024, 1, 1),
        end=date(2025, 12, 31),
        limit=None,
        fetcher=None,
        corpus_root=corpus,
        raw_root=tmp_path / "raw",
    )
    assert recovered.corpus_id == first.corpus_id, "rerunning from the raw cache recovers it"
    assert_disk_is_the_manifest(corpus, recovered)


@pytest.mark.parametrize("failure", ["d26_refusal", "roster_mismatch", "empty_window"])
def test_predictable_failures_happen_before_anything_is_cleared(tmp_path, monkeypatch, failure):
    """D27 clears only after every pre-write refusal has passed (§2.5)."""
    corpus, existing = an_authoritative_corpus(tmp_path)
    before = every_file_under(corpus)

    def must_not_clear(*args, **kwargs):
        raise AssertionError("the corpus was cleared before validation finished")

    monkeypatch.setattr("ingest.cli.clear_corpus", must_not_clear)

    common = {"source": "cfpb", "corpus_root": corpus}
    if failure == "d26_refusal":
        expected, run = (
            AuthoritativeCorpusExists,
            dict(
                start=date(2024, 1, 1),
                end=date(2025, 12, 31),
                limit=1,
                fetcher=None,
                raw_root=tmp_path / "raw",
            ),
        )
    elif failure == "roster_mismatch":
        expected, run = (
            RosterMismatch,
            dict(
                start=date(2024, 1, 1),
                end=date(2025, 12, 31),
                limit=None,
                fetcher=StubFetcher([cfpb_page([cfpb_row("9", product="Intruder")])]),
                raw_root=tmp_path / "later-raw",
            ),
        )
    else:
        expected, run = (
            EmptyWindow,
            dict(
                start=date(2023, 1, 1),
                end=date(2023, 12, 31),
                limit=None,
                fetcher=None,
                raw_root=tmp_path / "raw",
            ),
        )

    with pytest.raises(expected):
        ingest(**common, **run)

    assert every_file_under(corpus) == before, "byte-identical: nothing was cleared"
    assert read_manifest("cfpb", root=corpus).corpus_id == existing.corpus_id


# --- D28: --limit must be a positive integer ---------------------------------

INVALID_LIMITS = [0, -1]


def forbid_everything_after_the_limit_check(monkeypatch):
    """Patch every step D28 must precede to fail loudly if it is reached.

    A manifest lookup (the D26 check and the roster lock both start there),
    clearing, and either kind of write. The fetcher is deliberately left alone:
    whether it runs is asserted from its own call count, so that a check moved
    after `fetch_into_cache` is caught by the no-fetch invariant itself.
    """

    def reached(*args, **kwargs):
        raise AssertionError("a step D28 must precede ran before --limit was validated")

    for name in ("manifest_path", "read_manifest", "clear_corpus", "write_partition"):
        monkeypatch.setattr(f"ingest.cli.{name}", reached)
    monkeypatch.setattr("ingest.cli.write_manifest", reached)


def an_invalid_limit_run(tmp_path, limit, corpus_root, fetcher):
    return ingest(
        source="cfpb",
        start=date(2024, 1, 1),
        end=date(2025, 12, 31),
        limit=limit,
        fetcher=fetcher,
        corpus_root=corpus_root,
        raw_root=tmp_path / "later-raw",
    )


@pytest.mark.parametrize("limit", INVALID_LIMITS)
def test_an_invalid_limit_is_refused_on_a_fresh_root(tmp_path, monkeypatch, limit):
    corpus = tmp_path / "corpus"
    fetcher = StubFetcher([cfpb_page([cfpb_row("1")])])
    forbid_everything_after_the_limit_check(monkeypatch)

    with pytest.raises(InvalidLimit) as exc:
        an_invalid_limit_run(tmp_path, limit, corpus, fetcher)

    assert isinstance(exc.value, IngestError)
    message = str(exc.value)
    assert f"got {limit}" in message, "the error names the value supplied"
    assert "positive integer" in message
    assert fetcher.calls == 0
    assert not corpus.exists(), "nothing written"
    assert not (tmp_path / "later-raw").exists(), "not even the raw cache"


@pytest.mark.parametrize("limit", INVALID_LIMITS)
def test_an_invalid_limit_never_reaches_the_fetcher(tmp_path, limit):
    """No patches here: the fetcher's own call count is the evidence."""
    fetcher = StubFetcher([cfpb_page([cfpb_row("1")])])

    with pytest.raises(InvalidLimit):
        an_invalid_limit_run(tmp_path, limit, tmp_path / "corpus", fetcher)

    assert fetcher.calls == 0, "fetch_into_cache must not run"
    assert fetcher.pages_yielded == 0
    assert not (tmp_path / "later-raw").exists()


@pytest.mark.parametrize("limit", INVALID_LIMITS)
def test_an_invalid_limit_precedes_the_d26_refusal(tmp_path, monkeypatch, limit):
    """Against an authoritative root D26 would refuse too. D28 must answer
    first, and without so much as reading the manifest D26 would read."""
    corpus, existing = an_authoritative_corpus(tmp_path)
    before = every_file_under(corpus)
    fetcher = StubFetcher([cfpb_page([cfpb_row("3")])])
    forbid_everything_after_the_limit_check(monkeypatch)

    with pytest.raises(InvalidLimit):
        an_invalid_limit_run(tmp_path, limit, corpus, fetcher)
    monkeypatch.undo()

    assert fetcher.calls == 0
    assert every_file_under(corpus) == before, "the corpus is byte-identical"
    assert read_manifest("cfpb", root=corpus).corpus_id == existing.corpus_id


@pytest.mark.parametrize("root_kind", ["fresh", "authoritative"])
@pytest.mark.parametrize("limit", INVALID_LIMITS)
def test_the_command_line_reaches_the_same_limit_check(tmp_path, monkeypatch, limit, root_kind):
    if root_kind == "authoritative":
        corpus, _ = an_authoritative_corpus(tmp_path)
    else:
        corpus = tmp_path / "fresh-corpus"
    before = every_file_under(corpus) if corpus.exists() else {}
    monkeypatch.chdir(tmp_path)
    forbid_everything_after_the_limit_check(monkeypatch)

    with pytest.raises(InvalidLimit) as exc:
        main(
            [
                "--source",
                "cfpb",
                "--start",
                "2024-01-01",
                "--end",
                "2025-12-31",
                "--limit",
                str(limit),
                "--corpus-root",
                str(corpus),
            ]
        )
    monkeypatch.undo()

    assert f"got {limit}" in str(exc.value)
    assert (every_file_under(corpus) if corpus.exists() else {}) == before
    assert not (tmp_path / "data").exists(), "the default raw cache was never touched"


def test_the_parser_keeps_limit_as_a_plain_integer():
    """D28 lives in `ingest()` alone; the parser does not duplicate the rule."""
    for value in ("0", "-1", "1"):
        args = build_parser().parse_args(
            ["--source", "cfpb", "--start", "2024-01-01", "--end", "2025-12-31", "--limit", value]
        )
        assert args.limit == int(value)


def test_a_limit_of_one_is_still_accepted(tmp_path):
    manifest, _ = a_cfpb_run(tmp_path, [[cfpb_row("1"), cfpb_row("2")]], limit=1)
    assert manifest.limit == 1
    assert manifest.record_count == 1


def test_a_limit_of_one_is_still_accepted_from_the_command_line(tmp_path, monkeypatch):
    # Seed the default raw cache the command line reads, relative to tmp_path.
    ingest(
        source="cfpb",
        start=date(2024, 1, 1),
        end=date(2025, 12, 31),
        limit=None,
        fetcher=StubFetcher([cfpb_page([cfpb_row("1"), cfpb_row("2")])]),
        corpus_root=tmp_path / "seed-corpus",
        raw_root=tmp_path / RAW_ROOT,
    )
    monkeypatch.chdir(tmp_path)
    dev = tmp_path / "dev-corpus"

    code = main(
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
            str(dev),
        ]
    )

    assert code == 0
    written = read_manifest("cfpb", root=dev)
    assert written.limit == 1
    assert written.record_count == 1


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


# --- D37.1: the ingest run persists the outcome stream --------------------------------
#
# RED until the sidecar exists. `ingest/cli.py` currently normalises `(record,
# outcome)` pairs and then keeps the records alone, so the outcome stream the 311
# risk model is built on reaches no consumer at all.


def nyc311_resolved(
    external_id, *, created="2024-03-15T09:00:00.000", closed=None, complaint_type="Noise"
):
    """A 311 row that may carry a `closed_date`, and so a resolution time."""
    row = nyc311_row(external_id, created=created, complaint_type=complaint_type)
    if closed is not None:
        row["closed_date"] = closed
    return row


def a_311_run(tmp_path, rows, *, raw="raw", **kwargs):
    """Ingest one 311 page. ``raw`` names the cache, so a replacement run can be
    given a fresh one -- the cache is resumable by design, and a second run
    sharing it would legitimately re-ingest the first run's rows too."""
    fetcher = StubFetcher([rows])
    manifest = ingest(
        source="nyc311",
        start=date(2024, 1, 1),
        end=date(2025, 12, 31),
        limit=kwargs.pop("limit", None),
        fetcher=fetcher,
        corpus_root=tmp_path / "corpus",
        raw_root=tmp_path / raw,
        **kwargs,
    )
    return manifest, fetcher


def loaded_outcomes(tmp_path):
    from ingest.manifest import load_outcomes

    _, stream = load_outcomes("nyc311", root=tmp_path / "corpus")
    return list(stream)


def test_a_311_run_persists_the_outcome_stream(tmp_path):
    """D37.1. Mutation: keep `cli.py`'s `records = [record for record, _ in kept]`."""
    a_311_run(
        tmp_path,
        [
            nyc311_resolved("1", closed="2024-03-15T21:00:00.000"),
            nyc311_resolved("2"),
        ],
    )
    outcomes = loaded_outcomes(tmp_path)
    assert [o.external_id for o in outcomes] == ["1", "2"]
    assert outcomes[0].resolution_hours == 12.0
    assert outcomes[1].resolution_hours is None
    # D37: the source's own normalised close timestamp, not a derived one.
    assert outcomes[0].closed_at == datetime(2024, 3, 16, 1, 0, tzinfo=UTC)
    assert outcomes[1].closed_at is None


def test_a_311_run_preserves_the_normalised_close_timestamp(tmp_path):
    """D37: `closed_at` comes from `closed_date`, through the existing normaliser."""
    a_311_run(tmp_path, [nyc311_resolved("1", closed="2024-03-20T17:30:00.000")])
    (loaded,) = loaded_outcomes(tmp_path)
    assert loaded.closed_at == datetime(2024, 3, 20, 21, 30, tzinfo=UTC)
    assert loaded.resolution_hours is not None


def test_a_cfpb_run_persists_its_own_outcome_sidecar(tmp_path):
    """Task 19's O1 supersedes D37's scope: CFPB now persists its own sidecar.

    This test previously asserted the opposite. D37 deferred the CFPB schema to
    "the task that first consumes it"; Task 19 is that task, so the assertion is
    inverted rather than deleted and the same property stays pinned here.

    Mutation: leave `cli.py` gating outcome persistence on `source == "nyc311"`,
    which would discard the evaluation target the probe is scored against.
    """
    from ingest import storage as storage_module

    a_cfpb_run(tmp_path, [[cfpb_row("1"), cfpb_row("2")]])
    manifest = read_manifest("cfpb", root=tmp_path / "corpus")
    assert manifest.outcome_part_files, "the run declared no CFPB outcome sidecar"
    root = tmp_path / "corpus" / "cfpb" / f"v{SCHEMA_VERSION}"
    assert (root / "outcomes").is_dir()
    assert tuple(storage_module.CFPB_OUTCOME_ARROW_SCHEMA.names) == (
        "external_id",
        "timely_response",
        "date_sent_to_company",
    )


def test_the_nyc311_loader_never_deserialises_a_cfpb_sidecar(tmp_path):
    """Fail closed across sources: the typed schema check refuses those bytes.

    The other half of the inversion. A CFPB sidecar now exists, so the typed
    absence error is no longer the right answer for this corpus; what must hold
    instead is that NYC 311's loader never turns CFPB rows into `NYC311Outcome`.
    The property asserted is failure and non-deserialisation, not a particular
    exception class -- no contract fixes one for a cross-source read.
    """
    from ingest.manifest import load_outcomes
    from ingest.schema import NYC311Outcome

    a_cfpb_run(tmp_path, [[cfpb_row("1")]])
    produced: list[object] = []
    with pytest.raises(Exception):
        _, outcomes = load_outcomes("cfpb", root=tmp_path / "corpus")
        produced.extend(outcomes)
    assert produced == [], "CFPB rows were deserialised before the refusal"
    assert not any(isinstance(item, NYC311Outcome) for item in produced)


def test_records_and_outcomes_are_written_by_the_same_run(tmp_path):
    """One run, one manifest, both streams. Mutation: a second pass for outcomes."""
    manifest, _ = a_311_run(
        tmp_path, [nyc311_resolved(str(i), closed="2024-03-15T21:00:00.000") for i in range(1, 4)]
    )
    assert manifest.record_count == 3
    assert manifest.part_files and manifest.outcome_part_files
    _, records = load_corpus("nyc311", root=tmp_path / "corpus")
    assert len(list(records)) == 3
    assert len(loaded_outcomes(tmp_path)) == 3


def test_the_run_manifest_binds_both_streams(tmp_path):
    """D37.17: the two checksum maps are both populated and disjoint."""
    manifest, _ = a_311_run(tmp_path, [nyc311_resolved("1", closed="2024-03-15T21:00:00.000")])
    assert set(manifest.part_files) & set(manifest.outcome_part_files) == set()
    assert all("/outcomes/" in path for path in manifest.outcome_part_files)
    assert manifest.manifest_version == 4


def test_corpus_identity_binds_the_outcome_bytes_end_to_end(tmp_path):
    """D37.1: two runs whose records match but whose outcomes differ get different ids."""
    first, _ = a_311_run(tmp_path / "a", [nyc311_resolved("1", closed="2024-03-15T21:00:00.000")])
    second, _ = a_311_run(tmp_path / "b", [nyc311_resolved("1", closed="2024-03-16T09:00:00.000")])
    assert set(first.part_files.values()) == set(second.part_files.values())
    assert first.corpus_id != second.corpus_id


def test_a_replacement_run_leaves_no_stale_outcome_partition(tmp_path):
    """D27 + D37.17: the whole versioned tree goes, sidecar included."""
    first, _ = a_311_run(
        tmp_path,
        [nyc311_resolved(str(i), closed="2024-03-15T21:00:00.000") for i in range(1, 6)],
        raw="raw-first",
    )
    assert len(first.outcome_part_files) == 1
    manifest, _ = a_311_run(
        tmp_path,
        [nyc311_resolved("1", closed="2024-03-15T21:00:00.000")],
        raw="raw-second",
    )
    assert manifest.record_count == 1
    outcomes = loaded_outcomes(tmp_path)
    assert [o.external_id for o in outcomes] == ["1"]
    # Nothing from the five-record run may remain on disk under the sidecar.
    survivors = sorted(
        path.relative_to(tmp_path / "corpus").as_posix()
        for path in (tmp_path / "corpus" / "nyc311" / f"v{SCHEMA_VERSION}" / "outcomes").rglob(
            "*.parquet"
        )
    )
    assert survivors == sorted(manifest.outcome_part_files)


def test_every_persisted_record_has_exactly_one_persisted_outcome(tmp_path):
    """D37.1: the sidecar is written from the same truncated set as the records."""
    manifest, _ = a_311_run(
        tmp_path,
        [nyc311_resolved(str(i), closed="2024-03-15T21:00:00.000") for i in range(1, 8)],
        limit=4,
    )
    _, records = load_corpus("nyc311", root=tmp_path / "corpus")
    record_ids = [r.external_id for r in records]
    outcome_ids = [o.external_id for o in loaded_outcomes(tmp_path)]
    assert manifest.record_count == 4
    assert sorted(record_ids) == sorted(outcome_ids)
    assert len(set(outcome_ids)) == len(outcome_ids)


def test_a_cfpb_run_still_behaves_exactly_as_before(tmp_path):
    """D37.15: nothing about the existing CFPB path changes.

    CFPB has an outcome stream of its own, but D37.1 fixes a sidecar schema for
    NYC 311 only, so this pins the status quo rather than inventing one.
    """
    manifest, _ = a_cfpb_run(tmp_path, [[cfpb_row("1"), cfpb_row("2")]])
    assert manifest.record_count == 2
    assert manifest.part_files
    _, records = load_corpus("cfpb", root=tmp_path / "corpus")
    assert [r.external_id for r in records] == ["1", "2"]


def test_a_cfpb_corpus_remains_loadable_without_a_311_sidecar(tmp_path):
    """A record-only corpus is still a corpus for every record-only consumer."""
    a_cfpb_run(tmp_path, [[cfpb_row("1")]])
    manifest = read_manifest("cfpb", root=tmp_path / "corpus")
    assert manifest.schema_version == SCHEMA_VERSION == 1
    assert manifest.manifest_version == 4


# --- Task 23: the acquisition layer (addendum D44, D46) ----------------------------------
#
# Every transport below is a scripted fake: nothing here reaches the network, and the
# `closed_network` fixture holds every test that fetches to that.

ACQ_START, ACQ_END = date(2024, 1, 1), date(2024, 1, 3)
ACQ_DAYS = ("2024-01-01", "2024-01-02", "2024-01-03")
TEST_COMMIT = "c" * 40
TEST_CLIENT = ClientIdentity(user_agent="Sentinel-test/0", library="fake", library_version="0")
FIXED_NOW = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)


@pytest.fixture
def closed_network(monkeypatch):
    import socket

    def refuse(*args, **kwargs):
        raise AssertionError("a Task 23 test attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def day_rows(day):
    return [
        {
            "unique_key": f"{day}-{i}",
            "created_date": f"{day}T09:00:00.000",
            "complaint_type": "Noise",
            "descriptor": "Loud",
        }
        for i in range(2)
    ]


class DayTransport:
    """Serves one day's rows per request and records the days asked for."""

    def __init__(self, forbid=()):
        self.days = []
        self.forbid = set(forbid)

    def get(self, url, params, headers):
        day = dict(params)["day"]
        self.days.append(day)
        if day in self.forbid:
            return Response(status=403, headers={}, body=b"")
        return Response(status=200, headers={}, body=json.dumps(day_rows(day)).encode())


def quiet_client(transport):
    return HttpClient(
        transport, clock=lambda: 0.0, sleep=lambda seconds: None, now=lambda: FIXED_NOW
    )


def day_fetcher(directory, http, *, commit=TEST_COMMIT, client=TEST_CLIENT):
    """A test-only fetcher over the acquisition API: one slice per day, resumable."""

    def fetch(source, start, end):
        window_start, window_end = resolve_window(source, start, end)
        with Acquisition(
            directory,
            source=source,
            start=start,
            end=end,
            resolved_start=window_start,
            resolved_end=window_end,
            client=client,
            sentinel_commit=commit,
            now=lambda: FIXED_NOW,
        ) as acquisition:
            done = acquisition.completed_slices()
            for day in ACQ_DAYS:
                if day in done:
                    continue
                fetched = http.get("https://example.test/311", [("day", day)])
                page = json.loads(fetched.response.body)
                yield page
                acquisition.record_slice(
                    day, requests=[fetched.record], pages=[page_digest(page)], verification={}
                )
            acquisition.complete({})

    return fetch


def acquired(tmp_path, *, name="acq", transport=None, fetcher=None, corpus="corpus"):
    directory = tmp_path / name
    transport = transport or DayTransport()
    manifest = ingest(
        source="nyc311",
        start=ACQ_START,
        end=ACQ_END,
        limit=None,
        fetcher=fetcher or day_fetcher(directory, quiet_client(transport)),
        corpus_root=tmp_path / corpus,
        acquisition=directory,
    )
    return manifest, directory, transport


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def never_called(source, start, end):
    raise AssertionError("the fetcher must not be called")
    yield  # pragma: no cover


def test_the_page_digest_is_the_cli_page_checksum():
    pages = [
        {"hits": {"hits": [{"_source": {"b": 1, "a": "café"}}]}},
        [{"unique_key": "1", "descriptor": "ñ ✓"}, {"unique_key": "2"}],
        [],
        {"nested": [1, [2, {"z": None, "y": True}]]},
    ]
    for page in pages:
        assert page_digest(page) == page_checksum(page)


def test_a_page_write_killed_part_way_leaves_nothing_at_its_name(tmp_path, monkeypatch):
    import ingest.cli as cli

    def dies_part_way(value, stream, **kwargs):
        stream.write("[{")
        raise RuntimeError("killed mid-write")

    monkeypatch.setattr(cli.json, "dump", dies_part_way)
    with pytest.raises(RuntimeError):
        cache_page([{"unique_key": "1"}], "nyc311", tmp_path / "raw")
    assert list((tmp_path / "raw" / "nyc311").iterdir()) == []


@pytest.mark.parametrize(
    "damage",
    [
        lambda path: path.write_bytes(path.read_bytes()[:12]),
        lambda path: path.write_bytes(gzip.compress(b'[{"unique_key":"other"}]')),
        lambda path: path.write_bytes(b"not gzip at all"),
    ],
)
def test_a_cached_page_that_does_not_match_its_name_is_replaced(tmp_path, damage):
    page = [{"unique_key": "1", "descriptor": "d"}]
    path, _ = cache_page(page, "nyc311", tmp_path / "raw")
    damage(path)

    again, written = cache_page(page, "nyc311", tmp_path / "raw")
    assert again == path and written is True
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        assert page_checksum(json.load(handle)) == page_checksum(page)
    assert [p.name for p in path.parent.iterdir()] == [path.name], "no temporary file remains"


def test_a_duplicate_external_id_refuses_before_any_write(tmp_path):
    corpus = tmp_path / "corpus"
    a_311_run(tmp_path, [nyc311_row("1"), nyc311_row("2")])
    before = snapshot(corpus)

    later = nyc311_row("7", created="2024-03-16T10:00:00.000")
    with pytest.raises(DuplicateExternalId, match="nyc311:7") as caught:
        ingest(
            source="nyc311",
            start=date(2024, 1, 1),
            end=date(2024, 12, 31),
            limit=None,
            fetcher=StubFetcher([[nyc311_row("7"), nyc311_row("8")], [later]]),
            corpus_root=corpus,
            raw_root=tmp_path / "later-raw",
        )
    assert isinstance(caught.value, IngestError)
    assert snapshot(corpus) == before


def test_a_duplicate_within_one_cfpb_page_refuses_on_a_first_run(tmp_path):
    with pytest.raises(DuplicateExternalId, match="cfpb:5"):
        a_cfpb_run(tmp_path, [[cfpb_row("5"), cfpb_row("5"), cfpb_row("6")]])
    corpus = tmp_path / "corpus"
    assert not corpus.exists() or not list(corpus.rglob("*.*"))


def test_a_run_without_an_acquisition_records_no_acquisition_id(tmp_path):
    manifest, _ = a_cfpb_run(tmp_path, [[cfpb_row("1")]])
    assert manifest.acquisition_id is None
    assert read_manifest("cfpb", root=tmp_path / "corpus").acquisition_id is None


def test_an_acquisition_run_records_the_sha256_of_its_record(tmp_path, closed_network):
    manifest, directory, transport = acquired(tmp_path)
    data = record_path(directory).read_bytes()
    assert manifest.acquisition_id == hashlib.sha256(data).hexdigest()
    assert read_manifest("nyc311", root=tmp_path / "corpus").acquisition_id == (
        manifest.acquisition_id
    )
    assert canonical_bytes(json.loads(data)) == data
    assert transport.days == list(ACQ_DAYS)
    assert manifest.record_count == 6
    _, records = load_corpus("nyc311", root=tmp_path / "corpus")
    assert sorted(r.external_id for r in records) == sorted(
        row["unique_key"] for day in ACQ_DAYS for row in day_rows(day)
    )


def test_a_completed_acquisition_is_reused_with_zero_requests(tmp_path, closed_network):
    first, directory, _ = acquired(tmp_path)
    record_before = record_path(directory).read_bytes()
    stamp = record_path(directory).stat().st_mtime_ns

    transport = DayTransport()
    second, _, _ = acquired(tmp_path, transport=transport, fetcher=never_called)

    assert transport.days == []
    assert second.acquisition_id == first.acquisition_id
    assert second.corpus_id == first.corpus_id
    assert record_path(directory).read_bytes() == record_before
    assert record_path(directory).stat().st_mtime_ns == stamp


def test_an_interrupted_acquisition_resumes_from_its_journal(tmp_path, closed_network):
    clean, clean_dir, _ = acquired(tmp_path, name="clean", corpus="clean-corpus")

    broken = DayTransport(forbid={"2024-01-02"})
    with pytest.raises(FetchFailed, match="403"):
        acquired(tmp_path, transport=broken)
    directory = tmp_path / "acq"
    assert broken.days == ["2024-01-01", "2024-01-02"]
    assert not record_path(directory).exists(), "an incomplete acquisition has no record"
    corpus = tmp_path / "corpus"
    assert not corpus.exists() or not list(corpus.rglob("manifest.json"))

    resumed_transport = DayTransport()
    resumed, _, _ = acquired(tmp_path, transport=resumed_transport)
    assert resumed_transport.days == ["2024-01-02", "2024-01-03"], "journaled slices skipped"
    assert record_path(directory).read_bytes() == record_path(clean_dir).read_bytes()
    assert resumed.acquisition_id == clean.acquisition_id
    assert resumed.corpus_id == clean.corpus_id


def test_an_incomplete_acquisition_refuses_before_any_corpus_write(tmp_path, closed_network):
    corpus = tmp_path / "corpus"
    a_311_run(tmp_path, [nyc311_row("1")])
    before = snapshot(corpus)
    with pytest.raises(AcquisitionIncomplete):
        ingest(
            source="nyc311",
            start=ACQ_START,
            end=ACQ_END,
            limit=None,
            fetcher=None,
            corpus_root=corpus,
            acquisition=tmp_path / "never-acquired",
        )
    assert snapshot(corpus) == before
    assert not (tmp_path / "never-acquired").exists()


def _drop_a_page(directory):
    next((directory / "nyc311").glob("*.json.gz")).unlink()


def _add_an_unlisted_page(directory):
    path = directory / "nyc311" / ("e" * 64 + ".json.gz")
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump([], handle)


def _corrupt_a_page(directory):
    path = next((directory / "nyc311").glob("*.json.gz"))
    path.write_bytes(gzip.compress(b'[{"unique_key":"forged"}]'))


def _pretty_print_the_record(directory):
    path = record_path(directory)
    path.write_bytes(json.dumps(json.loads(path.read_bytes()), indent=2).encode() + b"\n")


@pytest.mark.parametrize(
    "damage", [_drop_a_page, _add_an_unlisted_page, _corrupt_a_page, _pretty_print_the_record]
)
def test_a_damaged_acquisition_refuses_before_any_corpus_write(tmp_path, closed_network, damage):
    _, directory, _ = acquired(tmp_path, corpus="scratch-corpus")
    damage(directory)
    corpus = tmp_path / "corpus"
    a_311_run(tmp_path, [nyc311_row("1")])
    before = snapshot(corpus)

    with pytest.raises(AcquisitionIntegrityError):
        acquired(tmp_path, fetcher=never_called)
    assert snapshot(corpus) == before


def test_an_acquisition_of_another_window_refuses(tmp_path, closed_network):
    _, directory, _ = acquired(tmp_path)
    with pytest.raises(AcquisitionIntegrityError, match="window"):
        ingest(
            source="nyc311",
            start=ACQ_START,
            end=date(2024, 1, 2),
            limit=None,
            fetcher=never_called,
            corpus_root=tmp_path / "other-corpus",
            acquisition=directory,
        )
    assert not (tmp_path / "other-corpus").exists()


@pytest.mark.parametrize("source", ["cfpb", "nyc311"])
def test_fetch_refuses_before_any_side_effect_while_no_fetcher_is_registered(
    tmp_path, monkeypatch, closed_network, source
):
    import ingest.cli as cli

    def forbidden(*args, **kwargs):
        raise AssertionError("nothing may be built or run before the refusal")

    # Tasks 24 and 25 registered both sources, so the refusal is reached by removing one.
    monkeypatch.delitem(cli.FETCHERS, source)
    monkeypatch.chdir(tmp_path)
    for name in ("RequestsTransport", "HttpClient", "FetchContext", "current_commit", "ingest"):
        monkeypatch.setattr(cli, name, forbidden)

    with pytest.raises(FetcherUnavailable, match=source) as caught:
        main(["--source", source, "--start", "2024-01-01", "--end", "2024-01-03", "--fetch"])
    assert isinstance(caught.value, IngestError)
    assert list(tmp_path.iterdir()) == [], "no directory created and nothing written"


def test_fetch_builds_the_real_transport_only_in_main_and_uses_the_acquisition_directory(
    tmp_path, monkeypatch, closed_network
):
    import ingest.cli as cli

    built = []

    class FakeRequestsTransport(DayTransport):
        def __init__(self):
            super().__init__()
            self.identity = TEST_CLIENT
            built.append(self)

    contexts = []

    def factory(context):
        contexts.append(context)
        return day_fetcher(context.directory, context.http, commit=context.sentinel_commit)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "RequestsTransport", FakeRequestsTransport)
    monkeypatch.setattr(cli, "HttpClient", quiet_client)
    monkeypatch.setattr(cli, "current_commit", lambda: TEST_COMMIT)
    monkeypatch.setitem(cli.FETCHERS, "nyc311", factory)

    code = main(
        [
            "--source",
            "nyc311",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-03",
            "--corpus-root",
            str(tmp_path / "corpus"),
            "--fetch",
        ]
    )
    assert code == 0
    assert len(built) == 1 and built[0].days == list(ACQ_DAYS)
    context = contexts[0]
    assert context.directory == Path("data/acquisitions/nyc311/2024-01-01_2024-01-03")
    assert (context.source, context.start, context.end) == ("nyc311", ACQ_START, ACQ_END)
    assert (context.resolved_start, context.resolved_end) == resolve_window(
        "nyc311", ACQ_START, ACQ_END
    )
    assert context.client == TEST_CLIENT and context.sentinel_commit == TEST_COMMIT
    record = tmp_path / context.directory / "acquisition.json"
    manifest = read_manifest("nyc311", root=tmp_path / "corpus")
    assert manifest.acquisition_id == hashlib.sha256(record.read_bytes()).hexdigest()


def test_without_fetch_the_command_line_passes_exactly_the_arguments_it_always_did(monkeypatch):
    seen = {}
    monkeypatch.setattr("ingest.cli.ingest", lambda **kw: seen.update(kw))
    main(["--source", "cfpb", "--start", "2024-01-01", "--end", "2025-12-31"])
    assert set(seen) == {"source", "start", "end", "limit", "corpus_root"}
    arguments = ["--source", "cfpb", "--start", "2024-01-01", "--end", "2024-01-02"]
    assert build_parser().parse_args(arguments).fetch is False


# --- Task 24: the registered NYC 311 fetcher behind --fetch (D48 (8)) ---------------------
#
# Nothing below patches FETCHERS: `--fetch` reaches the factory Task 24 registered, and
# the only fake is the transport the command line's main() would otherwise build.

NYC_ACQUISITION = Path("data/acquisitions/nyc311/2024-01-01_2024-01-03")
NYC_BOUNDS = re.compile(r"created_date >= '(.{10})T00:00:00' AND created_date < '(.{10})T00:00:00'")
NYC_METADATA = ("metadata",)
NYC_WINDOW = ("count", "2024-01-01", "2024-01-04")


def nyc_count(day):
    return ("count", day, (date.fromisoformat(day) + timedelta(days=1)).isoformat())


class SocrataTransport:
    """NYC 311's two endpoints over `day_rows`, answering 403 to any request in `forbid`."""

    identity = TEST_CLIENT

    def __init__(self, forbid=()):
        self.log = []
        self.forbid = set(forbid)

    def get(self, url, params, headers):
        params = dict(params)
        if url == "https://data.cityofnewyork.us/api/views/erm2-nwe9.json":
            request, body = NYC_METADATA, {"rowsUpdatedAt": 1790559478}
        else:
            assert url == "https://data.cityofnewyork.us/resource/erm2-nwe9.json", url
            first, last = NYC_BOUNDS.fullmatch(params["$where"]).groups()
            rows = [row for day in ACQ_DAYS if first <= day < last for row in day_rows(day)]
            if params["$select"] == "count(*) AS n":
                request, body = ("count", first, last), [{"n": str(len(rows))}]
            else:
                request, body = ("data", first), rows
        self.log.append(request)
        if request in self.forbid:
            return Response(status=403, headers={}, body=b"")
        return Response(status=200, headers={}, body=json.dumps(body).encode())


class OfflineTransport:
    identity = TEST_CLIENT

    def get(self, url, params, headers):
        raise AssertionError("a completed acquisition is reused without a request")


def nyc_fetch(tmp_path, monkeypatch, transport):
    import ingest.cli as cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "RequestsTransport", lambda: transport)
    monkeypatch.setattr(cli, "HttpClient", quiet_client)
    monkeypatch.setattr(cli, "current_commit", lambda: TEST_COMMIT)
    arguments = ["--source", "nyc311", "--start", "2024-01-01", "--end", "2024-01-03"]
    return main([*arguments, "--corpus-root", str(tmp_path / "corpus"), "--fetch"])


def test_fetch_runs_the_registered_nyc311_fetcher_end_to_end(tmp_path, monkeypatch, closed_network):
    from ingest.fetch.nyc311 import make_nyc311_fetcher
    from ingest.fetch.registry import FETCHERS

    assert FETCHERS["nyc311"] is make_nyc311_fetcher
    transport = SocrataTransport()
    assert nyc_fetch(tmp_path, monkeypatch, transport) == 0

    days = [(kind, day) for day in ACQ_DAYS for kind in ("count", "data")]
    requests = [nyc_count(day) if kind == "count" else (kind, day) for kind, day in days]
    assert transport.log == [NYC_METADATA, NYC_WINDOW, *requests, NYC_METADATA, NYC_WINDOW]
    data = record_path(tmp_path / NYC_ACQUISITION).read_bytes()
    record = json.loads(data)
    assert record["source_details"]["dataset_id"] == "erm2-nwe9"
    assert [entry["key"] for entry in record["slices"]] == list(ACQ_DAYS)
    manifest = read_manifest("nyc311", root=tmp_path / "corpus")
    assert manifest.acquisition_id == hashlib.sha256(data).hexdigest()
    assert manifest.record_count == 6


def test_a_403_under_fetch_stops_at_once_and_a_rerun_resumes_the_acquisition(
    tmp_path, monkeypatch, closed_network
):
    broken = SocrataTransport(forbid={("data", "2024-01-02")})
    with pytest.raises(FetchFailed, match="403"):
        nyc_fetch(tmp_path, monkeypatch, broken)
    assert broken.log[-1] == ("data", "2024-01-02")
    assert broken.log.count(("data", "2024-01-02")) == 1, "never retried"
    assert not record_path(tmp_path / NYC_ACQUISITION).exists()
    assert not list((tmp_path / "corpus").rglob("manifest.json"))

    resumed = SocrataTransport()
    assert nyc_fetch(tmp_path, monkeypatch, resumed) == 0
    assert resumed.log == [
        nyc_count("2024-01-02"),
        ("data", "2024-01-02"),
        nyc_count("2024-01-03"),
        ("data", "2024-01-03"),
        NYC_METADATA,
        NYC_WINDOW,
    ], "the start snapshot and day one come from the acquisition, not the source"
    assert read_manifest("nyc311", root=tmp_path / "corpus").record_count == 6


def test_a_completed_nyc311_acquisition_is_reused_offline_with_zero_requests(
    tmp_path, monkeypatch, closed_network
):
    assert nyc_fetch(tmp_path, monkeypatch, SocrataTransport()) == 0
    first = read_manifest("nyc311", root=tmp_path / "corpus")
    record = record_path(tmp_path / NYC_ACQUISITION).read_bytes()

    assert nyc_fetch(tmp_path, monkeypatch, OfflineTransport()) == 0
    again = read_manifest("nyc311", root=tmp_path / "corpus")
    assert again.acquisition_id == first.acquisition_id
    assert again.corpus_id == first.corpus_id
    assert record_path(tmp_path / NYC_ACQUISITION).read_bytes() == record


# --- Task 25: the registered CFPB fetcher behind --fetch (D49) ----------------------------
#
# FETCHERS is not patched: `--fetch` reaches the factory Task 25 registered, and the only
# fake is the transport, answering the reading room, the files host and the API.

CFPB_ACQUISITION = Path("data/acquisitions/cfpb/2024-01-01_2024-01-03")
CFPB_ROOM = (
    "https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/"
    "cfpb-consumer-complaint-database-narratives-archive/"
)
CFPB_API = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
CFPB_FILES = "https://files.consumerfinance.gov/f/documents/"
CFPB_EXPORTS = {
    "2023-12": "CCDB_Export_4_December_2023.zip",
    "2024-01": "CCDB_Export_5_January_2024.zip",
}
CFPB_COMPLAINTS = [
    ("20000001", "2023-12-31", "Before the window."),
    ("20000002", "2024-01-01", "One."),
    ("20000003", "2024-01-02", "Two."),
    ("20000004", "2024-01-02", ""),
    ("20000005", "2024-01-03", "Three."),
    ("20000006", "2024-01-04", "After the window."),
]
CFPB_ARCHIVE_HEADER = [
    "Date received", "Product", "Sub-product", "Issue", "Sub-issue",
    "Consumer complaint narrative", "Company public response", "Company", "State",
    "ZIP code", "Tags", "Submitted via", "Date sent to company",
    "Company response to consumer", "Timely response?", "Complaint ID",
]  # fmt: skip
CFPB_API_HEADER = [
    "Date received", "Product", "Sub-product", "Issue", "Sub-issue",
    "Company public response", "Company", "State", "ZIP code", "Tags", "Submitted via",
    "Date sent to company", "Company response to consumer", "Timely response?",
    "Complaint ID",
]  # fmt: skip


def cfpb_csv(header, rows):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


class CFPBTransport:
    """The reading room, the files host and the API over CFPB_COMPLAINTS; 403 on `forbid`."""

    identity = TEST_CLIENT

    def __init__(self, forbid=()):
        self.log = []
        self.forbid = set(forbid)

    def get(self, url, params, headers):
        params = dict(params)
        if url == CFPB_ROOM:
            request = ("room",)
            links = "".join(f'<a href="{CFPB_FILES}{n}">x</a>' for n in CFPB_EXPORTS.values())
            body = links.encode()
        elif url.startswith(CFPB_FILES):
            request = ("zip", url[len(CFPB_FILES) :])
            month = next(m for m, n in CFPB_EXPORTS.items() if n == request[1])
            rows = [
                [day, "Credit card", "", "Problem", "", story, "", "Bank", "NY", "10001", "",
                 "Web", day, "Closed with explanation", "Yes", cid]
                for cid, day, story in CFPB_COMPLAINTS if day.startswith(month)
            ]  # fmt: skip
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                info = zipfile.ZipInfo(request[1].replace(".zip", ".csv"), (2026, 9, 13, 0, 0, 0))
                archive.writestr(info, cfpb_csv(CFPB_ARCHIVE_HEADER, rows))
            body = buffer.getvalue()
        else:
            assert url == CFPB_API, url
            first, last = params["date_received_min"], params["date_received_max"]
            found = [c for c in CFPB_COMPLAINTS if first <= c[1] <= last]
            if params.get("format") == "csv":
                request = ("csv", first)
                body = cfpb_csv(CFPB_API_HEADER, [
                    [f"{day}T10:00:00.000Z", "Credit card", "", "Problem", "None", "None", "Bank",
                     "NY", "10001", "None", "Web", f"{day}T10:05:00.000Z",
                     "Closed with explanation", "Yes", cid]
                    for cid, day, _ in found
                ])  # fmt: skip
            else:
                request = ("count", first, last)
                total = {"value": len(found), "relation": "eq"}
                hits = [{"_index": "complaint-public-v1", "_source": {}}] if found else []
                meta = {"last_indexed": "2026-09-30T12:00:00-05:00"}
                body = json.dumps({"_meta": meta, "hits": {"total": total, "hits": hits}}).encode()
        self.log.append(request)
        if request in self.forbid:
            return Response(status=403, headers={}, body=b"")
        return Response(status=200, headers={}, body=body)


def cfpb_fetch(tmp_path, monkeypatch, transport):
    import ingest.cli as cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "RequestsTransport", lambda: transport)
    monkeypatch.setattr(cli, "HttpClient", quiet_client)
    monkeypatch.setattr(cli, "current_commit", lambda: TEST_COMMIT)
    arguments = ["--source", "cfpb", "--start", "2024-01-01", "--end", "2024-01-03"]
    return main([*arguments, "--corpus-root", str(tmp_path / "corpus"), "--fetch"])


def test_fetch_runs_the_registered_cfpb_fetcher_end_to_end(tmp_path, monkeypatch, closed_network):
    from ingest.fetch.cfpb import make_cfpb_fetcher
    from ingest.fetch.registry import FETCHERS

    assert FETCHERS["cfpb"] is make_cfpb_fetcher
    transport = CFPBTransport()
    assert cfpb_fetch(tmp_path, monkeypatch, transport) == 0

    assert transport.log[:3] == [
        ("room",),
        ("zip", CFPB_EXPORTS["2023-12"]),
        ("zip", CFPB_EXPORTS["2024-01"]),
    ]
    data = record_path(tmp_path / CFPB_ACQUISITION).read_bytes()
    record = json.loads(data)
    assert record["source_details"]["acquisition_kind"] == "cfpb-archive-api-reconstruction-v1"
    assert [entry["key"] for entry in record["slices"]] == [
        "2023-12-31", "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"
    ]  # fmt: skip
    manifest = read_manifest("cfpb", root=tmp_path / "corpus")
    assert manifest.acquisition_id == hashlib.sha256(data).hexdigest()
    assert manifest.record_count == 3
    _, records = load_corpus("cfpb", root=tmp_path / "corpus")
    assert sorted(r.external_id for r in records) == ["20000002", "20000003", "20000005"]


def test_a_403_under_cfpb_fetch_stops_at_once_and_a_rerun_resumes_from_the_pins(
    tmp_path, monkeypatch, closed_network
):
    broken = CFPBTransport(forbid={("csv", "2024-01-02")})
    with pytest.raises(FetchFailed, match="403"):
        cfpb_fetch(tmp_path, monkeypatch, broken)
    assert broken.log.count(("csv", "2024-01-02")) == 1, "never retried"
    assert not record_path(tmp_path / CFPB_ACQUISITION).exists()

    resumed = CFPBTransport()
    assert cfpb_fetch(tmp_path, monkeypatch, resumed) == 0
    assert resumed.log == [
        ("count", "2024-01-02", "2024-01-02"),
        ("csv", "2024-01-02"),
        ("count", "2024-01-03", "2024-01-03"),
        ("csv", "2024-01-03"),
        ("count", "2024-01-04", "2024-01-04"),
        ("csv", "2024-01-04"),
        ("count", "2024-01-01", "2024-01-03"),
    ], "no reading room, no export and no finished day is requested again"
    assert read_manifest("cfpb", root=tmp_path / "corpus").record_count == 3


def test_a_completed_cfpb_acquisition_is_reused_offline_with_zero_requests(
    tmp_path, monkeypatch, closed_network
):
    assert cfpb_fetch(tmp_path, monkeypatch, CFPBTransport()) == 0
    first = read_manifest("cfpb", root=tmp_path / "corpus")
    record = record_path(tmp_path / CFPB_ACQUISITION).read_bytes()

    assert cfpb_fetch(tmp_path, monkeypatch, OfflineTransport()) == 0
    again = read_manifest("cfpb", root=tmp_path / "corpus")
    assert again.acquisition_id == first.acquisition_id
    assert again.corpus_id == first.corpus_id
    assert record_path(tmp_path / CFPB_ACQUISITION).read_bytes() == record


# --- Task 26: D50's typed NYC 311 exclusions through ingest() (C1-C9) ----------------------
#
# Every otherwise-valid row of one of D45's three classes is excluded, counted by kind and
# by its created_date's New York civil year, and never refuses the run; every other
# refusal still refuses before any write. The fetcher is untouched: rows reach the cache
# exactly as served, and classification happens only once the pages are read back.


def without_descriptor(row):
    return {**row, "descriptor": None}


def kind_rows():
    """Two valid rows and one row of each D50 kind, all created inside 2024-2025."""
    return [
        nyc311_resolved("v1", created="2024-03-15T09:00:00.000", closed="2024-03-15T10:00:00.000"),
        nyc311_row("v2", created="2025-06-01T12:00:00.000"),
        without_descriptor(nyc311_row("d1", created="2024-05-01T08:00:00.000")),
        nyc311_row("cf", created="2024-11-03T01:30:00.000"),
        nyc311_row("cg", created="2025-03-09T02:30:00.000"),
        nyc311_resolved("xf", created="2025-11-01T12:00:00.000", closed="2025-11-02T01:15:00.000"),
        nyc311_resolved("xg", created="2024-03-09T12:00:00.000", closed="2024-03-10T02:30:00.000"),
        nyc311_resolved("ng", created="2025-02-01T12:00:00.000", closed="2025-02-01T11:00:00.000"),
    ]


KIND_COUNTS = {
    "MissingDescriptor": {2024: 1},
    "AmbiguousLocalTime:created_date": {2024: 1},
    "NonexistentLocalTime:created_date": {2025: 1},
    "AmbiguousLocalTime:closed_date": {2025: 1},
    "NonexistentLocalTime:closed_date": {2024: 1},
    "NegativeResolutionTime": {2025: 1},
}
NOTHING_EXCLUDED = {kind: {} for kind in EXCLUSION_KINDS}


def excluded_total(records):
    return sum(count for years in records.values() for count in years.values())


def test_c1_valid_rows_are_kept_and_each_kind_is_excluded_and_counted(tmp_path):
    rows = kind_rows()
    manifest, _ = a_311_run(tmp_path, rows)
    assert manifest.excluded_records == KIND_COUNTS
    assert read_manifest("nyc311", root=tmp_path / "corpus").excluded_records == KIND_COUNTS
    _, records = load_corpus("nyc311", root=tmp_path / "corpus")
    assert sorted(record.external_id for record in records) == ["v1", "v2"]
    assert sorted(outcome.external_id for outcome in loaded_outcomes(tmp_path)) == ["v1", "v2"]
    assert manifest.record_count + excluded_total(manifest.excluded_records) == len(rows)


def test_c1_an_exclusion_counts_under_created_date_s_civil_year(tmp_path):
    late = "2024-12-31T23:30:00.000"
    rows = [nyc311_row("kept", created=late), without_descriptor(nyc311_row("gone", created=late))]
    manifest, _ = a_311_run(tmp_path, rows)
    assert manifest.excluded_records == {**NOTHING_EXCLUDED, "MissingDescriptor": {2024: 1}}
    assert manifest.per_year_counts == {2025: 1}, "a kept record's partition is its UTC year"


def test_c1_exclusions_of_one_kind_in_one_year_are_summed(tmp_path):
    rows = [nyc311_row("1")] + [
        without_descriptor(nyc311_row(f"d{month}", created=f"2024-0{month}-15T09:00:00.000"))
        for month in (1, 2, 3)
    ]
    rows.append(without_descriptor(nyc311_row("d9", created="2025-01-15T09:00:00.000")))
    manifest, _ = a_311_run(tmp_path, rows)
    assert manifest.excluded_records == {
        **NOTHING_EXCLUDED,
        "MissingDescriptor": {2024: 3, 2025: 1},
    }


def test_c1_a_run_with_nothing_to_exclude_records_every_kind_empty(tmp_path):
    manifest, _ = a_311_run(tmp_path, [nyc311_row("1"), nyc311_row("2")])
    assert manifest.excluded_records == NOTHING_EXCLUDED
    assert manifest.record_count == 2


NON_D45_IN_RUN = {
    "a blank unique_key": {"unique_key": "  "},
    "a complaint_type that is not text": {"complaint_type": 7},
    "a created_date with an offset": {"created_date": "2024-03-15T09:00:00-04:00"},
    "a closed_date that is not ISO-8601": {"closed_date": "soon"},
}
D45_IN_RUN = {
    "no D45 condition": {},
    "MissingDescriptor": {"descriptor": None},
    "AmbiguousLocalTime:created_date": {"created_date": "2024-11-03T01:30:00.000"},
    "NegativeResolutionTime": {"closed_date": "2024-03-15T08:00:00.000"},
}
REFUSING = [
    (problem, condition)
    for problem in sorted(NON_D45_IN_RUN)
    for condition in sorted(D45_IN_RUN)
    if not set(NON_D45_IN_RUN[problem]) & set(D45_IN_RUN[condition])
]


@pytest.mark.parametrize("problem, condition", REFUSING)
def test_c2_every_other_refusal_still_refuses_before_any_write(tmp_path, problem, condition):
    a_311_run(tmp_path, [nyc311_row("1")])
    corpus = tmp_path / "corpus"
    before = snapshot(corpus)
    bad = {**nyc311_row("2"), **NON_D45_IN_RUN[problem], **D45_IN_RUN[condition]}
    with pytest.raises(MissingField):
        a_311_run(tmp_path, [nyc311_row("3"), bad], raw="later")
    assert snapshot(corpus) == before


def test_c3_a_window_whose_every_row_is_excluded_is_empty_and_names_the_exclusions(tmp_path):
    rows = [row for row in kind_rows() if not row["unique_key"].startswith("v")]
    with pytest.raises(EmptyWindow) as exc:
        a_311_run(tmp_path, rows)
    message = str(exc.value)
    assert "zero records" in message and "1 cached page" in message
    assert "6 rows were excluded" in message, message
    for kind, years in KIND_COUNTS.items():
        assert f"{kind} {sum(years.values())}" in message, message
    corpus = tmp_path / "corpus"
    assert not corpus.exists() or not [p for p in corpus.rglob("*") if p.is_file()]


def test_c3_a_single_excluded_row_is_named_in_the_singular(tmp_path):
    with pytest.raises(EmptyWindow, match="1 row was excluded under D50"):
        a_311_run(tmp_path, [without_descriptor(nyc311_row("1"))])


def test_c3_a_cfpb_empty_window_says_nothing_of_d50(tmp_path):
    with pytest.raises(EmptyWindow) as exc:
        a_cfpb_run(tmp_path, [[cfpb_row("1", received="2023-03-15T09:00:00-04:00")]])
    assert "D50" not in str(exc.value) and "excluded" not in str(exc.value)


def test_c4_the_counts_cover_the_whole_window_before_limit_truncates_it(tmp_path):
    manifest, _ = a_311_run(tmp_path, kind_rows(), limit=1)
    assert (manifest.limit, manifest.record_count) == (1, 1)
    assert manifest.excluded_records == KIND_COUNTS


@pytest.mark.parametrize(
    "rows",
    [
        [nyc311_row("7"), without_descriptor(nyc311_row("7", created="2024-04-01T09:00:00.000"))],
        [
            nyc311_row("1"),
            without_descriptor(nyc311_row("7")),
            nyc311_row("7", created="2024-11-03T01:30:00.000"),
        ],
    ],
    ids=["kept-and-excluded", "both-excluded"],
)
def test_c5_an_excluded_row_cannot_hide_a_duplicate_identity(tmp_path, rows):
    with pytest.raises(DuplicateExternalId, match="nyc311:7"):
        a_311_run(tmp_path, rows)
    corpus = tmp_path / "corpus"
    assert not corpus.exists() or not [p for p in corpus.rglob("*") if p.is_file()]


def test_c6_an_excluded_row_outside_the_window_is_neither_counted_nor_refused(tmp_path):
    rows = [
        nyc311_row("1"),
        without_descriptor(nyc311_row("before", created="2023-12-31T23:59:59.999")),
        without_descriptor(nyc311_row("first", created="2024-01-01T00:00:00.000")),
        without_descriptor(nyc311_row("last", created="2025-12-31T23:59:59.999")),
        without_descriptor(nyc311_row("after", created="2026-01-01T00:00:00.000")),
        nyc311_row("fold-2023", created="2023-11-05T01:30:00.000"),
    ]
    manifest, _ = a_311_run(tmp_path, rows)
    assert manifest.record_count == 1
    assert manifest.excluded_records == {
        **NOTHING_EXCLUDED,
        "MissingDescriptor": {2024: 1, 2025: 1},
    }


def test_c6_another_refusal_outside_the_window_still_refuses(tmp_path):
    outside = {**nyc311_row("old", created="2023-06-01T09:00:00.000"), "complaint_type": None}
    with pytest.raises(MissingField, match="complaint_type"):
        a_311_run(tmp_path, [nyc311_row("1"), without_descriptor(outside)])


def test_c7_cfpb_records_no_exclusions_and_keeps_its_refusals(tmp_path):
    manifest, _ = a_cfpb_run(tmp_path / "a", [[cfpb_row("1")]])
    assert manifest.excluded_records == {}
    assert read_manifest("cfpb", root=tmp_path / "a" / "corpus").excluded_records == {}
    blank = {**cfpb_row("2"), "complaint_what_happened": "   "}
    with pytest.raises(MissingNarrative):
        a_cfpb_run(tmp_path / "b", [[cfpb_row("1"), blank]])
    corpus = tmp_path / "b" / "corpus"
    assert not corpus.exists() or not [p for p in corpus.rglob("*") if p.is_file()]


class EdgeSocrata:
    """NYC 311's two endpoints over fixed rows per civil day, edge rows among them."""

    identity = TEST_CLIENT

    def __init__(self, days):
        self.days = days
        self.log = []

    def get(self, url, params, headers):
        params = dict(params)
        if url == "https://data.cityofnewyork.us/api/views/erm2-nwe9.json":
            self.log.append(NYC_METADATA)
            body = json.dumps({"rowsUpdatedAt": 1790559478}).encode()
            return Response(status=200, headers={}, body=body)
        assert url == "https://data.cityofnewyork.us/resource/erm2-nwe9.json", url
        first, last = NYC_BOUNDS.fullmatch(params["$where"]).groups()
        rows = [row for day in sorted(self.days) if first <= day < last for row in self.days[day]]
        if params["$select"] == "count(*) AS n":
            self.log.append(("count", first, last))
            return Response(
                status=200, headers={}, body=json.dumps([{"n": str(len(rows))}]).encode()
            )
        self.log.append(("data", first))
        return Response(status=200, headers={}, body=json.dumps(rows).encode())


def edge_fetch(tmp_path, monkeypatch, transport, start, end):
    import ingest.cli as cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "RequestsTransport", lambda: transport)
    monkeypatch.setattr(cli, "HttpClient", quiet_client)
    monkeypatch.setattr(cli, "current_commit", lambda: TEST_COMMIT)
    arguments = ["--source", "nyc311", "--start", start, "--end", end, "--fetch"]
    return main([*arguments, "--corpus-root", str(tmp_path / "corpus")])


EDGE_WINDOWS = {
    "spring": (
        "2024-03-09",
        "2024-03-11",
        {
            "2024-03-09": [
                nyc311_resolved(
                    "m1", created="2024-03-09T09:00:00.000", closed="2024-03-09T10:00:00.000"
                ),
                without_descriptor(nyc311_row("m2", created="2024-03-09T11:00:00.000")),
                nyc311_resolved(
                    "m3", created="2024-03-09T12:00:00.000", closed="2024-03-10T02:30:00.000"
                ),
            ],
            "2024-03-10": [
                nyc311_row("m4", created="2024-03-10T02:30:00.000"),
                nyc311_row("m5", created="2024-03-10T03:00:00.000"),
                nyc311_resolved(
                    "m6", created="2024-03-10T12:00:00.000", closed="2024-03-10T11:00:00.000"
                ),
            ],
            "2024-03-11": [
                nyc311_resolved(
                    "m7", created="2024-03-11T09:00:00.000", closed="2024-11-03T01:30:00.000"
                ),
                nyc311_row("m8", created="2024-03-11T10:00:00.000"),
            ],
        },
        {
            **NOTHING_EXCLUDED,
            "MissingDescriptor": {2024: 1},
            "NonexistentLocalTime:created_date": {2024: 1},
            "AmbiguousLocalTime:closed_date": {2024: 1},
            "NonexistentLocalTime:closed_date": {2024: 1},
            "NegativeResolutionTime": {2024: 1},
        },
    ),
    "autumn": (
        "2024-11-02",
        "2024-11-04",
        {
            "2024-11-02": [
                nyc311_row("n1", created="2024-11-02T09:00:00.000"),
                nyc311_resolved(
                    "n2", created="2024-11-02T12:00:00.000", closed="2024-11-03T01:30:00.000"
                ),
                nyc311_resolved(
                    "n3", created="2024-11-02T15:00:00.000", closed="2024-11-02T14:00:00.000"
                ),
            ],
            "2024-11-03": [
                nyc311_row("n4", created="2024-11-03T01:30:00.000"),
                nyc311_row("n5", created="2024-11-03T03:00:00.000"),
                without_descriptor(nyc311_row("n6", created="2024-11-03T10:00:00.000")),
            ],
            "2024-11-04": [
                nyc311_row("n7", created="2024-11-04T09:00:00.000"),
                nyc311_resolved(
                    "n8", created="2024-11-04T09:30:00.000", closed="2025-03-09T02:30:00.000"
                ),
            ],
        },
        {
            **NOTHING_EXCLUDED,
            "MissingDescriptor": {2024: 1},
            "AmbiguousLocalTime:created_date": {2024: 1},
            "AmbiguousLocalTime:closed_date": {2024: 1},
            "NonexistentLocalTime:closed_date": {2024: 1},
            "NegativeResolutionTime": {2024: 1},
        },
    ),
}


def keys_of(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from keys_of(item)
    elif isinstance(value, list):
        for item in value:
            yield from keys_of(item)


@pytest.mark.parametrize("window", sorted(EDGE_WINDOWS))
def test_c8_fetched_edge_rows_are_cached_as_served_then_excluded_and_counted(
    tmp_path, monkeypatch, closed_network, window
):
    start, end, days, expected = EDGE_WINDOWS[window]
    assert edge_fetch(tmp_path, monkeypatch, EdgeSocrata(days), start, end) == 0

    directory = tmp_path / "data" / "acquisitions" / "nyc311" / f"{start}_{end}"
    data = record_path(directory).read_bytes()
    record = json.loads(data)
    assert [entry["key"] for entry in record["slices"]] == sorted(days)
    for entry in record["slices"]:
        [digest] = entry["pages"]
        with gzip.open(directory / "nyc311" / f"{digest}.json.gz", "rt", encoding="utf-8") as page:
            assert json.load(page) == days[entry["key"]], "cached exactly as served"
    assert not [key for key in keys_of(record) if "exclu" in key.lower()]

    manifest = read_manifest("nyc311", root=tmp_path / "corpus")
    assert manifest.acquisition_id == hashlib.sha256(data).hexdigest()
    assert manifest.excluded_records == expected
    rows = sum(entry["verification"]["rows"] for entry in record["slices"])
    assert rows == manifest.record_count + excluded_total(expected)

    assert edge_fetch(tmp_path, monkeypatch, OfflineTransport(), start, end) == 0
    again = read_manifest("nyc311", root=tmp_path / "corpus")
    assert again.excluded_records == manifest.excluded_records
    assert (again.corpus_id, again.acquisition_id) == (manifest.corpus_id, manifest.acquisition_id)
    assert record_path(directory).read_bytes() == data


def test_c8_the_two_windows_cover_every_kind():
    covered = {
        kind
        for _, _, _, expected in EDGE_WINDOWS.values()
        for kind, years in expected.items()
        if years
    }
    assert covered == set(EXCLUSION_KINDS)

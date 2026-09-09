"""Resumable corpus ingestion, and the timestamp provenance diagnostic.

    python -m ingest.cli --source {cfpb,nyc311} --start YYYY-MM-DD --end YYYY-MM-DD [--limit N]

The order of operations is load-bearing::

    fetch (or reuse the raw cache)
      -> normalize through the source's adapter
      -> filter to the requested window
      -> apply --limit
      -> assert the label roster                <-- before any Parquet is written
      -> write partitions
      -> compute the timestamp diagnostic
      -> build and write the manifest

The roster assertion sits where it does deliberately (§1.1): a corpus written
before its taxonomy is checked has already changed the experimental population,
and deleting it afterwards is not the same as never having written it. A test
asserts no part file appears after a `RosterMismatch`.

**Resumability is content-addressed, not a checkpoint file.** Each fetched page
is stored gzipped at `data/raw/<source>/<sha256>.json.gz`, so a page whose
checksum matches an existing file is skipped. Re-running performs zero writes
and no fetch is required at all: normalize and load read the cache. An
interrupted run leaves whole pages behind, never half of one, and the resumed
run rewrites every partition from the full cache — so a resumed corpus is
byte-identical to a clean one rather than merely equivalent.

**Fetching is injected.** No approved document specifies an endpoint, a
pagination scheme, a retry policy or a rate limit for either source, so none is
invented here. `Fetcher` is the boundary; the concrete HTTP client belongs to
whichever task specifies those things. Passing `fetcher=None` runs entirely from
the cache, which is what the plan requires of the normalize-and-load pass.

**The diagnostic follows §2.3 exactly and adds nothing.** Its primary evidence
is the CFPB `date_received` -> `date_sent_to_company` delta; the hour and
weekday distributions are secondary and may only move a verdict toward doubt.
`hour_concentration >= 0.50` is D22's threshold and the only distributional one
that exists. Nothing here describes a verdict as proof of how a timestamp was
produced.

Django-independent: invoked as `python -m ingest.cli`, never as a management
command.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from scipy.stats import chi2

from ingest.manifest import build_manifest, manifest_path, read_manifest, write_manifest
from ingest.roster import assert_roster, derive_roster
from ingest.schema import CFPBOutcome, CorpusRecord, NYC311Outcome
from ingest.sources import cfpb, nyc311
from ingest.sources.base import SourcePage
from ingest.storage import CORPUS_ROOT, write_partition

RAW_ROOT = Path("data") / "raw"
"""Gitignored, like the corpus. Holds the fetched pages a rerun reuses."""

SOURCES = ("cfpb", "nyc311")

STRONGLY_SUSPICIOUS = "strongly_suspicious_load_timestamp"
SUPPORTED = "supported_plausible_event_time"
INSUFFICIENT = "suspicious_insufficient_evidence"

HOUR_CONCENTRATION_DOWNGRADE = 0.50
"""D22. A Sentinel project threshold, not a source-derived fact. It may move
`supported_plausible_event_time` toward doubt and do nothing else."""

VERDICT_RULE: dict[str, Any] = {
    "strongly_suspicious_median_delta_seconds_max": 60,
    "strongly_suspicious_frac_delta_le_1min_min": 0.50,
    "strongly_suspicious_frac_identical_timestamps_min": 0.20,
    "supported_median_delta_seconds_min": 3600,
    "supported_frac_delta_le_1min_max": 0.05,
    "supported_count_delta_negative_max": 0,
    "insufficient_pair_coverage_below": 0.50,
    "hour_concentration_downgrade_at": HOUR_CONCENTRATION_DOWNGRADE,
}
"""Recorded in every manifest beside the verdict, so a reader sees the numbers
that were applied rather than having to consult the addendum."""

_NO_PAIR_REASON = (
    "the created-to-closed interval is this source's target variable, so using "
    "it as a provenance check would test the label with the label (addendum 2.3)"
)

Fetcher = Callable[[str, date, date], Iterable[SourcePage]]


class IngestError(Exception):
    """Ingestion could not proceed."""


class InvalidDateRange(IngestError):
    """`--end` precedes `--start`. Never silently swapped."""


# --- the raw cache -----------------------------------------------------------


def page_checksum(page: SourcePage) -> str:
    """A page's identity: SHA256 over its canonical JSON.

    Keys are sorted, so a source that reorders a JSON object without changing
    its content does not look like a new page and get fetched twice.
    """
    canonical = json.dumps(page, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def raw_dir(source: str, raw_root: Path = RAW_ROOT) -> Path:
    return Path(raw_root) / source


def cache_page(page: SourcePage, source: str, raw_root: Path = RAW_ROOT) -> tuple[Path, bool]:
    """Store one page, or recognise it as already stored.

    Returns the path and whether anything was written. Content addressing is
    what makes a rerun free: the same page yields the same name.
    """
    directory = raw_dir(source, raw_root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{page_checksum(page)}.json.gz"
    if path.exists():
        return path, False
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(page, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return path, True


def iter_cached_pages(source: str, raw_root: Path = RAW_ROOT) -> Iterator[SourcePage]:
    """Every cached page, in a deterministic order (checksum, hence filename)."""
    directory = raw_dir(source, raw_root)
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            yield json.load(handle)


def fetch_into_cache(
    source: str, start: date, end: date, fetcher: Fetcher | None, raw_root: Path
) -> tuple[int, int]:
    """Pull pages into the cache. Returns (written, skipped)."""
    if fetcher is None:
        return 0, 0
    written = skipped = 0
    for page in fetcher(source, start, end):
        _, was_written = cache_page(page, source, raw_root)
        written += was_written
        skipped += not was_written
    return written, skipped


# --- normalization -----------------------------------------------------------


def _window_bounds(start: date, end: date) -> tuple[datetime, datetime]:
    """The window as instants. `end` is inclusive of its whole day."""
    return (
        datetime(start.year, start.month, start.day, tzinfo=UTC),
        datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=UTC),
    )


def _normalize_source(
    source: str, pages: Iterable[SourcePage]
) -> Iterator[tuple[CorpusRecord, CFPBOutcome | NYC311Outcome]]:
    adapter = cfpb if source == "cfpb" else nyc311
    for page in pages:
        for row in adapter.rows_from_page(page):
            yield adapter.normalize(row)


def _local_hour_source(source: str) -> Callable[[datetime], datetime]:
    """How a record's *submitted hour* is read, per source.

    311's floating timestamps are New York civil time and §2.4 binds its
    `submitted_hour` to that local representation, so the histogram is built
    from it. CFPB publishes real offsets and the corpus stores the instant, so
    its own local wall clock is not recoverable and the stored value is used.
    """
    if source == "nyc311":
        return nyc311.to_source_local
    return lambda instant: instant


# --- the diagnostic ----------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile over a sorted copy of `values`."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * (q / 100.0)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def chi_square_uniform(counts: Sequence[int]) -> tuple[float, float, int]:
    """Chi-square against a uniform distribution, with its p-value.

    Recorded for a reader. It does not gate the verdict: §2.3 admits exactly one
    distributional threshold and D22 fixes it as `hour_concentration`.
    """
    total = sum(counts)
    degrees = len(counts) - 1
    if total == 0:
        return 0.0, 1.0, degrees
    expected = total / len(counts)
    statistic = sum((count - expected) ** 2 / expected for count in counts)
    return float(statistic), float(chi2.sf(statistic, degrees)), degrees


def hour_concentration(hour_counts: Sequence[int]) -> float:
    """D22: the largest of the 24 submitted-hour counts over the record total."""
    total = sum(hour_counts)
    if total == 0:
        return 0.0
    return max(hour_counts) / total


def _field_delta_metrics(deltas: Sequence[float], paired: int, total: int) -> dict[str, Any]:
    coverage = (paired / total) if total else 0.0
    if not deltas:
        return {
            "evidence_class": "field_delta",
            "available": True,
            "pair_coverage": coverage,
            "median_delta_seconds": None,
            "delta_percentiles_seconds": {},
            "frac_delta_le_1min": 0.0,
            "frac_delta_le_10min": 0.0,
            "frac_delta_le_1h": 0.0,
            "count_delta_negative": 0,
            "count_delta_zero": 0,
            "frac_identical_timestamps": 0.0,
        }

    n = len(deltas)
    percentiles = {f"p{q}": percentile(deltas, q) for q in (5, 25, 50, 75, 95, 99)}
    return {
        "evidence_class": "field_delta",
        "available": True,
        "pair_coverage": coverage,
        "median_delta_seconds": percentiles["p50"],
        "delta_percentiles_seconds": percentiles,
        "frac_delta_le_1min": sum(d <= 60 for d in deltas) / n,
        "frac_delta_le_10min": sum(d <= 600 for d in deltas) / n,
        "frac_delta_le_1h": sum(d <= 3600 for d in deltas) / n,
        "count_delta_negative": sum(d < 0 for d in deltas),
        "count_delta_zero": sum(d == 0 for d in deltas),
        "frac_identical_timestamps": sum(d == 0 for d in deltas) / n,
    }


def decide_verdict(primary: dict[str, Any]) -> tuple[str, str]:
    """The §2.3 rule over delta metrics alone. Returns (verdict, branch).

    Coverage is checked first because §2.3 makes it unconditional: below half,
    the verdict is insufficient whatever the deltas say.
    """
    if not primary.get("available", False):
        return INSUFFICIENT, "no_testable_pair"

    if primary["pair_coverage"] < VERDICT_RULE["insufficient_pair_coverage_below"]:
        return INSUFFICIENT, "pair_coverage_below_threshold"

    median = primary["median_delta_seconds"]
    if median is None:
        return INSUFFICIENT, "no_pairs_measured"

    if (
        median <= VERDICT_RULE["strongly_suspicious_median_delta_seconds_max"]
        or primary["frac_delta_le_1min"]
        >= VERDICT_RULE["strongly_suspicious_frac_delta_le_1min_min"]
        or primary["frac_identical_timestamps"]
        >= VERDICT_RULE["strongly_suspicious_frac_identical_timestamps_min"]
    ):
        return STRONGLY_SUSPICIOUS, STRONGLY_SUSPICIOUS

    if (
        median >= VERDICT_RULE["supported_median_delta_seconds_min"]
        and primary["frac_delta_le_1min"] < VERDICT_RULE["supported_frac_delta_le_1min_max"]
        and primary["count_delta_negative"] == VERDICT_RULE["supported_count_delta_negative_max"]
    ):
        return SUPPORTED, SUPPORTED

    return INSUFFICIENT, "no_branch_matched"


def build_diagnostic(
    *,
    source: str,
    submitted_local: Sequence[datetime],
    deltas_seconds: Sequence[float] | None,
    paired_count: int,
    total_count: int,
) -> dict[str, Any]:
    """The §2.3 diagnostic object, exactly as plan §G documents it.

    `deltas_seconds is None` means the source exposes no testable pair, which is
    recorded as such rather than defaulted to supported: absence of evidence is
    not evidence of soundness.
    """
    hour_counts = [0] * 24
    weekday_counts = [0] * 7
    for moment in submitted_local:
        hour_counts[moment.hour] += 1
        weekday_counts[moment.weekday()] += 1

    statistic, p_value, degrees = chi_square_uniform(hour_counts)
    concentration = hour_concentration(hour_counts)

    if deltas_seconds is None:
        primary: dict[str, Any] = {
            "evidence_class": "field_delta",
            "available": False,
            "reason": _NO_PAIR_REASON,
        }
    else:
        primary = _field_delta_metrics(deltas_seconds, paired_count, total_count)

    verdict, branch = decide_verdict(primary)

    downgraded = verdict == SUPPORTED and concentration >= HOUR_CONCENTRATION_DOWNGRADE
    if downgraded:
        verdict = INSUFFICIENT

    return {
        "verdict": verdict,
        "verdict_branch": branch,
        "verdict_rule": dict(VERDICT_RULE),
        "not_directly_testable": deltas_seconds is None,
        "primary_evidence": primary,
        "secondary_evidence": {
            "evidence_class": "distributional_anomaly",
            "hour_counts": hour_counts,
            "weekday_counts": weekday_counts,
            "chi_square": {
                "statistic": statistic,
                "p_value": p_value,
                "degrees_of_freedom": degrees,
            },
            "hour_concentration": concentration,
            "downgraded_verdict": downgraded,
        },
    }


# --- ingestion ---------------------------------------------------------------


def _locked_roster(source: str, corpus_root: Path, observed_by_year: dict[int, set[str]]):
    """The roster to assert against.

    On a later run it is the one the manifest locked. On the first, it is
    derived from the data as the intersection across years (§1) — and the
    assertion still runs, so a label present in only part of the window fails
    the first ingest rather than quietly becoming part of the vocabulary.
    """
    if manifest_path(source, corpus_root).is_file():
        return frozenset(read_manifest(source, root=corpus_root).label_roster)
    return derive_roster(observed_by_year)


def ingest(
    *,
    source: str,
    start: date,
    end: date,
    limit: int | None,
    fetcher: Fetcher | None = None,
    corpus_root: Path = CORPUS_ROOT,
    raw_root: Path = RAW_ROOT,
):
    """Fetch, normalize, validate, write, and describe one source's corpus."""
    if end < start:
        raise InvalidDateRange(f"--end {end.isoformat()} precedes --start {start.isoformat()}")

    corpus_root = Path(corpus_root)
    raw_root = Path(raw_root)
    window_start, window_end = _window_bounds(start, end)

    fetch_into_cache(source, start, end, fetcher, raw_root)

    records: list[CorpusRecord] = []
    deltas: list[float] = []
    paired = 0
    for record, outcome in _normalize_source(source, iter_cached_pages(source, raw_root)):
        if not (window_start <= record.submitted_at <= window_end):
            continue
        records.append(record)
        if isinstance(outcome, CFPBOutcome) and outcome.sent_to_company_at is not None:
            paired += 1
            deltas.append((outcome.sent_to_company_at - record.submitted_at).total_seconds())
        if limit is not None and len(records) >= limit:
            break

    # Roster first. Nothing below this line may run before it.
    by_year: dict[int, set[str]] = {}
    counts: dict[str, int] = {}
    for record in records:
        by_year.setdefault(record.submitted_at.year, set()).add(record.label)
        counts[record.label] = counts.get(record.label, 0) + 1
    if source == "cfpb":
        assert_roster(counts, _locked_roster(source, corpus_root, by_year))

    by_partition: dict[int, list[CorpusRecord]] = {}
    for record in records:
        by_partition.setdefault(record.submitted_at.year, []).append(record)
    for year, partition in sorted(by_partition.items()):
        write_partition(partition, source, year, 0, root=corpus_root)

    local = _local_hour_source(source)
    diagnostic = build_diagnostic(
        source=source,
        submitted_local=[local(r.submitted_at) for r in records],
        deltas_seconds=deltas if source == "cfpb" else None,
        paired_count=paired,
        total_count=len(records),
    )

    api_version = cfpb.SOURCE_API_VERSION if source == "cfpb" else nyc311.SOURCE_API_VERSION
    manifest = build_manifest(
        source=source,
        window_start=window_start,
        window_end=window_end,
        source_api_version=api_version,
        limit=limit,
        timestamp_diagnostic=diagnostic,
        root=corpus_root,
    )
    write_manifest(manifest, root=corpus_root)
    return manifest


# --- command line ------------------------------------------------------------


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a YYYY-MM-DD date: {value!r}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ingest.cli")
    parser.add_argument("--source", required=True, choices=SOURCES)
    parser.add_argument("--start", required=True, type=_parse_date)
    parser.add_argument("--end", required=True, type=_parse_date)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="bound records for development; recorded in the manifest",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ingest(source=args.source, start=args.start, end=args.end, limit=args.limit)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

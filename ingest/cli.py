"""Resumable corpus ingestion, and the timestamp provenance diagnostic.

    python -m ingest.cli --source {cfpb,nyc311} --start YYYY-MM-DD --end YYYY-MM-DD
                         [--limit N] [--corpus-root PATH] [--fetch]

The order of operations is load-bearing::

    refuse an --end before --start, then a --limit below one (D28)
      -> refuse a --limit run into a root holding an authoritative corpus (D26)
      -> fetch (or reuse the raw cache, or a completed acquisition)
      -> verify the acquisition, when one is given (D46)
      -> normalize through the source's adapter, filtered to the window; NYC 311
         rows are classified, and an otherwise-valid row of a D45 class inside the
         window is excluded and counted by kind and civil year instead (D50)
      -> refuse an empty window (D23)
      -> assert the label roster over the whole window   <-- before the corpus changes
      -> refuse a duplicate (source, external_id) in the window (D44)
      -> apply --limit, and compute the timestamp diagnostic
      -> clear the source's tree, manifest first (D27)
      -> write partitions
      -> build the manifest and write it last, atomically

Every refusal comes before the clearing step. The roster assertion in
particular sits where it does deliberately (§1.1): a corpus written before its
taxonomy is checked has already changed the experimental population, and
deleting it afterwards is not the same as never having written it. Tests assert
that nothing on disk changes after any refusal.

**Resumability is content-addressed, not a checkpoint file.** Each fetched page
is stored gzipped at `<raw root>/<source>/<sha256>.json.gz`, so a page whose
checksum matches an existing file is skipped. Re-running performs zero writes
and no fetch is required at all: normalize and load read the cache. A page is
written to a temporary file beside its destination and moved into place, and a
page already on disk is digested again and replaced when its content does not
match its name (D44), so an interrupted run leaves whole pages behind, never
half of one. The resumed run replaces the source's whole tree from the full
cache (D27) — so a resumed corpus is byte-identical to a clean one rather than
merely equivalent.

**Fetching is injected.** `Fetcher` is the boundary, and passing `fetcher=None`
runs entirely from the cache, which is what the plan requires of the
normalize-and-load pass. Addendum D44 and D46 specify the concrete fetch: each
acquisition has its own directory, `data/acquisitions/<source>/<start>_<end>/`,
which is the raw root its pages are cached in, and `ingest.fetch` holds the
transport, the retry and pacing policy, the journal and the immutable acquisition
record. Given `acquisition=`, `ingest()` reads only the pages that acquisition's
completed record lists, verifying each one first, and records the record's
digest as the manifest's `acquisition_id`; without it, nothing here changes.
`--fetch` is opt-in: it uses the fetcher registered for the source and refuses
with `FetcherUnavailable` while none is. Without `--fetch` the command line reads
the raw cache exactly as it always has.

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
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from datetime import UTC, date, datetime, tzinfo
from pathlib import Path
from typing import Any

from scipy.stats import chi2

from ingest.fetch.acquisition import (
    acquisition_dir,
    current_commit,
    is_complete,
    verify_acquisition,
)
from ingest.fetch.http import HttpClient, RequestsTransport
from ingest.fetch.registry import FETCHERS, FetchContext
from ingest.manifest import (
    CorpusManifest,
    build_manifest,
    clear_corpus,
    manifest_path,
    no_exclusions,
    read_manifest,
    write_manifest,
)
from ingest.roster import assert_roster, derive_roster
from ingest.schema import CFPBOutcome, CorpusRecord, NYC311Outcome
from ingest.sources import cfpb, nyc311
from ingest.sources.base import SourcePage
from ingest.storage import (
    CORPUS_ROOT,
    write_cfpb_outcome_partition,
    write_outcome_partition,
    write_partition,
)

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


class InvalidLimit(IngestError):
    """`--limit` is zero or negative (D28, §2.8).

    A limit bounds the records persisted, and nothing below one bounds anything:
    zero would persist the empty corpus D23 already refuses, and a negative value
    would be applied as a slice that silently drops records. Raised immediately
    after `InvalidDateRange`, before any manifest is read, anything is fetched,
    or anything is written.
    """


class AuthoritativeCorpusExists(IngestError):
    """A `--limit`ed run targeted a root holding an authoritative corpus (D26, §2.6).

    Raised before `fetch_into_cache` and before any write, so the existing
    partitions and manifest are untouched. A development corpus belongs under a
    different `--corpus-root`.
    """


class EmptyWindow(IngestError):
    """The requested window normalized to zero records (D23, §2.5).

    Raised before roster derivation, so nothing is written. An empty run would
    otherwise derive an empty roster and lock it into the first manifest, after
    which every later ingest fails with every label unexpected — the taxonomy
    corruption §1.1 exists to prevent, self-inflicted.
    """


class DuplicateExternalId(IngestError):
    """The normalized window holds two records with one `(source, external_id)` (D44).

    Raised after the roster assertion and before any write, so nothing is written.
    A record's identity is the pair; two records sharing it would collide on every
    `RecordRef`, and neither is dropped in favour of the other.
    """


class FetcherUnavailable(IngestError):
    """`--fetch` named a source no fetcher is registered for (D46).

    Raised by the command line before any transport is built, any request is made,
    any directory is created and anything is written. Task 23 registers no source;
    Tasks 24 and 25 each register theirs.
    """


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


def _stored_checksum(path: Path) -> str | None:
    """The checksum of the page stored at `path`, or `None` when it cannot be read."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return page_checksum(json.load(handle))
    except (OSError, EOFError, ValueError):
        return None


def cache_page(page: SourcePage, source: str, raw_root: Path = RAW_ROOT) -> tuple[Path, bool]:
    """Store one page, or recognise it as already stored.

    Returns the path and whether anything was written. Content addressing is
    what makes a rerun free: the same page yields the same name.

    A page is never trusted by its name alone (D44). One already on disk is
    digested again, and replaced when its content does not match its name, so a
    truncated or altered page is repaired rather than read. A new page is written
    to a temporary file beside its destination and moved into place, so a write
    killed part-way leaves nothing at the final name.
    """
    directory = raw_dir(source, raw_root)
    directory.mkdir(parents=True, exist_ok=True)
    checksum = page_checksum(page)
    path = directory / f"{checksum}.json.gz"
    if path.exists() and _stored_checksum(path) == checksum:
        return path, False
    handle, temporary = tempfile.mkstemp(dir=directory, prefix=".page-", suffix=".tmp")
    os.close(handle)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            json.dump(page, stream, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
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


WINDOW_FRAMES: dict[str, tzinfo] = {
    "cfpb": UTC,
    "nyc311": nyc311.SOURCE_TIMEZONE,
}
"""§2.5: each source's dates resolve in its own civil frame.

311 is `America/New_York`, matching §2.4's reading of its floating timestamps —
a UTC window would cut its days four or five hours off. CFPB is UTC because it
publishes a per-record offset that normalization discards, so there is no single
civil frame a bare date could resolve against without inventing one.
"""


def resolve_window(source: str, start: date, end: date) -> tuple[datetime, datetime]:
    """Inclusive civil days in the source's frame, as UTC-aware instants (§2.5).

    Both endpoints are whole days: `--start` from its local midnight, `--end`
    through its local `23:59:59.999999`. Resolving each endpoint separately in
    the source frame is what makes a daylight-saving day come out 23 or 25 hours
    long rather than a fixed 24 — a UTC slice would be an hour wrong at one end.
    """
    frame = WINDOW_FRAMES[source]
    return (
        datetime(start.year, start.month, start.day, 0, 0, 0, 0, tzinfo=frame).astimezone(UTC),
        datetime(end.year, end.month, end.day, 23, 59, 59, 999999, tzinfo=frame).astimezone(UTC),
    )


def _normalize_source(
    source: str, pages: Iterable[SourcePage]
) -> Iterator[tuple[CorpusRecord, CFPBOutcome | NYC311Outcome] | nyc311.Exclusion]:
    """Each row's normalized pair, or, for NYC 311 only, the row's D50 `Exclusion`.

    NYC 311 rows go through `classify_row`, which refuses every problem outside D45's
    classes exactly as `normalize` does. CFPB has no D50 class and is unchanged.
    """
    for page in pages:
        if source == "cfpb":
            for row in cfpb.rows_from_page(page):
                yield cfpb.normalize(row)
        else:
            for row in nyc311.rows_from_page(page):
                yield nyc311.classify_row(row)


def _exclusion_counts(
    source: str, exclusions: Iterable[nyc311.Exclusion]
) -> dict[str, dict[int, int]]:
    """D50's `excluded_records`: kind -> created_date's civil year -> rows excluded."""
    counts = no_exclusions(source)
    for exclusion in exclusions:
        years = counts[exclusion.kind]
        year = exclusion.created_civil_date.year
        years[year] = years.get(year, 0) + 1
    return {kind: dict(sorted(years.items())) for kind, years in counts.items()}


def _exclusion_summary(excluded_records: dict[str, dict[int, int]]) -> str:
    """The exclusion counts, for a refusal message that must state them (D50)."""
    totals = {kind: sum(years.values()) for kind, years in excluded_records.items()}
    total = sum(totals.values())
    were = "row was" if total == 1 else "rows were"
    detail = ", ".join(f"{kind} {count}" for kind, count in totals.items())
    return f"{total} {were} excluded under D50 ({detail}). "


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


def authoritative_manifest(source: str, corpus_root: Path) -> CorpusManifest | None:
    """The manifest in `corpus_root` if it is authoritative, else `None`.

    Only an **unbounded** manifest (`limit` null) is authoritative (D24). This
    is the one predicate both the roster lock and the D26 refusal ask.
    """
    if not manifest_path(source, corpus_root).is_file():
        return None
    manifest = read_manifest(source, root=corpus_root)
    return manifest if manifest.limit is None else None


def authoritative_roster(source: str, corpus_root: Path) -> frozenset[str] | None:
    """The locked roster, or `None` when no authoritative corpus exists (D24).

    Only an **unbounded** manifest is authoritative. One written with a
    non-null `limit` describes a deliberately partial corpus, and its
    `label_roster` is the persisted subset rather than the window's taxonomy —
    adopting it would let a development run define the vocabulary a production
    run is judged against. That is the field earning §G's justification for it
    rather than merely stating it.

    The lock is read only from the run's own target root; no other root is
    consulted. Since D26 a limited run never reaches an authoritative lock at
    all — a limited run into a root holding one is refused before this is
    asked — so the lock found here only ever guards unbounded runs, and a
    limited run derives its roster from its own complete window.
    """
    manifest = authoritative_manifest(source, corpus_root)
    return None if manifest is None else frozenset(manifest.label_roster)


def _locked_roster(source: str, corpus_root: Path, observed_by_year: dict[int, set[str]]):
    """The roster to assert against.

    An unbounded manifest's roster when one exists; otherwise derived from the
    data as the intersection across years (§1). The assertion runs either way,
    so a label present in only part of the window fails rather than quietly
    becoming part of the vocabulary.
    """
    locked = authoritative_roster(source, corpus_root)
    if locked is not None:
        return locked
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
    acquisition: Path | None = None,
):
    """Fetch, normalize, validate, write, and describe one source's corpus.

    `acquisition` names an acquisition directory (D46). With `None`, the run reads
    `raw_root` exactly as it always has, checks no acquisition, and records
    `acquisition_id` as `None`. With a directory, that directory is the raw root for
    the fetch and for the read, a complete one is reused without calling `fetcher`,
    and the pages are normalized only after `verify_acquisition` has passed.
    """
    if end < start:
        raise InvalidDateRange(f"--end {end.isoformat()} precedes --start {start.isoformat()}")
    if limit is not None and limit < 1:
        raise InvalidLimit(f"--limit must be a positive integer; got {limit}")

    corpus_root = Path(corpus_root)
    raw_root = Path(raw_root) if acquisition is None else Path(acquisition)

    # D26: a truncated run may not replace a full corpus. Checked before the
    # fetch and before any write, so a refused run leaves no trace at all.
    if limit is not None:
        existing = authoritative_manifest(source, corpus_root)
        if existing is not None:
            raise AuthoritativeCorpusExists(
                f"{source}: {corpus_root} already holds an authoritative corpus "
                f"(corpus_id {existing.corpus_id}, {existing.record_count} records, "
                f"limit null). A --limit run may not replace it (D26); a limited run "
                f"must target a different root via --corpus-root."
            )

    window_start, window_end = resolve_window(source, start, end)

    acquisition_id: str | None = None
    if acquisition is None:
        fetch_into_cache(source, start, end, fetcher, raw_root)
    else:
        # D46: a completed acquisition is immutable and reused with zero requests;
        # whatever the fetch did, nothing is read until the record verifies.
        if not is_complete(raw_root):
            fetch_into_cache(source, start, end, fetcher, raw_root)
        acquisition_id = verify_acquisition(
            raw_root, source=source, start=start, end=end
        ).acquisition_id

    # The complete window, before any truncation. --limit bounds persistence
    # only (D24), so the roster below is derived from every candidate record.
    # D50: an otherwise-valid NYC 311 row of a D45 class is excluded rather than kept.
    # It is counted when its created_date's civil date lies in the window's civil
    # dates -- for a row that normalizes, the same test as the instant comparison --
    # and, like any row outside the window, it is not counted when it does not.
    pages_read = 0
    window: list[tuple[CorpusRecord, CFPBOutcome | NYC311Outcome]] = []
    exclusions: list[nyc311.Exclusion] = []
    for page in iter_cached_pages(source, raw_root):
        pages_read += 1
        for item in _normalize_source(source, [page]):
            if isinstance(item, nyc311.Exclusion):
                if start <= item.created_civil_date <= end:
                    exclusions.append(item)
                continue
            record, outcome = item
            if window_start <= record.submitted_at <= window_end:
                window.append((record, outcome))
    excluded_records = _exclusion_counts(source, exclusions)

    if not window:
        raise EmptyWindow(
            f"{source}: zero records fell inside "
            f"{window_start.isoformat()} .. {window_end.isoformat()} "
            f"({pages_read} cached page{'' if pages_read == 1 else 's'} read). "
            f"{_exclusion_summary(excluded_records) if source == 'nyc311' else ''}"
            "Nothing was written."
        )

    # Roster next, over the whole window. Nothing below may run before it.
    by_year: dict[int, set[str]] = {}
    counts: dict[str, int] = {}
    for record, _ in window:
        by_year.setdefault(record.submitted_at.year, set()).add(record.label)
        counts[record.label] = counts.get(record.label, 0) + 1
    if source == "cfpb":
        assert_roster(counts, _locked_roster(source, corpus_root, by_year))

    # D44: a record's identity is (source, external_id), so two records sharing one
    # refuse the run rather than one of them being kept. Still before any write; after
    # the roster, so a taxonomy failure is reported as the taxonomy failure it is. An
    # excluded row keeps its identity, so it cannot hide a duplicate (D50).
    identities = Counter((record.source, record.external_id) for record, _ in window)
    identities.update((source, exclusion.external_id) for exclusion in exclusions)
    duplicates = {key: count for key, count in identities.items() if count > 1}
    if duplicates:
        shown = ", ".join(
            f"{src}:{external_id} ({count} records)"
            for (src, external_id), count in sorted(duplicates.items())[:10]
        )
        raise DuplicateExternalId(
            f"{source}: {len(duplicates)} (source, external_id) pair"
            f"{'' if len(duplicates) == 1 else 's'} occur more than once in the window: "
            f"{shown}. Nothing was written."
        )

    # Only now does --limit decide what is persisted.
    kept = window if limit is None else window[:limit]
    records = [record for record, _ in kept]
    deltas = [
        (outcome.sent_to_company_at - record.submitted_at).total_seconds()
        for record, outcome in kept
        if isinstance(outcome, CFPBOutcome) and outcome.sent_to_company_at is not None
    ]
    paired = len(deltas)

    by_partition: dict[int, list[CorpusRecord]] = {}
    for record in records:
        by_partition.setdefault(record.submitted_at.year, []).append(record)

    # D37: NYC 311's outcome stream is persisted beside its records rather than
    # discarded here, because the risk model's entire target derives from it.
    # Task 19's O1 does the same for CFPB, whose `timely_response` is the probe's
    # evaluation target. Two sources, two sidecar schemas: each is written by its
    # own writer, and neither accepts the other's rows.
    outcomes_by_partition: dict[int, list[NYC311Outcome]] = {}
    cfpb_outcomes_by_partition: dict[int, list[CFPBOutcome]] = {}
    for record, outcome in kept:
        year = record.submitted_at.year
        if isinstance(outcome, NYC311Outcome):
            outcomes_by_partition.setdefault(year, []).append(outcome)
        else:
            cfpb_outcomes_by_partition.setdefault(year, []).append(outcome)

    local = _local_hour_source(source)
    diagnostic = build_diagnostic(
        source=source,
        submitted_local=[local(r.submitted_at) for r in records],
        deltas_seconds=deltas if source == "cfpb" else None,
        paired_count=paired,
        total_count=len(records),
    )

    api_version = cfpb.SOURCE_API_VERSION if source == "cfpb" else nyc311.SOURCE_API_VERSION

    # D27: every refusal has already run. From here the source's corpus is
    # replaced, and the manifest is the validity boundary — deleted first, written
    # last. A failure in between leaves no manifest, so no corpus; rerunning from
    # the raw cache recovers it. This is not atomic directory replacement.
    clear_corpus(source, root=corpus_root)
    for year, partition in sorted(by_partition.items()):
        write_partition(partition, source, year, 0, root=corpus_root)
    for year, outcome_partition in sorted(outcomes_by_partition.items()):
        write_outcome_partition(outcome_partition, source, year, 0, root=corpus_root)
    for year, cfpb_partition in sorted(cfpb_outcomes_by_partition.items()):
        write_cfpb_outcome_partition(cfpb_partition, source, year, 0, root=corpus_root)

    manifest = build_manifest(
        source=source,
        window_start=window_start,
        window_end=window_end,
        source_api_version=api_version,
        limit=limit,
        timestamp_diagnostic=diagnostic,
        root=corpus_root,
        acquisition_id=acquisition_id,
        excluded_records=excluded_records,
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
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=CORPUS_ROOT,
        help="corpus root to write; a --limit run needs one without an authoritative corpus",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="acquire the window from its source first (network); without it the run "
        "reads the raw cache only",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.fetch:
        ingest(
            source=args.source,
            start=args.start,
            end=args.end,
            limit=args.limit,
            corpus_root=args.corpus_root,
        )
        return 0

    # D46: the source's registered fetcher, or a refusal before any transport, any
    # request, any directory and any write.
    factory = FETCHERS.get(args.source)
    if factory is None:
        raise FetcherUnavailable(
            f"{args.source}: no fetcher is registered for this source, so --fetch cannot "
            "acquire it. Nothing was requested, created or written."
        )
    transport = RequestsTransport()
    resolved_start, resolved_end = resolve_window(args.source, args.start, args.end)
    context = FetchContext(
        source=args.source,
        start=args.start,
        end=args.end,
        resolved_start=resolved_start,
        resolved_end=resolved_end,
        directory=acquisition_dir(args.source, args.start, args.end),
        http=HttpClient(transport),
        client=transport.identity,
        sentinel_commit=current_commit(),
        now=lambda: datetime.now(UTC),
    )
    ingest(
        source=args.source,
        start=args.start,
        end=args.end,
        limit=args.limit,
        fetcher=factory(context),
        corpus_root=args.corpus_root,
        acquisition=context.directory,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

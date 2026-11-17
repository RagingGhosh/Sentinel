# Sentinel — Phase 2 Specification Addendum

**Date:** 2026-09-04
**Status:** Approved with amendments (C1/C2 resolved 2026-09-04; see D15–D19)
**Amends:** `docs/superpowers/specs/2026-09-03-sentinel-design.md` §6 (ML pipeline)
**Basis:** measured data reconnaissance, 2026-09-04. Every quantity below was
measured, not assumed; each headline figure was verified by a second method.

This addendum supersedes §6 of the Phase 1 design wherever the two conflict.
It changes nothing in §§1–5 or §§7–13: the Phase 1 architecture stands.

---

## 0. Measurements this addendum rests on

| Measurement | Value | How verified |
|---|---|---|
| CFPB complaints, total | 17,546,059 | API aggregate |
| CFPB with published narrative | 3,847,257 (21.93%) | aggregate **and** independent filtered query — identical |
| CFPB narrative length | median 1,202 chars (max 3,871) | 60-row sample |
| CFPB `timely` = Yes | 17,438,176 (99.39%) vs 107,883 No (0.61%) | API aggregate |
| CFPB `timely` within narratives | 98.88% / 1.12% (43,115 No) | API aggregate |
| CFPB `product` values | 21 total; **11 stable across 2023–2025** | per-year aggregates |
| CFPB narratives 2024 + 2025 | 814,385 + 1,222,049 = **2,036,434** | per-year aggregates |
| CFPB resolution date | **does not exist** — only `date_received`, `date_sent_to_company` | field list |
| NYC 311 rows, 2020–present | 22,355,568 (3,445,027 in 2024) | SoQL count |
| NYC 311 distinct `complaint_type` | 276 | SoQL count distinct |
| NYC 311 `descriptor` length | **median 15 chars** | 300-row sample |
| NYC 311 `closed_date` coverage | 290,596 / 297,763 = **97.6%** | June 2024 window |
| NYC 311 resolution hours | p50 9.8, p90 732, p99 8,183, max 19,600 | June 2024, n=4,801 |
| NYC 311 breach rate by threshold | 24h→37.4%, 72h→31.1%, 168h→19.5%, 720h→10.2% | June 2024 |
| — same, second window | 24h→39.7%, 72h→26.9%, 168h→17.4%, 720h→9.0% | Feb 2025, n=4,968 |
| Priority / assignee / queue depth | **absent from both sources** | CFPB 17 fields, 311 48 columns |

---

## 1. CFPB window and taxonomy

**Decision: train and evaluate on `date_received` in 2024-01-01 … 2025-12-31 only.**
Both endpoints are inclusive, and each source's dates resolve in its own
civil frame — see §2.5, which pins what a bare `YYYY-MM-DD` means.
That window holds 2,036,434 narratives under a label vocabulary measured stable
across 2023, 2024 and 2025.

**Why older eras are excluded.** The CFPB `product` field is not a stable
taxonomy that grew imbalanced; it is a **different vocabulary in each era**:

| Year | Narratives | Labels in use | Largest class |
|---|---|---|---|
| 2016 | 77,766 | 12 | `Credit reporting` — 19.4% |
| 2020 | 174,297 | 9 | `Credit reporting, credit repair services, or other personal consumer reports` — 55.0% |
| 2024 | 814,385 | 11 | `Credit reporting or other personal consumer reports` — 77.4% |

Those three largest classes are the same underlying concept under three
successive names; together with the residual `Credit reporting` they account for
65% of all narratives across history. `Credit card` / `Credit card or prepaid
card` / `Prepaid card` and three payday-loan variants split the same way.

Including older eras would force one of two bad outcomes. A **random** split
leaks era into both folds and scores well by learning era-specific label
priors — a number that means nothing. A **temporal** split trains on labels that
do not exist in the test period and tests on labels that did not exist in
training, which is not a generalisation test but a vocabulary mismatch.

Restricting the window makes the taxonomy coherent, and 2.0M narratives is far
more than the models need.

**The label roster is derived, not hardcoded.** Ingest computes the label set as
the intersection of the `product` values present in each year of the window. No
label list is transcribed into code from this document; the assumption is
executable rather than documentary.

### 1.1 Roster failure policy

Deriving the roster is not enough on its own — a roster can stay eleven labels
wide while one label is replaced by another. The rule is therefore about
membership, not count.

**Ingest reports the observed roster before processing any records, then fails
loudly if it differs from the locked roster in either direction:**

| Condition | Behaviour |
|---|---|
| A `product` value appears that is not in the locked roster | **Fail.** Report the unexpected label and its record count. |
| A locked roster label is absent from the window | **Fail.** Report which label vanished. |
| Roster matches exactly | Proceed, having printed the roster and per-label counts. |

**"The window" means every record the window normalizes to, before any
development truncation.** `--limit` bounds what is written to disk and nothing
else (D24): a locked label that a truncated run happened not to keep has not
vanished from the source, and reporting it as missing would be a taxonomy alarm
manufactured by a development flag. Roster derivation and assertion therefore
run over the complete normalized window.

**Silently dropping the record, mapping it to a neighbouring label, mapping it
to `other`, or auto-expanding the taxonomy are all prohibited.** Each would
change the experimental population underneath a published benchmark without
anyone deciding to, which is precisely the failure this window exists to
prevent — CFPB has renamed this taxonomy at least twice already, and will again.

A taxonomy change is a **spec decision with a version bump**, not an ingest-time
inference. When ingest fails this way, the correct response is to amend this
document, restate the window and roster, and note in `metadata.json` that
artifacts before and after are not comparable.

**Residual imbalance is accepted and reported, not engineered away.** The
largest class is ~77% of the 2024 window. Metrics are chosen accordingly (§5),
and the majority-class baseline is always reported beside the model.

---

## 2. Corpus versus operational data

**Decision: external records never enter the operational database.**

The `Complaint` table is operational state for a live application: every row has
a non-null `submitted_by` foreign key with `PROTECT`, appears in the agent
queue, and participates in the lifecycle. Inserting 2.0M CFPB narratives or
22.4M 311 records would require fabricating a submitter for each, would make the
queue unusable, and would put a corpus into a free-tier Postgres for no
operational reason.

**Corpus records live on disk as files** (Parquet under `data/`, gitignored),
read by training and evaluation code only, and always through `load_corpus`, the
manifest-gated corpus reader (§2.7, D27). They are never migrated into, joined
against, or synchronised with `complaints_complaint`.

### 2.1 The corpus record

The Phase 1 spec's `RawComplaint` is **withdrawn** — its `sla_met` field unified
two incomparable constructs (§4). It is replaced by a record carrying only what
both sources genuinely share, with outcomes held separately per source:

```python
@dataclass(frozen=True)
class CorpusRecord:
    source: str            # domain pack slug: "cfpb" | "nyc311"
    external_id: str       # the source's own identifier
    text: str
    label: str             # the source's own category label, from §1's roster
    submitted_at: datetime
```

```python
@dataclass(frozen=True)
class CFPBOutcome:
    external_id: str
    timely_response: bool                  # see §4.1
    sent_to_company_at: datetime | None    # provenance evidence only — see §2.3

@dataclass(frozen=True)
class NYC311Outcome:
    external_id: str
    closed_at: datetime | None
    resolution_hours: float | None   # see §4.2
```

`CFPBOutcome.sent_to_company_at` is **not a model feature and not a target**. It
exists solely so §2.3's field-delta measurement is computable from the corpus
rather than only from the raw cache, which keeps the provenance verdict
reproducible from committed artifacts. Using it as a feature would be a defect:
it is downstream of intake and unavailable for a live complaint.

A source contributes a `CorpusRecord` stream and, where it has one, an outcome
stream. Nothing forces a source to have both — CFPB has no resolution time at
all, and that asymmetry is now expressible rather than papered over.

### 2.2 Identity: corpus records have no complaint IDs

A corpus record is identified by `(source, external_id)` — never by an integer
that could be mistaken for a `Complaint` primary key. **No corpus record is ever
assigned a synthetic complaint ID, and no evaluation output is typed to
`Complaint.pk`.**

This has a direct consequence for the dedup work. `ml.base.Match` is typed
`complaint_id: int` and is correct as shipped — it describes a *serving* result
over live complaints. The duplicate-retrieval benchmark (§5.3) runs over corpus
records, which have no such id, so it must not reuse that type.

**Resolution: separate what is benchmarked from what is served.**

- **`TextEmbedder`** — new in Phase 2, the thing actually benchmarked: text in,
  vector out, plus a `model_version`. Both the MiniLM and TF-IDF candidates
  implement it. The benchmark measures embedders over `(source, external_id)`
  pairs and never touches `Match`.
- **`DedupIndex`** — unchanged from Phase 1. It remains the serving interface
  over live `Complaint` rows and continues to return `Match`. Phase 3 wires the
  winning embedder behind it.

Evaluation types live in the training package and are never imported by serving
code. The boundary is enforced by that import direction, not by convention.

### 2.3 Timestamp provenance diagnostic

`submitted_hour` and `submitted_weekday` are features in both `RiskFeaturesV1`
and `TransferFeaturesV1`. They are worthless if the timestamp they derive from
was stamped when the dataset was assembled rather than when the complaint was
filed. The reconnaissance raised that possibility for CFPB: on a sampled record,
`date_received` and `date_sent_to_company` were **three seconds apart**, which is
not a plausible interval for forwarding a complaint to a company.

**Distribution shape does not establish provenance.** An hour-of-day histogram
that looks non-diurnal proves nothing on its own: complaints aggregated across
time zones, batch-forwarded submissions, and web intake with automated retries
all produce flat or spiky distributions from genuine event times. A diurnal-
looking histogram proves nothing either. **The hour/weekday diagnostics are
therefore classified as distributional anomaly detection, not evidence of
timestamp provenance**, and they may never on their own establish that a
timestamp is an artifact.

#### Primary evidence: the field-delta measurement

Where a source exposes two timestamps whose real-world interval is known to be
non-trivial, the delta between them is direct evidence. For CFPB, `date_received`
and `date_sent_to_company` bracket a real administrative process; if both were
stamped at load time, that process would appear to take seconds.

Ingest measures, over records where both fields are present:

- `pair_coverage` — the fraction of records having both timestamps;
- the delta distribution in seconds, reported at **p5, p25, p50, p75, p95, p99**,
  plus the median stated explicitly;
- `frac_delta_le_1min`, `frac_delta_le_10min`, `frac_delta_le_1h`;
- `count_delta_negative`, `count_delta_zero`;
- `frac_identical_timestamps` — deltas of exactly zero at full precision.

#### The verdict rule, pre-specified

The verdict is a function of the delta metrics, fixed here **before any data is
seen**, so it is not a judgment made after looking at results:

| Verdict | Rule |
|---|---|
| `strongly_suspicious_load_timestamp` | `median_delta_seconds <= 60` **or** `frac_delta_le_1min >= 0.50` **or** `frac_identical_timestamps >= 0.20` |
| `supported_plausible_event_time` | `median_delta_seconds >= 3600` **and** `frac_delta_le_1min < 0.05` **and** `count_delta_negative == 0` |
| `suspicious_insufficient_evidence` | anything else, **and always** when `pair_coverage < 0.50` |

**The distributional diagnostics do not enter this rule.** They are recorded as
`secondary_evidence` and may downgrade `supported_plausible_event_time` to
`suspicious_insufficient_evidence` when they are extreme — a conservative move
toward doubt. They may never produce `strongly_suspicious_load_timestamp`, and
they may never upgrade a verdict.

#### What "extreme" means: the `hour_concentration` threshold

"Extreme" was left undefined above, which is a gap in a rule whose whole purpose
is to be fixed before the data is seen. It is now pinned.

**`hour_concentration` is the proportion of records belonging to the most
frequent submitted hour among the applicable records** — the largest of the 24
hour-of-day counts divided by their total. A perfectly uniform distribution
gives 1/24 ≈ 0.042; a source that stamped every record in one hour gives 1.0.

**Threshold: `hour_concentration >= 0.50` is extreme.** That is roughly twelve
times uniform, and means half of everything landed in a single hour of the day.

This threshold is a **Sentinel project decision, not a source-derived fact**.
Nothing in either source's documentation suggests it; it is a line this project
draws, fixed here before any corpus is evaluated, and recorded in the manifest
beside the verdict so a reader sees the number that was applied rather than
inferring it.

Its effect is bounded in one direction only:

| Condition | Effect |
|---|---|
| `hour_concentration >= 0.50` and verdict is `supported_plausible_event_time` | Downgrade to `suspicious_insufficient_evidence` |
| `hour_concentration >= 0.50` and verdict is anything else | **No effect** |
| `hour_concentration < 0.50` | No effect |

It can never produce `strongly_suspicious_load_timestamp` and can never upgrade
a verdict. The primary evidence remains the `date_received` →
`date_sent_to_company` delta distribution, and distribution shape still does not
prove timestamp provenance — a concentrated histogram is a reason to withhold
confidence, never a reason to assert an artifact.

No other distributional threshold exists. The chi-square statistic and its
p-value are recorded for a reader, and do not gate the verdict.

#### Sources without a testable pair

NYC 311 exposes `created_date` and `closed_date`, but their interval **is the
target variable** (`nyc311_resolution_hours`), so using it as a provenance check
would test the label with the label. 311 therefore receives the distributional
diagnostic only, and its verdict is `suspicious_insufficient_evidence` by
construction — recorded honestly as "not directly testable by this method"
rather than defaulted to `supported`. Absence of evidence is not recorded as
evidence of soundness.

#### What the verdict is allowed to decide

The verdict gates one downstream decision, stated in §5.4: whether the robustness
probe's figures are classified as substantive. Nothing else consumes it, and it
is never described as having proven anything about how the data was produced —
only as having measured an interval that is or is not consistent with a real
process.

### 2.4 NYC 311 timestamp interpretation

**Source fact.** The NYC Open Data schema types `created_date` and `closed_date`
as **Floating Timestamp**. That type carries no UTC offset, and the source
documentation does not state a timezone as part of the field type. A row arrives
spelled `2024-06-01T09:30:00.000` and parses to a naive datetime.

**Project decision.** Phase 2 interprets both fields as **`America/New_York`
civil (local) time**, and converts them to UTC-aware datetimes for storage. This
is Sentinel's interpretation of an unlabelled field, not a claim that the source
publishes an offset. It is recorded here so a reader can disagree with it
explicitly rather than discover it in code.

The decision is forced to be *some* decision: `ingest.storage.write_partition`
refuses a naive `submitted_at`, so a floating timestamp cannot reach the corpus
uninterpreted. The choice is therefore which interpretation, not whether to make
one. Assuming UTC directly is rejected — it would assert that New Yorkers
contact 311 on UTC wall-clock time.

| Rule | Behaviour |
|---|---|
| Both fields | Interpreted as `America/New_York` civil time, then converted to UTC |
| `CorpusRecord.submitted_at` | Stored as the UTC-aware instant |
| `submitted_hour`, `submitted_weekday` | Derived from the **local** representation, never from the UTC one |
| `nyc311_resolution_hours` | Computed from the UTC-aware **instants** |
| Ambiguous local time (autumn fold) | **Reject.** Never silently choose a fold |
| Nonexistent local time (spring gap) | **Reject.** Never heuristically shift |

**Why the features come from the local representation.** `submitted_hour` exists
to capture when a human contacted the service. A New Yorker calling at 09:00
EDT is 13:00 UTC, so a UTC-derived hour shifts the whole diurnal pattern — and
shifts it by a *different* amount either side of a DST boundary, smearing the
daily signal the feature exists to carry. CFPB's timestamps arrive with real
offsets, so its `submitted_hour` is already anchored to the filer's own clock;
deriving 311's from UTC would make one feature name mean two different things
across the two domains that §5.4's cross-domain probe compares directly.

**Why resolution hours come from the instants.** Elapsed time must be measured
between instants, not between wall-clock readings. A request opened before a DST
transition and closed after it spans a local clock that jumped an hour;
subtracting the two floating timestamps would report that hour as real work
done, or as an hour of work never done. Converting both to UTC first makes the
interval correct by construction.

**Why the two DST cases are rejected rather than resolved.** In the autumn
transition the hour 01:00–02:00 local occurs twice, and nothing in a floating
timestamp says which. Choosing a fold would assign a wrong instant to roughly
half the affected records, invisibly. In the spring transition 02:00–03:00 local
never occurs at all, so a value there means either the source's clock handling
is broken or this section's interpretation is wrong — both worth stopping for.
The affected volume is about one hour of records per year in each direction,
which is a negligible loss and a loud signal. This follows §1.1's standing
principle: a normalizer that silently repairs a record changes the experimental
population underneath a published benchmark without anyone deciding to.

**If this interpretation is wrong**, it is wrong in one place and by a fixed
offset, and every affected number is recomputable from the corpus. That is the
reason for naming it here rather than leaving it implicit.

### 2.5 Ingestion window semantics

The CLI takes `--start YYYY-MM-DD --end YYYY-MM-DD`. That surface is unchanged;
what a bare date *means* was never stated, and is pinned here.

**Both bounds are inclusive civil dates.** `--start` includes the whole of that
day and `--end` includes the whole of its own, through `23:59:59.999999`. §1
writes the window as `2024-01-01 … 2025-12-31`, which reads as both endpoints
included; an exclusive `--end` would silently drop the final day of every window
anyone typed from that sentence.

**The civil frame is the source's own.**

| Source | Frame | Because |
|---|---|---|
| `nyc311` | `America/New_York` | §2.4 already reads its floating timestamps as New York civil time. A window in any other frame would cut its days in the wrong place |
| `cfpb` | UTC | CFPB publishes a per-record offset and normalization converts to UTC, discarding it. There is no single CFPB civil frame a bare date could resolve against, and the stored instant is all that survives |

Both bounds resolve to UTC-aware instants and are compared against
`CorpusRecord.submitted_at`, which is UTC-aware for both sources. The manifest's
`window_start` and `window_end` record the **resolved instants**, not the dates
supplied, so a reader sees the interval that was actually applied rather than
having to re-derive it.

A single UTC rule for both sources was rejected: it would shift 311's day
boundaries by four or five hours, putting a request filed at 20:00 on the last
day of a window outside it. Giving CFPB a civil frame would be worse, since it
would mean inventing a timezone for records that carry their own.

#### An empty window is a failure, not an empty corpus

**If normalization over the requested window yields zero records, ingest fails
with a typed error and writes nothing** — no partition, no manifest.

The alternative is worse than an error. An empty run would derive an empty
roster and lock it into the first manifest, after which every later ingest fails
with every label unexpected: a self-inflicted version of exactly the taxonomy
corruption §1.1 exists to prevent. Failing at the point of emptiness keeps that
unreachable.

The error is raised **before** roster derivation, and reports the source, the
resolved window bounds, how many cached pages were read, and that none of their
records fell inside the window — enough to tell an empty cache apart from a
cache whose records all lie outside the window.

#### Every refusal that precedes a corpus write

The empty window is one of several refusals ingest raises before it writes
anything to a corpus root. They are listed together so none is found by accident:

| Refusal | Raised when |
|---|---|
| `InvalidDateRange` | `--end` precedes `--start` |
| `InvalidLimit` | `--limit` is not a positive integer (§2.8, D28) |
| `AuthoritativeCorpusExists` | a `--limit`ed run targets a root holding an authoritative corpus (§2.6, D26) |
| fetch and adapter normalization errors | a page cannot be fetched, or a row violates its source's contract — for example §2.4's DST rejections |
| `EmptyWindow` | the window normalizes to zero records (this section, D23) |
| `RosterMismatch` | the window's labels differ from the roster in either direction (§1.1) |

Each leaves an existing corpus untouched. Only once every one of them has passed
does a run begin replacing the corpus (§2.7, D27).

### 2.6 Corpus replacement

A successful run replaces the source's whole versioned tree with a corpus of the
current run's window, and nothing from the previous corpus survives it (§2.7,
D27). An unbounded rerun over an unbounded corpus is therefore idempotent — the
same window and raw cache produce the same part files and, within one
environment, the same `corpus_id` — and that rerun is the resume path. What was
never stated is what a `--limit`ed run may do to a root that already holds a
full corpus. Left unstated, it replaced it: a truncated run
overwrote the authoritative partitions and manifest, and every artifact citing
the old `corpus_id` was left naming a corpus that no longer existed.

**A limited run may not replace an authoritative corpus.** An authoritative
corpus is one whose manifest records `limit: null` (D24). When the target corpus
root already holds an authoritative manifest for the source, a run with
`--limit` raises `AuthoritativeCorpusExists`, a subclass of `IngestError`,
**before `fetch_into_cache` and before any write**. The message states the
source, the corpus root, the existing manifest's `corpus_id` and `record_count`,
and that a limited run must target a different root.

| Target corpus root holds | Unbounded run | `--limit`ed run |
|---|---|---|
| nothing | Writes an authoritative corpus | **Allowed.** Writes a truncated corpus recording its `limit` |
| a truncated corpus (`limit` non-null) | Allowed. Replaces it with an authoritative corpus | **Allowed.** Replaces one development corpus with another |
| an authoritative corpus (`limit: null`) | **Allowed.** Replaces it with the current window's corpus, even a narrower one (D27); idempotent when the window and cache are unchanged | **Refused** with `AuthoritativeCorpusExists` |

**A development corpus is created deliberately, elsewhere.** The CLI accepts
`--corpus-root PATH`, defaulting to the existing corpus root (`data/corpus/`). A
truncated corpus is produced with `--limit N --corpus-root <another path>`, and
succeeds against a fresh root or one already holding a truncated corpus. The
argument names where output is written and nothing more: it introduces no second
kind of corpus, and every root has the layout plan §G describes.

**Consequence for roster validation.** A run reads its roster lock only from its
own target corpus root (D24). A limited run therefore never consults an
authoritative roster: the only configuration in which one would be present in
its root is the configuration refused above. A limited run derives its roster
from its own window, as §1 prescribes when no lock exists, and asserts against
that. The authoritative roster continues to guard every unbounded run into the
authoritative root — the corpus artifacts are trained on. §1.1's rule is
unchanged: whenever roster validation runs, it runs over the complete normalized
window before `--limit` truncates anything.

**Rejected alternatives.** Writing limited runs to a distinct non-authoritative
tree is infrastructure no measurement justifies (invariant 4), and would leave
every downstream reader to decide which tree to read. Merging a limited run into
the existing corpus yields a population that is neither the full window nor the
truncated one — the unannounced change to the experimental population §1.1
exists to prevent — under a `corpus_id` naming a corpus no single run produced.
Permitting replacement contradicts the reason plan §G gives for recording
`limit`: an artifact can see whether the corpus behind its `corpus_id` was
complete only if a truncated run cannot silently take that corpus's place.
Requiring a separate root without adding `--corpus-root` would be a rule the CLI
gives no way to obey.

**A narrower unbounded window replaces the corpus completely.** An unbounded run
over a narrower window than an existing authoritative corpus is allowed, and
every record outside the new window is removed (§2.7, D27). §1.1 still applies:
a locked CFPB label absent from the narrower window fails `RosterMismatch`.

### 2.7 Corpus writes, validity and reading

Beyond the layout in plan §G, what a corpus root contains was never specified.
`write_partition` overwrote one path and deleted nothing, `build_manifest`
described whatever part files were on disk, and `read_corpus` read whatever part
files were on disk whether or not a manifest existed. A rerun that wrote fewer
year partitions than its predecessor left the rest behind and counted them, and a
run failing part-way left new partitions under the previous manifest. This
section fixes both (D27).

**A successful run replaces the source's corpus with the current window.** Once
every refusal listed in §2.5 has passed, ingest:

1. deletes the source's `manifest.json`;
2. deletes the rest of `<corpus root>/<source>/v<SCHEMA_VERSION>/`;
3. writes the run's partitions;
4. writes the manifest last, to a temporary file in the same directory that is
   then moved into place with `os.replace`.

The resulting corpus is the current run's window, whatever the previous corpus
held — including a window narrower than the one it replaces. Stale partitions
cannot survive. No other source, and no other schema version's tree, is touched:
the storage layout keeps an older `v<N>` findable precisely so that an artifact
citing it can still locate its bytes.

After a successful run:

- the part files on disk are exactly the manifest's `part_files`;
- every record lies within `window_start` .. `window_end`;
- a limited manifest satisfies `1 <= record_count <= limit` (§2.5, §2.8).

**The manifest is the validity boundary.** A source/version tree without a valid
manifest is not a corpus, whatever Parquet files it contains. The manifest is
deleted before any partition is removed or written, and written only after every
partition, so its presence means a run completed and wrote exactly what it lists.

**`load_corpus` is the only corpus reader.** It lives beside the manifest and:

- raises `ManifestNotFound` when there is no manifest;
- raises a `CorpusIntegrityError` when a part file exists on disk that
  `part_files` does not list;
- raises `ChecksumMismatch` when a listed part file is missing or its bytes
  differ from the recorded digest;
- otherwise streams exactly the listed files, in `(submitted_at, external_id)`
  order, holding at most one batch per part file.

Byte verification is required rather than optional because plan §R already
claims that `corpus_id` ties an artifact to exact input bytes, and that claim
holds only if the bytes are checked when the corpus is loaded.

**`read_corpus` is a low-level partition reader, not a corpus reader.** It keeps
Task 4's signature and behaviour: it reads whatever part files exist and asserts
nothing about validity. That is exactly what `build_manifest` needs — it reads
the partitions a run has just written, before the manifest describing them
exists — and it is what the storage tests exercise. Nothing else may use it. It
cannot itself require a manifest: the manifest module already depends on the
storage module, so the reverse dependency would be circular, and `build_manifest`
would be left with nothing to read with.

**A validity boundary, not atomic directory replacement.** D27 does not replace
the tree atomically, and nothing here claims that it does.

| | What D27 provides | What it does not provide |
|---|---|---|
| A reader using `load_corpus` | The previous completed corpus, no corpus, or the new completed corpus — never a mixture of two runs | Continuous availability: between deleting the old manifest and writing the new one, the source has no corpus |
| A run that fails after clearing | No manifest, so nothing that could be mistaken for a complete corpus | Survival of the previous corpus |
| The manifest file itself | A complete document or none, through a single-file `os.replace` | Any atomic swap of the directory tree around it |

Staging a complete tree and switching it in was rejected. On Windows `os.replace`
cannot move a directory over an existing non-empty one, so the switch becomes two
renames with a gap between them, and making it genuinely atomic needs a pointer
that changes plan §G's layout. That is infrastructure no measurement justifies
(invariant 4), spent protecting a corpus plan §X already treats as regenerable.

**Recovery after a failed load is a rerun from the raw cache.** A run that fails
after clearing leaves no manifest: that root is not authoritative (D26), holds no
roster lock (D24), and `load_corpus` refuses it. Rerunning over the same window
rebuilds the corpus from the raw cache without network access.

**Not decided here:** concurrent ingests into, or reads from, one corpus root.

### 2.8 The value of `--limit`

**`--limit` must be a positive integer.** Zero or a negative value raises
`InvalidLimit`, a subclass of `IngestError`, immediately after the
`InvalidDateRange` check — before the D26 refusal, before `fetch_into_cache`, and
before any write. The message names the value supplied, and the command line
reaches the same check because the CLI passes the value through unchanged (D28).

`--limit 0` would persist the empty corpus §2.5 already defines as a failure,
with an empty roster and a diagnostic computed over nothing. A negative value
bounds nothing: applied as a slice, `-1` silently drops the window's last record
and records `limit: -1`. Checking before D26 means a nonsensical value never
causes a manifest to be read.

This rule concerns the value of `limit` alone and is independent of how a run
replaces the corpus (§2.7).

---

---

## 3. RiskFeatures v1

**Decision: three features are removed from the historical training feature set,
because neither source corpus contains them.** Measured: CFPB exposes 17 fields
and NYC 311 exposes 48 columns; neither has priority, assignee, or any queue
signal.

Removed from v1 training:

| Feature | Why removed |
|---|---|
| `priority_rank` | Neither source has a priority concept. It is a Sentinel workflow field. |
| `queue_depth` | Depends on Sentinel's own backlog at submission time. Undefined for an external record. |
| `assignee_open_count` | Depends on Sentinel's own assignment state. Undefined for an external record. |

**They are not deleted from the codebase.** `ml.base.RiskFeatures` remains the
serving-time contract and keeps all ten fields, because a Phase 3+ model trained
on Sentinel's own accumulated history *will* have them. What changes is that the
**v1 model is trained on a strict subset**, and the artifact's `feature_spec`
records exactly which subset, so a model can never be served features it was not
trained on.

### 3.1 A fourth feature is also unusable, and this was not previously noticed

`age_hours` is defined as "age at prediction time". For a corpus record the
prediction point is intake, so `age_hours` is identically zero for every
training row. It is therefore **also excluded from v1 training** and retained
only in the serving contract.

**This is a zero-variance problem, not a missing-data problem, and the
distinction matters.** The value is not absent, unknown, or imputable — it is
known exactly, and it is the same for every row. No imputation strategy, default
value, or richer extraction recovers signal from it, because there is none to
recover: a feature with no variance cannot carry information about a target.
Treating it as missing would invite someone to "fix" it later by filling it in.
It becomes informative only under a prediction point where complaints have
genuinely differing ages — which is Sentinel's own operational history in
Phase 3+, not a historical corpus.

### 3.2 The v1 training feature set, and an honest warning

What remains computable from a 311 corpus record:

| Feature | Source | Note |
|---|---|---|
| `submitted_hour` | `created_date` | domain-free |
| `submitted_weekday` | `created_date` | domain-free |
| `text_length` | `descriptor` | weak — 311 descriptors have a median of 15 characters |
| `category_mean_resolution_hours` | training fold only | target-derived — see §6.3 |
| `category_breach_rate` | training fold only | target-derived — see §6.3 |

**`sla_hours` was removed from this set (see D15).** An earlier draft listed it as
a sixth feature, sourced from the §7 per-type threshold. That threshold is the
p75 of training *resolution hours* — the same outcomes that define the target —
so a training row's own resolution time contributes to the p75 that would become
that row's own feature. It is target-derived in exactly the sense §6.3 exists to
prevent, and the frozen-label design offers no leakage-safe construction for it:
the label requires one frozen threshold per type, while a leakage-safe feature
would require an out-of-fold one, and those cannot be the same number.
Maintaining two distinct thresholds — one for the label, one for the feature —
was rejected as a source of future error without a compelling reason to accept
it. **RiskFeaturesV1 is therefore five features.**

**This is a thin feature set, and the resulting model may be weak.** That is a
finding, not a failure. The spec's commitment is to measure and publish
honestly; if a v1 risk model does not beat its majority-class baseline by a
meaningful margin, the README says so and the model does not ship behind a
serving path. A weak, honestly-reported model is a better portfolio artifact
than a strong-looking one built on leakage.

### 3.3 The artifact is authoritative about its own features

Retaining a ten-field conceptual contract while training on five creates an
implementation trap: a serving path that must construct a ten-field
`RiskFeatures` in order to call a model that needs five will be asked for
`queue_depth` in an environment that cannot produce it. Three concepts are
therefore named separately:

| Concept | Meaning |
|---|---|
| **`RiskFeatures`** (the contract) | The complete conceptual interface. Every feature Sentinel may ever compute about a complaint. Stable across model versions. |
| **`RiskFeaturesV1`** (the trained subset) | The five features model v1 actually accepts, per §3.2. |
| **`TransferFeaturesV1`** (a separate, smaller subset) | The three features the reduced-feature cross-domain cross-target robustness probe accepts, per §5.4. Versioned independently of `RiskFeaturesV1`. |
| **`feature_spec`** (in `metadata.json`) | The exact, ordered, versioned feature list the artifact was trained on. |

**The rule: at inference, the artifact's `feature_spec` — not the breadth of the
serving interface — determines which features are built.** The inference adapter
reads `feature_spec`, constructs exactly those features in exactly that order,
and raises if a named feature is unavailable. Features outside the spec are
never computed, so an environment that cannot produce `queue_depth` is never
asked for it.

This makes `feature_spec` a **compatibility guard rather than documentation**: a
model can never be served a vector it was not trained on, and swapping in a v2
artifact that needs more features fails loudly at load time instead of silently
mis-scoring.

**Consequence for the Phase 1 interface, stated plainly:** the four excluded
fields become optional (`| None`) on `RiskFeatures`, because they genuinely are
unavailable in some contexts and a required field that cannot be supplied is a
lie in the type. That is a real change to a Phase 1 interface. It lands in
**Phase 3**, with the inference adapter that consumes it — Phase 2 does no
serving and needs no such change. §8's architecture table is corrected
accordingly.

---

## 4. SLA labels: two constructs, never unified

**Decision: the Phase 1 `sla_met` field is withdrawn.** The two sources measure
different things and are given different names, definitions and uses.

### 4.1 `cfpb_timely_response` (boolean)

The CFPB `timely` field: whether the company responded to the consumer within
the CFPB's own response window. Measured distribution: **99.39% true** across
all complaints, 98.88% within narratives.

It is a **regulatory responsiveness flag about a company**, not a measure of
whether a complaint was resolved before a service deadline. It says nothing
about how long resolution took — CFPB publishes no resolution date at all.

Permitted use: the reduced-feature cross-domain cross-target robustness probe
of §5.4, with imbalance-aware
metrics. **Not** permitted: use as a primary training target, or as a component
of any unified SLA label.

### 4.2 `nyc311_resolution_hours` (float) and `nyc311_sla_breach` (boolean)

`resolution_hours = closed_date − created_date`, available for 97.6% of
records, with no negative durations observed. `nyc311_sla_breach` is derived
from it by the threshold rule in §7.

This is a **genuine elapsed-time measure of resolution**, and it is the primary
training target for the risk model.

### 4.3 The prohibition

No field, column, dataclass or variable in Phase 2 may combine these two under a
single name. A model trained on one is never evaluated on the other except
through the explicitly-labelled cross-domain cross-target robustness probe in
§5.4, which is exploratory analysis rather than a claim that either target
substitutes for the other.

---

## 5. Evaluation metrics

### 5.1 Universal rules

Every reported metric is accompanied by a **majority-class baseline** computed
on the same split. Any metric quoted without its baseline is incomplete.

**Accuracy is never a headline metric** in this project. At the measured class
ratios — 77% for the CFPB majority label, 99.39% for `timely` — accuracy
describes the class prior, not the model.

### 5.2 Triage classifier (CFPB narratives → label)

Headline: **macro-F1** across the §1 roster. Reported alongside: per-class
precision/recall/F1, the confusion matrix, and top-3 accuracy (the UI presents a
shortlist). Baselines: majority-class and stratified-random.

### 5.3 Duplicate retrieval (MiniLM vs TF-IDF)

Unchanged in method from Phase 1 §6.3 and still labelled honestly as a
**synthetic duplicate-retrieval benchmark, not real-world duplicate-detection
accuracy**. Metric: recall@k over perturbed held-out records, measured over
`(source, external_id)` pairs per §2.2. The MiniLM candidate ships only if it
beats the TF-IDF candidate; the result is published either way.

**The comparison methodology is locked before either candidate is built.** A
benchmark where the two arms differ in more than one respect cannot attribute
its result to the representation — which is the only question it exists to
answer. Both candidates must therefore share, identically:

| Held constant | |
|---|---|
| Temporal train / validation / test boundaries | the same cut dates, per §6.1 |
| Evaluation population | the same held-out records and the same perturbations applied to them |
| Target definition | the same notion of a correct retrieval |
| Preprocessing leakage rules | §6.2 applies to both; in particular a TF-IDF vocabulary is built from training text only |
| Retrieval procedure | the same similarity function, the same k, the same index population |
| Downstream classifier family | where an embedder feeds a classifier, the same family and the same tuning budget |
| Primary metric and baselines | the same recall@k definition and the same baselines |

**The representation is the only permitted difference between the arms.** Any
other divergence — a different k, a differently-tuned classifier, a
differently-built index — invalidates the comparison, and the benchmark is
re-run rather than reported with a caveat.

Tuning budget is held equal rather than optimal: if one arm receives a
hyperparameter search, so does the other, over a comparable space. An
unequal-effort comparison measures effort, not representation.

**Embedding dimension is recorded and validated, never assumed.** The MiniLM
family does not share one output width — `all-MiniLM-L6-v2` and
`all-MiniLM-L12-v2` emit 384 dimensions, but other MiniLM checkpoints and
distillations do not, and an ONNX export can be produced with a pooling layer
that changes the width. Both arms therefore:

- record `embedding_dimension` in the artifact and benchmark metadata, read from
  the model's actual output rather than from a constant;
- record `embedding_model_id` and the ONNX file's SHA256, so the exact
  checkpoint behind a published number is recoverable;
- **validate** the observed dimension against the value recorded when the index
  was built, and fail loudly on mismatch rather than silently comparing vectors
  of different widths or relying on broadcasting to hide it.

A hardcoded `384` anywhere in the benchmark is a defect. The number is an
observation about a specific checkpoint, not a property of the approach.

### 5.4 Reduced-feature cross-domain cross-target robustness probe

Phase 1 §6.4 made cross-domain transfer a headline claim. **The measurement does
not support that weight**, and closer reading shows the experiment is not what
Phase 1 called it.

**It is not transfer. It crosses both the domain and the target.** A transfer
experiment holds the task fixed and changes the data. This one changes both:

| | Source | Target of evaluation |
|---|---|---|
| Domain | NYC 311 civic service requests | CFPB consumer financial complaints |
| Target variable | `nyc311_sla_breach` | `cfpb_timely_response` |
| Target means | resolution took longer than the type's training p75 | the company replied within CFPB's 15-day window |

Those two target variables are defined by §4 as **deliberately non-equivalent
constructs**, and §4.3 prohibits unifying them. A model trained to predict one
and scored against the other is therefore not being asked the same question in a
new domain — it is being asked a **different question** in a new domain.

**The experiment is renamed accordingly: the reduced-feature cross-domain
cross-target robustness probe.** It is exploratory robustness analysis, not
same-task transfer.

**Binding prohibition.** This probe must **never** be described, labelled,
summarised or tabulated as evidence that the 311 SLA-risk model transfers to the
CFPB task. It cannot support that claim, because the CFPB task is a different
task. Any wording implying otherwise — "transfers to", "generalises to", "works
on CFPB" — is a defect in the report, not a stylistic preference.

**It is a separate model, not the primary risk artifact scored on other data.**
The primary `RiskFeaturesV1` model cannot be evaluated on CFPB at all: two of
its five features — `category_mean_resolution_hours` and `category_breach_rate` —
require resolution times, and CFPB publishes no resolution date. Substituting
`timely` for a breach rate is prohibited by §4.3. Scoring the primary artifact on
CFPB is therefore impossible, not merely unwise. (Before D15 removed it,
`sla_hours` was a third such feature; its removal narrows the count but not the
conclusion.)

**`TransferFeaturesV1` — three features, versioned independently:**

1. `submitted_hour`
2. `submitted_weekday`
3. `text_length`

These are the only features computable in both corpora. A **distinct model** is
trained on NYC 311 using only these, targeting `nyc311_sla_breach`, and then
scored on CFPB records against `cfpb_timely_response`.

**Naming and reporting are binding.** The probe is called the **reduced-feature
cross-domain cross-target robustness probe** wherever it appears — artifact
metadata, README, and any published table. Its artifact carries its own
`model_name`, its own `model_version`, and a `feature_spec` of exactly the three
names above, so it cannot be confused with the primary model at load time.

**Every report of this probe must state, explicitly and in the output itself:**

1. **Source domain and source target** — NYC 311, `nyc311_sla_breach`, with the
   threshold rule that defines it.
2. **Evaluation domain and evaluation target** — CFPB, `cfpb_timely_response`,
   with what that field actually measures.
3. **The reduced feature set** — the three names, and that the primary model's
   five-feature set was not usable here.
4. **That the target semantics differ**, naming both constructs and stating that
   §4 defines them as non-equivalent.
5. **The polarity mapping** — the model outputs P(adverse), and "adverse" means
   `nyc311_sla_breach == True` in the source but `cfpb_timely_response == False`
   in the evaluation domain. That correspondence is an interpretive choice, not
   a fact about the data, so it is stated rather than left implicit.
6. **That this is exploratory robustness analysis, not same-task transfer**, and
   therefore cannot establish that the 311 model works on the CFPB task.

**The two evaluations' headline figures are not comparable to each other.**
Measured base rates differ by roughly 27× — 311 breach at 26.9–31.1% against
CFPB not-timely at 1.12% within narratives — and PR-AUC is base-rate dependent.
A lower cross-domain figure is therefore expected arithmetic before any question
of domain or target shift. Each figure is interpretable **only as lift over its
own baseline**; both baselines and both base rates are reported beside them, and
the two are never presented as a single before/after pair.

A report missing any of the six is incomplete, and the experiment's own output
carries them so they cannot be dropped in transcription.

**Result classification, driven by the §2.3 verdict.** Two of the probe's three
features derive from a CFPB timestamp whose provenance §2.3 measures. The probe
therefore reads that verdict and classifies its own result accordingly:

| §2.3 verdict | Probe `result_classification` | Meaning |
|---|---|---|
| `strongly_suspicious_load_timestamp` | `non-informative / diagnostic` | Figures must not be interpreted as substantive model evidence. With `submitted_hour` unusable and `text_length` degenerate across domains, the model would rest almost entirely on `submitted_weekday`. |
| `suspicious_insufficient_evidence` | `substantive_with_stated_caveat` | Figures reported, with the unresolved provenance question carried beside them. |
| `supported_plausible_event_time` | `substantive` | Figures reported normally. |

**The downgrade to `non-informative` requires the pre-specified §2.3 delta rule
to fire.** A non-diurnal hour histogram is not sufficient and never has been —
the distributional diagnostics are secondary evidence that can raise doubt but
cannot establish artifact status. The verdict, the metrics that produced it, and
the rule thresholds are all recorded in the probe's output, so a reader can see
which branch fired and why.

**What it can and cannot tell you.** A reduced-feature model that scores poorly
here may be failing because of domain shift, because the target means something
different, because three weak features are insufficient, or any combination — the
design cannot separate them. To give the number something honest to sit beside,
the same three-feature model is **also evaluated in-domain on held-out 311**
against its own target. That figure is the **reduced-feature in-domain reference
performance** — not a "ceiling", since a ceiling implies the cross-domain number
is measuring the same quantity less well, and it is not.

Required metrics when it is run:

- **PR-AUC** as the headline. ROC-AUC is prohibited as a headline at these
  ratios — it is optimistic and hard to interpret when negatives dominate
  ~163:1. ROC-AUC may appear as a secondary figure.
- **Minority-class precision, recall and F1**, reported explicitly.
- **Majority-class baseline** on the same split.
- The **absolute count of minority-class instances** in the evaluation split, so
  a reader can judge whether the estimate is stable. (There are 43,115 `timely =
  No` records within narratives in total — a large absolute number, which is why
  the experiment is worth running at all rather than abandoning.)

**The probe's result is never reported as a standalone number.** Every figure
appears beside the trivial baseline computed on the same split, in the same
table, so a reader sees the model and the do-nothing comparison together. A
PR-AUC quoted alone invites the reader to supply their own intuition about what
is good, and at a 1.12% base rate that intuition will be wrong. The baseline is
part of the result, not a footnote to it.

A poor result here is published rather than buried — but the design cannot say
what it is a finding *about*. Domain shift, the differing target semantics of
§4, feature degeneracy, or any combination could produce it, and the probe
cannot separate them. It is reported as an observation with its causes
enumerated and unresolved, never as a measurement of domain shift alone.

### 5.5 Risk model (311 → `nyc311_sla_breach`)

The Phase 1 design named these metrics in its §6.4; they are restated here
because §5.4 no longer carries them and they would otherwise be homeless.

Headline: **PR-AUC**. Reported alongside: ROC-AUC (secondary only), a
calibration curve, the absolute minority count per period, and the
majority-class and stratified-random baselines on the same split.

**`precision@k` is removed from the required set.** The Phase 1 design named it
on the reasoning that the model ranks an agent queue, so the top of the list
matters more than the global curve. That reasoning is sound but incomplete: it
never defined a `k`, and `k` is not a modelling choice — it is an operational
one, meaning "how many complaints an agent reviews in a sitting", which Sentinel
has no data to ground and no operational history to derive. A `precision@k`
reported against an invented `k` would look like an operational guarantee while
being an arbitrary slice.

It may return once Phase 3 has real queue-throughput data, at which point `k`
can be set from observed agent behaviour and given a stated operational meaning.
Until then the calibration curve carries the "is the top of the ranking
trustworthy" question, and does so without inventing a constant.

---

## 6. Temporal validation — mandatory

**Decision: all model evaluation uses a time-ordered split.** Random and
stratified-random splits are prohibited for any model in this project.

### 6.1 The split

Records are ordered by `submitted_at` and cut into three contiguous, disjoint
periods: **train (earliest) → validation → test (latest)**. No record appears in
more than one period. No period overlaps another in time.

Default proportions are **70 / 15 / 15 by record count** within the window,
which places the cuts by date rather than by row index so that no timestamp
straddles a boundary. The proportions are adjustable; the resulting **cut dates
and per-period record counts are recorded in the artifact's `metadata.json`**
whatever is chosen.

### 6.2 Prohibited leakage paths

Every one of the following must be fitted or derived on the **training period
alone** and then applied unchanged to validation and test. Each is a way a
temporal split can be silently defeated:

| Path | Rule |
|---|---|
| **Preprocessing** | Scalers, imputers, encoders and any fitted transform are fitted on train only. |
| **Vocabulary construction** | The TF-IDF vocabulary and IDF weights are built from training text only. Building them over the full corpus leaks future token statistics into the past — the single easiest mistake to make here. |
| **Feature selection** | Any selection that consults the target uses training folds only. |
| **Threshold tuning** | Decision thresholds (triage abstention, dedup similarity, risk banding) are tuned on **validation** and reported on **test**, untouched. |
| **Resampling** | Any class rebalancing applies to the training period only. Validation and test keep their natural distribution — resampling them would misstate real-world performance. |
| **Evaluation-index construction** | The dedup retrieval index for a given evaluation contains only records from that evaluation's own period or earlier. An index built over all periods lets a test query retrieve a future record. |
| **Target-derived features** | See §6.3. |

### 6.3 Target-derived features are the sharpest edge here

`category_mean_resolution_hours` and `category_breach_rate` are computed *from
the label*. Computing them over the full dataset injects test-period outcomes
into training features — leakage that is invisible in the code and produces
excellent, meaningless metrics.

Training-fold-only computation is **necessary but not sufficient**. If a
training row's own outcome contributes to the aggregate assigned to that row —
row A's 12 hours feeding the category mean that becomes A's feature — then A's
target has influenced A's feature. That is still target leakage, and it is
severe for small categories, where a single row can move the aggregate by a
large fraction.

**Rule, by row class:**

| Row class | How the aggregate is produced |
|---|---|
| **Training** | Out-of-fold: the value assigned to a row is computed **excluding that row's own outcome**. |
| **Validation / test** | Computed **exclusively from the training period**, applied unchanged. |
| **Unseen category** | The training-period global mean. Never a value derived from the period the category appears in. |

**Use K-fold out-of-fold, not leave-one-out.** Leave-one-out target encoding has
a well-known pathology: with the row's own target removed, the encoded value
becomes systematically anti-correlated with that target, and gradient-boosted
trees can recover the original label from it — producing a model that scores
superbly and generalises not at all. K-fold OOF does not have this failure mode.

**The folds are time-ordered, not random.** Within the training period, folds
are built by forward chaining (each fold's aggregate is computed from strictly
earlier training data) rather than random K-fold. Random folds inside the
training period would let later training records inform earlier ones — a weaker
version of the same leak this section exists to prevent, and inconsistent with
§6.1's decision that time ordering is what makes the evaluation interpretable.
This is stricter than common practice, deliberately.

### 6.4 Enforcement

Leakage prevention is tested, not merely documented. The evaluation harness
carries tests asserting that no validation or test record's `submitted_at`
precedes the training cut, that fitted transforms expose training-only
statistics, and that target-derived feature tables are keyed to the training
period. A leakage rule without a test is a comment.

---

## 7. The 311 SLA threshold

**Decision: the threshold is derived from the measured resolution-time
distribution of the training period. It is not chosen to match any number in
the Phase 1 spec.**

The measured distribution is heavy-tailed and reproducible across windows a year
apart:

| Threshold | Breach rate, June 2024 | Breach rate, Feb 2025 |
|---|---|---|
| 24h | 37.4% | 39.7% |
| 72h | 31.1% | 26.9% |
| 168h (1 week) | 19.5% | 17.4% |
| 720h (30 days) | 10.2% | 9.0% |

Median resolution is 9.8–17.1 hours, but p90 is 732 hours and p99 is 8,183
hours. Any single global threshold is therefore an arbitrary cut through a
distribution whose bulk and tail differ by three orders of magnitude.

**Rule:** the threshold is set **per complaint type**, at that type's own p75 of
resolution hours computed on the **training period only** (§6.3 applies — this
is a target-derived quantity). Types with **fewer than 100 training records**
fall back to the global training-period p75; 311 has 276 distinct complaint
types against millions of records per year, so the fallback should be rare, and
the count of types that hit it is itself reported. Both the per-type thresholds
and the resulting overall breach rate are written into the artifact's
`metadata.json` and published.

The rationale for p75 is that it produces a breach class large enough to learn
from at every measured window while still describing genuinely slow resolution,
rather than encoding "slower than typical" as failure. Any alternative choice is
acceptable if it is derived from the measured distribution and published with
its resulting class balance; what is prohibited is picking a round number
because it appeared in an earlier document.

**This threshold defines the label only. It is never used as a training
feature.** An earlier draft fed it in as `sla_hours`; that was removed because
the threshold is computed from the very outcomes the label encodes, making it
target-derived without a leakage-safe construction (§3.2, D15). The threshold's
sole role is to turn `nyc311_resolution_hours` into `nyc311_sla_breach`.

---

## 8. Phase 1 architecture preserved

This addendum changes the data and evaluation design. It changes none of the
architecture that Phase 1 established and its review validated:

| Property | Status |
|---|---|
| ML protocols in `ml/base.py` as the serving contract | **Unchanged in Phase 2.** `TriageModel`, `DedupIndex`, `RiskModel` keep their signatures. `TextEmbedder` is added for the benchmark (§2.2); it does not alter the existing three. |
| `RiskFeatures` field requirements | **Changes in Phase 3, not Phase 2** — see the correction below. |
| `model_version` provenance on every result object | **Unchanged and extended.** Artifacts additionally record `feature_spec` naming the exact trained subset (§3), split cut dates (§6.1), and thresholds (§7). |
| `Prediction` append-only, on both instance and bulk ORM paths | **Unchanged.** Phase 2 writes no `Prediction` rows at all — it produces artifacts, not predictions. |
| All lifecycle mutation through `complaints/services.py` | **Unchanged.** Phase 2 adds no mutation path. |
| Model suggests, human decides | **Unchanged.** No Phase 2 code writes `Complaint.category` or `Complaint.priority`. |
| ML degrades to absent, never to broken | **Unchanged.** The registry still resolves to null implementations; Phase 2 does not wire artifacts into serving. That is Phase 3. |
| Domain packs: the system knows the concept of a domain, never the meaning of one | **Unchanged, and extended.** Dataset adapters attach to pack classes in `domains/packs.py`. No source-specific literal enters `complaints/`. |
| Separation of training/evaluation data from live operational data | **Newly explicit** (§2), where Phase 1 left it unstated. |

**Correction to an earlier draft of this addendum.** A previous version claimed
the Phase 1 ML interfaces were untouched, without qualification. That was wrong.
§3.3's compatibility guard requires the four unavailable `RiskFeatures` fields to
become optional, because a required field that cannot be supplied is a lie in the
type. The change is small and lands in **Phase 3** alongside the inference
adapter that consumes it — Phase 2 performs no serving and needs no such change —
but it is a change to a Phase 1 interface and is recorded as one rather than
absorbed silently.

**Phase 2 produces no schema migration.** It adds no model and alters no table.

---

## 9. Decision log

Why each item changed from the Phase 1 design.

**D1 — CFPB restricted to 2024–2025 (§1).**
*Was:* the full CFPB corpus, treated as one taxonomy that was merely imbalanced.
*Now:* a two-year window with a stable 11-label vocabulary.
*Why:* measurement showed the vocabulary itself changes by era — the same
concept appears as `Credit reporting` (2016), `Credit reporting, credit repair
services, or other personal consumer reports` (2020), and `Credit reporting or
other personal consumer reports` (2024), while its share moves 19.4% → 55.0% →
77.4%. Neither a random nor a temporal split over that history measures
generalisation. The Phase 1 spec assumed a stable taxonomy without checking.

**D2 — Corpus separated from operational data (§2).**
*Was:* unspecified. The spec never said where ingested records live.
*Now:* files on disk, never in `complaints_complaint`.
*Why:* the ambiguity was a genuine hole rather than an oversight of wording.
Measurement made the answer obvious: 2.0M narratives and 22.4M 311 records
against a `submitted_by` non-null PROTECT FK and a live agent queue.

**D3 — `RawComplaint` withdrawn, replaced by `CorpusRecord` + per-source outcomes (§2.1).**
*Was:* one record type with a shared `sla_met` field.
*Now:* a shared record plus source-specific outcome types.
*Why:* `sla_met` unified two incomparable constructs (D5), and CFPB has no
resolution outcome at all — an asymmetry the single type could not express.

**D4 — Evaluation identity separated from serving identity (§2.2).**
*Was:* `Match.complaint_id: int` implicitly used for both serving and the dedup
benchmark.
*Now:* `Match` stays a serving type over live complaints; the benchmark measures
a new `TextEmbedder` over `(source, external_id)`.
*Why:* corpus records have no complaint IDs and must never be given synthetic
ones. Separating the benchmarked unit from the serving wrapper resolves the type
mismatch without weakening the Phase 1 interface.

**D5 — SLA constructs split (§4).**
*Was:* one `sla_met` boolean sourced from CFPB `timely` or a 311 computation.
*Now:* `cfpb_timely_response` and `nyc311_resolution_hours` /
`nyc311_sla_breach`, never combined.
*Why:* they measure different things. CFPB's is a regulatory responsiveness flag
about a company's reply, true 99.39% of the time, with no resolution duration
behind it. 311's is elapsed resolution time. Unifying them would have trained a
model on a target that meant two things.

**D6 — Transfer evaluation demoted to secondary (§5.4).**
*Was:* a headline claim, with the cross-domain gap presented as the project's
key ML result.
*Now:* a secondary robustness experiment with mandatory imbalance-aware metrics.
*Why:* the transfer target is 0.61% minority overall, 1.12% within narratives.
I designed this experiment in Phase 1 specifically to make the two-dataset
choice defensible; the measurement shows it cannot carry that weight. Keeping it
as a headline would have been the exact dishonesty the original design set out
to avoid. It is still worth running — 43,115 minority instances is a large
absolute number — but as robustness, not as proof.

**D7 — Accuracy and ROC-AUC removed as headline metrics (§5.1, §5.4).**
*Was:* ROC-AUC named as a primary risk-model metric.
*Now:* PR-AUC headline for the imbalanced experiment, with minority-class
precision/recall/F1 and a mandatory majority-class baseline.
*Why:* at 163:1 ROC-AUC is optimistic and hard to interpret, and accuracy simply
restates the class prior — a majority-class baseline scores 99.4%.

**D8 — Temporal validation made mandatory and leakage paths enumerated (§6).**
*Was:* the spec named no validation strategy at all.
*Now:* time-ordered splits, with seven named leakage paths each carrying a rule
and a test.
*Why:* an omission, not a wrong decision. Complaint data is timestamped and
drifts measurably — CFPB volume grew 10× and its majority class moved 58 points
across the measured years. A random split on such data leaks and flatters.

**D9 — Three features removed from v1 training, plus a fourth found unusable (§3).**
*Was:* ten `RiskFeatures`, described as domain-independent.
*Now:* v1 trains on a strict subset; the serving contract keeps all ten.
*Why:* `priority_rank`, `queue_depth` and `assignee_open_count` are Sentinel
operational state, absent from both corpora — confirmed against CFPB's 17 fields
and 311's 48 columns. `age_hours` was found during this analysis to be
identically zero for corpus records, since their prediction point is intake;
that fourth exclusion was not previously identified.

**D10 — 311 threshold derived from measurement (§7).**
*Was:* an SLA deadline implied by `category.sla_hours`, a Sentinel
configuration value with no empirical basis.
*Now:* per-type p75 computed on the training period, with published breach
rates.
*Why:* the measured distribution is heavy-tailed — median 9.8h, p90 732h, p99
8,183h. Any global round number is arbitrary, and choosing one to match an
earlier document would be fitting the data to the spec rather than the reverse.

---

## 10. Explicitly out of scope for Phase 2

- Wiring any artifact into the serving registry — that is Phase 3.
- Writing `Prediction` rows, populating `Complaint.embedding`, or any UI.
- Any schema migration.
- Any change to `complaints/services.py` or the lifecycle.
- Retraining the risk model on Sentinel's own operational history — that becomes
  possible only once Phase 3 has accumulated resolved complaints, and it is what
  restores the four excluded features.

**D11 — Target-derived aggregates require out-of-fold construction, not merely
training-fold construction (§6.3).**
*Was (first draft of this addendum):* computed from the training period, applied
as constants to validation and test.
*Now:* out-of-fold for training rows, training-period-only for validation and
test, via time-ordered forward-chaining folds; K-fold rather than leave-one-out.
*Why:* training-fold-only is necessary but not sufficient. A row whose own
outcome feeds the aggregate assigned to that row has leaked its target into its
feature, severely so for small categories. Leave-one-out was rejected in favour
of K-fold because LOO target encoding becomes systematically anti-correlated
with the target and boosted trees can invert it. Forward-chaining folds were
chosen over random ones so that the within-train construction obeys the same
time ordering §6.1 relies on.

**D12 — The artifact, not the serving interface, is authoritative about features
(§3.3).**
*Was (first draft):* the ten-field contract retained, with the trained subset
merely recorded in `feature_spec`.
*Now:* three named concepts — the conceptual contract, the v1 trained subset,
and `feature_spec` — with the inference adapter building exactly and only what
`feature_spec` names.
*Why:* the first draft left an implementation trap. A serving path required to
construct ten fields in order to call a model that accepts only its declared
subset (six at the time D12 was written, five after D15) would be asked for
`queue_depth` in an environment that cannot produce it. Making the artifact
authoritative turns `feature_spec` from documentation into a compatibility
guard that fails loudly at load time rather than mis-scoring silently. This
change requires the four unavailable fields to become optional on
`RiskFeatures` in Phase 3, which §8 now records as a real interface change
rather than claiming none occurred.

**D13 — Out-of-roster labels fail ingest loudly (§1.1).**
*Was (first draft):* the roster derived at ingest with an assertion on its
*count*.
*Now:* membership checked in both directions, the observed roster reported
before processing, and any difference failing the run.
*Why:* a count assertion passes when one label is swapped for another, which is
exactly how this taxonomy has changed before. Dropping, remapping or
auto-expanding would alter the experimental population underneath a published
benchmark with nobody deciding to. A taxonomy change is a spec decision with a
version bump.

**D14 — The dedup benchmark methodology is locked before either arm is built
(§5.3).**
*Was:* unspecified. Phase 1 named the comparison but not its conditions.
*Now:* splits, evaluation population, target definition, preprocessing rules,
retrieval procedure, downstream classifier family, metrics, baselines and tuning
budget all held identical; representation is the only permitted difference.
*Why:* a benchmark whose arms differ in more than one respect cannot attribute
its result to the representation, which is the only question it exists to
answer. Unequal tuning effort in particular measures effort, not representation
— and the decision to ship or cut MiniLM rests entirely on this comparison being
clean.

**D15 — `sla_hours` removed from `RiskFeaturesV1`; the set is five features
(§3.2, §7).**
*Was:* six features, with `sla_hours` sourced from the §7 per-type threshold.
*Now:* five features. The threshold defines the label and is never a feature.
*Why:* **the threshold is target-derived.** It is the p75 of training resolution
hours — the same outcomes that define `nyc311_sla_breach` — so a training row's
own resolution time contributes to the p75 that would become that row's own
feature. That is the exact defect §6.3 exists to prevent, in a feature §6.3 did
not name. The frozen-label design offers no leakage-safe construction for it: the
label needs one frozen threshold per type or its definition varies by row, while
a leakage-safe feature needs an out-of-fold one, and those cannot be the same
number. Maintaining two distinct thresholds — one for the label, one for the
feature — was rejected as a standing invitation to future error absent a
compelling reason. Redundancy with `category_mean_resolution_hours` is a
secondary observation, not the justification; the feature would be removed for
leakage even if it carried unique signal.

**D16 — The transfer experiment becomes a distinct reduced-feature model
(§5.4).** *(Further reframed by D19 — read both.)*
*Was:* evaluate the primary 311 risk artifact against CFPB
`cfpb_timely_response`.
*Now:* train a separate three-feature model (`TransferFeaturesV1`:
`submitted_hour`, `submitted_weekday`, `text_length`), versioned independently,
and evaluate it both in-domain on held-out 311 and cross-domain on CFPB.
*Why:* the original was impossible, not merely awkward. Three of the primary
model's five features require resolution times that CFPB does not publish, and
§4.3 prohibits substituting `timely` for a breach rate. Adding the in-domain
reduced-feature evaluation makes the transfer gap interpretable: comparing a
three-feature cross-domain score against a five-feature in-domain score would
attribute to domain shift what may simply be missing features. Naming is binding
so that a reduced-feature robustness probe is never read as the primary model's
performance.

**D17 — `precision@k` removed from the risk model's required metrics (§5.5).**
*Was:* named in the Phase 1 design as a primary risk metric.
*Now:* removed until a `k` with a stated operational meaning exists.
*Why:* the reasoning behind it — that the model ranks a queue, so the top
matters most — is sound, but no `k` was ever defined, and `k` is an operational
quantity ("how many complaints an agent reviews in a sitting") that Sentinel has
no history to ground. A `precision@k` against an invented `k` reads as an
operational guarantee while being an arbitrary slice. The calibration curve
carries the same question without inventing a constant. It may return in Phase 3
from observed throughput.

**D18 — Embedding dimension is recorded and validated, not assumed (§5.3).**
*Was:* an implicit assumption that MiniLM emits 384 dimensions.
*Now:* `embedding_dimension`, `embedding_model_id` and the ONNX SHA256 are
recorded from the model's actual output, and the observed dimension is validated
against the index's recorded value.
*Why:* 384 is a property of specific checkpoints (`all-MiniLM-L6-v2`,
`all-MiniLM-L12-v2`), not of the MiniLM family or of an arbitrary ONNX export
whose pooling layer may differ. A hardcoded width would either crash obscurely or,
worse, silently compare vectors of different widths.

**D19 — The experiment is cross-domain AND cross-target, and is renamed to say
so (§5.4).**
*Was (D16):* "reduced-feature cross-domain robustness experiment", framed as
transfer with a reduced feature set.
*Now:* the **reduced-feature cross-domain cross-target robustness probe**, with
a binding prohibition on describing it as evidence that the 311 SLA-risk model
transfers to the CFPB task, and six facts its report must state in its own
output.
*Why:* D16 fixed the feature mismatch but left the framing wrong. The model is
trained to predict `nyc311_sla_breach` — resolution slower than a type's
training p75 — and scored against `cfpb_timely_response` — whether a company
replied inside a 15-day regulatory window. §4 defines those as deliberately
non-equivalent constructs and §4.3 prohibits unifying them. Transfer, properly
used, holds the task fixed and varies the data; this varies both. Calling it
transfer would let a reader conclude the risk model "works on CFPB", which the
experiment cannot show and which §4.3 forbids asserting.
*Also changed:* "in-domain ceiling" became **"reduced-feature in-domain
reference performance"**. A ceiling implies the cross-domain figure measures the
same quantity less well. It does not measure the same quantity at all.

**D20 — Timestamp provenance is measured against a field-delta rule; distribution
shape is demoted to secondary evidence (§2.3, §5.4).**
*Was:* Task 8 emitted hour/weekday distributions with a verdict of
`plausible_diurnal` / `suspect_uniform` / `suspect_concentrated`, and the probe
downgraded its result to non-informative on that basis.
*Now:* a three-level verdict — `supported_plausible_event_time`,
`suspicious_insufficient_evidence`, `strongly_suspicious_load_timestamp` —
determined by a pre-specified rule over the `date_received` →
`date_sent_to_company` delta distribution. Hour and weekday diagnostics are
retained as `secondary_evidence`, may move a verdict only toward doubt, and can
never establish artifact status alone.
*Why:* distribution shape does not identify provenance. Complaints aggregated
across time zones, batch forwarding, and automated retries all produce flat or
spiky hour distributions from genuine event times, and a diurnal-looking
histogram would prove nothing either. The delta between two timestamps bracketing
a known administrative process is direct evidence: a median of seconds is not a
forwarding process. Fixing the thresholds before seeing data prevents the verdict
from becoming a judgment made after looking at the answer.
*Also recorded:* NYC 311 has no testable pair — the interval between its two
timestamps **is** the target variable — so it receives the distributional
diagnostic only and is classified `suspicious_insufficient_evidence` by
construction rather than defaulted to supported. Absence of evidence is not
recorded as evidence of soundness.

**D21 — NYC 311 floating timestamps are interpreted as `America/New_York` civil
time, with both DST edge cases rejected (§2.4).**
*Was:* unspecified. The plan mapped `created_date → submitted_at` without saying
how a field carrying no offset becomes an aware datetime.
*Now:* both 311 timestamps are read as `America/New_York` civil time and
converted to UTC for storage; `submitted_hour` and `submitted_weekday` derive
from the local representation while `nyc311_resolution_hours` derives from the
UTC instants; ambiguous and nonexistent local times are rejected rather than
folded or shifted.
*Why:* the gap was not optional to leave open — `write_partition` refuses a naive
`submitted_at`, so the floating timestamp had to be interpreted somehow, and the
only question was whether the interpretation would be written down or improvised
in an adapter. Assuming UTC would have shifted `submitted_hour` by four or five
hours and made that feature mean something different for 311 than for CFPB,
whose timestamps carry genuine offsets — the two domains §5.4's probe compares
directly. Deriving the hour locally and the elapsed time from instants keeps each
quantity measured in the frame it is actually about.
*Recorded as interpretation, not as source fact:* NYC Open Data types these
fields Floating Timestamp and does not state a zone for that type.
`America/New_York` is this project's reading of an unlabelled field, written
down so it can be disagreed with.

**D22 — The `hour_concentration` downgrade threshold is fixed at 0.50, and the
manifest carries the diagnostic and the record limit (§2.3, plan §G).**
*Was:* §2.3 allowed the distributional diagnostics to downgrade a `supported`
verdict "when they are extreme" without ever defining extreme, and the manifest
contract in plan §G listed neither `timestamp_diagnostic` nor `limit` — the two
things Task 8 is required to record in it.
*Now:* `hour_concentration` is defined as the share of records in the most
frequent submitted hour, and `>= 0.50` is extreme. The manifest gains
`timestamp_diagnostic` and `limit`, both required, plus a `manifest_version`
distinct from `schema_version`.
*Why the threshold:* a rule advertised as pre-specified cannot leave one of its
branches to be filled in later by whoever runs the ingest; that is the judgment-
after-looking §2.3 exists to prevent. 0.50 is about twelve times uniform. It is
a project decision rather than a source-derived fact, and is recorded in the
manifest beside the verdict so the applied number is visible rather than
inferred. Its effect stays one-directional — it may move `supported` toward
doubt and nothing else, because distribution shape does not establish
provenance.
*Why two version numbers:* `schema_version` versions `CorpusRecord` and is the
`v<N>` segment of the storage path. Incrementing it for a manifest change would
leave the records identical while every written partition became unreachable
under a new tree. `manifest_version` versions the manifest document alone. This
amendment is itself the demonstration that the two evolve independently.
*Compatibility:* both new fields are required rather than optional, and no
migration is defined, because no manifest exists — `data/` is gitignored and no
ingest has run, so Task 8 writes the first one. From Task 8 onward manifests
exist on disk, and any later change must increment `manifest_version` and say
how the previous version is read.

**D23 — A window that normalizes to zero records is a typed ingestion failure,
not an empty corpus (§2.5, plan Task 8).**
*Was:* unspecified. Ingest reached roster derivation with no years and surfaced
`ValueError: cannot derive a roster from no years` — Task 7's internal guard
leaking through Task 8 as an accidental error type.
*Now:* zero records in the requested window raises `EmptyWindow`, a subclass of
the CLI's `IngestError`, before roster derivation, writing no partition and no
manifest. The message states the source, the resolved window bounds, the number
of cached pages read, and that none of their records fell inside the window.
*Why:* an empty run would derive an empty roster and lock it into the first
manifest, after which every later ingest fails with every label unexpected — a
self-inflicted version of the taxonomy corruption §1.1 exists to prevent.
Failing at the point of emptiness makes that unreachable.
*Why a typed error rather than Task 7's `ValueError`:* `derive_roster`'s guard
concerns an undefined intersection, an internal precondition of a pure function.
It is not the ingest-level fact an operator needs, and a caller cannot tell it
from any other `ValueError`. Task 7 is unchanged; the CLI simply never reaches
it with no years.

**D24 — `--limit` bounds persistence; roster validation operates on the complete
window, and a truncated manifest never becomes the lock (§1.1, plan §G).**
*Was:* `--limit` truncated the normalized stream before roster validation, so a
limited run could report a locked label as missing purely because truncation
dropped it. Observed: a corpus holding two labels, re-ingested with `--limit 1`,
raised `RosterMismatch: missing (1): 'Beta'` — a taxonomy alarm manufactured by
a development flag.
*Now:* roster derivation and assertion run over the complete normalized window,
before any truncation. `--limit` bounds only what is written to disk. The
manifest records the supplied `limit` exactly as §G already describes.
*Why:* §1.1's failure condition is already written as "a locked roster label is
absent **from the window**". The window is a property of `--start`/`--end` and
the source's data; how many records were kept for development is not part of it.
*Also decided — the lock comes only from an unbounded corpus in the run's own
target root.* A manifest whose `limit` is non-null describes a deliberately
partial corpus and is **not** authoritative: a later run does not adopt its
roster. A run reads its lock only from the corpus root it writes to; no other
root is consulted. A limited run derives its roster from its own complete window
and never replaces an authoritative roster.
*Amended by D26:* as first written, this entry also had a limited run validate
against an authoritative roster present in its own root. D26 refuses a limited
run into a root holding an authoritative corpus before any write, so that case
cannot occur, and D26 takes precedence over it. This uses `limit` for precisely
the purpose §G already gives it — so a truncated corpus can never be mistaken
for a full one — and keeps a development run from defining the vocabulary a
production run is judged against.
*Also recorded — two populations, deliberately.* `label_roster`, `record_count`,
`per_year_counts` and `part_files` continue to describe the **persisted**
corpus. Only roster validation uses the full window. On a limited run
`label_roster` therefore need not contain every label the window held and must
not be read as the window's taxonomy. Recording the window roster instead was
considered and rejected for now: it would require changing `build_manifest`'s
signature a second time, and the manifest contract is better left stable.

**D25 — `--start` and `--end` are inclusive civil dates, resolved in each
source's own frame (§1, §2.4, §2.5).**
*Was:* unspecified. Plan Task 8 gave `--start YYYY-MM-DD --end YYYY-MM-DD`
without stating whether `--end` is inclusive or what timezone a bare date
denotes, and §1 states the window as `2024-01-01 … 2025-12-31` without either.
*Now:* the CLI surface is unchanged and both bounds are inclusive whole days.
311 resolves them in `America/New_York`; CFPB resolves them in UTC. Both become
UTC-aware instants for comparison, and the manifest records the resolved
instants rather than the supplied dates.
*Why per source:* a single UTC rule would shift 311's day boundaries by four or
five hours, putting a request filed at 20:00 on a window's last day outside it —
incoherent beside §2.4, which already reads that source's timestamps as New York
civil time and derives its `submitted_hour` from them. Giving CFPB a civil frame
would be worse: it carries a per-record offset that normalization discards, so
any single frame would be invented.
*Why inclusive:* §1's `2024-01-01 … 2025-12-31` reads as both endpoints
included, and an exclusive `--end` would silently drop the final day of every
window typed from that sentence.
*Consequence, accepted deliberately:* this changes which records a 311 window
admits relative to the ingestion CLI as first committed, which compared both
sources against UTC bounds. The shift is intended, not a defect, and the code is
amended to match this decision rather than the reverse.

**D26 — A limited run may not replace an authoritative corpus (§2.6, plan §G,
plan Task 8).**
*Was:* `--limit` bounded persistence and a truncated manifest never became the
roster lock (D24), but nothing stopped a limited run from overwriting the
partitions and manifest of an existing unbounded corpus. Observed: an
authoritative corpus holding two labels, re-ingested with `--limit 1` into the
same root, was replaced by a one-record corpus whose manifest carried
`limit: 1`. D24's protection then applied to a corpus that no longer existed.
*Now:* when the target corpus root already holds an authoritative manifest for
the source — one whose `limit` is `null` — a run with `--limit` raises
`AuthoritativeCorpusExists`, a subclass of `IngestError`, before
`fetch_into_cache` and before any write. The message states the source, the
corpus root, the existing manifest's `corpus_id` and `record_count`, and that a
limited run must target a different root. A limited run against a fresh root is
allowed, and so is one against a root holding a truncated corpus: nothing
authoritative is at risk, and replacing one development corpus with another is
the intended iteration loop. An unbounded rerun over an authoritative corpus is
the resume path, is idempotent, and is unaffected.
*Why refusal rather than a separate tree, a merge, or replacement:* a second
tree is infrastructure no measurement justifies (invariant 4) and leaves every
downstream reader asking which tree to read. A merge produces a corpus that is
neither the full window nor the truncated one, changing the experimental
population with nobody deciding to — the failure §1.1 exists to prevent — under
a `corpus_id` naming a population no single run produced. Permitting replacement
contradicts the reason plan §G gives for recording `limit`: an artifact citing a
`corpus_id` cannot see whether the corpus behind it was complete if a typo can
silently replace it.
*Also decided — the CLI gains `--corpus-root PATH`.* Without it the refusal has
no remedy: the CLI writes only to the default corpus root, so a developer
holding a full corpus could not produce a truncated one at all. The argument
defaults to the existing corpus root and names where output goes; it adds no
new corpus concept.
*Consequence, accepted deliberately — D26 takes precedence over D24's same-root
limited-run case.* D24 reads the roster lock only from the run's own target
root. A limited run therefore cannot consult an authoritative same-root roster,
because that configuration is refused before any write; it derives its roster
from its own complete window instead. No cross-root roster lookup and no separate
authoritative-root argument is introduced. The authoritative roster still guards
every unbounded run into the authoritative root, and D24's underlying invariant
is intact: whenever roster validation runs, it runs over the complete normalized
window before `--limit` truncates anything.
*Not decided here:* an unbounded run over a narrower window also discards the
records outside it. That is visible, because `window_start` and `window_end` are
recorded in the manifest, and it is a different question from truncation. It is
left open rather than folded into this decision.
*Resolved by D27:* an unbounded run over a narrower window is allowed and
replaces the corpus completely, removing every record outside the new window.
This entry's open question is closed. Its description of that outcome became
accurate only when D27 made replacement remove stale partitions.

**D27 — A successful run replaces the source's corpus with the current run's
window; the manifest is the validity boundary, and a corpus is read only through
it (§2, §2.5, §2.6, §2.7, plan §D, §F, §G, §P, §R, §X, plan Tasks 4, 8, 16, 17,
19).**
*Was:* unspecified, and inconsistent. `write_partition` overwrote one path and
deleted nothing; `build_manifest` described every part file on disk; `read_corpus`
read every part file on disk whether or not a manifest existed. A rerun that
wrote fewer year partitions left the rest in place and counted them. Observed: a
`limit: 1` run into a truncated root recorded `record_count: 2`, and an unbounded
2024-only run into an authoritative root kept its 2025 partition under a manifest
whose `window_end` was 2024-12-31. §2.6 and D26 described a replacement the code
did not perform. A run failing after its first partition write left new
partitions under the previous manifest, which D24 and D26 went on reading as
authoritative and `read_corpus` read as a mixed population.
*Now — replacement.* Once every pre-write refusal has passed (§2.5), ingest
deletes the source's manifest, then the rest of
`<corpus root>/<source>/v<SCHEMA_VERSION>/`, then writes its partitions, then
writes the manifest last to a same-directory temporary file moved into place with
`os.replace`. The corpus that results is the current run's window, whatever the
previous corpus held — including a window **narrower** than the corpus it
replaces, and an authoritative corpus replaced by a narrower one: records outside
the new window are removed, and no stale partition survives. After a successful
run the part files on disk are exactly `part_files`, every record lies within
`window_start` .. `window_end`, and a limited manifest satisfies
`1 <= record_count <= limit` (D23, D28). No other source and no other schema
version is touched.
*Now — validity.* A source/version tree without a valid manifest is not a corpus.
`load_corpus` is the only corpus reader: it raises `ManifestNotFound` without a
manifest, a `CorpusIntegrityError` for a part file on disk that `part_files` does
not list, and `ChecksumMismatch` for a listed file that is missing or whose bytes
differ from its digest; otherwise it streams exactly the listed files in
`(submitted_at, external_id)` order. `read_corpus` is reclassified as a low-level
partition reader. It keeps its signature and behaviour, remains usable before any
manifest exists, asserts nothing about validity, and is used only by
`build_manifest` and by tests. This changes the Task 4 contract, which presented
`read_corpus` as the corpus reader.
*A validity boundary, not atomic directory replacement.* A reader using
`load_corpus` sees the previous completed corpus, no corpus, or the new completed
corpus, and never a mixture of runs. The tree is not replaced atomically: the
previous corpus does not survive a failed write, and between deleting the old
manifest and writing the new one the source has no corpus. Only the manifest
file itself is replaced atomically.
*Why this rather than staging, retention, or refusal:* staging needs a directory
swap that is not atomic on every supported platform, or a pointer that changes
the §G layout — infrastructure no measurement justifies (invariant 4) for a
corpus plan §X already calls regenerable from the raw cache. Keeping old
partitions is the merge D26 rejected under §1.1. Refusing whenever old partitions
exist would block D26's permitted development loop. Making `read_corpus` itself
require a manifest would need a circular dependency between the storage and
manifest modules, and would leave `build_manifest` no way to read what it is
about to describe.
*Consequences, accepted deliberately.* An I/O failure after clearing loses the
previous corpus; no predictable failure can reach that step. Recovery is a rerun
from the raw cache. A run that fails after clearing leaves no manifest, so that
root is not authoritative (D26) and holds no roster lock (D24): the next CFPB run
derives its roster from its own window rather than comparing against the lost
lock. The failed run had already validated that window against the lock, so a
rerun over the same raw cache reproduces the same taxonomy. An artifact citing a
replaced `corpus_id` can no longer verify it against disk. Narrowing a CFPB
corpus remains subject to §1.1: a locked label absent from the narrower window
fails `RosterMismatch`.
*Resolves D26's open question:* an unbounded run over a narrower window may
replace an authoritative corpus, and replaces it completely.
*Not decided here:* concurrent ingests into, or reads from, one corpus root.

**D28 — `--limit` must be a positive integer (§2.5, §2.8, plan §G, plan Task
8).**
*Was:* unspecified. `--limit 0` wrote a zero-record corpus and manifest with an
empty roster and a diagnostic computed over nothing. `--limit -1` was applied as
a slice, silently dropped the window's last record, and recorded `limit: -1`.
*Now:* `ingest()` raises `InvalidLimit`, a subclass of `IngestError`, when
`limit` is not `None` and is less than 1. The check sits immediately after
`InvalidDateRange`, before the D26 refusal, before `fetch_into_cache`, and before
any write, and its message names the value supplied. The command line reaches the
same check because `main()` propagates it exactly as it does `InvalidDateRange`;
the argument parser keeps `type=int`, so the rule lives in one place for every
caller.
*Why:* plan §G defines `limit` as a bound on the records persisted, and a
negative number bounds nothing. A zero limit produces the empty corpus D23
already decided is a failure rather than a corpus. Checking before D26 means a
nonsensical value never causes a manifest to be read.
*Kept separate from D27:* this decision concerns the value of `limit` alone. It
is independent of how a run replaces or validates the corpus.

**D29 — The temporal split keeps plan §H's boundary rule and adds one explicit
fallback for a boundary no timestamp satisfies (§6.1, plan §H, plan Task 9).**
*Was:* inconsistent. Plan §H places each boundary at "the largest timestamp
whose cumulative count does not exceed the target", but that rule has no answer
when the earliest eligible group of records sharing a timestamp is itself larger
than the target. Plan Task 9 nevertheless requires that "a single-timestamp
input puts everything in train and reports it" — an outcome the rule as written
cannot produce, because a single timestamp's only cumulative count is the whole
corpus, which exceeds the train target of `0.70 × n`.
*Now:* §H's rule stands unchanged. Unique timestamps are sorted ascending with
their cumulative record counts. The train boundary is the largest timestamp
whose cumulative count does not exceed `train_fraction × n`; the validation
boundary is the largest whose cumulative count does not exceed
`(train_fraction + validation_fraction) × n`. Every record at a boundary
timestamp belongs to the earlier period, so a group of records sharing a
timestamp is never split across periods. When no timestamp satisfies a
boundary, one explicit fallback applies, and nothing else changes. For the train
boundary, the boundary is the earliest timestamp, which places that entire
oversized group in train. For the validation boundary, if no timestamp after the
train boundary satisfies the validation target, `val_end` equals `train_end` and
validation is empty. Empty periods are valid, and are reported with a zero count
rather than raised. A single-timestamp corpus therefore places every record in
train and reports empty validation and test periods.
*Why this fallback:* it is the smallest addition that lets §H's rule produce
Task 9's required single-timestamp outcome, and it preserves the rule's
guarantees everywhere else. Outside the fallback, train never exceeds its
requested share and test never falls below its own, so ties can never shrink the
evaluation population.
*Rejected — the crossing-boundary interpretation:* assigning the group whose
cumulative count crosses a target to the earlier period would also yield the
single-timestamp outcome, but it contradicts "does not exceed", and under heavy
ties it can empty the test period entirely: 5,000 / 3,000 / 2,000 tied records
would split 80 / 20 / 0 rather than 50 / 30 / 20. The original rule is kept, not
replaced.
*Scope:* a record of the behaviour Task 9 implements. It changes no other
Task 9 behaviour.

**D30 — Forward-chaining folds use exactly `n_folds` apply blocks over the
records after the realised warm-up, cut by the Task 9 / D29 rule (§6.3, D11,
D29, plan §J, plan Task 10).**
*Was:* inconsistent and incomplete. Plan §J describes "the four date cuts
dividing the remaining 80% into four equal-count blocks, plus the warm-up
boundary — five apply blocks total", while the same section sizes each apply
block at ~16% of the training period and the function defaults to `n_folds=5`.
Four cuts inside the remainder make five blocks, not four, and four blocks of
the remaining 80% would hold 20% each, not ~16%. Plan Task 10 also left open how
fold cuts are measured once ties move the warm-up boundary, what an empty fold
is, and whether degenerate inputs "behave or raise".
*Now:* `forward_chaining_folds(timestamps, n_folds=5, warmup_fraction=0.20) ->
list[Fold]` runs over the training period's timestamps only; validation and test
take no part, and Task 9's `temporal_split` is unchanged. The warm-up is realised
by Task 9's date-cut semantics: its boundary is the largest timestamp whose
cumulative record count does not exceed `warmup_fraction × n`, with D29's
fallback to the earliest timestamp when none does. With W the realised warm-up
record count, the remaining `n − W` records are divided into exactly `n_folds`
approximately equal-count apply blocks — five of ~16% each by default — whose
boundary targets are `W + (n − W) × k / n_folds` for `k = 1 … n_folds`. Every
boundary uses the Task 9 / D29 rule: the largest timestamp whose cumulative
count does not exceed the target; the records at a boundary timestamp join the
earlier block, so a timestamp group is never split; and when no timestamp beyond
the previous boundary satisfies a target, the boundary stays at the previous one
and that fold's apply block is empty. The final target is `n`, so the apply
blocks together cover every record after the warm-up. The windows expand: fold
`i`'s fit block is every record strictly before its apply block, and its apply
block holds the records after the previous boundary up to and including its own.
*Also decided — degenerate inputs.* The result always contains exactly `n_folds`
`Fold` objects, and empty apply folds are valid and reported rather than raised.
`n_folds` must be an integer of at least 1, and `bool` is not accepted.
`warmup_fraction` must be finite and strictly between 0 and 1. Empty `timestamps`
raise `ValueError`. A single-timestamp input puts every record in the warm-up and
returns `n_folds` empty apply folds. `n_folds=1` is valid and yields one apply
block holding everything after the warm-up.
*Also decided — the `Fold` fields.* `fit_indices` and `apply_indices` are tuples
of original input positions in ascending chronological order; records sharing a
timestamp appear in ascending input position, so the order is deterministic.
`fit_end` is the previous boundary, the last timestamp in the fit block.
`apply_end` is the fold's inclusive boundary, and equals `fit_end` for an empty
fold. `apply_start` is the earliest timestamp in the apply block, or `None` for
an empty fold. The warm-up size is reported as `len(folds[0].fit_indices)`,
because the first fold's fit block is exactly the warm-up.
*Why:* five apply blocks is the only reading consistent with ~16%, "five apply
blocks total" and `n_folds=5`. Measuring the fold cuts against the realised
remainder keeps the apply blocks equal-count among themselves when ties move the
warm-up, which is what "dividing the remaining … into equal-count blocks"
describes; measuring them as fixed shares of the whole period would let a large
tied first group shrink the first fold instead. Reusing the Task 9 / D29 rule
means one boundary rule governs every date cut in the project, and reporting
empty folds follows D29's treatment of empty periods.
*Scope:* resolves Task 10's ambiguities only. The expanding-window design of
§6.3 and D11 and the behaviour of `temporal_split` are unchanged.

**D31 — Out-of-fold category aggregates take both their observations and their
breach threshold from the fold's fit block alone (§3.2, §4.2, §6.3, §7, D11,
D15, D30, plan §J, §K, plan Tasks 11 and 13).**
*Was:* incomplete, with a leakage path. `category_breach_rate` is a rate of
`nyc311_sla_breach`, which exists only once §7's per-type threshold turns
`resolution_hours` into a label, yet plan Task 11 depends only on Task 10 and its
API receives outcomes that carry no label. Taking the labels from Task 13's
frozen training-period thresholds would let a training row's own
`resolution_hours` move the p75 that labels the earlier rows in its fold's fit
block, and so move that row's own `category_breach_rate` — the defect D15
removed `sla_hours` for, and a breach of Task 11's own defining property. §6.3's
unseen-category rule, "the training-period global mean", is stated for validation
and test; applied to a training row it would include that row's own outcome,
which plan §J forbids for warm-up rows for exactly that reason. Open requests
and the shapes of the Task 11 structures were unspecified.
*Now — the breach rate.* For each Task 10 fold, the breach thresholds are fitted
exclusively on that fold's fit block, with Task 13's threshold semantics: each
category's p75 of `resolution_hours`, and the global p75 of the same fit block as
the fallback for a category with fewer than 100 observations. Breach labels are
derived only from outcomes in that fit block, `category_breach_rate` is computed
from those labels, and the result is applied to that fold's apply block. For
validation and test rows, the thresholds are the training-period thresholds and
the rates come from the whole training period. The threshold calculation is one
small, pure, reusable training utility with exactly Task 13's semantics; it is
not duplicated anywhere, and Task 13 reuses it. Introducing it does not
implement Task 13.
*Now — a category absent from a fold's fit block.* An out-of-fold apply row whose
category does not occur in that fold's fit block receives that fit block's
global aggregate. The whole-training-period global statistic is never used for an
out-of-fold apply row; it applies only to validation and test rows whose category
was unseen in training. Warm-up rows remain `NaN`.
*Now — open requests.* A row whose `resolution_hours` is `None` contributes to
neither the category mean nor the breach-rate statistic. It may still receive
aggregates computed from eligible observations. A statistic with no eligible
observations is `NaN`. Whether open requests remain in the eventual
model-training population is left to the risk-model task.
*Now — shapes and definitions.* `records`, `outcomes` and fold indices align by
position, and their `external_id`s are checked for consistency. `ValueError` is
raised for mismatched lengths, mismatched external IDs, outcomes that are not NYC
311 outcomes, or a fold collection that does not partition the expected row
positions exactly. Means and rates are plain and record-weighted, with no
smoothing and no minimum-count rule for aggregates, and sums use `math.fsum` so
they are stable and deterministic. `AggregateColumns` is a frozen dataclass
holding `category_mean_resolution_hours` and `category_breach_rate`, each aligned
by position to `records`. `FrozenAggregates` is a frozen dataclass holding
read-only per-category mappings and the two global values. Standard library only.
*Now — the p75 threshold definition.* A p75 is linear interpolation between
adjacent order statistics: with the values sorted ascending as `x[0] … x[n − 1]`
and `h = (n − 1) × 0.75`, the percentile is
`x[floor(h)] + fractional_part(h) × (x[ceil(h)] − x[floor(h)])`. It is computed by
a small pure training utility that imports neither `ingest` nor `numpy`, and this
definition is the shared Sentinel threshold definition that Task 13 also uses.
*Now — what the fallback counts.* The fewer-than-100 rule counts only eligible
observations, those whose `resolution_hours` is not `None`. Open requests
contribute to no category p75, no global p75 and no breach-rate statistic, and a
fit block's global fallback likewise uses only that same fit block's eligible
observations. With zero eligible observations the threshold is undefined, and
the corresponding breach-rate statistic is `NaN`.
*Now — when a category counts as seen.* For category-specific aggregates and
thresholds, a category is seen only through its eligible observations, those
whose `resolution_hours` is not `None`. A category that appears in a fit block
with zero eligible observations therefore receives that fit block's global
aggregate where the global value is defined, rather than `NaN`. The same rule
applies on the frozen training path: for validation and test rows, a training
category with no eligible observations receives the training global aggregate
where defined. Where the relevant global statistic itself has no eligible
observations, the aggregate is `NaN`. `fallback_categories` reports the
categories with 1 to 99 eligible observations; it is metadata only and does not
alter any Task 11 feature value.
*Now — what a valid fold collection is.* Partitioning the row positions is
necessary but not sufficient. Each fold's fit block must equal the warm-up plus
every preceding apply block, so it can never contain its own apply block or any
later one. A fold collection that fails either check raises `ValueError`.
*The invariant.* For every out-of-fold apply value, both (A) the observations used
to compute the category aggregate and (B) the threshold used to turn
`resolution_hours` into a breach label come exclusively from that fold's fit
block. No current-row, same-timestamp, later, validation or test target can
influence either.
*Why:* it is the only construction in which a row's own outcome cannot reach its
own breach-rate feature, which §6.3, D11 and D15 already require. Per-fold
thresholds serve only as an intermediate step of the aggregate and never become a
feature, so D15's rejection of a second threshold used as a feature does not
apply; one shared primitive keeps the per-fold thresholds and the label thresholds
from ever diverging in definition. Using the fit block's own global value for an
unseen category is the out-of-fold counterpart of §6.3's validation and test rule.
*Scope:* resolves Task 11's semantics only. Task 9's `temporal_split`, Task 10's
folds, and the meaning of Task 13's frozen label thresholds are unchanged.
**D32 — §2.4's claim that a CFPB `submitted_hour` is already anchored to the
filer's own clock is stale; the stored UTC instant is what survives (§2.4, §2.5,
plan §J, plan Task 12).**
*Was:* §2.4 justifies deriving 311's hour and weekday from the local
representation partly by asserting that "CFPB's timestamps arrive with real
offsets, so its `submitted_hour` is already anchored to the filer's own clock".
That describes the field as *published*, not as *stored*. `ingest.sources.cfpb`
rejects a naive timestamp and then converts the parsed value with
`.astimezone(UTC)`, so the published offset is consumed and discarded during
normalization and `CorpusRecord.submitted_at` holds the UTC instant alone. §2.5
already states this correctly — CFPB "publishes a per-record offset and
normalization converts to UTC, discarding it", and "the stored instant is all
that survives". The addendum therefore asserted both, and the two readings
disagree about what a CFPB corpus record still carries.
*Now:* §2.5's statement is the correct one and governs. A CFPB corpus record
retains no filer-local wall clock, so its `submitted_hour` and
`submitted_weekday` are read from the stored UTC instant, that being the only
representation which exists. NYC 311 is unchanged: its hour and weekday come
from the `America/New_York` civil representation via `nyc311.to_source_local`,
per §2.4's table and D21. Feature assembly therefore dispatches on
`CorpusRecord.source` — `nyc311` converts to New York civil time, `cfpb` uses the
stored instant as it is. That is the same rule `ingest.cli` already applies to
Task 8's hour and weekday diagnostics, so the diagnostic and the feature have one
stated basis rather than two.
*Why:* the stale sentence reads as a guarantee that a CFPB hour carries diurnal
meaning in the filer's own time, which no consumer of the corpus can rely on,
and it was the stated reason the two domains' `submitted_hour` were held to mean
the same thing. They do not: 311's is anchored to New York civil time and CFPB's
to UTC, so the two are not on a common civil frame. That is a finding §5.4's
reduced-feature cross-domain cross-target robustness probe must state about its
own inputs — the probe already carries a required feature-distribution
diagnostic and a binding prohibition on overclaiming — and it is recorded here
rather than left for a reader to rediscover from the adapters.
*Scope:* documentation only. No ingestion, normalization, schema, storage or
manifest behaviour changes; no corpus is re-ingested; no threshold, split, fold
or aggregate semantics are touched. D1–D31 are unaltered, and §2.4's table rows,
its DST rejections and D21 all stand.
**D33 — Task 13's record-facing threshold API lives beside the pure primitive,
refuses unresolved requests, and counts eligible observations (§7, §K, D10, D31,
plan Task 13).**
*Was:* plan Task 13 placed `fit_thresholds`, `apply_thresholds` and
`FrozenThresholds` in `ml/training/thresholds.py`, took `min_records=100`, and
returned `np.ndarray[bool]`. Four things collided with what already exists. D31
states the p75 is computed by a utility importing neither `ingest` nor `numpy`,
yet the Task 13 signatures need `CorpusRecord`, `NYC311Outcome` and `numpy` in
that same file — and `ml.training.aggregates` imports that module, so numpy would
become a Task 11 dependency. §7 and §K say "fewer than 100 training *records*"
while D31 counts only eligible observations, which disagree for any type whose
requests are largely still open. A `bool` array has no representation for a
request with no `resolution_hours`, forcing a population decision D31 explicitly
left to the risk-model task. And an undefined threshold is `NaN`, against which
every comparison is false, so an undefined threshold would silently label every
record "not breached".
*Now — where the API lives.* `ml/training/thresholds.py` stays the pure shared
primitive, importing neither `ingest` nor `numpy`, and `CategoryThresholds`,
`linear_percentile`, `fit_category_thresholds` and `is_breach` keep their current
names and semantics. Task 13's record- and outcome-facing API lives in
`ml/training/labels.py`, which may import `numpy` and `ingest` and which **calls**
the primitive rather than restating it. No percentile, interpolation,
minimum-count, global-fallback or breach comparison is written a second time.
*Now — the minimum-count rule.* D31 governs: the hundred counts **eligible
observations**, those whose `resolution_hours` is not `None`, never raw records.
The authoritative parameter is `min_eligible`, defaulting to 100. Plan Task 13's
`min_records` spelling is stale and does not override D31.
*Now — unresolved requests.* `apply_thresholds` raises `ValueError` when any
supplied outcome has `resolution_hours` of `None`. It does not coerce one to
`False`, drop it, shorten the returned array, or invent a third boolean state:
each would either manufacture a "not breached" label for a request that was never
resolved or break positional alignment with `records`. Whether open requests
remain in the model-training population stays with the risk-model task, as D31
says.
*Now — undefined thresholds.* Fitting may still report an undefined threshold as
`NaN` exactly as D31 says. **Applying one may not.** If the threshold that would
apply to a record is `NaN`, `apply_thresholds` raises `ValueError` rather than
emitting labels that are false only because every comparison with `NaN` is false.
*Now — the frozen object.* `FrozenThresholds` is a new immutable type wrapping
the primitive's `CategoryThresholds`, carrying Task 13's own field names:
`per_type`, `global_fallback` and `fallback_type_count`. Task 11's type is
neither renamed nor altered. The two fallback counters are deliberately different
quantities and are not interchangeable: `CategoryThresholds.fallback_categories`
is the categories with **1 to 99** eligible observations, while
`fallback_type_count` is **every** type whose applied threshold is the global
fallback, including a type with **zero** eligible observations that
`fallback_categories` omits.
*Now — breach rate.* Task 13 owns a small pure per-period breach-rate helper, as
§7 and §K require the resulting rate per period to be published. It consumes
applied labels and computes no threshold of its own.
*Why:* every clause keeps one definition in one place. Splitting the module is
what lets D31's "imports neither `ingest` nor `numpy`" stay literally true while
Task 13 still receives the record and outcome objects its callers hold, and it
keeps Task 11's tests running in the CI job that installs no training tier.
Refusing unresolved requests and undefined thresholds converts two silent
false-label paths into loud ones, which is the same principle §1.1 applies to a
normalizer that quietly repairs a record.
*Scope:* Task 13 only. Task 9's split, Task 10's folds, Task 11's aggregates and
Task 12's feature assembly are unchanged, as is every existing name in
`ml/training/thresholds.py`. D1–D32 are unaltered. The stale "Create
`ml/training/thresholds.py`" and "Prerequisites: Task 9" lines in plan Task 13
predate Task 11 creating that file; correcting them is deferred to a separate
documentation cleanup rather than mixed into this task.
**D34 — Task 14 metrics: the full scope except recall@k, baseline priors fitted
on training labels, and every class order and polarity supplied by the caller
(§5.1, §5.2, §5.3, §5.4, §5.5, D7, D17, plan Task 14, plan Task 18).**
*Was:* plan Task 14 listed `macro_f1`, `per_class_report`, `top_k_accuracy`,
`pr_auc`, `minority_report`, `recall_at_k`, `majority_baseline` and
`stratified_baseline`, and required every function to carry its baseline, but
left several published numbers undecided: whether the confusion matrix §5.2
requires and the ROC-AUC §5.4 and §5.5 name as secondary belong to it; whether a
baseline's class prior comes from training or evaluation labels, §5.1 saying only
"computed on the same split"; how many random draws a stratified baseline makes
and from what seed; which definition of PR-AUC applies; how a class absent from
the evaluation population enters the macro average; how top-k treats ties and a
`k` wider than the roster; which baseline recall@k carries; and who decides class
order, the minority class and positive polarity.
*Now — scope.* Task 14 implements `macro_f1`, `per_class_report`,
`confusion_matrix`, `top_k_accuracy`, `pr_auc`, a secondary `roc_auc`,
`minority_report`, `majority_baseline` and `stratified_baseline`. **`recall_at_k`
is deferred to Task 18.** §5.3 requires "the same baselines" for both retrieval
arms but defines none, and a majority or stratified classifier baseline has no
meaning for retrieval, while plan Task 18 already requires recall@k over
`RecordRef`. Task 18 therefore owns recall@k's semantics and its baseline, and no
retrieval baseline such as k/N is invented here.
*Now — results carry baselines.* Every public metric returns a structured result
holding the primary score and the named baseline scores applicable to that
metric, never a bare float. Baseline generation lives only in `majority_baseline`
and `stratified_baseline`; metrics consume the predictions or rankings those
produce and never reconstruct a baseline classifier themselves. A metric for
which the contract defines no baseline is not given an invented one.
*Now — baseline priors.* Both baselines fit their class prior from **training
labels only** and are then scored on the evaluation population. Neither the
majority class nor the stratified distribution is ever derived from validation or
test targets: §5.1's "computed on the same split" means evaluated on the same
split. A majority-class tie resolves to the earliest label in the caller-supplied
roster order.
*Now — the stratified-random baseline.* It draws its predictions from the
training-label class distribution in **exactly one** draw from
`numpy.random.default_rng(seed)`. `seed` is required and has no default; there
are no repeated Monte Carlo draws and no analytic expected score. The caller
records the seed in experiment metadata. This is an evaluation baseline only and
is unrelated to the prohibition on stratified-random data *splits*.
*Now — PR-AUC.* `pr_auc` is Average Precision in its step-wise definition,
the sum over score thresholds of (Rₙ − Rₙ₋₁) × Pₙ. The trapezoidal area under the
precision–recall curve is not used: its linear interpolation between operating
points is optimistic at exactly the imbalances §5.4 describes. ROC-AUC is
computed as a secondary figure only (D7).
*Now — implementation.* The metric arithmetic is written explicitly over NumPy and
the standard library, and hand-verified against hand-computed fixtures.
scikit-learn is not introduced for it, and no dependency is added.
*Now — the roster and zero denominators.* The caller supplies the complete,
ordered class roster. For per-class precision, recall and F1, an undefined
zero-denominator case resolves explicitly to `0.0`, support is reported, and a
class absent from the evaluation population stays represented. Macro-F1 averages
over the **full supplied roster**, never over only the classes present in the
targets or predictions.
*Now — top-k.* `top_k_accuracy` takes a score matrix whose columns correspond
exactly to the supplied labels, in that order. `k` must be at least 1; a `k`
greater than the number of classes raises `ValueError` and is never clamped. Tied
scores break deterministically by roster order, and unknown or misaligned labels
and a target/score length mismatch raise.
*Now — order, taxonomy and polarity.* Metrics never infer a class taxonomy, a
class order, a minority class or a positive polarity. The caller supplies the
ordered labels, and a `positive_label` wherever a metric is binary or
polarity-specific, as §5.4 already requires a polarity mapping to be stated
rather than implied. Unknown labels in targets or predictions raise. That one
roster order governs the confusion matrix, the per-class report, macro-F1, the
top-k score columns, top-k tie-breaking and majority-class tie-breaking.
*Now — ranking baselines.* The majority baseline has one definition across
classification and ranking metrics: the training class-prior vector, fitted from
training labels only, with its entries in the supplied roster order and ties in
the prior resolved by that order. For a label metric it predicts the class with
the highest training prior. For a ranking metric the complete prior vector is the
score vector, identical for every evaluation row. Majority Average Precision
therefore equals the positive-class base rate wherever the evaluation population
holds positives, majority ROC-AUC equals 0.5 wherever it holds both positives and
negatives, and majority top-k ranks the k classes with the highest training
prior. The stratified-random baseline is reported **only** for the label metrics
`macro_f1`, `per_class_report`, `confusion_matrix` and `minority_report`, and
**never** for `pr_auc`, `roc_auc` or `top_k_accuracy`. Its single seeded draw
yields one label per row and defines neither a continuous score nor a ranking;
manufacturing either — including using the 0/1 draw as a score — would be the
invented baseline the clause on results forbids. On a 99:1 population that
substitution reports an Average Precision anywhere from 0.01 to 1.0 depending on
the seed alone.
*Now — undefined ranking metrics.* `pr_auc` raises `ValueError` when the
evaluation population holds no example of `positive_label`. `roc_auc` raises
`ValueError` when it holds no positive example or no negative example. Neither
returns `0.0` or `NaN`, and the message names the metric and the missing class
condition.
*Now — an empty evaluation population.* Every public metric raises `ValueError`
when the evaluation population is empty. The `0.0` zero-denominator rule applies
only to an individual roster class absent from a **non-empty** evaluation
population; applied to no data at all it would publish a macro-F1 of `0.0` that
describes nothing.
*Now — confusion-matrix baselines.* The confusion-matrix result carries the
model's matrix beside the majority baseline's and the stratified baseline's,
each built by the same function, with rows the true class and columns the
predicted class, in the supplied roster order. A confusion matrix is never
reduced to a scalar to satisfy the baseline rule.
*Now — the majority baseline is mandatory.* As §5.1 requires, every applicable
public metric result carries its majority-class baseline, in the ranking form
above wherever the metric needs scores or a ranking, and no metric may return a
model score without it. The stratified-random baseline is carried only by the
four label metrics named above.
*Now — CI.* The stratified-random baseline requires
`numpy.random.default_rng`, so the Task 14 tests need NumPy, which CI's
application job does not install. `tests/ml/training/test_metrics.py` is therefore
added to the ML job's existing path-selected test command. This is an explicit
exception to plan Task 14's "must not change anything outside `ml/training/`",
and no other CI change is made.
*Why:* every clause above moves a published figure, so each is fixed before any
metric is computed rather than discovered in code. Fitting baseline priors on
training labels keeps the do-nothing comparison free of evaluation targets by the
same rule §6.3 applies to target-derived features; a single seeded draw keeps the
stratified figure reproducible under plan §R; and a caller-supplied order and
polarity keep the metrics from becoming a second, unreviewed taxonomy beside the
roster §1 derives from data.
*Scope:* Task 14 only. Tasks 9–13 are unchanged, `recall_at_k` moves to Task 18,
the one CI test-path line above is the only change outside `ml/training/` and
its tests, and D1–D33 are unaltered.
**D35 — Task 15 artifacts: joblib only, a closed metadata schema written last, an
immutable version directory, and a feature builder bound to the artifact's own
spec (§3.3, §6.1, §7, D12, D27, D34, plan §P, plan §R, plan Task 15).**
*Was:* plan §P fixed the directory layout and named the `metadata.json` fields,
and plan Task 15 fixed `write_artifact(model, metadata, path)` and
`load_artifact(path) -> LoadedArtifact` exposing `feature_spec` and a
`build_features(records)`. Left open were: which of the two model formats Task 15
supports; which fields are required, which may be null, and what types they must
have; what happens to unknown keys and to values JSON cannot represent exactly;
whether a version directory may be written twice and what a failed write leaves
behind; how the directory relates to the metadata; what else `LoadedArtifact`
exposes; and, decisively, how a feature builder taking only `records` could build
`RiskFeaturesV1`, whose two aggregate columns Task 12 can only take from Task 11's
`AggregateColumns`. As written, the primary risk artifact could never build its
own features.
*Now — format and layout.* Task 15 supports **joblib only**: the model is
`model.joblib` and the metadata is `metadata.json`, both directly inside the
version directory. ONNX is not written here and no dependency is added; the ONNX
MiniLM checkpoint belongs to Task 18. Loading a joblib file unpickles arbitrary
code, so only trusted, self-produced artifacts may be loaded.
*Now — the feature builder.* `LoadedArtifact.build_features(records,
aggregates=None)` delegates to Task 12's `build_features` with the artifact's own
`FeatureSpec`. The artifact binds the spec; it never freezes or embeds aggregate
values. Scoring a spec that names an aggregate feature without supplying
aggregates therefore raises Task 12's `FeatureUnavailable`, which is exactly the
"primary artifact cannot score CFPB" behaviour plan Task 19 asserts.
*Now — the feature spec.* `feature_spec` is a non-empty JSON array of non-empty,
distinct strings, and `feature_spec_version` a non-empty string. They load as
`FeatureSpec(names=tuple(feature_spec), version=feature_spec_version)` in exactly
the stored order: nothing is sorted, deduplicated, expanded to the ten-field
`RiskFeatures` interface, or read from the model object. A duplicated, empty or
non-string name raises `ArtifactSchemaError` rather than being repaired.
*Now — the load guard.* At load, every name in `feature_spec` is checked against
what this environment can produce, using Task 12's public `build_features` one
name at a time. If any cannot be produced, loading raises `FeatureSpecMismatch`
naming **every** unavailable feature, in spec order. The guard runs at load, as
plan Task 15 states; `write_artifact` does not check feature availability.
*Now — required fields.* Every artifact carries all sixteen base keys:
`model_name`, `model_version`, `trained_at`, `git_sha`, `corpus_id`,
`corpus_schema_version`, `source_window`, `split`, `feature_spec`,
`feature_spec_version`, `label_roster`, `thresholds`, `metrics`,
`warmup_row_count`, `seeds` and `dependency_versions`. A missing key raises
`ArtifactSchemaError` naming it. Only `label_roster`, `thresholds` and
`warmup_row_count` may be an explicit `null`, meaning not applicable to this kind
of artifact; `null` anywhere else is invalid. The optional keys
`embedding_dimension`, `embedding_model_id`, `embedding_model_sha256` and
`experiment_label` may be absent but, when present, must be non-null and valid.
Neither function ever supplies a missing value.
*Now — types.* Validation is shallow and structural. `model_name`,
`model_version`, `git_sha`, `corpus_id`, `feature_spec_version`,
`embedding_model_id`, `embedding_model_sha256` and `experiment_label` are
non-empty strings; `trained_at` is an ISO-8601 timestamp with a timezone;
`corpus_schema_version` is an integer, `warmup_row_count` a non-negative integer
and `embedding_dimension` a positive integer, booleans excluded; `split`,
`thresholds`, `metrics`, `seeds` and `dependency_versions` are JSON objects.
`source_window` and `label_roster` are checked only for presence and
nullability. The inner structure of every object belongs to the task that
produces it, and Task 15 defines none of it.
*Now — unknown keys and strict JSON.* An unknown top-level key raises
`ArtifactSchemaError`, so a misspelt field cannot pass unnoticed. Serialization is
strict JSON: a non-finite float, a non-string object key, or any value JSON has no
exact representation for raises `ArtifactSchemaError` instead of being converted —
non-finite values are never written as `NaN` or as `null`. Loading is equally
strict: `NaN` or `Infinity` tokens, duplicate keys, invalid UTF-8 and a non-object
document all raise `ArtifactSchemaError`.
*Now — the directory.* `path` is the caller-provided version directory. Its final
component must equal `model_version` and its parent's final component must equal
`model_name`, checked by both `write_artifact` and `load_artifact`; a mismatch
raises `ArtifactSchemaError`. The comparison is lexical and uses the path as the
caller gave it: `.` and `..` are normalised away as text, so `<model>/v1/../v1`
names `<model>/v1`, but symbolic links and junctions are never resolved, so a link
`v2` pointing at `v1` is checked as `v2` and cannot load `v1`'s artifact. Nothing
is derived from the path or written into the metadata from it.
*Now — immutability and the validity boundary.* A published version directory is
immutable: `write_artifact` raises `FileExistsError` when the directory already
holds `metadata.json` or `model.joblib`. All validation runs before anything is
written. The model is written first; `metadata.json` is written **last**,
through a temporary file in the same directory and `os.replace`, the D27
pattern. `metadata.json` is the validity boundary: a directory without it is not
an artifact, so a write that fails part-way leaves nothing loadable. Loading
raises `FileNotFoundError` for a missing `metadata.json` or `model.joblib`, and
unpickles the model only after the metadata, the directory and the feature guard
have all passed.
*Now — the loaded artifact.* `LoadedArtifact` is frozen and training-side: it
exposes `model`, a deeply read-only `metadata`, `feature_spec` and
`build_features`. It has no prediction or serving API, and nothing here touches
`ml/registry.py`. `ArtifactSchemaError` and `FeatureSpecMismatch` are both
`ValueError`s, like the rest of `ml.training`. The writer records the metadata it
is given exactly and never recomputes a split, a threshold, a metric, an
aggregate or a label.
*Now — deferred.* What `feature_spec` and `build_features` mean for a text triage
artifact or an embedder artifact is not defined here; Tasks 16 and 18 define those
contracts.
*Now — CI.* The artifact tests need joblib and scikit-learn, which CI's
application job does not install, so `tests/ml/training/test_artifacts.py` is
added to the ML job's existing path-selected test command. No other CI change is
made.
*Why:* an artifact exists to be trusted later by a process that did not train
it. Every clause above closes a way that trust could be misplaced without an
error: a builder quietly given different columns, a field quietly invented or
dropped, a value quietly rounded into valid JSON, a published version quietly
overwritten, or a half-written directory quietly accepted. Binding the spec while
leaving aggregates to the caller is the only reading under which the primary risk
artifact can build its features at all, and it keeps Task 11's out-of-fold and
frozen aggregate paths where D31 put them.
*Scope:* Task 15 only. Tasks 9–14 are unchanged, `ml/registry.py` is unchanged
and nothing is wired into serving; triage and embedder feature semantics are
deferred to Tasks 16 and 18; the one CI test-path line is the only change outside
`ml/training/` and its tests; D1–D34 are unaltered.
**D36 — Task 16 CFPB triage: TF-IDF blocks declared in the artifact, argmax
metrics beside a separate abstention rate, a derived alphabetical roster order and
frozen hyperparameters (§1, §1.1, §5.1, §5.2, §6.1, §6.2, D12, D34, D35, plan §M,
plan §P, plan §R, plan Task 16).**
*Was:* plan §M fixed the recipe — word and character TF-IDF fitted on train text
only, into `LogisticRegression` wrapped in `CalibratedClassifierCV`, against the
derived roster, with macro-F1 as the headline and an abstention threshold tuned on
validation. D35 deferred what `feature_spec` means for a text artifact, and plan
Task 16 simultaneously required an artifact per §P and forbade changing the
artifact module — which together were unsatisfiable, because Task 15's load guard
probes every stored feature name through Task 12's five-name vocabulary and would
refuse any name a TF-IDF model could honestly declare. Left open besides: what
quantity abstention thresholds and how an abstained record enters a metric; where
the fitted threshold is recorded; the artifact's name, version and experiment
label; the roster's order, which D34 makes decisive for five published figures;
whether class weighting or resampling applies to a 77%-majority target; every
estimator hyperparameter except the two n-gram ranges; and where the fixture
corpus is generated.
*Now — the text feature contract.* The triage artifact declares
`feature_spec = ["tfidf_word_1_2", "tfidf_char_3_5"]` with
`feature_spec_version = "triage_tfidf_v1"`, the word block before the character
block. These names denote ordered **feature blocks**, not columns: a triage
spec describes the design matrix's block structure and its order, where Task 12's
names each denote exactly one column. That difference is the contract, not an
accident of width, and it is why a triage spec is two names long while its matrix
is not two columns wide.
*Now — the compatibility seam.* A minimal additive extension to
`ml/training/artifacts.py` is authorised, so that those two names validate under
`load_artifact` and rebuild through `LoadedArtifact.build_features`. It is bounded
in four directions: the recognised text-block spec is a **closed constant**, never
a generic registry of arbitrary feature builders; Task 12's vocabulary, ordering
and `FeatureUnavailable` behaviour are unchanged, and a risk or transfer spec
still resolves exactly as it does today; no artifact file, metadata field or
filename is added, so D35's schema and layout stand unaltered; and nothing is
wired into `ml/registry.py` or any serving path. This supersedes plan Task 16's
"must not change: metrics or artifact modules" **only** to the extent of this
seam. `ml/training/metrics.py` is not changed at all.
*Now — abstention, and what it does not touch.* Confidence is the maximum
calibrated class probability, and a record is abstained when that confidence is
strictly below the fitted threshold. Task 14's classification metrics are computed
over the **complete** evaluation population from the argmax class prediction:
macro-F1, the per-class report, the confusion matrix and top-3 accuracy all see
every record, abstained or not. No `abstain` label joins the roster, and no
abstention sentinel is ever passed to a metric function — both would make the
published macro-F1 describe a different population than the one the model scored.
Top-3 accuracy is computed from the calibrated score matrix, whose columns follow
the roster order below. The abstention rate is reported as its own figure beside
the metrics, never folded into one.
*Now — the abstention objective is deferred, not chosen.* The threshold is tuned
on validation and applied unchanged to test (§6.2), but the tuning procedure
itself is **explicitly unresolved**: the candidate grid, the objective being
optimised, whether abstained records are excluded from that objective, any minimum
coverage requirement, and the deterministic tie-break among equal-scoring
candidates. No implementation may select any of these; until they are decided
here, Task 16's threshold fitting has no specification.
*Now — threshold metadata.* The fitted value is recorded in the existing nullable
`thresholds` object as `{"abstention": {"value": <threshold>, "quantity":
"max_calibrated_probability"}}`. No new top-level artifact field is introduced,
and D35's closed schema therefore still refuses every unknown key.
*Now — artifact identity.* `model_name` is `"cfpb_triage_tfidf"`, `model_version`
is `"v1"`, and `experiment_label` is `"cfpb triage tfidf logistic regression"`.
The version directory is consequently `.../cfpb_triage_tfidf/v1/`, which is what
D35's lexical directory check compares against.
*Now — roster order, derived.* The CFPB label list is never transcribed into code
(§1.1). The order is derived at runtime as `tuple(sorted(manifest.label_roster))`
and that one order governs everything D34 says it must: the classifier's class
ordering, the per-class report, the confusion matrix's rows and columns, macro-F1,
the top-k score columns, top-k tie-breaking, majority-baseline tie-breaking, and
the `label_roster` written to metadata.
*Now — class imbalance.* `class_weight=None` and no resampling of any period. The
~77% majority class is reported beside its majority-class and stratified-random
baselines (§5.1), which is what §1.1 means by accepting residual imbalance rather
than engineering it away.
*Now — hyperparameters, frozen rather than tuned.* The word vectoriser is
`analyzer="word"`, `ngram_range=(1, 2)`; the character vectoriser is
`analyzer="char"`, `ngram_range=(3, 5)`; the word block is combined before the
character block. `LogisticRegression` takes `C=1.0`, `solver="lbfgs"`,
`max_iter=1000`, `class_weight=None` and an explicit `random_state`.
`CalibratedClassifierCV` takes `method="sigmoid"`, `cv=5` and `ensemble=True`.
There is **no hyperparameter search**: the only quantity tuned in Task 16 is the
abstention threshold. Every parameter not named here takes the pinned
scikit-learn default, and `dependency_versions` records the versions that supplied
those defaults, so a default that moves between releases is visible in the
artifact rather than silent.
*Now — determinism, stated accurately.* The pinned scikit-learn 1.9.0 exposes no
`random_state` on `CalibratedClassifierCV`; its parameters are `estimator`,
`method`, `cv`, `n_jobs` and `ensemble`. An integer `cv` selects a non-shuffled
`StratifiedKFold`, so calibration is already deterministic given a fixed input
order, which plan §R's deterministic corpus ordering supplies. `lbfgs` is likewise
deterministic, so `LogisticRegression`'s `random_state` is recorded for
provenance rather than because it changes a result. The seeds actually governing a
published figure — the estimator seed and the stratified baseline's required seed
— are recorded in `seeds`. Plan §R's cross-platform caveat is unchanged.
*Now — `warmup_row_count`.* `null` for this artifact. Warm-up is a property of
forward-chaining out-of-fold aggregate construction (§6.3, D30), and triage has no
target-derived feature, so there is no warm-up prefix to count. This is exactly
D35's "not applicable to this kind of artifact" case.
*Now — the fixture corpus.* Task 16 adds no fixture-generator module. The
controlled corpus is generated inside
`tests/ml/training/test_triage_experiment.py` into a temporary directory, written
through the ordinary corpus path so `load_corpus` verifies it, and no generated
data is committed.
*Now — CI.* Task 16 removes the exit-code-5 tolerance from the `ml` job, restoring
the step to a plain `pytest -m ml`. The marker gains its first users here, so an
empty selection now means a mis-typed marker rather than an expected state. No
other CI change is made; the triage tests are selected by the marker and need no
path entry.
*Why:* every clause here fixes a number or a name that a later reader would
otherwise have to reverse-engineer from code. Declaring blocks rather than columns
keeps `feature_spec` an honest description of what the model consumes while
leaving D12's guarantee intact — a text artifact still cannot be handed a matrix
it was not trained on. Keeping metrics over the complete population makes the
headline comparable with every other model in this project and keeps abstention a
reported operating choice rather than a quiet population filter, which is the
mechanism by which a coverage policy would otherwise flatter a macro-F1. Deriving
the roster order from the manifest keeps §1.1's prohibition on transcribed label
lists true in the one place where an order, not just a membership, is published.
Freezing the hyperparameters keeps Task 16 an experiment whose result is
attributable to the recipe rather than to a search nobody recorded.
*Scope:* Task 16 only, and the abstention tuning objective remains open within it.
Tasks 9–15 keep their behaviour; `ml/training/metrics.py` is untouched; Task 12's
feature vocabulary is untouched; the authorised change to
`ml/training/artifacts.py` is limited to the compatibility seam above; no
dependency is added; `ml/registry.py`, serving, Django and the database schema are
untouched; the one CI tolerance removal is the only change outside `ml/training/`
and its tests; and D1–D35 are unaltered.
*Now — the abstention threshold, completed. This clause supersedes the paragraph
above beginning "the abstention objective is deferred, not chosen", and discharges
the caveat in the scope line above; D36 is no longer open in any part.* Confidence
remains the maximum calibrated class probability, and a record is abstained when
that confidence is strictly below the threshold. The threshold is selected on the
validation period alone, as follows.
The **candidate grid is fixed** at `0.00, 0.05, 0.10, …, 0.95` — twenty candidates,
arithmetic, inclusive of `0.00` and stopping at `0.95`. It is not derived from the
data, so the search space cannot drift with the corpus.
Each candidate is evaluated on the **validation period only**. A candidate retains
the validation records whose confidence is greater than or equal to it, and a
candidate is **feasible** when its retained set covers at least **70% of
validation records**. A candidate whose retained set is empty is invalid, never
merely unfeasible.
**The 70% minimum retained coverage is a Sentinel project design constant
introduced here by D36.** It is not derived from §5, §6, plan §M or any measured
quantity, and it must not be attributed to an earlier source. Without it the
selection is degenerate: the highest thresholds retain only the handful of
most-confident records and score near-perfectly on them.
Among feasible candidates the selected threshold **maximises macro-F1 on the
retained, non-abstained validation subset**, computed through Task 14 with the
**complete project label roster** supplied in the established deterministic roster
order — `tuple(sorted(manifest.label_roster))` — so that a class emptied by
filtering still enters the macro average at D34's explicit `0.0` rather than
disappearing from the denominator. Where two or more feasible candidates score an
identical retained macro-F1, the **lowest** threshold is selected, which is the one
retaining the most records. That tie-break is not decorative: on a small corpus
adjacent grid points frequently retain the identical record set.
The selected value is the fitted abstention threshold, recorded in `thresholds` as
already specified, and is **applied unchanged to the test period** (§6.2).
This objective is an **internal validation-selection criterion, not a published
Task 14 metric.** The retained-subset macro-F1 is never reported as the model's
macro-F1, and it is not added to the artifact's `metrics`. Published evaluation is
exactly what Task 14 and §5.2 already fix: macro-F1, per-class
precision/recall/F1/support, the confusion matrix, top-3 accuracy, each beside its
required baselines, **all computed over the complete test population using argmax
class predictions**. Abstention adds no label, removes no row from a published test
metric, and does not alter the label roster.
Everything else already decided in D36 stands unchanged: the two-name
`feature_spec` and its `triage_tfidf_v1` version, `model_name`
`"cfpb_triage_tfidf"`, `model_version` `"v1"`, `experiment_label` `"cfpb triage
tfidf logistic regression"`, the alphabetically derived roster order,
`class_weight=None` with no resampling, the frozen hyperparameters, a `null`
`warmup_row_count`, the fixture corpus generated inside the test module, and the
bounded artifact compatibility seam. The determinism correction stands too: the
pinned scikit-learn 1.9.0 accepts no `random_state` on `CalibratedClassifierCV`,
`cv=5` is used as specified and yields the deterministic non-shuffled
`StratifiedKFold`, and `LogisticRegression` receives its explicit `random_state`.
*Scope of this clause:* Task 16's abstention threshold only. No published metric,
metric function, roster, artifact field or earlier decision changes; Tasks 9–15
keep their behaviour; and D1–D35 remain unaltered.
*Now — four details closed before implementation, surfaced by the Task 16 RED
tests. This is a further completion of D36, not a new decision.*
**The abstention selector's signature.** Threshold selection is a separable
function whose shape is fixed here:

```
select_abstention_threshold(confidences, y_true, y_pred, roster, *, train_labels, seed)
```

The first four parameters are positional; `train_labels` and `seed` are
keyword-only. The two extra parameters are not convenience: D36's objective scores
each retained subset **through Task 14**, and `macro_f1` structurally requires a
`majority` and a `stratified` baseline. D34 fixes both priors to **training labels
only**, and the stratified baseline requires an explicit seed and a draw whose
count matches the candidate's retained size, so a fresh draw is made per candidate
from that one seed. The selector therefore **must not infer a prior from validation
or test labels**; passing no training labels is an error rather than an invitation
to fall back on the evaluation population.
**The text feature blocks stay in their natural representation.** A fitted
`TfidfVectorizer` produces a SciPy sparse matrix, and the fitted classifier
consumes one directly. `build_feature_blocks(texts)` therefore returns exactly what
the two fitted vectorisers produce, horizontally stacked with the **word block
first and the character block second**, and **no densification is performed merely
to satisfy a type annotation**. The compatibility seam passes that representation
through the existing public artifact API unchanged. Densifying a TF-IDF matrix for
appearance's sake would turn a narrow, sparse design matrix into a dense one whose
size is the product of the corpus and the vocabulary, which is a real cost paid for
nothing.
**Aggregates are refused, never ignored.** A triage artifact is text-only.
`LoadedArtifact.build_features(records, aggregates=...)` with a non-`None`
`aggregates` raises `ValueError`, and the message names `aggregates`. Silently
discarding a supplied argument is the defect class this project refuses
everywhere else, and it would let a caller believe an aggregate influenced a
score that never saw it.
**`abstention_rate` is ancillary, and lives inside `metrics`.** D35's schema is
closed and D36 fixes `thresholds` to exactly the abstention-threshold object, so
the rate is recorded under the existing `metrics` object at the period it
describes, in a field named `abstention_rate`. No new top-level metadata field is
created. It is **not** a member of the Task 14 metric set, **not** the headline,
and it neither replaces nor alters macro-F1, the per-class report, the confusion
matrix or top-3 accuracy — all four of which remain computed over the complete
evaluation population from argmax predictions, exactly as this decision already
fixed. It is computed **after** the frozen, validation-selected threshold has been
applied, as the share of that period's records whose confidence falls below it, and
it carries no baseline because it is a description of the operating point rather
than a score to beat.
*Scope of this clause:* Task 16 implementation detail only. No metric function,
roster, published figure, artifact field or earlier decision changes; Tasks 9–15
keep their behaviour; and D1–D35 remain unaltered.
**D37 — Task 17 the 311 risk model: outcomes persisted in a sidecar, open
requests split from the labelled population, a two-band decision threshold tuned
on validation, and a frozen HistGradientBoosting recipe (§2.1, §2.7, §4.2, §5.5,
§6, §7, §G, D1, D3, D11, D15, D21, D27, D30, D31, D33, D34, D35, plan §J, §K,
§L, plan Task 17).**
*Was:* Task 17 was unimplementable and several of its published figures were
undefined. The corpus stores `CorpusRecord` only — `ingest/cli.py` normalises
`(record, outcome)` pairs and then persists the records alone — so
`NYC311Outcome.resolution_hours`, from which the entire target derives, reached
no consumer. Every Task 11 and Task 13 API Task 17 must call takes a sequence of
those outcomes. Left open besides: the 311 training window, §1's window being
CFPB's alone; whether open requests belong to the model population, which D31 and
D33 both deferred here by name; what "decision banding" means, named three times
and defined nowhere; the positive class and roster order for a boolean target;
which thresholded metrics carry §5.5's stratified baseline, given D34 forbids it
on any ranking metric; who owns the calibration curve §5.5 requires and Task 14
never implemented; class weighting; every `HistGradientBoostingClassifier`
hyperparameter; the artifact's identity; and the shape of its `thresholds` object.
*Now — outcome persistence (D37.1).* `SCHEMA_VERSION` stays **1** and no outcome
field joins `CorpusRecord`. A source that has an outcome stream persists it as an
authoritative **sidecar inside that source's existing corpus root**, beside the
record partitions and the manifest, rather than discarding it at ingest. For NYC
311 the sidecar carries `external_id` and `resolution_hours`, the latter nullable
for an open request. Outcome parts join the manifest's integrity check and the
`corpus_id`, so an artifact citing that identity is bound to both its record and
its outcome bytes. A public `load_outcomes(...)` joins the manifest layer and
reads only manifest-declared outcome files, verifies their checksums before
yielding, never reads `data/raw/`, never silently skips a malformed row, and
preserves `external_id` identity. `load_corpus` remains the authoritative
`CorpusRecord` reader and is unchanged. Together the record corpus and the
outcome sidecar are Task 17's authoritative input. A corpus without a sidecar
stays valid for every task that needs no outcome; Task 17 requires one. No
database or ORM representation of the outcome stream is introduced.
*Now — the training window (D37.2).* 311 uses **2024-01-01 through 2025-12-31
inclusive**, under D21's NYC civil-time interpretation. This is a Sentinel design
decision taken here, **not** a claim that §1 or any earlier document already
fixed a 311 window; §1's window is CFPB's.
*Now — open requests (D37.3).* An open request stays in the corpus, in the
temporal split, in fold construction and in aggregate construction, and may
receive aggregate features. It is **not** a member of the labelled population: it
never receives `nyc311_sla_breach`, contributes to no threshold statistic, and is
excluded from classifier fitting and from every evaluation metric. The order is
therefore: all records → temporal split → thresholds and aggregates → feature
construction → resolved-only model population. An unresolved outcome is never
coerced to `False`, which is the refusal D33 already built into
`apply_thresholds`.
*Now — decision banding (D37.4).* Two bands: **low** where the score is below the
threshold and **high** where it is at or above it. The score is the model's
predicted probability of the positive class `True`. The candidate grid is
`0.05, 0.10, … 0.95`. The threshold is selected on the **validation period
alone**, maximising the positive class's F1 for `True`; an exact tie takes the
**lowest** threshold. No test record may influence the selection, and the chosen
value is applied unchanged to validation and test.
*Now — the positive class and the roster (D37.5).* `False` is the negative,
non-breach class and `True` the positive, breach class. The roster order is
exactly `(False, True)`, written to metadata as the JSON list `[false, true]`.
That one order governs the classifier's class ordering, the probability columns,
the thresholded predictions, every metric and the recorded roster.
*Now — published metrics (D37.6).* The ranking metrics are **PR-AUC (headline)**
and **ROC-AUC (secondary)**, both scored on the probability of `True` with
`positive_label=True` and carrying the majority baseline only, as D34 requires.
The thresholded supporting metrics are `minority_report` and `confusion_matrix`,
computed from the frozen decision threshold over the full `(False, True)` roster,
carrying the majority and stratified-random baselines wherever D34 permits.
Macro-F1 is **not** introduced as a risk headline, and the stratified-random
baseline is **never** used as a ranking baseline.
*Now — breach rate (D37.7).* Task 13's `breach_rate` is published per validation
and test period, computed over that period's resolved, labelled population. Open
requests are not in the denominator.
*Now — the calibration curve (D37.8).* Validation and test each carry a
diagnostic calibration curve over **10 uniform probability bins**, reporting only
the populated bins, each with `mean_predicted_probability`, `fraction_positive`
and `count`. It is ancillary diagnostic information: not a scalar headline, not a
Task 14 metric, and it carries no baseline. No Brier score is added.
*Now — class imbalance (D37.9).* `class_weight=None` and no resampling. The
residual imbalance the p75 threshold produces is reported, not engineered away,
which is the same rule §1.1 applies to CFPB's.
*Now — the model (D37.10).* `HistGradientBoostingClassifier` with
`learning_rate=0.1`, `max_iter=100`, `max_leaf_nodes=31`, `max_depth=None`,
`min_samples_leaf=20`, `l2_regularization=0.0`, `early_stopping=False`,
`class_weight=None` and `random_state=17`. No hyperparameter search; every other
parameter takes the pinned scikit-learn 1.9.0 default. `early_stopping=False` is
explicit and load-bearing: the estimator's `'auto'` default would carve an
internal **random** validation split out of the training rows, which is exactly
the non-temporal evaluation §6 prohibits.
*Now — artifact identity (D37.11).* `model_name` is `"nyc311_sla_risk"`,
`model_version` is `"v1"`, and `experiment_label` is
`"nyc311 sla risk histgradientboosting"`, giving the version directory
`nyc311/nyc311_sla_risk/v1/`. `feature_spec` is exactly `RiskFeaturesV1`'s five
names in order — `submitted_hour`, `submitted_weekday`, `text_length`,
`category_mean_resolution_hours`, `category_breach_rate` — under
`feature_spec_version` `"risk_features_v1"`. `warmup_row_count` is
`len(folds[0].fit_indices)`, the realised warm-up D30 defines, and is **not**
null here: unlike triage, this model has an out-of-fold construction and so has a
warm-up to count.
*Now — threshold metadata (D37.12).* The fitted Task 13 information is recorded
under `thresholds` as `min_eligible` (100), `percentile` (0.75), `per_type`,
`global_fallback` and `fallback_type_count`, taken directly from
`FrozenThresholds`. Those five fields are exactly the `FrozenThresholds`
content, and remain so. Beside them, inside that same `thresholds` object, the
frozen decision band of D37.4 is recorded as `{"decision": {"value":
<threshold>, "quantity": "probability_of_true"}}`, where `value` is the
validation-selected threshold the evaluation was frozen at and `quantity` names
what it thresholds. This mirrors how D36 records triage's abstention threshold,
and is what lets a loaded artifact re-apply its own operating point without
consulting anything outside itself. Nothing further about the selection is
recorded — not the candidate grid, not the objective, not the tie-break, not the
validation population size, not the achieved validation F1. Those are specified
in D37.4 and live in the experiment code, which the artifact's `git_sha` already
identifies. `decision` is additional nested metadata within the existing
nullable `thresholds` object, not a new top-level artifact metadata field: no
threshold mathematics is recomputed in Task 17 and no new top-level artifact
field is created, so D35's closed schema still refuses every unknown key.
*Now — features and aggregates (D37.13).* Training rows use
`forward_chaining_folds(...)` then `oof_category_aggregates(...)`; validation and
test rows use `fit_category_aggregates(train_records, train_outcomes)` then
`apply_category_aggregates(...)`. These existing public Task 11 APIs are called,
never restated, and the columns reach Task 12 through `AggregateColumns` and
`build_features`. `RiskFeaturesV1` remains authoritative, and `sla_hours`,
`age_hours`, `priority_rank`, `queue_depth` and `assignee_open_count` remain
outside it.
*Now — dependency versions (D37.14).* The artifact records the versions Task 17
actually uses: numpy, scipy, scikit-learn and joblib. `onnxruntime` is not
recorded merely because plan §R's generic provenance sentence names it; this
experiment does not use it.
*Now — scope (D37.15).* Task 17 may touch the minimum `ingest` storage and
manifest surface D37.1 requires in order to persist and load the outcome sidecar.
It must not change the `CorpusRecord` schema, aggregate semantics, threshold
semantics, `ml/training/metrics.py`, `features.py`, `labels.py`, `thresholds.py`
or `splits.py`, the registry or any serving path, Django, or the database schema,
and it adds no dependency unless a pinned one is genuinely insufficient.
*Why:* the sidecar is the only resolution that leaves `CorpusRecord` — and
therefore every existing corpus tree, every `corpus_id` already citable and
Task 16's published artifact — untouched while making the risk target reachable
at all; putting outcomes into the record schema would bump `SCHEMA_VERSION` and
strand every partition already written. Keeping open requests in the split and
the aggregates but out of the labelled population is what lets them inform a
category's feature history without ever acquiring a label that would have to be
invented. Two bands rather than three is the smallest banding that yields the
thresholded predictions §5.5's stratified baseline needs, since D34 allows that
baseline on no ranking metric. And fixing the hyperparameters keeps the published
PR-AUC attributable to the recipe rather than to an unrecorded search.
*Scope:* Task 17 only, plus the sidecar surface D37.1 names. Tasks 9–16 keep
their behaviour, `CorpusRecord` and `SCHEMA_VERSION` are unchanged, Task 16's
artifact and decisions are untouched, and D1–D36 are unaltered.
*Now — manifest versioning (D37.16), completing D37.1.* `MANIFEST_VERSION`
becomes **2**, while `CorpusManifest.schema_version` stays **1**: the manifest
document gains a field, the `CorpusRecord` schema does not, and §G separates the
two numbers for exactly this case. Manifest v1 files stay **readable**. Reading a
v1 manifest interprets the absent `outcome_part_files` as `{}` and leaves every
existing `part_files` semantic untouched. That compatibility is **read-only**: a
new write always emits v2. A v2 manifest carries `outcome_part_files`, which may
be empty for a source with no outcome stream. `verify_manifest` verifies the
outcome parts wherever they are declared, and `load_corpus` continues to load
record parts and only record parts. `load_outcomes` on a manifest declaring no
outcome parts raises a **typed absence error**; it does not return an empty
iterator, because "this corpus has no outcome sidecar" and "this sidecar is empty"
are different facts and a caller must not confuse them. Manifest-version
compatibility may never change the record corpus's `CorpusRecord` schema version
or silently reinterpret existing record data. The deliberate default for
`outcome_part_files` is an exception granted to this one new optional field and
does not extend to `manifest_version`, `limit` or `timestamp_diagnostic`, which
keep their no-silent-default rule.
*Now — the sidecar layout (D37.17).* **Existing record partitions do not move.**
They stay in `year=YYYY/` under the source's versioned root, and the sidecar is
added beside them as `outcomes/year=YYYY/part-*.parquet` under that same root.
Outcome filenames need not mirror record filenames. The manifest keeps the two
checksum sets **separate**, as `part_files` and `outcome_part_files`, so a reader
can tell which bytes are which without parsing a path. Outcome parts take part in
manifest verification, in corpus identity and in source-tree replacement and
deletion — the last for free, since `remove_source_tree` removes the whole
versioned root. `compute_corpus_id` is **unchanged**; `build_manifest` passes it
the merged record-and-outcome checksum set. A record-only corpus therefore keeps
the identity it already has, because merging an empty outcome set changes
nothing, while a corpus holding sidecar bytes has an identity that binds them.
`load_outcomes` reads only manifest-declared outcome parts, verifies their
checksums before yielding anything, rejects an outcome file on disk the manifest
does not list, never reads `data/raw/`, preserves `external_id`, raises on a
malformed row, and raises the typed absence error above when no sidecar is
declared.
*Why these two:* keeping the record partitions where they are is what makes
"existing corpora remain valid" literally true — relocating them inside `v1`
would strand every manifest already written and amount to a schema change under
another name. Separate checksum maps keep `part_files` meaning exactly what it
has always meant, so no existing reader is reinterpreted. And leaving
`compute_corpus_id` alone while merging at the call site is what lets a
record-only corpus keep its published identity while a corpus with outcomes gets
one that genuinely covers its inputs.
*Scope of these two clauses:* the manifest document and the corpus layout only.
`CorpusRecord`, `SCHEMA_VERSION`, `part_files` semantics, `load_corpus`,
`compute_corpus_id` and every Task 9–16 behaviour are unchanged, and D1–D36
remain unaltered.
*Now — a correction to D37.1's sidecar schema, and the scope of outcome
persistence.* The NYC 311 outcome sidecar carries **three** columns, not two:
`external_id`, `resolution_hours` and `closed_at`. D37.1's original two-column
sketch was unimplementable: `NYC311Outcome` has three fields, and both
`ml/training/aggregates.py` and `ml/training/labels.py` validate that the objects
they receive **are** `NYC311Outcome` instances — which D37.15 forbids changing —
so a loader reconstructing only two fields would have had to invent the third. A
`closed_at` of `None` means "still open at ingest" in §2.1, so pairing it with a
non-null `resolution_hours` would have produced a self-contradictory object.
*The semantics.* A **resolved** request has a non-null `external_id`, a non-null
`resolution_hours` and a non-null `closed_at`, the latter being the **actual
normalised NYC 311 close timestamp from the source**. An **open** request has a
non-null `external_id` with `resolution_hours` and `closed_at` both null.
`closed_at` is **never reconstructed** as `submitted_at + resolution_hours`: the
sidecar preserves what the source published, and a derived value would silently
become authoritative the moment the two disagreed — a rounding difference, a
timezone normalisation, or a corrected close time upstream. `load_outcomes`
therefore returns real `NYC311Outcome` instances carrying all three fields, which
is what keeps the Task 11 and Task 13 validation paths working with no adapter.
*The scope of outcome persistence.* D37's requirement to persist an outcome
stream applies **specifically to the NYC 311 stream Task 17 consumes**. A future
source-specific outcome stream defines its own sidecar schema in the task that
first consumes it. Concretely: Task 17 defines and persists NYC 311 outcomes;
**Task 19 may later define a CFPB outcome sidecar** for the CFPB stream it
consumes; and Task 17 must **not** invent or implement that future CFPB schema.
The Task 17 ingest changes therefore stay NYC 311-specific, and CFPB ingest
behaviour is unchanged by them.
*Why:* a sidecar exists so a later process can trust what an earlier one
observed. Two of the three fields would have forced the loader to fabricate the
third, and the only fabrication available contradicted the field's documented
meaning. Fixing the schema at three columns costs one nullable timestamp per row
and removes the contradiction entirely. Confining the requirement to NYC 311
keeps Task 17 from designing a schema for data it never reads, which is the same
discipline D35 applied when it left triage and embedder feature semantics to the
tasks that own them.
*Scope of this clause:* the NYC 311 sidecar schema and the reach of D37's
persistence requirement. `CorpusRecord`, `SCHEMA_VERSION`, `part_files`,
`load_corpus`, `compute_corpus_id`, CFPB ingest and every Task 9–16 behaviour are
unchanged, and D1–D36 remain unaltered.
**D38 — Task 18 duplicate retrieval: both arms frozen under one shared
configuration, the MiniLM asset pinned by digest as an external prerequisite,
no embedder artifact written in Phase 2, and D18's artifact clause deferred to
Phase 3 (§2.2, §5.3, §6.1, §6.2, D18, D34, D35, plan §N, plan Task 18).**
*Was:* §5.3 locked the comparison methodology before either candidate was built
and D18 fixed the dimension discipline, but no clause elected the values either
arm needs. Undecided on entry: the MiniLM checkpoint, its pooling rule, sequence
length and tokenizer; where the ONNX weights come from and whether a test may
fetch them; the benchmark's own TF-IDF recipe, D36's being Task 16-scoped; `k`,
the evaluation and perturbation populations, their seed and the tuning budget;
recall@k's definition and the baseline §5.3 requires but does not define, which
D34 deferred here; whether Task 18 writes an embedder artifact, which D35 also
deferred here; and the `TextEmbedder` protocol's return type, fit seam and input
contract.
*Now — the checkpoint, chosen here rather than assumed.* The MiniLM arm is
**`sentence-transformers/all-MiniLM-L6-v2`** at repository revision
**`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`**, and the export is that
revision's **`onnx/model.onnx`**. The optimised exports `model_O1` through
`model_O4` and every quantized export are excluded: a benchmark that cannot say
which bytes produced its number is not a benchmark. D18's warning stands
unweakened — this clause fixes *which* checkpoint Sentinel measures and fixes
nothing about its output width.
*Now — the graph, and what Sentinel adds to it.* The export takes three int64
inputs, `input_ids`, `attention_mask` and `token_type_ids`, and emits a single
output `last_hidden_state` of shape `(batch, sequence, hidden)`. **The graph
performs no pooling and no normalisation**; the published model's pooling and
normalising modules are outside it. Sentinel therefore applies
**attention-mask weighted mean pooling, then L2 normalisation**, which
reproduces the checkpoint's documented representation rather than inventing one.
`token_type_ids` are zeros, every input being a single segment. Batches are
padded to their longest member, which the graph requires; because pooling is
mask-weighted, **padding changes no vector**, and that is asserted rather than
assumed. `embedding_dimension` is read from the observed output width on every
run, and the literal `384` appears nowhere under `ml/embedders/`.
*Now — the model is frozen.* Phase 2 performs inference only: no fine-tuning, no
gradient computation, no training-mode execution. ONNX Runtime is the only
execution path.
*Now — tokenization is delegated, never hand-written.* Token ids come from
**`tokenizers==0.23.2`**, pinned in **`requirements/ml.txt`**, reading the
checkpoint's own `tokenizer.json` from the same revision as the weights.
Sentinel does not implement word-piece segmentation, text normalisation or
special-token placement: a hand-rolled tokenizer disagreeing with the
checkpoint's by a single token yields embeddings that are wrong in a way no test
of ours would catch, and the benchmark would then measure our tokenizer rather
than the representation.
*Now — sequence length, and whose number each one is.* Sentinel truncates to
**256 word-piece tokens**, from the right, and records the count of truncated
inputs. Three distinct limits exist and must not be conflated:
| Limit | Whose it is |
|---|---|
| 128 | the packaged `tokenizer.json`'s own default truncation and fixed padding |
| 256 | **Sentinel's benchmark limit**, set by this decision |
| 512 | the ONNX graph's positional limit, which it enforces by failing |
**256 is Sentinel's benchmark setting and is not the checkpoint's native
maximum.** Sentinel **explicitly overrides** the tokenizer's packaged 128-token
truncation and fixed-padding defaults; loaded as shipped, that file would
silently shorten every input to 128 while this decision said 256. The effective
limit in force is asserted by test, so a tokenizer whose defaults change cannot
quietly change the benchmark.
*Now — the model asset is an external prerequisite, verified by digest.* The
weights and `tokenizer.json` live under
**`ml/artifacts/embedders/all_minilm_l6_v2/v1/`**, relocatable with the optional
**`SENTINEL_MINILM_DIR`** environment variable. Neither file is committed: the
weights are already excluded by `.gitignore`'s `ml/artifacts/**/*.onnx` rule,
and the rule excluding `tokenizer.json` is added when Task 18 is implemented.
Both carry a required expected SHA256, verified on every load:
| Asset | Bytes | Expected SHA256 |
|---|---|---|
| `model.onnx` | 90,405,214 | `6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452` |
| `tokenizer.json` | 466,247 | `be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037` |
The pair of digests **is** the model identity: weights and tokenizer are pinned
together, because either swapped alone produces silently wrong vectors. A
missing asset raises **`ModelAssetUnavailable`**; a digest mismatch raises
**`ModelAssetMismatch`**. **No digest is ever adopted from whatever file is
found**: a first-observed bootstrap would verify nothing, and is prohibited.
*Now — no test touches the network, and no test downloads a model.* This extends
§2's ingest-scoped rule to the whole project. The TF-IDF arm, the shared
configuration, every leakage assertion and every perturbation assertion run with
no assets present. MiniLM-specific tests **skip** when the assets are absent, and
the skip names the missing path rather than passing silently; setting
**`SENTINEL_REQUIRE_MINILM=1`** converts that skip into a failure, so an
environment that is supposed to hold the assets cannot go green without them. **A
benchmark run fails closed**: unlike a test, the benchmark never skips the MiniLM
arm — a missing or mismatched asset aborts the run rather than publishing a
one-armed comparison.
*Now — the benchmark's TF-IDF arm, frozen here and not inherited.* D36's
word-plus-character recipe belongs to Task 16's classifier and is not carried
over; a representation chosen for a classifier is not thereby a representation
for retrieval. The benchmark arm is a single character-n-gram block:
`analyzer="char_wb"`, `ngram_range=(3, 5)`, `lowercase=True`, `min_df=2`,
`max_df=1.0`, `max_features=2048`, `norm="l2"`, `sublinear_tf=False`,
`dtype=np.float32`, **dense output**. Every parameter not named takes the pinned
scikit-learn 1.9.0 default, and `dependency_versions` records the version that
supplied those defaults. **The vocabulary and IDF weights are fitted on
training-period text only** (§6.2, vocabulary construction). `max_features` is
load-bearing rather than cosmetic: it bounds this arm's observed
`embedding_dimension` so both arms return dense `float32` matrices of comparable
width, which is what lets the protocol's return type stay honest. No
dimensionality reduction, stemming, stop-word list or tuned parameter is added.
*Now — one frozen `BenchmarkConfig`, with its values.* The corpus is **CFPB**,
loaded through `load_corpus`; the window is the one its manifest records; the
split is §6.1's at **`DEFAULT_FRACTIONS`, 70 / 15 / 15**. The **index
population** is every record whose `submitted_at` falls at or before the
**evaluation window's end** — the `window_end` the corpus manifest recorded at
ingest. Records later than that boundary are excluded (§6.2,
evaluation-index construction), and the boundary is deliberately independent of
both the records loaded and the queries sampled: derived from the records it
would be circular, since the split partitions the very records handed in and
the newest of them is always inside it; derived from the sampled queries it
would make the candidate population depend on which 500 records were drawn and
would drop legitimate candidates from later in the same period. Both arms
search the one bounded population that results, and its `candidate_refs` are
the same for each. The **query population** is **500 test-period
records**, drawn deterministically. The **seed is 18**. **`k` is 10** and is the
headline; **recall@1, recall@5 and recall@10** are reported, the same set for
both arms. Similarity is **cosine over L2-normalised vectors**. The **tuning
budget is zero for both arms** — neither receives a hyperparameter search, which
satisfies §5.3's equal-effort rule at its only defensible point. **Both arms
receive the same `BenchmarkConfig` object**, asserted by identity rather than
equality, and **the perturbed texts are generated once from that object and
passed identically to both arms** — an arm never perturbs for itself, that being
the likeliest way two arms silently stop comparing the same thing.
*Now — the perturbations.* Three types, each reported separately, all applied to
**the same 500 records** under **the same seed**: **synonym substitution** from a
frozen in-repo substitution table, on at most **20% of eligible tokens**;
**truncation** to the **leading 60% of characters**, never below one character;
and **typo injection** by **adjacent-character transposition** at **2% of
characters**. No external or downloaded synonym corpus is used: a downloaded
lexicon would make the benchmark non-reproducible in exactly the way §5.3 exists
to prevent. **Generation is deterministic** — perturbed text is a function of
`(record text, perturbation type, seed)` alone.
*Now — recall@k, defined here and owned here.* `recall_at_k` is the **fraction of
perturbed queries whose original `RecordRef` appears among the top `k`
candidates** ranked by similarity. Queries are the perturbed records; candidates
are the index population, which contains the originals. **Identity is `RecordRef`
throughout and never a positional index.** A **duplicate `RecordRef` in the index
raises**, a silent deduplication being a silent change of denominator. A **`k`
exceeding the candidate population raises**, matching Task 14's treatment of a
`k` wider than the roster. The baseline is **a seeded random ranking**: for each
evaluation query it produces one random ranking of the candidate population,
drawn from a single seeded `numpy.random.default_rng(seed)` stream per
invocation. Exactly one ranking per query, no repeated Monte Carlo rounds and
no averaging over generated baseline rounds; the seed is recorded and the same
recall@k definition scores it. No analytic k/N stand-in is used — the same
discipline D34 fixed for the stratified baseline, and the reason D34 declined to
invent a retrieval baseline in Task 14. **The metric lives in
`ml/training/experiments/dedup.py`, and Task 14's prohibition on `recall_at_k` in
`ml/training/metrics.py` is unchanged**, because this definition is
retrieval-specific and has no meaning for a classifier.
*Now — Task 18 writes no embedder artifact, and D18's artifact clause is
deferred.* Task 18 publishes a benchmark report, not a model. **The benchmark
report is authoritative** for the embedding provenance and the observed
dimension. **D18 is clarified, not weakened: its requirement that
`embedding_dimension`, `embedding_model_id` and the ONNX SHA256 appear in
*artifact* metadata is deferred to the first serving embedder artifact, which
Phase 3 creates when it wires the winning embedder behind `DedupIndex`.** D18's
core discipline remains fully binding on Task 18: the dimension is **observed,
recorded and validated, and never assumed from a checkpoint-family stereotype**.
`ml/training/artifacts.py` is not modified, `_ARTIFACT_PRODUCED_SPECS` does not
grow, and what `feature_spec` and `build_features` mean for an embedder artifact
stays undefined — D35 deferred that question here, and this decision answers it
by deciding that Phase 2 writes no such artifact. The report is returned as a
frozen object and serialized to a caller-supplied path; nothing is written inside
the repository by default.
*Now — provenance, recorded per arm.* The benchmark report records, for each arm:
| Field | MiniLM | TF-IDF |
|---|---|---|
| `embedding_model_id` | `sentence-transformers/all-MiniLM-L6-v2` | `tfidf_char_wb_3_5_v1` |
| `embedding_model_sha256` | the ONNX digest above | `null` |
| `tokenizer_sha256` | the tokenizer digest above | `null` |
| `embedding_dimension` | observed at run time | observed at run time |
The two nulls are a decision, not an omission: a vectoriser fitted at run time
has no model file to digest, and **TF-IDF's provenance is its frozen
configuration together with the `corpus_id` it was fitted on**. `tokenizer_sha256`
is a field of the benchmark report and not of any artifact, so D35's closed
metadata schema is untouched by it.
*Now — model versions.* `model_version` is **`all_minilm_l6_v2_onnx_v1`** for the
MiniLM arm and **`tfidf_char_wb_3_5_v1`** for the TF-IDF arm. Each names what
would have to change for the number to mean something different: a differently
produced export, or a different frozen vectoriser recipe.
*Now — the `TextEmbedder` protocol.* It declares exactly
`embed(texts: Sequence[str]) -> np.ndarray`, `model_version: str`,
`embedding_dimension: int` and `embedding_model_id: str`. **numpy stays
`TYPE_CHECKING`-only in `ml/base.py`**, so the module Django imports at startup
still imports nothing heavy at run time. **The protocol has no `fit`**: each arm
is constructed already fitted — TF-IDF through `fit_tfidf(train_texts, config)`,
MiniLM through `load_minilm(...)` — so an unfitted embedder is not a state that
can exist, and there is no embed-before-fit path to define. `embed` accepts
**non-empty strings only**: an empty or whitespace-only string raises
`ValueError`, and a non-string element raises `ValueError`. Output is **dense
`float32`, one row per input, in input order, L2-normalised**, and
`embedding_dimension` is read from that output.
*Now — reproducibility, claimed exactly as far as it holds.* Given identical
**model bytes, tokenizer bytes, dependency versions, execution provider,
configuration and inputs**, the benchmark reproduces: the same embeddings, the
same rankings, the same recall figures. **No claim of bitwise identity across
arbitrary machines is made** — plan §R already declines that claim for
BLAS-backed operations, and ONNX Runtime is subject to the same reality. Those
six conditions are recorded in the benchmark report, so a discrepancy can be
attributed rather than argued about.
*Now — dimension validation.* The index records the dimension it was built with,
and every subsequent embedding is validated against it. A mismatch raises
**`EmbeddingDimensionMismatch`**. There is no broadcasting, no truncation, no
padding, no warn-and-continue and no silent comparison of vectors of different
widths (D18).
*Now — boundaries, tests and CI.* **`ml/embedders/` joins `ingest/` and
`ml/training/` in the Django-independence assertion** of
`tests/test_import_boundaries.py`; it is already forbidden to serving, and the
missing half of that guard is closed here. **Every Task 18 test is marked `ml`**,
so CI's existing ML job selects them and **no CI workflow is modified**.
**`tokenizers==0.23.2` is the only new *direct* dependency Task 18 introduces.**
It is not dependency-free: `tokenizers` declares a mandatory dependency on
`huggingface-hub`, so that package and its own mandatory closure — `filelock`,
`fsspec`, `hf-xet`, `PyYAML` and `tqdm` — are **pinned explicitly in
`requirements/ml.txt`**, as this project pins every transitive. They are
transitive installation dependencies, not Sentinel dependencies:
**Sentinel imports none of `huggingface_hub`, `fsspec`, `filelock`, `tqdm`,
`PyYAML` or `hf_xet`.** Tokenization is `tokenizers.Tokenizer.from_file` over the
pinned local `tokenizer.json`; `from_pretrained`, `hf_hub_download` and
`snapshot_download` are never called, and no hub cache or download path is
reached — which is what keeps the no-network rule above true of the
implementation and not merely of the tests. `huggingface-hub` is held at `0.36.2`
deliberately: the 1.x line replaces `requests`, which `base.txt` already carries,
with an `httpx` stack nothing else here needs. `hf-xet` carries upstream's own
platform marker rather than being installed unconditionally, and `colorama`,
which `tqdm` needs on Windows alone, stays pinned in `dev.txt` only, a Phase 1
package belonging to exactly one tier. `base.txt` is untouched, so the production
runtime budget is unaffected in Phase 2.
*Now — the implementation API surface is frozen with the RED tests.* Task 18's
tests name the production surface, so those names are part of this decision
rather than an implementation detail: `TextEmbedder`, the protocol in
`ml/base.py`; `fit_tfidf(train_texts, config)` and `load_minilm(...)`, the two
construction seams; `asset_dir()`, the MiniLM asset-location surface that honours
`SENTINEL_MINILM_DIR`; and, owned by `ml/training/experiments/dedup.py`,
`BENCHMARK_CONFIG`, `index_population`, `query_population`,
`build_perturbations`, `build_index`, `recall_at_k`, `random_ranking_baseline`
and `run_benchmark`. `BENCHMARK_CONFIG` is the single frozen `BenchmarkConfig`
instance both arms receive. **Renaming any of these requires amending this
decision and the tests together**; renaming in code alone is a silent contract
change. These names are Task 18's Phase 2 experiment surface and nothing more.
They are not a serving API: `ml/registry.py`, `Match` and `DedupIndex` are
unchanged, and Phase 3 still owns wiring the winning embedder behind
`DedupIndex`.
*Now — what the benchmark report records.* Per arm: `model_version`,
`embedding_model_id`, `embedding_model_sha256`, `tokenizer_sha256`, the observed
`embedding_dimension`, the configuration the arm was run under, and recall@k for
every reported `k` under every perturbation type, each beside its baseline. Once
for the run: the seed, the `corpus_id` of the manifest the benchmark loaded
through `load_corpus`, and — for the MiniLM arm — the count of truncated inputs.
Nothing further is added here: a report field that no clause of this decision
requires is not part of the contract, and adding one is an amendment rather than
an implementation choice.
*Now — what the truncation count counts.* For each `run_benchmark` invocation,
the MiniLM truncation count is the total number of input texts processed by that
benchmark's MiniLM arm that were right-truncated to Sentinel's 256-token limit.
It is a run-level quantity, not a cross-run persistent statistic. Reusing an
embedder in a different benchmark run must not cause the earlier run's count to
appear in the later report. The embedder's counter is per instance and starts at
zero, carrying no shared or class-level state, so a run satisfies this either by
loading its own arm or by recording the difference across the run; the report
carries that run's number and no other's.
*Now — the report names the tests read.* Five names on the report are fixed,
because Task 18's tests read them to prove the clauses above rather than to
inspect an implementation: `label`, which identifies the run as the **synthetic
duplicate-retrieval benchmark** §5.3 requires it to be labelled; `arms`, the two
representation arms, keyed `tfidf` and `minilm`; `candidate_refs`, which exposes
the shared retrieval population so both arms can be shown to have searched the
same candidates; and `queries`, which exposes the shared perturbed-query sets by
perturbation type so the single generation can be asserted by identity rather
than by equality. Nothing else about the report or the embedders is fixed here:
how an arm tokenizes, counts its truncated inputs, reaches its ONNX session or
holds its fitted vectoriser is implementation mechanics, and those accessors are
deliberately not named.
*Why:* §5.3's comparison is worth running only if the single difference between
the arms is the representation, and that is a property of values, not of prose.
Freezing every constant in one shared object — and generating the perturbations
once, outside both arms — makes divergence require editing the thing both arms
read, which is what plan §N asked for. Pinning the checkpoint together with its
tokenizer, by digest, keeps a published number traceable to exact bytes while
keeping a large binary out of a public repository and out of every ordinary test.
Naming the three sequence limits separately prevents the quiet failure this
investigation actually found: a packaged tokenizer that would have truncated at
128 while the decision said 256. Declining to write an artifact keeps D35's
closed schema and its closed spec tuple untouched by a task that has no model to
serve, and leaves the embedder-artifact question to the phase that serves one.
*Scope:* Task 18 only. D1–D37 are unaltered; D18 is clarified as to where its
fields are recorded in Phase 2 and is otherwise unchanged, its dimension
discipline remaining binding. Unchanged and untouched: `ml/registry.py`,
`ml/null.py`, `Match`, `DedupIndex` and every serving path; `ml/training/metrics.py`
and `tests/ml/training/test_metrics.py`; `ml/training/artifacts.py`; Task 16's
`triage.py` and its tests; Task 17's `risk.py`, `ingest/storage.py`,
`ingest/manifest.py`, `ingest/cli.py` and their tests. The only changes outside
Task 18's plan file list are the single `tokenizers==0.23.2` line in
`requirements/ml.txt`, the one-tuple addition in `tests/test_import_boundaries.py`,
and one `.gitignore` line excluding `tokenizer.json` under the external MiniLM
asset directory. No other dependency is added and no CI workflow changes.

**D39 — Task 19 the reduced-feature cross-domain cross-target robustness probe:
CFPB outcomes persisted in their own sidecar, one model fitted once on 311 TRAIN
and scored twice, one frozen source-training baseline for both evaluations, and
Task 19's diagnostics in a standalone report rather than in D35's metadata (§4,
§4.3, §5.4, D15, D16, D19, D20, D34, D35, D37, D38, plan Task 19).**
*Was:* §5.4 fixed what the probe is and D19 made the transfer claim a defect
rather than a matter of phrasing, but nothing said how it would run. Undecided on
entry: where the CFPB target lives, since `normalize` produced
`cfpb_timely_response` and nothing persisted it, which made the probe's
evaluation target unreachable and D37's scope clause assigned the sidecar schema
to whichever task first consumed it; the estimator and its preprocessing; the
model version and artifact identity; which baseline scores two populations with
two different targets, where D34 fixes priors to training labels; which 311
period is the in-domain evaluation and what becomes of validation; how open 311
requests behave in a probe that has no resolution times on one side; and whether
the diagnostics D35's closed schema has no field for widen it or go elsewhere.
*Now — O1, the CFPB target is persisted in a source-specific sidecar.* Exactly
three fields per record: `external_id`, `timely_response`, and
`date_sent_to_company`, which stays provenance evidence and is never a feature or
a target. `timely_response` is **never** written into `CorpusRecord.label`, which
remains the CFPB product taxonomy Task 16's roster consumes. The sidecar is
manifest-backed and integrity-checked on Task 17's terms: the manifest decides
which files exist and every listed file's bytes are verified before one outcome
is yielded. NYC 311's `load_outcomes()` keeps its name, signature,
`Iterator[NYC311Outcome]` return type and `OutcomeSidecarNotFound`, and Task 17's
outcome schema is not redesigned; a CFPB-specific loader is added instead of
making the established outcome type contract generic. The four public symbols are
`CFPB_OUTCOME_ARROW_SCHEMA`, `write_cfpb_outcome_partition`,
`read_cfpb_outcome_parts` and `load_cfpb_outcomes(years=None, root=)`, each
mirroring its NYC 311 counterpart. No writer-level source-slug guard is added:
`ingest/sources/__init__.py` keeps source names out of storage, so the writers
work in terms of schema types only.
*Now — O2, the estimator.* Task 17's exact configuration, taken as a decision
here rather than inherited, because the probe is a distinct model and D37.10's
scope is Task 17: `learning_rate=0.1`, `max_iter=100`, `max_leaf_nodes=31`,
`max_depth=None`, `min_samples_leaf=20`, `l2_regularization=0`,
`early_stopping=False`, `class_weight=None`, `random_state=17`. Every parameter
not named takes the pinned scikit-learn default and the report records the
version that supplied it. No hyperparameter search, no preprocessing, no
sampling correction, no class weighting, no threshold selection and no banding:
§5.4 requires the class imbalance to be reported, not engineered away.
*Now — O3, the artifact identity.* `model_name` `xdomain_xtarget_probe`,
`model_version` `xdomain_xtarget_probe_v1`, `experiment_label` "reduced-feature
cross-domain cross-target robustness probe", `feature_spec` the three
`transfer_features_v1` names, and `thresholds` null because the probe bands
nothing. The naming is binding so the probe cannot be confused with the primary
model at load time.
*Now — O4, one frozen source-training baseline.* A majority/prior baseline
derived from **NYC 311 TRAIN labels only**, the positive prior being the fraction
of `True` breach labels in the training population, frozen before either
evaluation and used unchanged for both. CFPB labels never fit a baseline, which
is how the tension with D34 resolves: D34 fixes priors to training labels, and
the probe's training labels are the 311 ones in both evaluations. Each evaluation
publishes PR-AUC as the headline, ROC-AUC as secondary only, minority precision,
recall and F1, the minority count and the base rate, each beside that baseline —
no figure alone. The two headline figures are never presented as a before/after
pair, an improvement, a degradation or a delta: base rates differ by roughly 27×,
PR-AUC is base-rate dependent, and each figure is interpretable only as lift over
its own baseline. That prohibition is on comparative presentation **inside the
metrics**, not on a vocabulary anywhere in the document: Task 8's provenance
evidence, which the report must copy, is built from field names such as
`median_delta_seconds` and `frac_delta_le_1min`, and those are explicitly
permitted.
*Now — O5, the populations.* The source is NYC 311 through `load_corpus`, over
Task 17's window `2024-01-01T00:00:00Z` to `2025-12-31T23:59:59Z` inclusive, cut
by `temporal_split` at `DEFAULT_FRACTIONS`. The probe applies those frozen inputs
rather than importing Task 17's module, and because the split is deterministic
given the same timestamps and fractions it reproduces Task 17's cut dates
exactly — which is what keeps the reduced-feature in-domain reference and the
primary model's in-domain figure resting on the same data. Training uses the
TRAIN period only. The validation period is produced and then left unused,
because the probe tunes nothing and selects no threshold. The in-domain
evaluation is that split's TEST period, scored once. The cross-domain evaluation
is the CFPB Task 16 test population with persisted `timely_response` outcomes —
the test period of `temporal_split` over the CFPB corpus at `DEFAULT_FRACTIONS`,
restricted to records the sidecar holds an outcome for — scored once. No record
that tuned Task 16's abstention threshold appears, because that threshold was
selected on CFPB validation. Both period identifiers are recorded in the report.
*Now — O6, open records keep their existing semantics.* An open NYC 311 request
stays in the corpus and in split construction, receives no fabricated label,
contributes no supervised training or evaluation label, and may still carry the
three aggregate-free structural features. Nothing coerces an unresolved outcome
to `False`, which is the refusal D33 built into `apply_thresholds`. No analogous
"open" case is invented for CFPB: `ingest/sources/cfpb.py` already refuses any
row whose `timely` is not exactly `Yes` or `No`, so every persisted CFPB record
carries a definite outcome.
*Now — O7, reporting.* D35's closed artifact metadata schema is **not** extended.
`artifacts.py` refuses unknown top-level keys and no Task 19 diagnostic is added
to it; `warmup_row_count` is null because the probe has no out-of-fold
construction to count. A standalone Task 19 experiment report is persisted
instead, following the convention D38 established: returned as a frozen object
and serialized to a caller-supplied path, with nothing written inside the
repository by default and no in-repository location invented. It carries the
source training population, both evaluation populations, the exact feature names
in order, the estimator and its version, the frozen baseline prior, the metrics
for each evaluation beside that baseline, `feature_distribution_shift` with all
nine quantiles per feature and both percentage-outside measures, the six framing
facts, `result_classification`, the CFPB `timestamp_diagnostic` verdict with the
delta metrics and rule thresholds that produced it, and §12's interpretation
limits. The result classification follows Task 8's CFPB verdict alone and
downgrades only: `strongly_suspicious_load_timestamp` gives
"non-informative / diagnostic", `suspicious_insufficient_evidence` gives
"substantive_with_stated_caveat", `supported_plausible_event_time` gives
"substantive", and a missing or unknown verdict is a refusal with no fallback
value. The distribution-shift block is evidence and context and can never
upgrade that classification. The expected `text_length` finding — 311 descriptor
medians against CFPB narrative medians an order of magnitude longer — is
published, never normalised away.
*Now — the two superseded Task 17 assertions.* Two `tests/ingest/test_cli.py`
tests were written to stop Task 17 from pre-empting the schema O1 now defines:
one asserted that CFPB ingest creates no outcome sidecar, the other that
`load_outcomes("cfpb")` raises the typed absence error. Task 19 has arrived, so
both describe a scope that no longer holds, and they are **inverted rather than
deleted** so the same two properties stay pinned in the same place: CFPB
ingestion declares and persists its sidecar, and NYC 311's `load_outcomes()`
never successfully deserialises a CFPB sidecar as `NYC311Outcome`. The second
property is failure and non-deserialisation, **not** a particular exception
class: no existing contract fixes one for a cross-source read, and the typed
schema rejection already in `ingest/storage.py` is sufficient fail-closed
behaviour. NYC 311 production behaviour is unchanged by both.
*Scope:* Task 19 only. D1–D38 are unaltered; D37's scope clause, which assigned a
future source's sidecar schema to the task that first consumes it, is satisfied
here rather than amended. The working contract these rulings ratify is
`docs/superpowers/specs/2026-09-24-task-19-robustness-probe-contract.md`, and the
two documents agree. Changed for Task 19: `ingest/storage.py`,
`ingest/manifest.py` and `ingest/cli.py`, only as far as the source-specific
sidecar requires; `ml/training/experiments/robustness_probe.py`, created; two
assertions in `tests/ingest/test_cli.py`, inverted as above; and one path added
to the ML job's existing path-selected command in `.github/workflows/ci.yml`, so
that `tests/ingest/test_cfpb_outcomes.py` runs in the PyArrow-capable job rather
than skipping silently in the application one. Unchanged and untouched:
`ingest/schema.py`; NYC 311's outcome schema, `load_outcomes()` and their
semantics; `ml/training/{metrics,artifacts,features,labels,thresholds,aggregates,splits}.py`;
Task 16's `triage.py`, Task 17's `risk.py` and Task 18's `dedup.py` and the
embedders, with their tests; `ml/base.py`, `ml/registry.py`, `ml/null.py` and
every serving path; `requirements/`; `pyproject.toml`. No dependency is added:
scikit-learn, numpy and pyarrow are already pinned. No migration, no `Prediction`
row, no `Complaint.embedding` value and no change to the lifecycle or to
authorization — the Phase 2 boundary holds.

**D40 — Task 20 inference-environment resource measurement: all five §S
categories retained, the index primitives extracted so a pandas- and
pyarrow-free environment can import them, peak RSS from the standard library, and
artifact absence recorded rather than invented (plan §E, §S, §U Task 20, §10,
D18, D27, D35, D38).**
*Was:* plan §S fixed the five quantities and the `ml.txt`-only environment, and
plan §U fixed the files, the inference-only guard and the acceptance rule, but
nothing said how three of the five figures could be produced. Undecided on entry,
and found by reconnaissance rather than assumed: the index primitives Task 20
must time are coupled at module-import time to the ingest/Parquet stack, so they
cannot be imported in the very environment the task mandates; peak RSS has no
available measurement API, since `psutil` is in no tier and `tracemalloc` cannot
see the ONNX runtime's native allocation; "artifact sizes on disk" has no input
in a fresh clone, because artifacts are git-ignored and produced by runs; the
vector population for the 10k and 50k figures was unspecified, and no corpus is
loadable in the mandated environment; Task 20 rested on the plan alone with no
decision-log entry, unlike every task since D36; and plan §P carried a forward
reference assigning an artifact-compatibility test to Task 20.
*Now — the five categories stand, and the coupling is extracted rather than the
task narrowed.* All five §S categories are retained: MiniLM peak RSS during
load; embedding throughput at batch sizes 1, 8 and 32; artifact sizes on disk;
index build time **and** peak memory at 10,000 **and** 50,000 vectors; and
single-query latency. Because `ml/training/experiments/dedup.py` imports
`ingest.manifest`, which imports `ingest.storage`, which imports `pyarrow`, the
index primitives cannot be imported where `pandas` and `pyarrow` are absent — and
the harness is required to refuse to run where they are present. The resolution
is a narrow, behaviour-preserving extraction: the pyarrow-free retrieval and
index primitives Task 20 needs move to `ml/training/index.py`, importable without
either package; Task 18's public benchmark behaviour is unchanged; `dedup.py`
consumes and re-exports the extracted primitives rather than duplicating them;
no index implementation is duplicated inside `measure.py`; the Task 18 benchmark
contract, metric semantics and report semantics do not change; and regression
coverage proves Task 18's behaviour is unchanged. **This is the only permitted
exception to plan §U's "must not change experiment code", and the extraction
exists solely to separate inference and index mechanics from corpus and Parquet
I/O so that the mandated clean measurement environment can measure the actual
Sentinel index rather than a copy of it.**
*Now — peak RSS from the standard library.* No `psutil` and no other new runtime
dependency: plan §E's four tiers stand, and `ml.txt` becomes a Phase 3 *runtime*
tier, so a measurement-only package must not enter it. Peak process RSS is
measured with platform-specific standard-library mechanisms — `ctypes` calling
`GetProcessMemoryInfo` on Windows, `resource.getrusage` on POSIX — and the report
identifies which backend produced each figure, because the two do not measure
quite the same thing and differ in units by platform. `tracemalloc` is not used
for these figures: it sees the Python heap, and the quantity of interest is
dominated by native allocation it cannot observe. An unsupported platform is a
refusal rather than a zero.
*Now — artifact size, and absence as a first-class result.* No prior training run
is required as hidden setup, and the harness triggers none. It receives a
caller-supplied artifact root. Each expected artifact class that exists is
measured for its actual on-disk byte size during the recorded run; each that does
not exist is recorded as absent or unavailable, with no invented size, and
absence is never reported as zero bytes. The report distinguishes measured sizes
from absent artifacts structurally. Because D38 decided that Phase 2 writes no
embedder artifact, that class is expected to be absent, and recording it as
absent is the correct outcome rather than a gap.
*Now — deterministic synthetic vectors, at the embedder's own width.* The 10k and
50k index figures use deterministic synthetic `float32` vectors. Their dimension
comes from the actual MiniLM embedder — `load_minilm().embedding_dimension`,
which `ml/embedders/minilm.py` sets from an observed forward pass — and is never
hard-coded, which keeps D18's discipline intact and keeps the literal `384` out
of the harness as it is out of `ml/embedders/`. All benchmark vectors are
generated before timing begins; index timing excludes vector generation and
embedding generation; and deterministic synthetic `RecordRef` identities are used
as the index API requires, distinct by construction because `build_index` refuses
a repeated reference. The purpose is to measure Sentinel index construction and
query cost rather than conflate it with embedding throughput. Synthetic vectors
are used because the mandated environment cannot load a corpus, and because index
cost as a function of population size is a property of the index rather than of
any corpus.
*Now — spec authority, and the report rule.* Task 20's requirements are frozen in
`docs/superpowers/specs/2026-09-25-task-20-resource-measurement-contract.md`,
which this decision ratifies, rather than resting on the plan alone. The
environment is created and run from `requirements/ml.txt` only; `pandas` and
`pyarrow` must not be importable; if either is, the harness refuses to write the
report; and Python version, CPU model, core count and the relevant library
versions are recorded with every figure. The report path is caller-supplied with
no default in-repository location, following the convention D38 established;
every numeric figure carries its environment provenance; every figure is produced
by an actual measurement run; and no hand-entered measured value is published.
*Now — the stale compatibility cross-reference is closed, not implemented twice.*
Plan §P's sentence assigning the artifact compatibility-guard test to Task 20 is
already satisfied by Task 15: `tests/ml/training/test_artifacts.py` carries the
deliberately mismatched artifact fixture, whose docstring names it as plan Task
15's acceptance, and asserts `FeatureSpecMismatch` naming the unproducible
feature before the model is unpickled. Task 20 therefore adds no duplicate
compatibility test and performs no implementation or test work for that sentence.
*Scope:* Task 20 only. D1–D39 are unaltered; D18's dimension discipline, D27's
whole-or-nothing write pattern, D35's closed artifact metadata schema and D38's
no-embedder-artifact ruling are all applied here rather than amended. Changed for
Task 20: `ml/training/measure.py` and `tests/ml/training/test_measure.py`,
created; `docs/phase-2-resource-measurements.md`, created from a recorded run;
`ml/training/index.py`, created by the extraction above; and
`ml/training/experiments/dedup.py`, which receives the import and re-export of
the extracted primitives and no other change. Unchanged and untouched:
`tests/ml/training/test_dedup_benchmark.py`; Task 16's `triage.py`, Task 17's
`risk.py` and Task 19's `robustness_probe.py` and their tests; `ml/embedders/*`
and their tests;
`ml/training/{metrics,artifacts,features,labels,thresholds,aggregates,splits}.py`;
`ml/base.py`, `ml/registry.py`, `ml/null.py` and every serving path; `ingest/*`;
`tests/test_import_boundaries.py`, which already forbids serving from importing
`ml.training` and so needs no change for a module created there; `requirements/`;
`pyproject.toml`; and `.github/workflows/ci.yml`. No dependency is added and no
CI workflow changes. Whether the index eventually belongs beside a serving path
is Phase 3's question and is not answered here.

**D40 addendum — exception identity under a reloadable module (D40.1, §7 of the
Task 20 contract).**
*Was:* D40.1 authorised the index extraction and required Task 18's behaviour to
be unchanged, but said only that `EmbeddingDimensionMismatch` stays in
`ml/embedders/minilm.py` and is imported from there. It did not say *when* the
class is resolved, and the distinction turned out to matter.
*Now:* the four extracted symbols retain their observable signatures, defaults,
validation, exception types, exception messages, ordering and numerical
behaviour. Within that, **`RetrievalIndex` may resolve
`EmbeddingDimensionMismatch` through the currently loaded `ml.embedders.minilm`
module at raise time rather than capturing the class object when
`ml/training/index.py` is imported.** This is required rather than preferred:
that module is reloadable and is reloaded by its own test suite,
`importlib.reload` rebinds the class to a new object, and a class captured at
index-module import time then differs from the one a caller reads off the module,
so `except` stops matching. The defect was real and observed — three Task 18
tests failed in a full-suite run while passing in isolation, which is how this
class of problem hides. Before the extraction the question could not arise,
because `rank` and the benchmark resolved the name from one namespace.
**This changes implementation binding, not observable exception type or message
behaviour**; the type and the message are exactly Task 18's, and a regression test
reloads the embedder module and then asserts that the class `rank` raises is the
one the module currently exposes. The purpose is **compatibility preservation,
not refactoring**, and **no other extracted body may receive an analogous change
without a separate contract decision** — the deviation from lifting the bodies
verbatim is confined to that single raise site.
*Scope:* D40.1 only. D1–D39 and the rest of D40 are unaltered. No production file
other than `ml/training/index.py` is affected, and that file is the one this
clarification describes; `ml/embedders/minilm.py`,
`ml/training/experiments/dedup.py` and Task 18's tests are untouched.

**D41 — Task 20's measurement API surface is frozen, because the RED suite needs
observable interfaces to assert on (§15 of the Task 20 contract, D40, plan §S,
plan §U Task 20).**
*Was:* D40 froze the five measurements, the clean-environment rule, the RSS
backends, the artifact-absence semantics, the synthetic vector source and the
report rule — but named no identifier. The RED suite then could not express any of
it without choosing names, so it chose them, exactly as Tasks 18 and 19's RED
phases did before their surfaces were frozen. Three of its choices were design
decisions rather than transcriptions of D40, and are ratified here.
*Now — the names are authoritative.* Constants: `BATCH_SIZES` `(1, 8, 32)`,
`INDEX_POPULATIONS` `(10_000, 50_000)`, `FORBIDDEN_PACKAGES`
`("pandas", "pyarrow")`, `SEED` `20`, `WINDOWS_RSS_BACKEND`, `POSIX_RSS_BACKEND`
and `ARTIFACT_CLASSES`. Structure: `Figure(value, unit, environment)`. Functions:
`environment()`, `peak_rss_bytes()`, `rss_backend()`,
`require_clean_environment()`, `synthetic_vectors(count, dimension)`,
`synthetic_refs(count)`, `load_embedder()`, `measure_minilm_load()`,
`measure_embedding_throughput(embedder)`, `measure_artifact_sizes(artifact_root)`,
`measure_index_build(dimension)`, `measure_query_latency(index)`, and
`run_measurements(*, artifact_root, report_path) -> ResourceReport`. Renaming one
means amending the contract and the tests together, never the code alone. No
further public API is invented.
*Now — one entry point, five internal seams.* **`run_measurements` is the
harness's entry point** and the only function an outside caller is expected to
use; both its arguments are keyword-only and neither has a default. **The five
`measure_*` functions are internal stage seams**, existing so that D40's
whole-or-nothing failure contract can be exercised — a test injects a failure at
exactly one stage and asserts no report survives — and they are **not** intended
as stable external APIs. The constants are frozen because the measurement
protocol depends on them, and are exposed as no runtime knob; a test may patch a
constant to avoid waiting for a real 50,000-vector build, which is a test
affordance rather than a supported production configuration.
*Now — a figure cannot exist unprovenanced.* `environment` is a **required**
constructor argument of `Figure`, so an unprovenanced figure is a construction
error rather than a serialization-time omission — which is what makes D40.5's
"every numeric figure carries its environment provenance" checkable rather than
aspirational. The attached environment records the CPU model, the CPU core count,
the Python version, the versions of the libraries the figure depended on, and,
for a memory figure, the RSS backend that produced it. **A figure whose
provenance is missing or incomplete does not serialize successfully**, and neither
does a non-finite value.
*Now — absence has its own shape.* An existing artifact serializes as
`{"value": <integer bytes>, "unit": "bytes"}`; an absent one as
`{"status": "absent"}`, carrying no `value` key. Absence is never zero bytes,
never a null value and never an ordinary measurement holding a sentinel number,
and the distinction **survives serialization** so that a reader of the persisted
report can tell the two apart. The converse binds equally: a genuinely empty
artifact directory measures zero bytes and is reported as a measurement, because
zero is a fact and absence is the lack of one.
*Now — `ARTIFACT_CLASSES` is declared, not discovered.* A filesystem scan cannot
report a class that is missing, and D40.3 requires exactly that, absence being the
normal result in a fresh clone where artifacts are git-ignored. A scan would also
let directory enumeration order decide the report's order, which D40's
reproducibility rule forbids. So the tuple is part of the measurement protocol,
and the report's artifact membership and order equal it whatever the filesystem
holds.
*Scope:* Task 20's API surface only. **D1–D40 are unaltered**, including D40's six
rulings and its exception-identity addendum; this decision names the interfaces
through which those rulings are observed, and changes none of them. D40's
environment rule in particular is not relaxed because the development environment
installs the training tier: the harness correctly refuses there, which is the rule
working. No production file exists yet — `ml/training/measure.py` is unwritten —
and this decision authorises no code, no dependency, no `pyproject.toml` change
and no CI change.

**D41 addendum — `minilm_assets` is an external model-asset class and resolves
through the MiniLM asset directory (§15.8 of the Task 20 contract, D40.3, D18,
D38).**
*Was:* D40.3 fixed the artifact-size semantics around a **caller-supplied artifact
root**, and D41 froze `ARTIFACT_CLASSES` as a declared tuple whose fourth member
is `minilm_assets`. Neither said where that member resolves, and the contract's own
§5 went further than it should have: reading D38's "Phase 2 writes no embedder
artifact" as meaning the class is simply expected to be absent. That conflated two
different things.
*Now:* three of the four classes are experiment artifacts and resolve under the
caller-supplied `artifact_root`, where Tasks 16, 17 and 19 write them.
**`minilm_assets` is an external model-asset class, not a run-produced experiment
artifact**, so it is **intentionally resolved through the MiniLM asset directory**
— `SENTINEL_MINILM_DIR` when configured, the embedder's packaged default otherwise
— and not through `artifact_root`. D38 remains exactly as written: Phase 2 writes
no embedder *artifact*, and D18's artifact-metadata clause still waits for Phase
3's first serving embedder artifact. That is a statement about artifacts, not about
the assets: the pinned `model.onnx` and `tokenizer.json` are real files with a real
on-disk size, and their cost is the kind of figure plan §S asks for. **The measured
size is a real on-disk byte measurement**, taken during the recorded run and
reported in the same shape as any other measured class. **If the asset directory is
unavailable the class follows the ordinary absence and failure semantics** of the
contract's §5 and §9 — recorded absent, or the run fails — and **no value is
invented**, absence never being zero. `ARTIFACT_CLASSES` is unchanged and the
assets are not moved into `artifact_root`.
*Also recorded:* the filesystem-ordering mutation the negative sweep left
uncaught stays **contract-equivalent**, and no ordering rule is invented to change
that. Two variants were tried and neither is detectable: a class directory chosen
without sorting is the same path whenever class names are unique, which they are in
a `<domain>/<model>/<version>` tree; and an unsorted byte total is equal by
construction, because the figure is a sum and addition is commutative. The
determinism that matters is already required and already proven — the report's
artifact membership and order come from the declared `ARTIFACT_CLASSES` tuple, and
a mutation that discovers classes from the filesystem instead is caught by six
tests.
*Scope:* Task 20 only. **D1–D40 are unaltered**, including D38's no-embedder-artifact
ruling, which this clarification applies rather than amends, and D40.3's
caller-supplied artifact root, which continues to govern the three experiment
classes. The rest of D41 is unchanged. No production code changes: the harness
already behaves this way, and this decision records why.

**D41 addendum — a publishable memory figure needs a fresh process (§4.1 and §8 of
the Task 20 contract, D40.2).**
*Was:* D40.2 froze the backends — `GetProcessMemoryInfo` on Windows,
`getrusage` on POSIX — and the quantity, peak process RSS. It did not say how many
runs one interpreter may contribute, because the question only appears once the
harness is actually run twice.
*Now:* **any memory figure intended for publication must be produced in a fresh
process that has performed no earlier measurement run.** Both backends report a
process high-water mark for the lifetime of the process, so a second run inside one
interpreter inherits whatever the first reached; **repeated runs in a single
interpreter are not independent memory measurements** and the later ones are not
publishable as such. This was observed rather than predicted: in the
clean-environment validation the second run reported a MiniLM *load* peak equal to
the first run's *50,000-vector index* peak, because the process had already been
there. The first run's figures were sound and the second run's memory figures were
not. Time and throughput figures do not have this property and may be repeated in
one process; the rule is about memory alone.
**Even in a fresh process, a memory figure is the process peak observed through the
measured stage, not an incremental allocation attributable to that stage alone** — a
load figure includes the interpreter, the imported libraries and the ONNX session,
and an index figure includes everything the process had already reached. The
published documentation must not imply otherwise, and `docs/phase-2-resource-measurements.md`
may publish only figures from fresh-process runs in the clean environment.
*Scope:* a measurement procedure, and nothing more. **D1–D40 are unaltered**,
including D40.2's two backends, the quantity they report and the
standard-library-only dependency rule; the existing D41 rulings are unaltered; and
the five measurement categories of D40.1 are untouched. **No production code and no
test changes**: `ml/training/measure.py` and `tests/ml/training/test_measure.py` are
byte-identical to the state this clause describes, because the clause constrains how
a run is *conducted* rather than what the harness does.

**D42 — Task 21's real-corpus metrics have never been measured, so Task 21 is
re-scoped to publish only what has been; the model evaluations, the ship-or-cut
decision and the cross-platform tolerance are deferred, and Phase 2 is not yet
complete (plan §U Task 21, §T, §R, §W, §F, I1, addendum §3.2, §5.4, D19, D35,
D38, D40, D41).**
*Was:* plan §U defines Task 21 as "Reproducibility documentation and README
metrics", with prerequisites Tasks 16, 17, 18 and 20. Per §T it publishes, in the
README's "Model performance" section, "the measured table — every figure beside
its baseline, the dedup benchmark labelled as synthetic retrieval, and any
experiment that did not clear its baseline reported as such", and it writes a
reproducibility guide covering "corpus ingestion commands with expected runtimes".
Per §R and its own entry it *measures* the cross-platform metric tolerance "on at
least two environments". Its acceptance rule is that "no unmeasured figure appears
in either document". Reconnaissance before implementation found that none of those
inputs exists, and that the repository cannot currently produce them.
*Now — the blocked portion, stated as fact.* **Task 21's real-corpus metrics are
unmeasured.** No corpus of either source exists: `data/` is absent. No Task 16,
17 or 19 artifact has ever been produced, and no `metadata.json` exists under
`ml/artifacts`. The Task 18 benchmark has never run on a real corpus, the Task 19
probe has run only on test fixtures, and no MiniLM ship-or-cut decision has been
recorded. Tasks 16 to 19 delivered tested code, not measured results. **The
repository intentionally has no concrete fetcher for either source.** Plan §F
defines a fetch *stage* and a paginated, resumable `fetch` interface, but no
approved document specifies an endpoint, a pagination scheme, a retry policy or a
rate limit for CFPB or NYC 311. `ingest/cli.py` therefore stops at an injected
`Fetcher` boundary and records that "the concrete HTTP client belongs to whichever
task specifies those things"; its command line passes no fetcher and runs from a
raw cache that nothing populates. That was a deliberate refusal to invent
specification, and D42 does not reverse it. **A second measurement environment has
not been available**: the only measured figures in the repository, Task 20's, come
from one Windows machine.
*Now — no fabricated or substituted metric.* No figure may be invented, estimated,
extrapolated or copied from another source. No metric computed on a test fixture may
be presented as model performance: the fixtures exist to test behaviour, and a
figure over a few dozen synthetic rows describes the fixture, not a model. No
synthetic corpus may stand in for a real one. No metric from a partial or
`--limit`ed corpus may be presented as a model's performance. And no resource
figure may be presented as model performance: Task 20 measures cost, not quality.
*Now — Task 20's figures remain valid evidence.* `docs/phase-2-resource-measurements.md`
and the figures in it are unaffected. They were measured in a clean `requirements/ml.txt`
environment under D40 and D41, each traces to a recorded run, and they are the only
measured figures Task 21 may publish.
*Now — what Task 21 may publish.* **`docs/phase-2-reproducibility.md`** may state,
each limited to what exists and has been verified: environment setup for the four
dependency tiers `base`, `dev`, `ml` and `train`, as the files under `requirements/`
define them; the ingestion pipeline as implemented — plan §F's three stages, the
cache-driven normalize-and-load pass, and the plain statement that no concrete
fetcher exists, so neither corpus can currently be obtained from its source, **with
no ingestion runtime stated**, because none has been measured; how each Task 16 to 19
experiment is invoked, together with the statement that each needs a corpus that
cannot currently be obtained and that none has therefore been run on real data; how
to read artifact metadata under D35's closed schema; how to rerun the Task 20
measurement, including `SENTINEL_MINILM_DIR` and the fresh-process rule of D41's
second addendum; and **the plan I1 Phase 3 prerequisite**, stated explicitly as I1
requires — the risk model's two category aggregates are derived from NYC 311 and keyed
to its complaint types, so at Phase 3 serving time they may have no source for a
complaint in another domain. **`README.md`'s "Model performance" section** may state
plainly that no model has yet been evaluated on a real corpus and why, and may refer
readers to `docs/phase-2-resource-measurements.md` for the measured inference-time
resource cost, identified as cost rather than model performance. It carries **no
figure table**: there is no model metric to tabulate, and the resource figures stay
published once, in the document that traces each one to its run, rather than being
restated where they would read as model results. The section keeps its heading.
The robustness probe keeps its full binding name wherever either document mentions
it (§5.4), and D19's prohibition applies to both documents in full.
*Now — the three doc tests are retained, and what they currently prove is
disclosed.* The test that every metric table in `README.md` has a baseline column
is retained; the README carries no metric table under D42, so the test guards the
deferred publication and fails the moment a metric table without a baseline column
is added. The D19 test is retained and fully applicable to both documents. The test
that the probe's published table carries its `result_classification` is retained
as a conditional guard: any table in either document presenting the probe's figures
must carry its classification, and since D42 publishes no probe figure, the guard
currently asserts against their premature appearance. **The first and third tests
pass without a table to inspect, and that is disclosed rather than presented as
their original purpose having been met.** The acceptance rule is unchanged: every
measured figure either document carries must be a Task 20 figure traceable to its
source run. A protocol constant or a version string is not a measurement and may
appear as what it is.
*Now — what is deferred.* These are **not** Task 21's under D42; they remain
required, and are the target of later work. First, the specification and
implementation of a concrete fetcher for each source — endpoint, pagination, retry
and rate limit — belong to a future task that specifies them, as `ingest/cli.py`
already records. Second, real ingestion of both corpora, and the ingestion runtimes
§T asks the guide to state. Third, **real-corpus evaluation of the triage, risk,
dedup and probe work of Tasks 16 to 19**, and the README table §T describes. Fourth,
the **MiniLM ship-or-cut decision** plan §W requires, which rests on the Task 18
benchmark having run on a real corpus. Fifth, the probe's published real-corpus
table with its six framing facts and its `result_classification`. Sixth, the risk
model's README verdict against its majority baseline that addendum §3.2 requires;
until it exists, neither document may claim the risk model cleared its baseline or
failed to, and no model is represented as fit to serve. Seventh, the **cross-platform
metric tolerance** of §R and Task 21, **until a second environment is actually
measured**.
*Now — the original requirements stand.* Plan §U's Task 21 entry, §T, §R, §W and I1
are not edited, weakened or deleted. **D42 records a deviation from them; it does not
rewrite them.** When the deferred work exists, the original Task 21 content becomes
deliverable, and a later decision closes this deviation rather than this one being
quietly forgotten.
*Now — Phase 2 is not yet complete.* **The original Phase 2 completion criteria of
plan §W are not fully satisfied, and this re-scope does not change that.** Unmet
because of this block: both corpora ingesting reproducibly from their sources; the
triage and risk experiments' loadable artifacts with real-corpus metrics beside their
baselines; the dedup benchmark's report of both arms with a ship-or-cut decision
recorded; and the probe's published real-corpus figures, framing facts and result
classification. Task 22's integration guarantees have not yet run. D42 makes no
finding about the criteria it does not touch. No document may describe Phase 2 as
complete while any §W criterion is unmet.
*Scope:* the Task 21 block only, and no other decision. **D1–D41 are unaltered.**
D19 applies in full; §5.4's binding name is kept; addendum §3.2's README commitment is
deferred together with the evaluation it depends on, not withdrawn; D35's schema is
what the guide describes; D38's report convention and its no-embedder-artifact ruling
stand; and D40, D41 and D41's two addenda are unaffected, their figures being the
only ones Task 21 publishes. D42 adds no fetcher, specifies no endpoint, downloads
nothing and creates no synthetic substitute. It changes no code, no test, no
requirement and no CI, and `README.md` and `docs/phase-2-reproducibility.md` remain
unchanged until Task 21 is implemented under it.

**D43 — The 2024–2025 CFPB corpus is a Sentinel reconstruction joining the official
CFPB Narratives Archive to the official CFPB Consumer Complaint Database API by exact
Complaint ID, and any disagreement between the two refuses the ingestion run; NYC 311
is read from the official NYC Open Data dataset; the concrete fetch builds against the
implemented injected `Fetcher` boundary (plan §F, §G, §X, Tasks 5, 6, 8; addendum §0,
§1, §2.3, §2.4, §2.5; D21, D37.2, D42).**
*Was:* D42 recorded that no approved document specified an endpoint, a pagination
scheme, a retry policy or a rate limit for either source, and deferred the concrete
fetcher to a future task that specifies them. The design assumed CFPB records came
from the live Consumer Complaint Database search API: `ingest/sources/cfpb.py` names
it (`SOURCE_API_VERSION = "cfpb-ccdb-v1"`, "Elasticsearch-shaped response"), and
§0's CFPB measurements were taken from it. Plan §F described a per-adapter fetch
abstraction (`SourceAdapter.fetch`, `PageCursor`, `FetchPage`, `dataset_adapter`)
that the code never implemented; the implemented boundary is the injected `Fetcher`
in `ingest/cli.py`.
*Now — why the CFPB source changes.* These are external facts, each read from an
official source. The CFPB ceased publishing complaint narratives: its announcement is
dated Aug 14, 2026 and states no effective date
(https://www.consumerfinance.gov/about-us/newsroom/the-cfpb-to-cease-discretionary-publication-of-complaint-narratives-and-visualizations/).
Release 24 of the Consumer Complaint Database (September 2026) records that
"Consumers' complaint narratives and complaint data visualizations have been removed
from the database" (https://cfpb.github.io/api/ccdb/release-notes.html). Previously
published narratives were placed in the CFPB's FOIA Reading Room as the CFPB Consumer
Complaint Database Narratives Archive, described as an "Archive of consumer
complaints received December 1, 2011 through August 14, 2026 that were previously
published in the Consumer Complaint Database", published as 21 ZIP exports with no
record counts, checksums or data dictionary
(https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/).
The live API still serves complaint metadata, and no narratives.
*Now — what one archive export shows, and what it does not.* Export #8,
`CCDB_Export_8_November_2024_through_December_2024.zip` (52,563,546 bytes, SHA256
`24b3326a954c0edeff33318b157d12eef5a59547f6f274f9236f1ceaba988a74`), was inspected
directly. It holds one CSV of 591,058 rows with 591,058 distinct Complaint IDs, under
16 display-named columns (`Date received`, `Product`, `Consumer complaint narrative`,
`Date sent to company`, `Timely response?`, `Complaint ID` among them) rather than the
adapter's `hits.hits[]._source` row shape. Its `Date received` spans 2024-11-01 …
2024-12-31, matching the file's name. In every row `Date received` and
`Date sent to company` are bare `YYYY-MM-DD` dates, with no time of day and no UTC
offset, and 73.8% of rows carry no narrative. **These are facts about export #8
alone.** It is one compatibility sample: the other 20 exports, including every export
covering 2025, have not been inspected, and nothing here asserts that they share its
format or coverage. Nor does the archive's stated date range, on its own, establish
that a corpus built from it is complete.
*Now — the archive alone is incompatible.* Task 5 requires an ISO-8601 timestamp with
an offset, parsed to aware UTC, and the existing normalizer refuses a date-only value
as `NaiveTimestamp`. No conversion can satisfy that contract without inventing a time
of day or an offset. A day-resolution contract is rejected rather than adopted: with
97.97% of export #8's received and sent dates falling on the same day, §2.3's
field-delta rule would return `strongly_suspicious_load_timestamp` by construction,
and the reduced-feature cross-domain cross-target robustness probe's CFPB
`submitted_hour` would lose its meaning.
*Now — the source model.* Sentinel's CFPB corpus for 2024-01-01 … 2025-12-31 is a
**Sentinel reconstruction**, not an official CFPB combined dataset. It joins two
source views of the same complaints — the Narratives Archive and the Consumer
Complaint Database API — by exact Complaint ID, which the official field reference
defines as "The unique identification number for a complaint"
(https://cfpb.github.io/api/ccdb/fields.html). The API is the official
`https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/`,
with `GET /{complaintId}` documented as "Find consumer complaint by ID"
(https://raw.githubusercontent.com/cfpb/ccdb5-api/main/swagger-config.yaml). The
reconstruction is recorded here as a project decision on a documented shared key, as
D21 records an interpretation, so that it can be disagreed with explicitly.
*Now — per-field authority.*

| Field | Authority | Rule |
|---|---|---|
| Complaint ID | both views | The join key, compared as exact strings. The API returns it as a string although its specification types it as an integer. |
| narrative | the archive only | The API no longer publishes narratives (Release 24). |
| `date_received` | the API only | Documented as "date & time". The archive's date-only value is a cross-check and never a substitute. |
| `date_sent_to_company` | the API only | As `date_received`. |
| `product` | both views | They must agree exactly, or the run is refused. |
| `timely` | both views | They must agree exactly, or the run is refused. |

*Now — join semantics.* A pair joins only on exact Complaint ID string equality. The
reconstructed row must satisfy the existing CFPB normalization contract unchanged: one
row carrying an API record's metadata and the archive's narrative field was accepted
by the unmodified normalizer, giving `submitted_at 2024-11-01T05:55:55+00:00`, while
the same row with the archive's date-only value was refused with `NaiveTimestamp`. How
the API is read — by Complaint ID, or by a date-window search followed by a local join
— and where the join is performed are the fetch task's to specify. The API's own
limits bound that choice: `size` at most 100, `frm` at most 100,000, deep paging
through `search_after`, and `date_received_min` inclusive with `date_received_max`
exclusive at date granularity.
*Now — refusals.* Each of the following refuses the ingestion run, before any corpus
write, as every refusal in §2.5 does:

| Case | Behaviour |
|---|---|
| `product` differs between the two views | Refuse the run. Neither view is silently preferred. |
| `timely` differs between the two views | Refuse the run. It is a model outcome field and is never reconciled. |
| An archive record included in the reconstruction has no valid API record: none exists, the retrieval fails permanently, or the returned `complaint_id` differs from the requested one | Refuse the run. What counts as a permanent failure is set by the fetch task's retry contract. |
| An API record lacks a required field or carries an unparseable one | Unchanged: the normalizer raises (`MissingField`, `NaiveTimestamp`) and §2.5 refuses the run. No exclude-and-report behaviour is introduced. |
| The archive's `Date received` differs from the UTC calendar date of the API's `date_received` | Refuse the run. |

The calendar-date comparison uses the API timestamp's UTC date because that is the
frame in which all 37 compared records agreed, including at least five where the New
York date would have differed. The archive's own date frame is not documented, so a
disagreement refuses rather than being reinterpreted. In every case the API timestamp
remains the only timestamp source: the archive's date is a cross-check and is never
substituted. Which archive records the reconstruction includes is not decided here
(see the narrative-filter decision below); these refusals apply to whichever records
that decision includes.
*Now — the compatibility sample, and what it is not.* Forty Complaint IDs were drawn
deterministically from export #8 — every n-th of its sorted narrative-bearing IDs
(30) and of its narrative-free IDs (10) — and requested one at a time, half a second
apart, from `GET /{complaintId}`. Thirty-seven returned HTTP 200 and three returned
HTTP 429. Among the 37, the Complaint ID, `product`, `timely` and the calendar date of
`date_received` agreed in 37 of 37, and every `date_received` and
`date_sent_to_company` had the form `YYYY-MM-DDTHH:MM:SS.000Z`, with varying times and
an explicit UTC offset. **This is a source-compatibility sample, not a completeness
result.** It covers November–December 2024 only, says nothing about 2025, and supports
no claim about how many archive records the API serves.
*Now — temporal consistency.* The sampled API records carry
`date_indexed = 2026-09-14T20:06:00Z`, so the API index was rebuilt then; the archive
export's member is stamped 2026-09-13 and its server reports `Last-Modified` of
Sep 14, 2026. The API's metadata is a current publication view. No official source
states that it never changes, and nothing here assumes it does. A reconstructed corpus
is therefore reproducible only if **both sides are retained and pinned**: the archive
ZIPs by digest, and the raw API responses as retrieved. Querying the API again later
may return different data.
*Now — provenance.* Every reconstructed CFPB corpus must record, at minimum: for each
archive ZIP, its URL, byte size, SHA256 and `Last-Modified` when served; the API base
URL, the exact endpoint and every query parameter; the retrieval time of each API
response; the API index name and `date_indexed` where a response carries them; the
join key, stated as exact Complaint ID string equality; the matched, missing and
mismatched counts, each broken down by kind; and a digest of every cached API
response, which the content-addressed raw cache already provides. The current
manifest (`manifest_version` 2) cannot hold all of this: its only field for source
identity is `source_api_version`, and it has none for archive digests, API query
parameters, retrieval times, index metadata or join counts. **The fetch task must
choose between a manifest-versioned provenance extension — which, per plan §G,
increments `manifest_version` and states how version-2 manifests are read — and a
separate provenance record.** D43 adds no schema field and makes neither choice.
Plan §X's "regenerable … from the source APIs given the recorded window" no longer
holds for CFPB narratives; the original reproducibility requirement stands, and only
the way it is met changes.
*Now — rate limiting and retry.* Three of forty requests sent half a second apart
returned HTTP 429. Separately, one client configuration was refused with HTTP 403 on
every request while standard curl requests succeeded; that behaviour is not
documented. The official material reviewed — the API specification, the field
reference, the release notes and the data-use page — states no numeric rate limit.
D43 therefore freezes no rate, interval, concurrency or backoff: retry, backoff and
pacing are part of the fetch task's implementation contract.
*Now — NYC 311 source.* The dataset is "311 Service Requests from 2020 to Present",
`erm2-nwe9`, official and published; the records for 2010–2019 were split into
`76ig-c548`
(https://www.nyc.gov/opendata/news/all-news/311-Service-Requests-Updates), and
D37.2's window lies wholly inside `erm2-nwe9`. Its SODA2 endpoint,
`https://data.cityofnewyork.us/resource/erm2-nwe9.json`, returns a bare JSON array,
with `unique_key` as text and `created_date` as a `floating_timestamp` such as
`"2020-09-25T15:43:34.000"`, which is the shape the existing adapter reads. Socrata's
documentation (https://dev.socrata.com/docs/queries/limit.html,
https://dev.socrata.com/docs/queries/order.html,
https://dev.socrata.com/docs/app-tokens.html,
https://dev.socrata.com/consumers/getting-started.html,
https://dev.socrata.com/docs/response-codes.html,
https://dev.socrata.com/docs/datatypes/floating_timestamp.html) states that `$limit`
defaults to 1,000 with a maximum of 50,000 on SODA 2.0 and none on 2.1 or 3.0; that
results are not implicitly ordered, so paging requires `$order`, "at a minimum
`$order=:id`"; that app tokens are optional for SODA2 and sent as `X-App-Token`, with
unauthenticated requests drawn from a throttled pool and throttling answered with HTTP
429; and that SODA3 (https://dev.socrata.com/docs/queries/) requires authentication
or an app token. Two official Socrata pages conflict about throttling with a token
("do not throttle … unless abusive" against "up to 1000 requests per rolling hour"),
and D43 resolves neither. The dataset "is updated daily and expected values for many
fields will change over time"; no snapshot mechanism is documented, and whether
`between` includes its endpoints is not documented.
*Now — fetch abstraction.* The fetch task builds against
`Fetcher = Callable[[str, date, date], Iterable[SourcePage]]` in `ingest/cli.py`,
passed into `ingest(..., fetcher=…)` and cached by `fetch_into_cache`. Plan §F's
`SourceAdapter.fetch`, `PageCursor`, `FetchPage` and `dataset_adapter` are superseded
for Phase 2 and are not revived. For `cfpb`, one `Fetcher` call covers both
acquisitions — the archive and the API — and yields pages in the adapter's existing
input shape; how and where the join is done is the fetch task's.
*Now — decisions deferred to the fetch task.* D43 decides none of these, and none may
be settled implicitly:
1. **The CFPB narrative filter.** Export #8 has no narrative in 73.8% of its rows, and
   the existing normalizer refuses such a row with `MissingNarrative`, which §2.5
   makes refuse the whole run. The fetch task must decide explicitly whether the
   reconstruction's archive side is restricted to narrative-bearing rows, and how that
   decision interacts with §2.5.
2. **Where provenance lives**: a manifest-versioned extension or a separate record.
3. **The reconstruction's source identity**, including its final `source_api_version`,
   whose constant lives in `ingest/sources/cfpb.py`, which Task 6 lists as
   must-not-change and so needs explicit authorization.
4. **Retry, backoff and pacing** for both sources, including what counts as a
   permanently failed retrieval.
5. **NYC 311 page size, ordering key and app-token use.**

*Now — ingestion conflicts that remain unresolved.* Besides the narrative filter, four
conflicts identified before D43 stay open, each needing its own decision before any
real ingest it affects: NYC 311's DST edge rows, which §2.4 calls a "negligible loss"
while §2.5 and the code refuse the run, inside a D37.2 window that contains two autumn
folds and two spring gaps; raw-cache pages written non-atomically and trusted by name
rather than by content; duplicate `external_id`s after a source changes, which nothing
in `ingest/` checks; and the memory held by normalizing a whole window at once.
*Now — what D43 does not authorize.* No invented time of day, offset or timezone for
any CFPB record, and no day-resolution CFPB timestamp contract; the archive's dates
never substitute for API timestamps. No silent reconciliation of any mismatch. No
synthetic, partial or `--limit`ed corpus standing in for a real one, and no download
of either full corpus. No fetcher, and no network access in tests or CI. No change to
the adapters, to normalization, to the adapter protocol, to the manifest schema or to
the `source_api_version` constant. No retry interval, rate limit, page size or
concurrency for either source. No treating the compatibility sample as a completeness
result, and no claim that the API's metadata never changes.
*Now — D42 is preserved.* D43 narrows D42's first deferred item — the concrete fetcher
— by naming the official sources and their endpoints; pagination choices, retry and
rate limiting remain the fetch task's, so that item is not closed. No figure may be
fabricated or substituted. D42's deferred items stay open until the deferred work
actually lands, and a later decision closes the deviation. **Phase 2 is not complete,
and no document may describe it as complete.**
*Scope:* the CFPB source model and its refusal policy, NYC 311's source, the
provenance minimum and the fetch boundary. **D1–D42 are unaltered.** D43 changes no
code, no test, no requirement, no CI and no other document.

**D44 — The concrete fetch contract: each acquisition is scoped to its own directory,
verified in UTC-day slices for CFPB and in New York civil-day slices for NYC 311, and
described by an immutable acquisition record that manifest version 3 cites by digest;
the fetchers build behind the implemented injected `Fetcher`, and plan Tasks 23–25 are
added to implement them, none of which authorizes real ingestion (plan §F, §G, §Q, §U,
§V, Tasks 6, 8, 22; addendum §1, §2.5, §2.7; D1, D24, D27, D37.2, D42, D43).**
*Was:* D43 left five choices to the fetch task — the CFPB narrative filter, where
provenance lives, the reconstruction's source identity including `source_api_version`,
retry, backoff and pacing, and NYC 311's page size, ordering key and app-token use —
and said none may be settled implicitly. It left four ingestion conflicts open, and it
authorized no change to the adapters, normalization, the adapter protocol, the
manifest schema or the `source_api_version` constant. Plan §U ends at Task 22, so the
fetch work had no task. The facts below were measured on 2026-09-28 by the fetch-task
contract reconnaissance, from aggregate counts, response headers, the archive's list
of exports, one day of CFPB metadata and the export #8 file D43 inspected; no corpus
was downloaded.
*Now — the API behaviour this entry rests on.* The CFPB API documents
`date_received_max` as "date < date_received_max", which D43 quotes; in use it
includes its own UTC calendar day. `date_received_min=date_received_max=2024-11-01`
returned 9,417 records timestamped 2024-11-01T00:00:09Z … 2024-11-01T23:59:55Z, and
`date_received_min=2024-11-01` with `date_received_max=2024-11-02` returned 14,810:
those 9,417 and the 5,393 of UTC 2024-11-02. D43 is unaltered, since its sentence
records the documentation; the fetcher relies on neither reading, querying
`date_received_min=date_received_max=D` and checking the timestamps returned. The
documented `format=csv` export, with which "frm/size will be ignored", returned that
whole 14,810-record UTC day in one response: 14,810 rows with 14,810 distinct
Complaint IDs, equal to the same day's JSON `hits.total`, each `Date received` of the
form `YYYY-MM-DDTHH:MM:SS.000Z`, in 15 columns with no narrative. Across 104 records
compared with the JSON response every compared field was equal, except that the CSV
spells a JSON null as the literal string `None` (97 `tags`, 34
`company_public_response` and 1 `sub_issue` cell among 100 compared records). For
inclusive UTC days the API holds 2,734,269 complaints in 2024 and 5,442,963 in 2025,
8,177,232 in the window; 2024-11-01 … 2024-12-31 holds 591,058, which is export #8's
row count — an equality of counts, not a proven equality of ID sets. Its
`_meta.last_indexed` was 2026-09-27T12:00:00-05:00, so the API is still being updated
after the archive's snapshot.
*Now — the archive and NYC 311 as measured.* The Narratives Archive lists 21 ZIP
exports. The window falls in export #5 (September 2023 through March 2024, which
begins before the window) through export #14 (November through December 2025),
736,201,829 bytes by `Content-Length`; export #15 is 36,832,036 bytes. Their ETags
carry a multipart suffix and are not content digests. NYC 311's `erm2-nwe9` holds
3,456,769 records in 2024 and 3,655,041 in 2025 by `created_date` inside D37.2's
window, its largest single day holds 16,388 (2025-12-15), and its `rowsUpdatedAt`
shows that it is updated daily.
*Now — (1) the CFPB narrative filter.* The archive side of the reconstruction is
restricted to rows whose narrative is a string that is not blank after `strip()` — the
adapter's own `MissingNarrative` test — applied by the CFPB fetcher before the
archive/API join. Every excluded row is counted in the acquisition record; none is
dropped silently. The CFPB adapter is not altered: a blank narrative that reaches
normalization still raises `MissingNarrative`, and §2.5 still refuses the run. D43's
refusals apply to the rows this filter includes. This is the population §1 and D1
already describe — the window "holds 2,036,434 narratives" — and plan Task 5's "so the
caller decides" names the caller that decides, which is the fetcher. Without a filter
every real run is refused, since 73.8% of export #8's rows carry no narrative;
skipping in the adapter would change normalization, which Task 8 protects, and would
hide the drop.
*Now — (2) provenance: an immutable acquisition record, cited by manifest version 3.*
Provenance lives in a separate acquisition record, one per acquisition, written last
through a same-directory temporary file and `os.replace`, and never modified once
written. `acquisition_id` is the SHA256 of the record's bytes as written. The corpus
manifest cites it: `manifest_version` 3 adds exactly one field,
`acquisition_id: str | None`. A version 1 or version 2 manifest stays readable, a
missing `acquisition_id` being read as `None` — the read-only default D37.16 gave
`outcome_part_files` — and every new write emits version 3. `schema_version`, the
storage path and the definition of `corpus_id` are unchanged, as plan §G requires of a
manifest-only change. Recording the whole provenance in the manifest was rejected,
because provenance is produced once, at fetch time, and is large, while the manifest
is rewritten by every normalize pass; a separate record with no link was rejected,
because an artifact cites only a `corpus_id` and could then never reach its
provenance. The record holds at least:

| Part | Recorded |
|---|---|
| Every acquisition | the source; the window as supplied and as resolved; start and completion times; the client identity (User-Agent, library and version); the retry and pacing values in force; the journal of completed slices, each with its request URL and every query parameter, its retrieval time, the digest and size of the raw response, the page digests it produced and its verification counts; the Sentinel commit |
| CFPB identity and archive | `acquisition_kind`; the reading-room page's URL, SHA256 and retrieval time; for each export its URL, byte size, SHA256, `Last-Modified`, the ETag as served (recorded, never used as a digest), member name and size, header, row count and observed `Date received` range |
| CFPB API | the base URL and endpoint; for each UTC day the JSON count's `hits.total`, `_meta.last_indexed` and the hit's `_index`, and the digest of the raw CSV body, which is retained gzipped |
| CFPB join | the join key, stated as exact Complaint ID string equality; the narrative-filter predicate and its excluded count; the matched, archive-only, excluded and API-only populations; mismatched and missing records by kind |
| NYC 311 | the dataset id, endpoint and SODA version; for each slice its `count(*)`, row count and page digest; `rowsUpdatedAt` and the window's `count(*)` at the start and at the end |

*Now — (3) source identity.* The reconstruction's acquisition identity is
`cfpb-archive-api-reconstruction-v1`, recorded as the acquisition record's
`acquisition_kind`. `SOURCE_API_VERSION = "cfpb-ccdb-v1"` in `ingest/sources/cfpb.py`
is kept unchanged: it versions the page shape the adapter reads
(`hits.hits[]._source`), and the reconstruction yields that shape. Task 6's and Task
8's protections of that file therefore stand, and D43's statement that no change to
the constant is authorized is not overridden.
*Now — (4) CFPB API retrieval.* For each UTC day D of the window, and one margin day
at each end, the fetcher makes one CSV export request (`format=csv`,
`date_received_min=date_received_max=D`) and one same-day JSON count request
(`size=1`). `GET /{complaintId}` is used only for a record whose CSV cell in an
adapter-relevant column — `Complaint ID`, `Product`, `Timely response?`,
`Date received` or `Date sent to company` — is the literal `None`, which is ambiguous
between a null and a value; the JSON value is then used, and no cell is ever mapped
silently. **The CSV export is treated as the API surface for this reconstruction —
"the API" of D43's per-field authority. That is an interpretation, recorded so it can
be disagreed with:** it is the same documented endpoint with a format parameter, and
its values equalled the JSON response's on every compared field apart from the
spelling of null. The choice rests on these constraints:

| Strategy | Requests for the window | Evidence | Cost |
|---|---|---|---|
| By Complaint ID, narrative rows only | at least 2,036,434 (§0, measured before narratives were removed) | three of forty requests half a second apart returned HTTP 429 (D43) | at least 11.8 days at that spacing and 23.6 days at one second; no view of API records the archive lacks |
| JSON search, `size=100` with `search_after` | about 81,773 | documented: `size` at most 100, `frm` at most 100,000; observed: chaining works, with a strictly increasing (epoch milliseconds, Complaint ID) key and no overlap, but one chained page took 11.7 s | about 2.7 to 11 days |
| **CSV export, one request per UTC day** | about 733, and as many count requests | documented to ignore `frm` and `size`; observed: a 14,810-record day returned complete in about 4 s | about two hours (an estimate) |

*Now — (5) CFPB completeness.* A CFPB acquisition is complete only when all of the
following hold, and its record is written only then. **Archive:** every export used —
exports #5 through #14, and export #15 solely to classify records at the window's end
boundary — is pinned by SHA256; its header equals the 16-column header verified in
export #8, or the run is refused; the verified content, not the export's name, covers
every window day; and no Complaint ID appears in two exports, or the run is refused.
**API:** for every day D, margin days included, the CSV's row count equals the JSON
`hits.total`, its Complaint IDs are distinct, and every row's `date_received` falls on
UTC day D. **Join:** every archive row the narrative filter includes joins exactly one
API row with an equal `product`, an equal `timely` and an equal date, or the run is
refused under D43; D43's case of a returned ID that differs arises only from a
`GET /{complaintId}` request. **Populations recorded:** matched; archive-only;
excluded by the narrative filter; and API-only, each API-only record being looked up
in every loaded export so that a date mismatch at a window edge is refused rather than
mislabelled. API-only records are recorded, not refused: they carry no obtainable
narrative and cannot enter the population. For export #8's range the net difference is
zero, 591,058 on each side; Task 25's reconnaissance measures the API-only population
of every other export used and reports it before the implementation is frozen, and
strict refusal remains the alternative a later decision may choose. **Completeness is
scoped to the retained archive population:** a completed acquisition claims that every
included archive record was joined and verified, never that CFPB's historical
narrative population is complete. §0's narrative counts, measured before narratives
were removed from the API, are recorded beside the result as a comparison and are not
a completeness gate. D43's compatibility sample is not a completeness result.
*Now — (6) failure, retry and pacing.* These values are **engineering policy, not
source requirements**: neither CFPB nor Socrata documents a numeric rate limit for
these endpoints, and Socrata's own pages conflict (D43).

| Aspect | Policy |
|---|---|
| Transient | a connection error or timeout; HTTP 429, 500, 502, 503 and 504; an unparseable or short body, such as a CSV whose row count differs from `hits.total` |
| Permanent | HTTP 400, 401, 403, 410 and any other 4xx except 429; a `GET /{complaintId}` 404 is a missing record, which refuses the run (D43) |
| Attempts | at most 6 per request |
| Backoff | 2, 4, 8, 16 and 32 seconds, with no jitter |
| `Retry-After` | when present, the wait is the larger of the backoff and the header, capped at 300 seconds |
| Pacing | at least 1 second between request starts, per host |
| HTTP 403 | never retried: the run stops immediately, and the client identity is never changed to get past it |
| Exhaustion | a typed `FetchFailed` naming the request and its last status |

An acquisition that ends in `FetchFailed` stays incomplete: no acquisition record is
written, so it never becomes a completed acquisition; `ingest` refuses before any
corpus write (§2.5, "a page cannot be fetched"); and a rerun resumes at the first
slice its journal does not hold. The client identifies itself honestly by a User-Agent
naming Sentinel, and never impersonates a browser or another client.
*Now — (7) NYC 311 page size.* SODA 2.0 with `$limit=50000`, its documented maximum,
and one request per New York civil day D: `$where` of
`created_date >= 'DT00:00:00' AND created_date < 'D+1T00:00:00'`, written with
explicit comparisons because whether `between` includes its endpoints is not
documented (D43), and
`$select=unique_key,created_date,closed_date,complaint_type,descriptor`, exactly the
adapter's fields. Literal floating-timestamp strings make every daylight-saving day
come out right with no timezone conversion. A slice holding 50,000 rows or more
refuses the run; it is never truncated. The largest measured day, 16,388, leaves about
three times that headroom. `SOURCE_API_VERSION = "socrata-soda2"` already names the
SODA 2.0 bare-array shape, and SODA 3 requires a token.
*Now — (8) NYC 311 ordering.* `$order=unique_key` within each slice: the source's own
identifier, which makes a page's bytes deterministic for unchanged data. Socrata's
documented minimum, `$order=:id`, exists for paging, and no slice is paged by offset;
a slice at the limit refuses under (7), and any future split would be by time, never
by offset.
*Now — (9) NYC 311 authentication.* No app token. The fetcher never reads, sends or
logs one. About 1,500 requests at one-second spacing do not need one, the two official
statements about throttling with a token conflict and D43 resolved neither, and an
unauthenticated client needs no secret in a public repository. An HTTP 429 goes
through (6); throttling that persists would be a new decision.
*Now — (10) NYC 311 completeness.* An NYC 311 acquisition is complete only when, for
every slice, `count(*)` equals the number of rows returned and the number of distinct
`unique_key`s, and is below 50,000; the slice totals reconcile to the window's
`count(*)` taken at the end of the run; and no `unique_key` appears in more than one
slice. If `rowsUpdatedAt` or the window count changes during the acquisition, every
slice's count is verified again and each changed slice is fetched again, once; an
inconsistency that remains refuses the run.
*Now — an acquisition-scoped raw cache.* Today every run's pages accumulate in one
cache per source that every normalize pass reads whole, and a real fetcher resumed
after an interruption would download every page again, skipping only the cache writes.
Each acquisition now has its own directory under `data/acquisitions/<source>/`, which
is the `raw_root` its pages are cached in; pages stay content-addressed. A journal of
completed slices lets an interrupted acquisition resume at the first slice it does not
hold, without repeating a request. The acquisition record is written last, and a
completed acquisition never changes. Normalization reads only the pages a completed
acquisition record lists, verifying each page's digest; an unlisted, missing or
corrupt page, or an acquisition without a completed record, refuses the run before any
corpus write.
*Now — atomic, checked cache writes, and duplicate identity refused.* D43 left two
conflicts open that the fetch path must touch, and D44 resolves both. **Partially
written raw-cache pages.** Today `cache_page` gzips straight to its final
content-addressed name and skips any name that exists, so a write killed part-way
leaves a truncated page that is trusted by name from then on and breaks every later
read: `ingest/cli.py`'s statement that an interrupted run leaves "whole pages behind,
never half of one" does not hold for such a write, and plan §F's "verified by
checksum, not existence" is not what the code does. A page is now written to a
temporary file in the same directory and moved into place with `os.replace`, and a
page whose name already exists is digested again and replaced when its content does
not match its name. **Duplicate `external_id`s.** Nothing checks for them today. They
can come from pages accumulated across acquisitions of a source updated daily, from
duplicates inside a source, and from overlapping CFPB exports. Acquisition-scoped raw
roots remove the first; the fetchers refuse any duplicate identifier within an
acquisition; and a normalized window holding two records with the same
`(source, external_id)` refuses the run before any corpus write, a refusal D44 adds to
those §2.5 lists. Nothing is ever deduplicated silently.
*Now — the two conflicts D44 does not resolve.* NYC 311's daylight-saving edge rows
are not the fetch path's to change: the fetchers pass them through untouched, and D45
records them together with the other NYC 311 refusals measured beside them.
Full-window normalization memory is unchanged: `ingest()` holds the whole window
because §1.1 and D24 require the roster to be asserted over it before anything is
written. The reconnaissance measured 1,661 bytes per held CFPB record-and-outcome pair
with a 1,202-character narrative and 472 bytes per NYC 311 pair, both undercounts, and
estimated a peak of about 4 to 6 GB per source. That peak is to be measured on the
first real run, and a two-pass redesign would be its own decision, taken only if the
measured peak does not fit.
*Now — the fetch boundary.* The fetchers are built with an injected transport, clock,
sleeper and acquisition directory, and passed to `ingest(..., fetcher=…)` as the
unchanged `Fetcher = Callable[[str, date, date], Iterable[SourcePage]]`. Plan §F's
`SourceAdapter.fetch`, `PageCursor`, `FetchPage` and `dataset_adapter` stay superseded
and are not revived. A new `--fetch` flag is opt-in; without it the command line stays
cache-only and network-free, which keeps Task 8's "Normalize and load are re-run from
the raw cache without network access" and every existing test. The fetch code lives in
a new `ingest/fetch/` package that imports only the standard library and `requests`,
already pinned in `requirements/base.txt`; it imports no pyarrow, no scipy, not
`ingest.cli`, and no Django. Tests never touch the network: every new fetch test
module blocks sockets, no fetcher can be built without an explicit transport, and the
real transport is built only in the command line's `main()`. Nothing under `data/`,
which is gitignored, is ever committed.
*Now — what D44 authorizes, and what it does not.*

| Surface | Under D44 |
|---|---|
| `ingest/cli.py` | **Modified:** an atomic, checked `cache_page`; verification of the acquisition record before normalization; the duplicate `external_id` refusal; the opt-in `--fetch`; `acquisition_id` passed to the manifest; its docstring's account of fetching |
| `ingest/manifest.py` | **Modified:** version 3, `acquisition_id`, and reading versions 1 and 2 |
| `tests/ingest/test_manifest.py` | **Modified:** the seven assertions that pin version 2 are updated, tests reading versions 1 and 2 are added, and nothing is weakened |
| `tests/ingest/test_cli.py` | Additions only |
| `ingest/fetch/`, `tests/ingest/fetch/`, `tests/fixtures/fetch/` | **Created**; fixtures are tiny and synthetic |
| `ingest/sources/cfpb.py`, `ingest/sources/nyc311.py`, `ingest/sources/base.py` | **Unchanged** |
| Normalization logic | **Unchanged** |
| `ingest/storage.py`, `ingest/schema.py`, `ingest/roster.py`, `ingest/identity.py` | **Unchanged** |
| `requirements/*`, `.github/workflows/ci.yml` | **Unchanged:** pure fetch tests run in the application job, and end-to-end tests live in `test_cli.py` and `test_manifest.py`, which the ML job already runs |
| `tests/test_phase2_integration.py`, the root `conftest.py` | **Unchanged**; a suite-wide network guard is not authorized here |
| `README.md`, `docs/phase-2-reproducibility.md`, Task 20's evidence | **Unchanged** until real ingestion |

*Now — real data.* **Tasks 23, 24 and 25 do not authorize full-window ingestion of
either source.** No corpus was downloaded to write this entry, and no README or
model-metric change follows from it. Downloading the archive exports that Task 25's
reconnaissance needs, about 0.77 GB, requires its own authorization. Real ingestion
happens only after D45 is decided and after a separate authorization of the hardware
and of the run; it remains D42's second deferred item. **Phase 2 is not complete, and
no document may describe it as complete.**
*Now — D42 and D43 are preserved.* D44 completes the specification of D42's first
deferred item — endpoint, pagination, retry and rate limit — but not its
implementation, so that item stays open until Tasks 23 to 25 land, and a later
decision closes the deviation. D42's other deferred items are untouched. D43's source
model, per-field authority and refusals stand as written; D44 adds how they are met.
*Scope:* the concrete fetch contract for both sources, the acquisition record and
manifest version 3, the two cache-and-identity resolutions, and plan Tasks 23 to 25.
**D1–D43 are unaltered.** The addendum's sections and plan §G are not edited: where
D44 adds a §2.5 refusal and a manifest version, this entry is their record. D44
changes no code, no test, no requirement and no CI; the code and test changes it
authorizes belong to Tasks 23 to 25.

**D45 — NYC 311's measured real-data refusals mean every real ingestion of D37.2's
window is refused under the current contract; how to treat them is left open here, and
must be decided before any real NYC 311 ingestion, though not before the fetcher is
built and tested (addendum §1.1, §2.4, §2.5; plan Task 6; D21, D37.2, D43, D44).**
*Was:* §2.4 rejects an ambiguous (autumn fold) or nonexistent (spring gap) local time
and calls the effect "a negligible loss and a loud signal", while §2.5 lists adapter
normalization errors — "for example §2.4's DST rejections" — among the refusals of the
whole run, which is what the code does. Plan Task 6 refuses a null `descriptor`
(`MissingDescriptor`) and a `closed_date` before its `created_date`
(`NegativeResolutionTime`), the latter because "the reconnaissance measured zero
negatives". D43 recorded the daylight-saving conflict as unresolved, and D44 leaves
those rows to this entry.
*Now — the measured populations.* Measured on 2026-09-28 by SoQL aggregate queries
against `erm2-nwe9`, over records whose `created_date` lies in D37.2's window; counts
only, no rows retrieved. The dataset is updated daily, so these counts are a snapshot,
and a real acquisition records its own.

| Row class | Adapter refusal | Records |
|---|---|---|
| `descriptor` null | `MissingDescriptor` | 47,844 |
| `closed_date` before `created_date` | `NegativeResolutionTime` | 1,904 |
| `created_date` in the autumn fold, 2024-11-03 01:00–02:00 | `AmbiguousLocalTime` | 659 |
| `created_date` in the autumn fold, 2025-11-02 01:00–02:00 | `AmbiguousLocalTime` | 796 |
| `created_date` in the spring gap, 2024-03-10 02:00–03:00 | `NonexistentLocalTime` | 1 |
| `created_date` in the spring gap, 2025-03-09 02:00–03:00 | `NonexistentLocalTime` | 0 |
| `closed_date` in the autumn fold, 2024-11-03 01:00–02:00 | `AmbiguousLocalTime` | 890 |
| `closed_date` in the autumn fold, 2025-11-02 01:00–02:00 | `AmbiguousLocalTime` | 759 |
| `closed_date` in the spring gap, 2024-03-10 02:00–03:00 | `NonexistentLocalTime` | 1 |
| `closed_date` in the spring gap, 2025-03-09 02:00–03:00 | `NonexistentLocalTime` | 1 |
| `closed_date` in the spring gap, 2026-03-08 02:00–03:00 | `NonexistentLocalTime` | 0 |

The daylight-saving rows number between 1,651 and 3,107: a record created and closed
in the same edge hour is counted twice above, and that overlap was not measured. Nor
was the overlap between the three row classes. Also measured: `complaint_type` null in
0 records, and `closed_date` null in 118,649, which are open requests the adapter
accepts. Whether `unique_key` is unique across the window was not measured; that query
timed out.
*Now — the conflict.* One such row refuses the whole run under §2.5 and the code, so
with these counts every real NYC 311 ingestion of D37.2's window is refused. For the
daylight-saving rows §2.4's own words describe a loss of records, while §2.5 and the
code refuse the run. For null descriptors and negative durations the refusal is
specified consistently, and the counts above are what it costs.
*Now — the two policy options.* For each of the three row classes the deciding entry
chooses one of:
(a) **Retain the refusal.** §2.4, §2.5, Task 6 and the adapter stay as they are. No
NYC 311 corpus can be produced for D37.2's window while any such row is present, so
the work that needs one stays deferred under D42.
(b) **Exclude the typed rows, counted and reported.** Only rows raising exactly that
class's typed adapter error are excluded; each exclusion is counted by class and year,
recorded in the corpus's provenance and reported; no other refusal is weakened, and
nothing is dropped silently. This changes how those typed errors are handled — in
`ingest/cli.py` or in the adapter — and §2.5's list of refusals, which Tasks 6 and 8
protect, so the entry that chooses it must authorize that change explicitly.
*Now — when the decision is required.* **D45 must be decided before any real NYC 311
ingestion.** It is **not** required to build or test the fetcher: Task 24's fetcher
passes every row through untouched, its tests use synthetic fixtures, and it
implements neither option. D45 chooses neither option.
*Now — what D45 does not do.* It changes no refusal, no adapter, no section of this
addendum and no plan task; it downloads nothing; and it makes no completeness claim.
D42's deferred items stay open. **Phase 2 is not complete, and no document may
describe it as complete.**
*Scope:* the NYC 311 real-data refusals only. **D1–D44 are unaltered.** D45 changes no
code, no test, no requirement and no CI.

**D46 — A correction to D44 and plan Task 23: nine existing assertions pin manifest
version 2, not seven; acquisition verification applies only to an explicitly supplied
acquisition; `--fetch` refuses until a source registers a fetcher; each source and
window has one acquisition directory; and the acquisition record's contents and exact
bytes are frozen before Task 23 is implemented (addendum D44; plan Task 23, §G; D27,
D37.16).**
*Was:* D44 authorized updating "the seven assertions that pin version 2" in
`tests/ingest/test_manifest.py`, kept `tests/ingest/test_cli.py` to additions only,
and plan Task 23's acceptance allowed "no existing assertion changes except the seven
that pin manifest version 2". Task 23's reconnaissance found two more such assertions
in `tests/ingest/test_cli.py`, so manifest version 3 could not be implemented within
D44 as written. It also found three points D44 left open: which normalization runs the
acquisition checks govern, given that every existing ingestion test normalizes a raw
root that has no acquisition record; what `--fetch` does while no source-specific
fetcher exists; and how an acquisition is found, resumed and reused. D44 listed the
acquisition record's minimum contents but not its bytes, although `acquisition_id` is
their SHA256.
*Now — (A) nine assertions, not seven.* Exactly these nine existing assertions may
change, each from 2 to 3. Each asserts the version a newly written manifest carries,
which becomes 3:

| File | Test | Assertion |
|---|---|---|
| `tests/ingest/test_manifest.py` | `test_the_manifest_carries_the_three_contract_fields` | `m.manifest_version == MANIFEST_VERSION == 2` |
| `tests/ingest/test_manifest.py` | `test_write_manifest_emits_the_three_new_keys` | `payload["manifest_version"] == 2` |
| `tests/ingest/test_manifest.py` | `test_read_manifest_reconstructs_the_new_fields_exactly` | `back.manifest_version == 2` |
| `tests/ingest/test_manifest.py` | `test_a_version_one_manifest_is_read_only_compatibility` | `manifest.manifest_version == 2` |
| `tests/ingest/test_manifest.py` | `test_a_version_one_manifest_is_read_only_compatibility` | `payload["manifest_version"] == 2` |
| `tests/ingest/test_manifest.py` | `test_a_source_with_no_outcome_stream_writes_an_empty_outcome_map` | `manifest.manifest_version == 2` |
| `tests/ingest/test_manifest.py` | `test_the_record_schema_version_is_untouched_by_the_manifest_bump` | `manifest.manifest_version == 2` |
| `tests/ingest/test_cli.py` | `test_the_run_manifest_binds_both_streams` | `manifest.manifest_version == 2` |
| `tests/ingest/test_cli.py` | `test_a_cfpb_corpus_remains_loadable_without_a_311_sidecar` | `manifest.manifest_version == 2` |

No other existing assertion may change.
`test_a_version_one_manifest_without_the_new_field_still_reads` keeps asserting that a
version 1 manifest reads as 1, and `tests/ingest/test_cli.py` is otherwise additions
only.
*Now — (B1) where acquisition verification applies.* `ingest()`, the pass that
normalizes cached pages into the authoritative corpus, gains an optional keyword
argument `acquisition`, the path of an acquisition directory, defaulting to `None`.
With `None`, nothing changes: `ingest()` reads the raw root exactly as before, applies
no acquisition check, and writes `acquisition_id` as `None`. With an acquisition
supplied, that directory is the raw root for the fetch and for the read; the
completeness, page-list and digest checks below apply; and the manifest's
`acquisition_id` is the SHA256 of that acquisition's record. When the supplied
acquisition is already complete, `ingest()` does not call the fetcher. The command
line without `--fetch` passes no acquisition, so it behaves exactly as before.
`load_corpus`, the reader of a written corpus, is unchanged.
*Now — (B2) `--fetch` before a source fetcher exists.* `--fetch` looks up a fetcher
registered for the requested source. Task 23 builds the registration point and
registers nothing, so for every source `--fetch` raises `FetcherUnavailable`, a
subclass of `IngestError` whose message names the source, before any transport is
built, any request is made, any directory is created and anything is written. Task 23
builds no generic fetcher and gives no source any fetch behaviour; Tasks 24 and 25
each register their own source.
*Now — (B3) one acquisition directory per source and window.* An acquisition lives at
`data/acquisitions/<source>/<start>_<end>/`, where `<start>` and `<end>` are the run's
`--start` and `--end` as `YYYY-MM-DD`. An acquisition whose record is absent is
incomplete, and a `--fetch` run resumes it from its journal. An acquisition whose
record is present is complete and immutable: a `--fetch` run reuses it with zero
requests, and nothing in Sentinel deletes, overwrites or appends to it. A record that
is present but fails verification refuses the run and is left as it is. A fresh
snapshot of the same source and window is an explicit operator action that Task 23
does not provide. The directory holds:

| Path | Holds |
|---|---|
| `acquisition.json` | the acquisition record, written last; its presence marks the acquisition complete |
| `journal.jsonl` | one entry per completed slice; kept after completion, and not part of `acquisition_id` |
| `<source>/<sha256>.json.gz` | the cached pages, content-addressed as today, since the directory is the raw root |

Any further subdirectory, such as one retaining raw source responses, is defined by
the task that writes it, and is never `<source>/` itself.
*Now — the acquisition record's contents.* The record is a JSON object with at least
these keys; D44's minimum is met by them together with each source's `source_details`:

| Key | Type | Content |
|---|---|---|
| `record_version` | integer | `1`, this contract |
| `source` | string | the source slug |
| `window` | object | `start` and `end`, the dates as supplied; `resolved_start` and `resolved_end`, the instants `resolve_window` gives |
| `started_at`, `completed_at` | string | timestamps of the acquisition's first request and of its completion |
| `client` | object | `user_agent`, `library` and `library_version` |
| `policy` | object | the D44 values in force: `max_attempts`, `backoff_seconds`, `retry_after_cap_seconds` and `min_request_interval_ms` |
| `sentinel_commit` | string | the 40-character commit of the Sentinel checkout that ran the acquisition; if it cannot be determined, the acquisition does not start |
| `slices` | array | one object per completed slice, sorted by `key`: `key`, a string unique in the record; `requests`, in the order sent, one per request whose response the slice used, each with `url`, `params` as `[name, value]` pairs in the order sent, `retrieved_at`, `status`, `response_sha256` and `response_bytes`; `pages`, the page digests in the order yielded; and `verification`, an object of the slice's verification counts |
| `pages` | array | every page digest the slices list, each once, in ascending order |
| `source_details` | object | the source-specific provenance D44 requires, including `acquisition_kind` for CFPB, with its keys defined by Task 24 or Task 25; empty in Task 23's tests |

*Now — the acquisition record's bytes.* `acquisition_id` is the SHA256 of the record
file's bytes, so those bytes are fixed by these rules. **Values:** objects, arrays,
strings, integers, `true`, `false` and `null` only — no floating-point number, NaN or
infinity; durations are whole seconds or milliseconds, as each key's name says.
**Timestamps:** UTC strings of the form `YYYY-MM-DDTHH:MM:SS.ffffff+00:00`, always
with six fractional digits. **Digests:** 64 lowercase hexadecimal characters.
**Serialization:** keys sorted by Unicode code point at every level; no whitespace
between tokens; non-ASCII characters written as UTF-8 rather than escaped; strings
kept exactly as received, with no Unicode normalization; the file is that JSON in
UTF-8 with no byte-order mark, followed by exactly one line feed (0x0A) and nothing
else. This is the canonical form `page_checksum` already applies to pages —
`json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)`
— with the final line feed added. **Writing:** in binary mode, so no platform newline
translation occurs, to a temporary file in the same directory, moved into place with
`os.replace`. **Hash input:** exactly the file's bytes, final line feed included;
`acquisition_id` is their SHA256 in lowercase hexadecimal. **Reading:** a reader
parses the file, serializes it again by these rules, and refuses the acquisition
unless the result equals the bytes on disk.
*Now — the journal.* Each line of `journal.jsonl` is one completed slice's object,
serialized by the rules above and ended by a line feed, appended and flushed to disk
before the next slice begins. A slice counts as completed only when its line is whole
and parses, and only while every page it lists is present under its digest; a trailing
incomplete line is removed before anything more is appended. The record's `slices` are
the journal's slices, sorted by `key`.
*Now — verifying a completed acquisition.* When `ingest()` is given an acquisition,
before any page is normalized: the record must be present, or the run is refused as
incomplete; its bytes must be canonical, it must parse, carry every required key and
`record_version` 1, and name the run's source and the run's `--start` and `--end`, or
the run is refused; every page it lists must be present under `<source>/`, and the
`page_checksum` of each page, as decompressed and parsed, must equal its name, or the
run is refused; and any `*.json.gz` file directly under `<source>/` that the record
does not list refuses the run. The pages are then read in the order `ingest()` already
uses, by file name, and every refusal precedes any corpus write.
*Now — what D46 does not change.* D44's ten decisions, its two cache-and-identity
resolutions and its authorized and protected surfaces stand, except where this entry
corrects them. Tasks 23 to 25 still do not authorize full-window ingestion, nothing is
downloaded, and D45 stays undecided. **Phase 2 is not complete, and no document may
describe it as complete.**
*Scope:* the correction of D44's count of assertions and the four points D44 left
open, recorded before Task 23 is implemented. **D1–D45 are unaltered.** Plan Task 23
is corrected by reference, not rewritten. D46 changes no code, no test, no requirement
and no CI.

**D47 — Hardening the Task 23 acquisition layer before Task 24: which HTTP statuses
are retried, redirects refused, the transport's timeouts frozen and recorded, orphaned
pages quarantined rather than adopted or deleted, a damaged journal truncated from its
first damaged slice, one writer per acquisition, and a version 3 manifest's
`acquisition_id` validated (addendum D44, D46; plan Task 23).**
*Was:* a senior audit of commit 9db0977, which implements Task 23 under D44 and D46,
found behaviour neither entry fixed. Statuses outside D44's two lists — 3xx, 501, 505
and every other 5xx — were not retried. The real transport followed up to 30 redirects
on its own while the request's provenance kept the URL first requested. Its 30-second
connect and 300-second read timeouts were neither frozen nor recorded. A page cached
just before a crash, and orphaned when the source then returned different content,
made completion refuse with no way to recover. Damage before the journal's last line
dropped every later slice under no stated rule. Nothing stopped two writers from
sharing one acquisition. A version 3 manifest's `acquisition_id` was read without its
value being checked. D46 is authoritative (commit 9f0430c) and 9db0977 implements it;
neither is rewritten.
*Now — (1) HTTP statuses.* Transient statuses are retried under D44's attempts,
backoff, `Retry-After` and pacing; permanent ones raise `FetchFailed` at once.

| Status | Class |
|---|---|
| 2xx | success, then the caller's body validation, as before |
| 429 | transient, as before |
| 500, 502, 503, 504 | transient, as before |
| every other 5xx except 501 and 505 | transient |
| 501, 505 | permanent |
| 403 | permanent, and stops the run at once, as before |
| every other 4xx except 429 | permanent, as before |
| 3xx | not a success: permanent, under (2) |
| any status outside 1xx to 5xx, and 1xx | permanent, as before |

*Now — (2) redirects.* `RequestsTransport` does not follow redirects. A 3xx reaches
the client unfollowed and is a permanent `FetchFailed` naming its status, so the URL a
request's provenance records is always the URL its response came from. Sentinel never
moves silently to another URL or host.
*Now — (3) timeouts.* The real transport's connect timeout is 30 seconds and its read
timeout 300 seconds. The read timeout bounds each wait for data, not a whole response,
and no overall deadline is introduced. A timeout is transient, as D44 says. Both
values are frozen HTTP policy, and every acquisition record's `policy` carries them as
the integers `connect_timeout_seconds` (30) and `read_timeout_seconds` (300), beside
D46's four keys. D46 lets a record carry more keys than it lists, so `record_version`
stays 1.
*Now — (4) orphaned pages.* A page is orphaned when it lies directly under `<source>/`
in an incomplete acquisition and no completed journal slice lists it: a page cached
before a crash, or one only a slice removed under (5) listed. An orphaned page is not
a completed slice and is never written into the journal. When an incomplete
acquisition is opened for writing, after its journal has been checked under (5), every
orphaned page is moved, never deleted, with `os.replace` to
`quarantine/<sha256 of the file's bytes>/<its file name>` inside the acquisition
directory. The destination is fixed by the file's bytes, so a rerun that meets
identical bytes replaces them with themselves, and different bytes never overwrite one
another. Nothing else moves: journal, listed pages, record and any other file stay
where they are, including a temporary file a killed write left behind, which is not a
page. Quarantine happens only when an acquisition is opened; completion stays strict
and refuses while any unlisted page lies under `<source>/`. D46's verification of a
completed acquisition is unchanged: every listed page must be present and verify, and
no unlisted page may remain under `<source>/`. `quarantine/` is never `<source>/`, so
nothing in it is a page.
*Now — (5) damage before the journal's last line.* When an acquisition is opened for
writing, its journal is read from the first line, and the first line that is not a
completed slice is the resume point: a line that is incomplete, unparseable, not
canonical or malformed, that repeats an earlier key, or that lists a page which is
missing or does not verify. The journal is truncated there, and that slice and every
slice after it are unfinished and may be fetched again. No line is repaired or
rewritten in place. The resume point depends only on the journal's bytes and the pages
on disk, so it is deterministic. Pages that only the removed slices listed are
orphaned and quarantined under (4). Completion never truncates: before it writes the
record it verifies every page the journaled slices list, and if any is missing or does
not verify it refuses and writes no record, leaving the next open to apply this rule.
*Now — (6) one writer per acquisition.* An acquisition directory has one active
writer. Opening an acquisition for writing first creates `writer.lock` in its
directory with an atomic exclusive create (`O_CREAT | O_EXCL`), which fails if the
file already exists; the file records its holder's process id, host and start time. An
open that finds the lock refuses, naming the file and its recorded holder. The writer
removes the lock when it closes, whether it succeeded or failed, and removes nothing
else. A lock left by a process that could not close, after a hard kill or a power
loss, refuses every later open until an operator removes that one file, having
confirmed that no writer is running; Sentinel never removes a lock it did not create,
and never guesses whether a holder is alive. Removing the lock touches no journal
line, page or record, so an interrupted acquisition resumes exactly as it would have.
Reading a completed acquisition, whether to verify it or to reuse it, takes no lock.
*Now — (7) `acquisition_id` in a manifest.* In a version 3 manifest `acquisition_id`
is required: its key must be present, and its value is either null — a run given no
acquisition, exactly as D46 says — or a SHA-256 digest written as 64 lowercase
hexadecimal characters. Any other value, or a missing key, refuses the manifest on
read, and `build_manifest` refuses any value that is neither. Versions 1 and 2 still
read it as `None`.
*Now — two defects fixed without a contract change.* A `Retry-After` header is read as
whole seconds only when it consists of ASCII decimal digits, surrounding whitespace
aside; anything else, including non-ASCII numerals, is ignored rather than raising or
being read as a number, as D44's "when present" already implies. And the journal's
pages are checked once, when an acquisition is opened, and each new slice's pages when
it is recorded, rather than every journaled page on every append; with completion's
check under (5) and D46's verification before normalization, no check is weakened.
*Now — what D47 does not change.* The adapters, normalization, `ingest/storage.py`,
`ingest/schema.py`, `ingest/roster.py` and `ingest/identity.py` are unchanged. D44,
D45 and D46 stand as written: D47 settles only what they left open. Task 22 is
unchanged. Tasks 24 and 25 remain unimplemented, `FETCHERS` stays empty, and Tasks 23
to 25 still do not authorize full-window ingestion. Nothing is downloaded, and D45
stays undecided. **Phase 2 is not complete, and no document may describe it as
complete.**
*Scope:* the seven points above, and the two defect fixes, recorded before Task 24.
**D1–D46 are unaltered.** Plan Task 23 is extended by reference, not rewritten: its
hardening lands after 9db0977 as two further commits. D47 changes no code, no test, no
requirement and no CI.

**D48 — Task 24's source and drift contract: a completed slice whose source has moved
is fetched again once, through a generic rewind that truncates the journal from that
slice and quarantines its pages; the acquisition's start state is persisted once and
never replaced; a count that disagrees with its rows is requested again together with
its data, once; NYC 311's freshness headers are recorded; and Task 24 may register its
fetcher (addendum D44, D45, D46, D47; plan Tasks 23, 24).**
*Was:* D44 (10) requires that when NYC 311's source moves during an acquisition, every
completed slice's count is verified again and each changed slice is fetched again,
once. Task 24's reconnaissance found that nothing in Task 23's acquisition layer can
do that: D46's journal is append-only, `record_slice` refuses a key it already holds,
and D47 (5) treats a repeated key as damage, so an already-journaled slice cannot be
replaced. It found no place, other than a pseudo-slice, to keep the acquisition's
start state across a resume, so "the start" of a resumed acquisition was undefined. A
count that differed from the rows returned was retried only on the data side, by the
per-request policy, which cannot help when the count is the stale side. Registering a
fetcher touches files plan Task 24 does not list and two Task 23 tests whose
assertions require an empty registry. And the data responses carry freshness headers
D44 does not mention.
*Now — (1) rewinding a drifted slice.* D44 (10) stands: when the source moves,
completed slices are checked again, a changed slice is fetched again once, and a slice
still inconsistent after that refuses the acquisition. To make that representable
under D47, the acquisition layer gains one generic operation, a rewind from a named
completed slice. It runs only in an open, incomplete acquisition held by its single
writer (D47 (6)). It truncates the journal immediately before the named slice's line,
so that slice and every slice journaled after it become unfinished; no line is
rewritten in place, and no superseding or repeated line is ever written, so D47's rule
that a key appears in the journal once holds. Pages that only the removed slices
listed are moved at once to D47's quarantine, by the same path rule, and never
deleted; a page a retained slice still lists stays. This is an intentional extension
of D47 (4)'s quarantine timing, under which quarantine otherwise happens only when an
acquisition is opened: without it, strict verification would reject the rewound
acquisition in the same run, because the removed slices' pages would lie unlisted
under `<source>/`. The rewind is recorded first, as one canonical line appended to
`rewinds.jsonl` in the acquisition directory holding the named slice, the removed
keys, the reason and the time, and only then is the journal truncated; an open that
finds the latest rewind's named slice still journaled completes that rewind before
anything else, so an interrupted rewind resumes deterministically. The fetch that
follows writes fresh pages and fresh journal lines. Completion and D46's verification
stay strict. The completed record carries every rewind, in order, under a top-level
`rewinds` key, which D46 permits, since a record may carry more keys than it lists,
and `record_version` stays 1. This is a deliberate extension of Task 23's generic API,
authorized here only because Task 24 needs D44's drift rule.
*Now — (2) the acquisition's start state.* A new acquisition records its source's
start snapshot before its first slice, in `start.json` in the acquisition directory:
canonical bytes under D46's rules, written once through a same-directory temporary
file and `os.replace`. It belongs to the acquisition, not to a slice, and is never
part of the journal. An open, a resume or a rewind never replaces it: the start of a
resumed acquisition remains the start of its first run, and every later observation is
an end observation. The completed record carries it unchanged as
`source_details.start`, and completion refuses if persisted `start.json` and
`source_details.start` differ.
*Now — (3) a count that disagrees with its rows.* For each slice the fetcher requests
the count, then the data. A count of 50,000 or more refuses before the data is
requested (D44 (7)). When the rows returned differ from the count, that is a
source-level disagreement, distinct from an HTTP retry: the fetcher requests both the
count and the data again, once, and accepts the slice only if the second pair agrees.
A second disagreement refuses the acquisition: no completed record is written, and the
incomplete acquisition is kept for diagnosis and resume. The stale side is never
silently accepted. D47's per-request retry policy still applies beneath each of those
requests. A `unique_key` repeated within a slice's data refuses at once, without a
source-level retry, since a second request cannot remove a duplicate the source
publishes.
*Now — (4) drift re-verification.* The fetcher checks the source state after its last
unfinished slice, and again after the slices a rewind made unfinished have been
fetched. A check takes a snapshot, `rowsUpdatedAt` and the window's `count(*)`, and
compares it with the start snapshot and with the latest snapshot at which every
completed slice was verified. That latest snapshot is kept for the run, and a resumed
run begins from the start snapshot. Any difference is movement, and movement is only a
trigger for verification, not a definition of correctness. Every completed slice is
then checked again by requesting its `count(*)`, and a count that differs from its
journaled count is a change. Rows and distinct `unique_key`s cannot be checked again
without the data, and at fetch time they equalled the count. Drift verification proves
count/identity invariants, not byte-for-byte immutability of historical rows. A slice
whose count and distinct-key checks remain unchanged is not re-downloaded solely to
detect same-count content substitutions. The earliest changed slice in journal order
is the rewind point: the acquisition rewinds from it under (1) and fetches from there.
Each changed slice is fetched again at most once: the rewind records the changed keys,
so a slice that changes again, or that fails verification after its re-fetch, refuses
the acquisition. A slice that appears in a previous rewind event's changed-key set
MUST NOT be rewound a second time. If that slice is found changed again after its
refetch, the acquisition refuses. A pass that finds no change ends the checks. Its
snapshot's window count must then equal the sum of the slice counts, and no
`unique_key` may appear in two slices; either failure refuses.
*Now — (5) start and end provenance.* The completed record's `source_details` carries
the immutable start snapshot as `start` and the snapshot of the final pass that found
no change as `end`. Differences between them are provenance, and nothing claims the
dataset was immutable during the run. Each slice's `verification` keeps its `count`,
`rows` and `distinct_unique_keys` together, and nowhere else; the record keeps D46's
slice history and the rewinds.
*Now — (6) freshness headers.* When a snapshot's response carries
`X-SODA2-Truth-Last-Modified` or `X-SODA2-Data-Out-Of-Date`, each value is recorded
exactly as sent in that snapshot, as `truth_last_modified` and `data_out_of_date`.
They are snapshot provenance only: they are not part of any slice's `verification`,
whose semantic content stays `count`, `rows` and `distinct_unique_keys`, and they add
no refusal.
*Now — (7) URL encoding.* The contract is the request parameters, not any one wire
encoding: tests assert the semantic parameters the fetcher passes to its transport,
and the real transport encodes them as `requests` does. No request is made to settle
whether Socrata reads a space written as `+` or as `%20`.
*Now — (8) registration, and the files Task 24 may touch.* As D46 says, Tasks 24 and
25 each register their own source. Task 24 may modify `ingest/fetch/registry.py` to
register NYC 311; the two tests whose assertions require an empty registry,
`test_task_23_registers_no_source_fetcher` in `tests/ingest/fetch/test_acquisition.py`
and the NYC 311 case of
`test_fetch_refuses_before_any_side_effect_while_no_fetcher_is_registered` in
`tests/ingest/test_cli.py`; `tests/ingest/test_cli.py` for the `--fetch` integration
tests registration makes possible; and `ingest/fetch/acquisition.py` only for (1) and
(2): the immutable start state, the rewind from a slice, the rewind history, the
quarantine of rewound pages and the completion of an interrupted rewind. No other
surface is added.
*Now — (9) the D45 boundary.* D45 stays undecided. The fetcher passes null or absent
descriptors, negative durations and daylight-saving-edge timestamps through exactly as
served, with no filter, repair or conversion; what normalization does with them
remains D45's to decide, and under the current contract it refuses the run. Neither of
D45's options is implemented.
*Now — the NYC 311 source contract Task 24 implements.* Dataset `erm2-nwe9`, SODA 2.0,
whose responses are bare JSON arrays. One slice per New York civil day D, with exactly
one page per day. Each slice's data request is
`$select=unique_key,created_date,closed_date,complaint_type,descriptor`, `$where` of
`created_date >= 'DT00:00:00' AND created_date < 'D+1T00:00:00'`, `$order=unique_key`
and `$limit=50000`. Its count request is `$select=count(*) AS n` with the same
`$where`, and the count arrives as a string. No token is sent. Daylight-saving days
use the same literals, unchanged. The record's `source_details` holds exactly:

| Key | Value |
|---|---|
| `dataset_id` | `erm2-nwe9` |
| `endpoint` | `https://data.cityofnewyork.us/resource/erm2-nwe9.json` |
| `metadata_endpoint` | `https://data.cityofnewyork.us/api/views/erm2-nwe9.json`, where `rowsUpdatedAt` is read |
| `soda_version` | `2.0` |
| `source_api_version` | `socrata-soda2`, the adapter's constant |
| `start` | the immutable start snapshot: `rows_updated_at` (integer seconds), `window_count`, the freshness headers when present, and the request records of its metadata and window-count requests |
| `end` | the final snapshot, in the same form |

*Now — what D48 does not change.* The adapters, normalization, `ingest/cli.py`,
`ingest/manifest.py`, `ingest/fetch/http.py`, `ingest/fetch/canonical.py`,
`ingest/storage.py`, `ingest/schema.py`, `ingest/roster.py` and `ingest/identity.py`
are unchanged. D44, D46 and D47 stand as written, except for the extensions this entry
makes explicit: quarantine at a rewind, and `start.json`, `rewinds.jsonl` and the
record's `rewinds` key. D45 stays undecided. Task 24 remains the NYC 311 source
fetcher; nothing here authorizes real ingestion, a download or a request to the
source, and `FETCHERS` stays empty until Task 24 lands. **Phase 2 is not complete, and
no document may describe it as complete.**
*Scope:* the drift, start-state, count-disagreement, freshness, encoding and
registration decisions Task 24 needs, recorded before it is implemented. **D1–D47 are
unaltered.** Plan Task 24 is extended by reference, not rewritten. D48 changes no
code, no test, no requirement and no CI.

**D49 — Task 25's CFPB reconstruction contract: each API day is verified when it is
fetched and its snapshot is kept as provenance, with no drift recount and no rewind; a
count that disagrees with its CSV is requested again together with the CSV, once; the
verified archive pins live in the acquisition's start state, and a resume refuses a
missing or changed export; a literal `None` Complaint ID refuses; and the served
`_index` is provenance only (addendum D43, D44, D46, D47, D48; plan Task 25).**
*Was:* D43 froze the reconstruction's source model, its per-field authority and its
refusals; D44 froze its narrative filter, its archive and API retrieval, its
completeness and its provenance; and plan Task 25 records its behaviour. Task 25's
reconnaissance, from evidence captured on 2026-09-28 and one read-only request on
2026-09-30, found five points none of them settles. D48's drift rule, which counts
completed slices again and rewinds a changed one, and its rule for a count that
disagrees with its rows are written for NYC 311 and Task 24. D44 (5) defines CFPB
completeness per UTC day and states no drift rule, and D44 (6) calls a CSV whose row
count differs from `hits.total` a transient body, retried alone under the per-request
policy, which cannot help when the count is the stale side. Nothing says where the
pins of the downloaded archive exports live across a resume, or what a resume does
when an export is missing or its bytes have changed. D44 (4) resolves a literal `None`
in an adapter-relevant CSV column by `GET /{complaintId}`, which cannot apply when the
`None` is in the Complaint ID column itself. And D44 (2) records "the hit's `_index`"
without saying what it identifies.
*Now — the API's own state, as measured.* The API documents `_meta.last_indexed` as
"The timestamp of the most recently indexed complaint" and `_meta.last_updated` as
"The timestamp of the most recent complaint"; neither is a modification time of
historical records. Between 2026-09-28 and 2026-09-30 both moved from
2026-09-27T12:00:00-05:00 to 2026-09-30T12:00:00-05:00, and `total_record_count` grew
from 18,040,765 to 18,091,520, while the count for UTC day 2024-11-01 stayed 9,417.
Over the same interval the served `_index` changed from `complaint-public-v2` to
`complaint-public-v1`. The global metadata therefore moves daily whether or not any
window day changes, and the index name alternates.
*Now — (1) the drift model: a snapshot per day.* CFPB does not use D48's
count-recount-rewind model. The archive side is immutable once downloaded and
verified, because its exact bytes are retained and pinned. Each API day is verified
internally when it is fetched, under (2), and that day's source snapshot — its
`hits.total`, its `_meta.last_indexed` and its served `_index` — is kept as provenance
together with its retained raw responses. The start and end snapshots are provenance
only. Movement of `_meta.last_indexed`, `_meta.last_updated`, a total count or the
served `_index` does not by itself trigger a recount or a rewind, and there is no
automatic CFPB rewind. CFPB historical stability is not asserted; reproducibility is
established from retained, content-addressed acquisition inputs. Those inputs are the
archive exports, pinned by SHA256, and the raw API responses, retained and digested. A
reconstructed day reflects the API as it was when that day was fetched, and an
acquisition may span more than one API index state, which its per-day provenance
shows. A change to an API record that leaves every count unchanged, a same-count
content substitution, cannot be detected. The population does not rest on the API
holding still: it is the pinned archive's, and every included archive record is joined
and checked when its day is fetched, under D43's refusals.
*Now — (2) a count that disagrees with its CSV.* For each UTC day the fetcher requests
the same-day JSON count, then the CSV. The day is accepted only when the count equals
the CSV's rows, the distinct Complaint IDs equal the rows, and every row's
`date_received` falls on the requested UTC day. When the rows differ from the count,
the count and the CSV are both requested again, exactly once, and the day is accepted
only if the second pair agrees; a second disagreement refuses the acquisition, which
is kept incomplete for diagnosis and resume. The stale side is never silently
accepted. D47's per-request retry policy still applies beneath each of those requests.
A Complaint ID repeated within a day's CSV, or a row outside the requested UTC day, is
not a count disagreement: it refuses at once, without a source-level retry. For Task
25 this rule takes the place of D44 (6)'s example of a transient body, a CSV whose row
count differs from `hits.total`; a CSV that cannot be parsed stays a transient body
under D44 (6) and D47. It is a CFPB source-level rule, not a change to D48 or to the
generic acquisition layer.
*Now — (3) archive pins across a resume.* The verified archive pins, the URL and
SHA256 of each export used, are persisted in the acquisition's start state (D48 (2)),
which is written once, before the first day, and never replaced. The exports
themselves are retained in the acquisition directory, in a subdirectory Task 25
defines under D46 (B3). On resume the same ZIPs must be present, and the SHA256 of
each one's bytes must equal its recorded pin; a missing or changed ZIP refuses the
resume. An archive is never re-pinned as a replacement, and archive discovery never
selects a new export in place of a pinned one. The start state remains immutable.
*Now — (4) a literal `None` in the Complaint ID column.* D43's and D44 (4)'s
resolution of a literal `None` by `GET /{complaintId}` cannot apply when the Complaint
ID itself is `None`. A row of a day's CSV whose Complaint ID cell is the literal
`None` refuses the acquisition as an unusable API record: without a usable identifier
it can be neither joined to a retained archive row nor classified, so whether it
belongs to the reconstruction's join cannot be known. No by-ID request is attempted
without a usable requested ID, and no identifier is invented.
*Now — (5) the served `_index`.* The `_index` each day's count response serves is
recorded as provenance only, and never used as a stable source identity. The stable
API identity remains `SOURCE_API_VERSION = "cfpb-ccdb-v1"`, unchanged under D44 (3),
and the acquisition kind remains `cfpb-archive-api-reconstruction-v1`.
*Now — the acquisition record's `source_details` for CFPB.* It holds exactly these
keys:

| Key | Holds |
|---|---|
| `acquisition_kind` | `cfpb-archive-api-reconstruction-v1` |
| `source_api_version` | `cfpb-ccdb-v1`, the adapter's constant |
| `archive` | the reading-room page's URL, SHA256 and retrieval time; for each export used, its pin and D44's pinned facts (byte size, `Last-Modified` and the ETag as served, member name and size), its role (a population export, or boundary only), and the content observed in it (header, row count and `Date received` range); and the narrative-filter predicate |
| `api` | the base URL, the endpoints and the fixed query parameters; and for each UTC day, the margin days included, its `hits_total`, `last_indexed` and `_index`. The raw-response digests are in that day's request records and are not repeated |
| `join` | the key, stated as exact Complaint ID string equality, and the totals `matched`, `excluded`, `archive_only`, `api_only` and `resolved_by_id` |
| `start` | the start state as persisted under (3): the archive pins, and the API snapshot taken before the first day |
| `end` | the API snapshot taken once every day is fetched, provenance only |

An API snapshot is one same-window JSON count request's `hits.total`,
`_meta.last_indexed`, `_meta.last_updated`, `_meta.total_record_count` and served
`_index`, with that request's record. Each day's slice `verification` holds counts
only, as D46 requires: `api_count`, `api_rows`, `api_distinct_ids`, `included`,
`matched`, `excluded`, `archive_only`, `api_only` and `resolved_by_id`. Generic
request provenance is kept once, in each slice's `requests`, and nowhere else.
*Now — the Task 25 source contract, as reconciled.*

| Rule | Contract |
|---|---|
| Exports | #5 to #14 are the population exports; #15 is boundary only, used solely to classify records at the window's end; all are selected from the reading room's list, and never replaced under (3) |
| Coverage | the archive's content, not a file name alone, establishes date coverage; every window day must appear in the verified content |
| Archive header | exactly export #8's 16 columns |
| Archive date | `Date received` is `YYYY-MM-DD` in every row |
| Narratives | a row whose narrative is not a string that is non-blank after `strip()` is excluded and counted (D44 (1)) |
| API day | per UTC day D, one CSV request (`format=csv`, `no_aggs=true`, `date_received_min=D`, `date_received_max=D`) and one same-day JSON count (`size=1`, `no_aggs=true`, the same bounds), whose `hits.total.relation` must be `eq`; and one margin day at each end of the window |
| `None` | as D43 and D44 (4) for Product, Timely response?, Date received and Date sent to company; as (4) above for Complaint ID |
| Pages | one `SourcePage` per window day, in the adapter's `hits.hits[]._source` shape with its six fields, its rows sorted by (`date_received`, `complaint_id`) and its date strings copied exactly as served; the margin days produce no page |
| Time | no invented time of day, offset or timezone (D43) |

*Now — what D49 does not change.* D43 and D44 stand as written, except for the two
points (2) and (4) make explicit: the archive alone is authoritative for the
narrative, and the API alone for `date_received` and `date_sent_to_company`; records
join on exact Complaint ID string equality; `product` and `timely` must agree, and the
archive's date must equal the UTC calendar date of the API's `date_received`; a
missing or unusable API record refuses, and so does a returned Complaint ID that
differs from the one requested; the narrative filter stands; API-only records are
recorded and are not part of the narrative population; and completeness is scoped to
the retained archive population, never to CFPB's historical narratives. D44's and
D47's retry, pacing, redirect and timeout policy stands, and so does D48's generic
acquisition implementation. D45 stays undecided and is untouched.
*Now — D49's scope and what it authorizes.* D49 is specific to Task 25. No change to
the generic acquisition API is needed: D48's start state and rewind remain available,
Task 25 uses the start state under (3), and it never invokes a rewind for API drift.
Task 25 is independent of D45. D49 authorizes no real acquisition, no download and no
request to either source: the reconnaissance plan Task 25 requires before its parser
is frozen, which downloads about 0.77 GB of archive exports, needs its own
authorization, and no real ingestion follows from D49. **Phase 2 is not complete, and
no document may describe it as complete.**
*Scope:* the five Task 25 decisions above, CFPB's `source_details`, and the Task 25
source contract as reconciled, recorded before Task 25 is implemented. **D1–D48 are
unaltered.** Plan Task 25 is extended by reference, not rewritten. D49 changes no
code, no test, no requirement and no CI.

**D50 — D45 decided: every otherwise-valid NYC 311 row in one of D45's three classes is
excluded from the corpus, never silently, typed by kind and counted by kind and New York
civil year in manifest version 4; a row with any other refusal is still refused, the
fetcher is unchanged, and the implementation is plan Task 26, which is authorized
separately (addendum §1.1, §2.3, §2.4, §2.5, §3.2, §4.2, §6, §7; plan Tasks 6, 8, 24, 26;
D21, D23, D24, D26, D27, D37, D44, D45, D46, D47, D48).**
*Was:* D45 measured on 2026-09-28, by SoQL aggregate queries against `erm2-nwe9` and
counts only, 47,844 rows with a null `descriptor`, 1,904 with `closed_date` before
`created_date`, and between 1,651 and 3,107 daylight-saving-edge rows among the
7,111,810 in D37.2's window. It recorded that one such row refuses every real NYC 311
ingestion under §2.5 and the code, offered for each class (a) retaining the refusal or
(b) excluding the typed rows, counted and reported, and chose neither. D48 (9) kept the
fetcher passing these rows through untouched. A review of D45 found four points option
(b) leaves open: which refusal a row with several problems raises, where the exclusion
is made, where the counts are recorded, and which year an excluded row counts under.
*Now — the decision.* **Option (b) for all three classes:** `MissingDescriptor`;
`NegativeResolutionTime`; and `AmbiguousLocalTime` and `NonexistentLocalTime` on either
timestamp. Only an otherwise-valid row is excluded: a row with any refusal outside
those classes is refused, exactly as today. D45's counts are a snapshot taken by
predicate; no corpus records them, and each run records its own under (4).
*Now — (1) the exclusion kinds.* Each class is the complete existing adapter error
class, and the two daylight-saving errors are counted separately for each timestamp:

| Kind | Excluded when |
|---|---|
| `MissingDescriptor` | `descriptor` is absent, null or not text; an empty or whitespace-only descriptor is valid and kept |
| `AmbiguousLocalTime:created_date` | `created_date`'s wall clock falls in an autumn fold |
| `NonexistentLocalTime:created_date` | `created_date`'s wall clock falls in a spring gap |
| `AmbiguousLocalTime:closed_date` | `closed_date`'s wall clock falls in an autumn fold |
| `NonexistentLocalTime:closed_date` | `closed_date`'s wall clock falls in a spring gap |
| `NegativeResolutionTime` | the closed instant precedes the created instant; a zero duration is kept |

The 47,844 rows D45 measured are null descriptors. The `MissingDescriptor` kind also
covers every other row that raises that existing typed error, an absent key or a value
that is not text, although reconnaissance observed no such value; the class is not
split.
*Now — (2) precedence.* A row is classified in two stages. **Stage one** applies every
check outside the D45 classes, in the adapter's existing relative order: `unique_key`;
`complaint_type`; `created_date`, which must be present, ISO-8601 and without an
offset; and `closed_date`, which is absent or null for an open request and otherwise
must be present, ISO-8601 and without an offset. A failing check raises exactly the
existing typed error, `MissingField`, and a row that is not a mapping fails as it does
today. A row that fails stage one refuses the run before any write, whatever D45
condition it also has. **Stage two** applies the D45 conditions in the adapter's
existing relative order — the descriptor, then a `created_date` edge, then a
`closed_date` edge, then a negative duration — and the first that applies is the row's
single kind. The kinds are not evaluated independently: a duration is undefined when
either timestamp has no instant; D45 (b) speaks of rows raising exactly one class's
typed adapter error, and a row raises one; and single attribution makes the counts sum
to the rows excluded. Per-kind counts are therefore attributions, not prevalences: a
fold row with a null descriptor counts only as `MissingDescriptor`. For a row with a
problem outside the D45 classes the run's outcome is unchanged: it is refused before
any write. Only the error reported for such a row that also has a D45 condition
changes, from the D45 error the adapter meets first today to the error outside them,
and that change is what keeps every other refusal from being weakened.
*Now — (3) where the exclusion is made.* The NYC 311 adapter gains a pure function
beside `normalize`, `classify_row(row)`, which returns either the
`(CorpusRecord, NYC311Outcome)` pair `normalize` returns or a frozen
`Exclusion(external_id, kind, created_civil_date)`. `ingest()` in `ingest/cli.py` holds
the policy: for NYC 311 only, it calls `classify_row`, keeps the pairs and counts the
exclusions. `normalize`, its typed errors, `NYC311Adapter` and the `SourceAdapter`
protocol are unchanged, so `normalize` still raises for every D45 row and plan Task 6's
tests stand. The boundary holds four properties. *Adapter purity:* classification is a
pure function of one row, with no I/O, clock, window or counter; the policy and the
counting live in the command line. *One validator:* a command line that only caught
`normalize`'s errors could not see the checks `normalize` never reached after its first
error, so it could not honour (2) without duplicating the adapter's validation.
*Fetcher pass-through:* nothing under `ingest/fetch/` changes, rows are cached exactly
as served (D48 (9)), and the fetch layer filters nothing. *Acquisition guarantees:*
classification reads only pages `verify_acquisition` has passed, and the acquisition
record, its slices, its page digests and `acquisition_id` are unchanged and hold no
exclusion count. CFPB is untouched, and D44 (1)'s narrative filter remains CFPB's own.
*Now — (4) provenance: manifest version 4.* Manifest version 3 has no field for
exclusions, and the acquisition record is immutable and written by the fetcher before
normalization. D50 authorizes the smallest change: `MANIFEST_VERSION` becomes 4, and
the manifest gains one field, `excluded_records`, mapping each kind to a mapping from
New York civil year to count, serialized as `per_year_counts` is. For NYC 311 it holds
exactly the six kinds, each listing only years with a count of at least 1, and an
empty mapping when there are none; for CFPB it is an empty mapping, since no kind
applies. `ingest()` passes the counts to `build_manifest`; they are never derived from
disk, where no excluded row is. They cover every row in the window before `--limit`
truncates anything, as D24 requires of the roster. The field is required in a version
4 document and validated on read; a version 1, 2 or 3 manifest reads it as an empty
mapping, which is truthful, because under those versions such a row refused the run.
`schema_version` stays 1, and `compute_corpus_id` and what it covers are unchanged.
The counts can be recomputed: the pure classifier over an acquisition's immutable
pages yields the same rows. The manifest is the report of record. The command line
prints nothing today and D50 adds no other channel; the README and the
reproducibility guide report the counts only when real ingestion is documented (D44).
*Now — (5) the year.* An excluded row counts under the year of its published
`created_date` literal, read as a New York civil date (§2.4, D21), for all six kinds,
including a `closed_date` edge and a negative duration. The year always exists,
because stage one has validated `created_date`'s form before any D45 condition is
applied, and it needs no UTC instant: a `created_date` edge has none, and none is
invented. `per_year_counts` counts kept records by the UTC year of `submitted_at`, and
the two years differ for a record created on 31 December after 19:00 New York time, so
reconciliation is by window total, not by year: for an unbounded run, `record_count`
plus the sum of `excluded_records` equals the rows in the window.
*Now — (6) the window, and what still refuses.* An excluded row is in the window when
its New York civil date lies within the run's inclusive civil dates, §2.5's frame; for
a row that normalizes this is equivalent to the instant comparison, because New York
midnight is never a daylight-saving transition. A row outside the window is outside
the population, whether it would normalize or not, and is not counted; in an
acquisition every row is in the window by construction. These still refuse before any
write, unchanged: `InvalidDateRange`, `InvalidLimit`, `AuthoritativeCorpusExists`,
every fetch and acquisition refusal (D44, D46, D47, D48), every stage-one error, and
CFPB's refusals, `MissingNarrative` among them. `EmptyWindow` is raised when every
in-window row is excluded, and its message also states the exclusion counts.
`DuplicateExternalId` counts the `unique_key` of every in-window row, kept or
excluded, so an exclusion cannot hide a duplicate. The timestamp diagnostic and the
outcome sidecar are built from kept records only.
*Now — (7) the corpus.* For a window, the NYC 311 corpus is all source rows that
successfully normalize after excluding exactly the authorized D45 classes and still
satisfy every other refusal and invariant: every in-window row of the verified
acquisition, or of the raw cache when no acquisition is given, that `normalize`
accepts, provided every row passes stage one and every other refusal passes. Each
excluded in-window row is counted in the manifest by kind and year. The whole row is
excluded, record and outcome, for every kind, including a `closed_date` edge or a
negative duration whose record fields are themselves valid: D45 offers no option that
keeps a record without its outcome. Open requests remain in the corpus (D37.3).
*Now — (8) research consequences.* The retained population lacks rows without a text
descriptor, rows with an edge-hour timestamp and closed rows with a negative duration,
so every later NYC 311 quantity is computed over it: resolution-time distributions
(§4.2, and §7's figures, which were measured before any exclusion); complaint-type
aggregates (§6.3); breach rates (D37.7); the per-type p75 thresholds and their
fallback (§7); `text_length` and the hour and weekday features (§3.2), and §2.3's
histogram; the temporal split's counts (§6.1); and the risk model's training and
evaluation population (D37), with the NYC 311 side of §5.4's probe. The exclusions are
not random. A `closed_date` edge and a negative duration are selected on the outcome;
a `created_date` edge removes particular local hours on four dates; and how null
descriptors fall across complaint types was not measured. NYC 311 has no locked
roster, so a type whose rows were all excluded would leave `label_roster` without any
refusal, and the manifest counts exclusions by kind and year, not by type. The
2026-09-28 snapshot puts the three classes at 47,844 to 52,855 rows, 0.67% to 0.74% of
the window; each run records its own. Every published NYC 311 figure must state that it
is computed over D50's population and cite the manifest's counts.
*Now — (9) the test contract, written first by plan Task 26.* In
`tests/ingest/test_nyc311_normalize.py`, additions only, every existing test
unchanged: (T1) every row `normalize` accepts yields the identical pair from
`classify_row` — fixture rows, an empty descriptor, an open request, a zero duration
and an interval spanning a transition; (T2) a null descriptor, an absent descriptor
key and a non-text descriptor are `MissingDescriptor`, and empty and whitespace-only
descriptors are kept; (T3) `created_date` at 01:00:00.000, 01:30 and 01:59:59.999 on
2024-11-03 and 2025-11-02 is `AmbiguousLocalTime:created_date`, at 02:00:00.000, 02:30
and 02:59:59.999 on 2024-03-10 and 2025-03-09 is `NonexistentLocalTime:created_date`,
and 00:59:59.999, 02:00:00.000 on a fold day and 03:00:00.000 on a gap day are kept;
(T4) the same edges on `closed_date`, including the 2026-03-08 gap, give the
`closed_date` kinds, counted under `created_date`'s year; (T5) a close one second
before the open is `NegativeResolutionTime`, a zero duration is kept, and the
comparison is between instants; (T6) overlaps take the first applicable kind — a null
descriptor with a `created_date` fold is `MissingDescriptor`, a `created_date` fold
with a `closed_date` gap is `AmbiguousLocalTime:created_date`, a `closed_date` fold
whose wall clock precedes `created_date` is `AmbiguousLocalTime:closed_date`, and a
null descriptor with a negative duration is `MissingDescriptor`; (T7) each check
outside the D45 classes — `unique_key` absent, blank or not text; `complaint_type`
absent or not text; `created_date` absent, blank, not ISO-8601 or carrying an offset;
`closed_date` blank, not ISO-8601 or carrying an offset — combined with each kind
raises the same `MissingField` the check raises alone, and a row that is not a mapping
fails as it does under `normalize`; (T8) a null-descriptor row created at
2024-12-31T23:30:00.000 counts under 2024, though its UTC instant is in 2025; (T9)
`normalize` still raises each D45 error for every D45 row of T2 to T6; (T10)
`classify_row` is deterministic, does not mutate its input, touches no network,
filesystem or clock, does not filter by window and imports no new module, as
`normalize`'s tests assert. In `tests/ingest/test_cli.py`: (C1) a raw-cache NYC 311 run
holding valid rows and one row of each kind succeeds, writes only the valid records and
outcomes, and records exact `excluded_records`, with `record_count` plus the excluded
sum equal to the rows; (C2) each check outside the D45 classes, alone and combined
with a kind, refuses before any write and leaves an existing corpus byte-identical;
(C3) a window whose every row is excluded raises `EmptyWindow` naming the counts and
writes nothing; (C4) under `--limit` the counts cover the whole window; (C5) a kept
row and an excluded row sharing a `unique_key` raise `DuplicateExternalId`; (C6) a row
of a kind outside the window is neither counted nor refused, while a row with another
refusal outside the window still refuses; (C7) a CFPB manifest's `excluded_records` is
empty, and a row the CFPB adapter refuses still refuses the run; (C8) a `--fetch` run
with the network closed and a fake transport serving rows of each kind caches its
pages exactly as served, its acquisition record holds no exclusion data and its digest
is the manifest's `acquisition_id`, the slices' `rows` sum to `record_count` plus the
excluded sum, and a rerun makes no request and yields the same counts and `corpus_id`.
In `tests/ingest/test_manifest.py`: (M1) a new write emits version 4 with
`excluded_records`, the round trip is exact, and serialization is deterministic; (M2)
version 1, 2 and 3 documents read it as an empty mapping; (M3) a version 4 document
without it raises `KeyError` naming it; (M4) malformed contents are refused — an
unknown kind, a missing NYC 311 kind, a year that is not an integer, a count below 1,
or a non-empty mapping for CFPB; (M5) `corpus_id` ignores `excluded_records`. The nine
existing assertions that pin manifest version 3 — the same nine D46 changed from 2 to
3, seven in `tests/ingest/test_manifest.py` and two in `tests/ingest/test_cli.py` —
assert 4, and no other existing assertion changes. Every test under
`tests/ingest/fetch/`, among them the test that rows reach the cache exactly as served,
`tests/ingest/test_cfpb_normalize.py` and `tests/test_phase2_integration.py` stay
unchanged and passing, and no test touches the network.
*Now — (10) history.* D45 stays unaltered as the historical record of the open
question, and **D1–D49 are unaltered**. §2.4, §2.5, §4.2 and plan Task 6 are not
edited; D50 is their record, as D44 was for its §2.5 refusal and its manifest version.
Task 6's "the reconnaissance measured zero negatives" and §4.2's "no negative
durations observed" are superseded by the 1,904 rows with `closed_date` before
`created_date` measured on 2026-09-28; that is a wall-clock predicate, so the
adapter's own count is at most 1,904. §2.4's "a negligible loss" now describes the
run's effect literally; its "Reject" stands at the adapter, which still raises; and
its reason that a spring-gap value is "worth stopping for" is overridden at the run
level, where a gap row is excluded and counted under its own kind instead. Task 26
rewords the adapter's statements that a negative duration is never dropped and that
both daylight-saving edges are refused, to say that `normalize` never repairs a row
while the run excludes and counts it under D50.
*Now — what D50 authorizes, and what it does not.* D50 decides the policy above and
records plan Task 26, and nothing more. **Authorization to implement Task 26 is
separate from this D50 documentation decision.** D50 does not authorize implementing
Task 26 or any change to code, tests or manifests; real ingestion; any request to, or
download from, either source; model training; threshold fitting; README metrics; or
any Phase 3 wiring. D42's deferred items stay open. **Phase 2 is not complete, and no
document may describe it as complete.**
*Now — the next dependency.* D50; then plan Task 26, test-first, under its own
authorization; then the hardware and resource gate, at which D44's estimated
full-window normalization peak of 4 to 6 GB per source is measured on the first real
run; then a separately authorized real NYC 311 acquisition and run. CFPB's real
ingestion does not depend on D50.
*Scope:* the treatment of D45's three NYC 311 row classes, manifest version 4's one
field, and plan Task 26. **D1–D49 are unaltered.** D50 changes no code, no test, no
requirement and no CI.

**D51 — The resource-readiness framework for real acquisition. The measured environment
and its test gate are recorded. Every real run, offline rerun, archive replay and
continuation is conducted under an external harness that becomes usable only when its
validation passes, with stage-specific, fail-closed network and integrity guards; CFPB
by-ID requests are checked, before each is sent, against the run's own production archive
keys and the active API day, so no separate index is built. Launch requires every sample
of a ten-minute quiet period to show system commit headroom of at least 6,000,000,000
bytes plus the variation observed in it; during a run, triggers request a graceful
interruption at the next safe checkpoint, a trigger and not a cap. Every execution has its
own run identifier, reserved by an exclusive-create claim, and ends in exactly one
outcome, classified from a dedicated terminal artifact verified against its launch file
and the orchestrator's own evidence; an invalid artifact or a demonstrated contradiction
is an unclassified termination, which can never be continued. An eligible interruption may
be continued only by a new execution, under a fresh explicit authorization and a new run
identifier. A CFPB-first sequence of separately authorized stages — A, A′, A2, the further
measurement, B, then C — is frozen; Stage B's peak-memory projection is unsupported, so
Stage B, and with it Stage C, cannot be authorized under this decision. The identity of
the measured process (U6) blocks every measured execution, and no acquisition is
authorized (addendum §2.5; plan §F, §S, §U, Tasks 20, 22–26; D24, D26, D27, D37.2, D40,
D41, D42, D43, D44, D46, D47, D48, D49, D50).**
*Was:* D44 made real ingestion conditional on "a separate authorization of the hardware
and of the run" and recorded a full-window normalization peak of about 4 to 6 GB per
source as an estimate to be measured on the first real run; D50 named that hardware and
resource gate as the next dependency. Task 20's figures, the only measured resource
figures, come from one Windows machine (Intel Family 6 Model 154, 12 cores, Python
3.13.3). The working copy has since moved to a second Windows machine. Nothing in the
repository measures an acquisition's own cost: `ingest.cli` records no memory or timing,
and `RequestRecord` keeps only the response each request finally used, so failed attempts,
retried statuses and backoff waits leave no evidence.

*Decision record (2026-11-16):* The owner conditionally approved D51 on 2026-11-16 as a
specification for future external harness development and for the conduct of separately
authorized executions. The approval applies to the specification only. It does not
authorize harness construction; validation activities; acquisition, offline reruns or
archive replays; any measured execution; or any repository modification beyond the
separately authorized D51 documentation entry. All prerequisites, technical blockers,
owner decisions, platform limitations and stage restrictions documented in D51 remain in
force. Every subsequent activity requires its own explicit authorization, and no
validation result or framework approval automatically authorizes an execution. Stage B and
Stage C remain prohibited under D51. Continuations remain prohibited until the
requirements of E6 are satisfied.

*Now — (0) Conventions.*

**Status words.**
- *Conditionally approved*: the status of everything in D51 since the owner's decision of
  2026-11-16 (17). It means approved as a specification only; it implements, validates and
  authorizes nothing.
- *Proposed*: the status of D51 before that decision. Where D51 still says that it
  "proposes" a procedure or method, it means one that D51 specifies; the word no longer
  means that D51 awaits approval.
- *Measured*: recorded on this machine on the date stated; a snapshot, not a guarantee.
- *Estimate*: not measured.
- *Specified*: a requirement D51 places on the harness, which does not exist.
- *Unverified*: a platform, library or mechanism property D51 relies on; each is listed in
  (18), and none is established.

**No test, validation, source review, measurement or acquisition described in D51 has been
built, run or performed.**

**Processes.** The *orchestrator* is the external harness process that prepares, launches,
observes and classifies an execution. The *child* is a fresh interpreter performing
exactly one measured run, the one measurement D41 allows a process.

**Run kinds.** A run kind is fixed by the execution's authorization and recorded in its
launch file (11.1). It is one of:
- a *real run*: `ingest.cli.main` with source contact;
- an *offline rerun*: `ingest.cli.main` with `--fetch` on a completed acquisition, reusing
  it with zero requests;
- an *archive replay* (13.9).

**Executions and run identifiers.** Every execution — of any run kind, first or
continuation — has its own *run identifier* (11.1). It is named in that execution's own
explicit authorization, is reserved by an exclusive-create *identifier claim* (11.1a), is
never reused, and is never derived from another identifier. A *continuation* is a new
execution, under its own fresh authorization and run identifier, that continues an earlier
execution whose outcome was an eligible interruption (12.7). No execution is ever resumed
under the authorization or run identifier of the execution it continues.

**Two kinds of activity.** A *validation activity* is a unit test, a mock validation of
(4.6), or a Windows filesystem integration test (4.6a). A *measured execution* is a real
run, an offline rerun, an archive replay or a continuation. (4.6a) keeps them apart.

**Time.** Fields ending `_qpc` are performance-counter readings, and they alone order
events across processes (10.1). Fields ending `_utc` are for audit only.

**Evidence classes.** A *contradiction* is well-formed evidence that conflicts with other
well-formed evidence. An *evidence defect*, written **E**, is evidence that is missing,
malformed, unusable, or whose ordering cannot be established. An *unclassified
termination* is the outcome of an invalid terminal artifact or of any contradiction
(12.3).

**Other terms**, each defined once: the refusal and trigger-stop exceptions (4.5); the
active API-day context, the integrity latch and the exit record (4.4); the terminal marker
(6.1); checkpoints, check events and the forced-action rule (9.2, 9.4, 10.3); the delivery
medium (9.3); the launch lifecycle, lifecycle nonce, identifier claim and run-directory
states (11.1a); the claims directory (11.1b); the terminal artifact, launch file, exit
snapshot and classification record (11); the acquisition manifest (11.9); the outcome
categories (12.6).

*Now — (1) The environment, as measured on 2026-10-10.*

These are measured snapshots, not guarantees about the machine at any later time.

| Item | Measured |
|---|---|
| Machine | Alienware 16 Area-51 AA16250; Windows 11 Home Single Language 25H2, build 26200.9168 |
| CPU | Intel Core Ultra 9 275HX (Family 6 Model 198), 24 cores, 24 logical processors |
| Memory | 34,359,738,368 bytes installed; in two snapshots, 8.0–8.5 GB available and 91–92% of a 54,284,386,304-byte commit limit in use, leaving 4,507,217,920 and 4,756,672,512 bytes of commit headroom, with `llama-server`, LM Studio and Ollama holding about 16.3 GB of private memory |
| Storage | `C:\sentinel` on the internal NVMe SSD (NTFS), about 1.7 TB free |
| Network | Cloudflare WARP installed with its tunnel disconnected; no proxy in the environment, WinINet or WinHTTP; direct route over Wi-Fi |
| Power | on AC; the active plan sleeps after 600 seconds idle on AC |
| Python | CPython 3.13.13 at `C:\venvs\sentinel-py313`; the 57 pins of `requirements/train.txt` and `requirements/dev.txt` installed exactly; `requests` 2.34.2 and `urllib3` 2.7.0, with neither brotli nor zstandard installed |
| Venv launcher (measured 2026-10-11) | `C:\venvs\sentinel-py313\Scripts\python.exe` is byte-identical to the standard library's `venvlauncher.exe` (SHA256 `e6c07188984d41b80f822dcdaeaceeb217cdf977f18f79c421a3a53edba895c3`) |

**Stage A cannot start on this evidence.** The latest headroom evidence is two historical
snapshots taken on 2026-10-10: 4,507,217,920 and 4,756,672,512 bytes of commit headroom,
each below the 6,000,000,000-byte minimum on its own, with substantial memory held by
local model servers. No ten-minute quiet-period evaluation under (7) has been performed,
so the launch gate has **not been evaluated**. The snapshots indicate it would not have
passed under the conditions recorded then. Only a fresh quiet-period evaluation,
immediately before launch, can decide it.

Two gates stand before Stage A. Its authorization may be sought only once every
authorization prerequisite of (16) has been met, and they are not: the harness does not
exist, so its validation has not passed, and the process-identity blocker U6 (4.2) is
unresolved. Its launch, once authorized, further requires the full lifecycle of (11.1a).
Neither gate is met now. Nothing here assumes the servers' memory use has changed, and
nothing terminates them.

*Now — (2) The test gate on this machine.*

`LongPathsEnabled` is 0. Under the default temporary directory, 47 tests failed on paths
of 260 to 266 characters. With a process-local short `TEMP` and `TMP` and a short
`--basetemp`, the full suite passed 2,417 tests, and ruff, mypy on both platforms, the
migrations check and the diff checks passed. No code change and no persistent registry or
user-environment change is required. Every test run on this machine uses that
process-local workaround. Paths that a real acquisition under `C:\sentinel` creates are
estimated at about 161 characters at most.

*Now — (3) What remains estimated or unknown.*

**Estimates.** These are all estimates:
- the 4 to 6 GB normalization peak (D44);
- the owner's conservative estimate of about 17 GB of disk for a full CFPB run, covering
  inputs, retained acquisition evidence, intermediate files and corpus. It was supplied
  with the 2026-10-10 migration task, appears in no repository document, and has
  components that are not yet measured;
- every projection from either figure.

**Reconnaissance drivers used in (13).**
- Narrative-bearing records held: 10,050 in A's window, 43,920 in A2's and 2,036,435 in
  the full window, a count fixed by the pinned export bytes.
- Normalization pages: 7, 31 and 731, the largest holding 1,936, 1,993 and 20,524 records.
- API days and their rows: 9 days and 37,422 rows for A; 33 and 150,837 for A2; 733 and
  8,188,022 for the full window. The largest day is 6,958 rows and 2,398,980 bytes in A2's
  span, against 34,091 rows and 9,962,247 bytes in the full window.
- The full window's partitions: 814,385 records in 2024 and 1,222,050 in 2025.
- Archive rows: 945,532 in export #5; 1,038,383 in the largest export by rows (#15);
  8,643,971 in the full window's ten exports before the last; 9,682,354 across all eleven.
- Export bytes: 773,033,865.

**The archive key array.** The production archive's key array, which `read_archive` builds
in the measured child and the by-ID guard of (6.8) searches, holds one signed 64-bit
integer per archive row: 7,564,256 bytes of keys for export #5 and 77,458,832 bytes for
the eleven exports, by arithmetic on those row counts. That arithmetic is not a
measurement of the array's footprint, nor of `read_archive`'s retained or peak cost, which
are measured under (5) and are unknown. The cost of the guard's lookups is unknown.

**Other unknowns.** NYC 311's raw and processed disk requirements, request counts and
duration are unknown. No real acquisition's peak, duration, retry rate or interruption
latency has been measured on any machine.

*Now — (4) The measurement harness, conditionally approved as a specification.*

*(4.1) Components.*

Every measured execution is conducted under an external harness kept outside the
repository, at `C:\sentinel-tools\acqmeter`. Each run record holds a copy of every harness
script and its SHA256.

The orchestrator:
- runs the launch lifecycle of (11.1a): L0 for a continuation, then L1 to L5 for every
  execution;
- prepares an archive replay's inputs (13.9);
- takes the readings of (5) from outside the measured process;
- fires and delivers the triggers of (9);
- verifies the terminal artifact (11.6), runs the postflight and classifies the outcome
  (12);
- writes the run record.

A fresh child performs exactly one measured run of its run kind, with the hooks of (4.3).
The harness does not import `ml/training/measure.py`, whose dependencies would inflate the
measured baseline.

*(4.2) Child process identity — blocker U6.*

The venv's `python.exe` is byte-identical to `venvlauncher.exe` (1). CPython's own
`multiprocessing` treats that file as a redirector, and bypasses it by launching
`sys._base_executable` with `__PYVENV_LAUNCHER__` set
(`Lib/multiprocessing/popen_spawn_win32.py`, bpo-35797). That starting it yields a second,
interpreter process is **unverified** platform behaviour.

**Specified.** The measured child is the interpreter process itself. It is launched as the
base interpreter with `__PYVENV_LAUNCHER__` set to the venv's `python.exe`, or by an
equivalent method, so that each of the following refers to that interpreter process:
- the orchestrator's process handle;
- its once-per-second readings;
- the budget trigger;
- the terminal artifact's `child_pid`.

The child's first event reports its own PID, its `sys.prefix` and the argument vector it
received. Validation must show that the orchestrator's handle and the child's self-report
name the same PID (MV-10).

**U6 is an independent blocker. No measured execution may be authorized or launched until
this requirement is designed, implemented and validated.**

*(4.3) Hooks.*

These are pass-through hooks in the child:

| Hook | Records | Behaviour |
|---|---|---|
| H1 `RequestsTransport` subclass | For every attempt: timing (counter and UTC); the host as taken from the URL H6 validated; the request class of (6.7); the parameters; the sanitized representation of (6.3) — never the raw prepared request, and no request-header value outside H6's permitted pairs; the status; the harness response metadata of (6.3), and no other response-header value; the body size; any transport error and whether it is transient; the child's counters of (5) at the attempt's start and end; and the active context's sequence number, if any. For a by-ID attempt, also: the active API day, the guard's decision under (6.8) and, when a lookup ran, the number of keys carrying the ID, with the matched key's inclusion bit and archive date — never a copy of the archive or its keys. For a refused attempt: the refusal event of (6.1), written before the refusal is raised. | Inherits the session and identity. Passes the same arguments to the real `get`, whose result or exception it returns unchanged, unless a guard of (6) refuses before the call. |
| H2 `HttpClient` subclass | Each logical request, its attempts and retries, validator rejections, and each wait, classed as pacing or backoff and checked against D44's and D47's policy. For each successful logical request, the `response_sha256` of the production `RequestRecord` it returns, and the active context's sequence number, if any. | The same constructor defaults; the original `time.sleep` with the same durations. It overrides `get` exactly once, which MV-5 confirms from the built source. |
| H3 stage wrappers on `ingest.cli`'s fetch, verification, page, roster, write and manifest functions, and on the CFPB fetcher's `discover`, `download`, `read_archive`, `read_export`, `snapshot` and `fetch_day` | The child's counters of (5) at every stage boundary, with the counter brackets of (5.3), and each instance's driver value. | Arguments, pages and results are passed through unchanged, and exceptions propagate. The `read_archive` and `fetch_day` wrappers also do what (4.4) specifies. |
| H4 resolver allowlist | The addresses each permitted host resolved to. | Refuses any host outside the stage's allowlist of (6.6). |
| H5 event log | Every event, appended and flushed as it occurs. Each child event carries a contiguous `seq` (10.3). Request and response details come only from the sanitized representations of (6.3), and exception information only through the descriptors of (11.5). A refusal event is flushed before the refusal is raised. | — |

*(4.4) The active API-day context, cleanup and the integrity latch.*

*The integrity latch.* At startup the child creates one integrity latch. It is owned by
the child, is independent of every context, and holds scalars only:
- `cleanup_failed` and `cleanup_verified`, both initially false;
- a reason, `primary_raised` or `primary_readback_failed`;
- the failing context's sequence number and ISO day.

It holds no reference to a context, a run, the archive, its keys or a day. Each flag can
only change from false to true, and nothing resets it within the process.

*The exit record.* A scalar exit record is overwritten at every `fetch_day` exit. It holds
the sequence number and, as integers, the identities of any exception in flight and of its
`__cause__` and `__context__`.

*The `read_archive` wrapper.*
- It records a weak reference (`weakref.ref`) to the archive it returns, in the run's
  guard state on the H1 transport, and keeps no strong reference. `_Archive` is a plain
  `@dataclass` without `slots` (`ingest/fetch/cfpb.py`). That such an instance supports
  weak references is CPython behaviour, which is unverified (D-5).
- No cache, registry or module-level object holds the archive or its keys.
- It refuses, by raising the refusal exception, if the weak reference cannot be created,
  or if a weak reference from an earlier return in the same run is still alive. Production
  calls `read_archive` once per run.

*The `fetch_day` wrapper*, in this order:
1. It takes the stage-boundary reading of (5).
2. It locates the guarded transport through the run's own client,
   `run.context.http._transport`. It refuses if any of these holds:
   - that object is not an H1 instance;
   - its context slot is occupied;
   - no weak reference is recorded for the run;
   - the weak reference has expired;
   - dereferencing it does not yield, by identity, the `archive` argument. The
     dereferenced object is used only in that comparison and is bound to no name that
     outlives it.

   A refusal records and flushes a refusal event, then raises the refusal exception
   without calling the original.
3. It stores a new context in the slot. The context holds scalar evidence — a sequence
   number and the ISO day — and references to the run, the `day` and the `archive`, never
   a copy of the keys.
4. It calls the original with the same arguments.
5. **Cleanup** runs before anything else, on every exit, normal or exceptional. It has
   three parts:
   - *detach:* the slot, which must hold this context, is emptied;
   - *clear:* the context's references to the run, the day and the archive are set to
     none, and its scalars are kept;
   - *release:* the wrapper's own locals holding the context, the day and the archive are
     set to none.

   The **primary** cleanup detaches and clears through the context's own methods, then
   reads back the slot and the three fields. If the primary raises, or its read-back
   fails, the wrapper:
   1. sets `cleanup_failed` and the reason first;
   2. attempts to set the transport's terminal marker (6.1);
   3. attempts the **fallback** — the four stores made directly and separately, whatever
      the primary achieved — and reads them back;
   4. sets `cleanup_verified` only if that read-back confirms all four.

   Release always follows. On an exceptional exit the wrapper re-raises with a bare
   `raise`, so it holds no local reference to the exception.
6. **Outcomes of cleanup.**
   - *Primary verified:* a normal exit returns the original's result unchanged; an
     exceptional exit re-raises the original exception object, with its class, message,
     `__cause__` and `__context__` unchanged and its traceback extended only by the
     wrapper's own entry.
   - *Primary failed:* the latch is set before anything propagates. A normal exit raises
     the refusal exception, with its integrity attribute set, instead of returning. An
     exceptional exit re-raises the original exception unchanged.
   - *`cleanup_verified` still false:* the wrapper records, where it can, only what its
     read-backs showed, and claims nothing more. Neither it nor any later record states
     that the context is empty or that its references are cleared.

*The outermost harness frame.* It receives the run's normal return or its propagating
exception, holding the exception through its own local only. Before any ordinary
classification it checks:
- the latch;
- the transport's slot, read directly;
- whether the exception is the refusal exception with its integrity attribute set.

Any indication, or any failure to complete these checks, makes the terminal state
`integrity`, whatever the exception's ordinary classification. The frame then:
- **On `integrity`:**
  1. it attempts to write the terminal artifact of (11) — guarded, and with a failure
     ignored — with state `integrity`, the integrity exit status, the latch's scalars, the
     exception descriptor of (11.5), and whether the exception's identity equals the exit
     record's;
  2. it does not modify, wrap, chain, annotate or re-raise the exception;
  3. it ends the process at once with the integrity exit status, without normal
     interpreter shutdown (for example via `os._exit`), so that no finalizer, generator
     `close()` or `atexit` handler runs, and no reading, stage or request follows.
- **Otherwise:**
  1. it takes the exit stage's final reading of (5);
  2. it determines the ordinary terminal state;
  3. it writes the `terminal_begin` event (10.3) and then the terminal artifact;
  4. it exits with that state's status.

Failure to write the artifact never prevents termination. If no valid artifact exists, the
orchestrator classifies the execution as an unclassified termination (11.8).

*(4.5) Harness exceptions.*

The refusal exception and the trigger-stop exception are dedicated, ordinary exceptions.
Each derives directly from `Exception`, and from no class that any handler on the fetch
path catches: in particular not from `TransportError` or `InvalidResponse`, which
`HttpClient` catches (`ingest/fetch/http.py`), nor from `FetchFailed`, which `by_id`
catches (`ingest/fetch/cfpb.py`). Every broad `except BaseException` on the path cleans up
and re-raises (`ingest/cli.py`, `ingest/fetch/acquisition.py`), so either exception
propagates out of the fetcher unchanged and ends the run.

The refusal exception carries an in-process boolean integrity attribute, which is never
persisted. The trigger-stop exception is raised by the checkpoint routine when it acts
(9.4) at a checkpoint that is not the run kind's end checkpoint (9.2).

*(4.6) Mock validation.*

Mock validation:
- runs with the network closed, in a scratch directory outside the repository;
- runs at the commit and harness digests its validation authorization names (4.6a);
- uses the repository's own fake transports, or the real `RequestsTransport` over a
  mounted fake adapter where the session's own behaviour is under test, together with
  synthetic exports, mock authorization fixtures and a fixture-owned fake delivery medium
  (9.3).

Each class is judged by its own standard. Hooked and unhooked runs are not required to
match when a guard, trigger, fault or interruption is triggered on purpose.

| Class | Scenarios | Passes when |
|---|---|---|
| MV-1 Passive equivalence — nothing fires | A clean run; HTTP 429 with `Retry-After`; repeated 5xx; a timeout; a validator rejection; HTTP 403; a 3xx; exhausted retries; a repeated count disagreement; a by-ID resolution the active-day guard of (6.8) permits; the production resume (D46) of an incomplete acquisition, exercised by launching the hooked child directly in mock mode, outside the continuation lifecycle; a rerun of a completed acquisition with zero requests; an archive replay of a completed acquisition's fake exports | Hooked and unhooked runs are identical in request URLs, parameters, application headers and their order, waits requested, acquisition contents with wall-clock fields masked, corpus records and outcomes, and exit status. Every prepared request carries exactly the headers (6.2) permits. The 3xx produces exactly one request and a `FetchFailed`. The replay's spill files and archive facts equal the fetch run's on the same exports. No fault is injected and no production code is substituted. After every `fetch_day` exit, the primary cleanup succeeds and no prohibited reference (MV-5) exists. The latch is clear at the end. Every hooked run ends with a valid terminal artifact (11.6) whose exit status matches its state. Where the unhooked run raises, the hooked run's exception meets MV-5's comparison. |
| MV-2 Network guards | For each stage's allowlist: a fake source or command leading the fetcher to a disallowed host, endpoint, export, parameter, date or by-ID identifier; a redirect to a host outside the allowlist; an altered export body; three independent header cases (a prohibited header such as `Authorization`, an unexpected value of a permitted header such as a different `User-Agent`, an explicit `Host`), each with a fresh session, its own fake adapter and log, and a distinctive synthetic value; two independent cookie cases, matching and non-matching, each with a fresh session, adapter and log | Every refusal is raised before the adapter receives the request, as its log proves. A redirect target is neither resolved nor requested. H7 refuses before the body reaches the fetcher, and no later request is made. Each header case is refused, with its record keeping the header's name, its presence and the refusal, never the value. In the matching cookie case the second request is refused before the adapter, and the cookie's name, domain and path are recorded without its value. In the non-matching case the second request reaches the adapter with no `Cookie` header. In each case, no retained artifact contains that case's synthetic value. Every stage's allowlist refuses each request that only another stage's allowlist permits. |
| MV-3 Active-day guard | Through the real `RequestsTransport` over a fake adapter, with synthetic exports read by the production `read_archive`, and — where production never sends the request — a U3 substitute for `_Run.values`: included and excluded IDs; blank narratives; IDs outside the window; another day and a margin day; IDs absent, repeated (synthetic archive), below the first key, above the last, at the first and last key, and between present neighbours; non-canonical IDs; IDs at or above 2^41; malformed by-ID URLs; no context, two contexts, a non-H1 transport, an expired or mismatched weak reference; a `fetch_day` that returns and one that raises; transient failures of a permitted by-ID request; one synthetic export read under A's and A2's windows; day counts and CSVs for the active day and for another; window snapshots inside and outside a context; discovery and downloads inside a context; a single-day window; a set terminal marker; the exact-day audit's six failure cases (6.9) | Permitted cases reach the adapter exactly as unhooked, retries included. Every other case is refused before the adapter, with the refusal event flushed first, the dedicated refusal exception, no further attempt and no backoff wait. A request made while the terminal marker is set is refused before any other rule. A row inside A2's window and outside A's is permitted only in A2. Each audit failure is a refusal-class integrity event. |
| MV-4 Triggers | Simulated system-floor, budget and disk readings injected at every safe checkpoint of (9.2) and inside every stretch without one, including the corpus write | The child stops at the next safe checkpoint, never inside a request, a non-atomic write or the corpus-write sequence. Evidence is retained. The outcome is classified as (12) defines. A budget trigger during the corpus write ends in "Completion, not compliant". A budget trigger is a refusal, so no continuation is ever eligible after one. Each checkpoint's latency is recorded in counter ticks. The continuation cases MV-4a to MV-4f are in the lifecycle table below. |
| MV-5 Exception preservation | (a) a normal return; (b) production exceptions in hooked and unhooked runs — exhausted retries (`FetchFailed`, no cause), a non-transient transport error (a `FetchFailed` caused by a `TransportError`), HTTP 403, and a synthetic `FieldDisagreement`; (c) an H6 refusal (U3 substitute) | (a) The slot is empty, the context's references are cleared, its scalars are retained, the latch is clear, and no prohibited reference exists. Prohibited means no reference to the archive, its keys or the day held by: the transport's slot; the context's fields; the locals of the H3, H1 or H6 frames, whether live or retained by a traceback; or any harness record, registry or module object. Legitimate holders are production frames and objects, test-fixture objects registered by identity, and the inspecting code's temporaries. The evidence, kept separate, is source review, direct assertions on slot and fields, direct inspection of harness frame locals, and identity checks through the weak reference; `gc.get_referrers` is supporting only, and a traced allocation peak below `8 × len(keys)` during one check is a diagnostic only. (b) Within the hooked run, the exit record's identity equals that of the object the outermost frame receives. Across runs, the class, `str()`, `args`, cause and context are compared in-process against fixture-supplied, synthetic expectations, and only booleans and descriptors are persisted. The production- and fixture-owned traceback entries, from `ingest.cli.main` to the raise site, are equal in code location and order. The only additional entries are: two H3 entries (the `ingest.cli` fetch wrapper and the `fetch_day` wrapper) in every case; one H2 `get` entry in the first three cases; and one H1 `get` entry in the `TransportError`'s own traceback in the second. (c) The dedicated refusal exception; the refusal event, naming condition 5 of (6.8), flushed before the outcome; zero adapter attempts; no backoff; a verified cleanup; and an ordinary refusal exit status. No unhooked counterpart is required. |
| MV-6 Cleanup failure | Each case on a normal return and on an exceptional exit, with the mock child run as a subprocess. The fixture keeps its own adapter log and an `atexit` marker. (a) The primary fails and the fallback verifies. (b) Every cleanup store fails. (c) Every harness event write fails as well, combined in turn with (a) and (b). | (a) and (b): the latch is set before propagation, as event order shows and as source review confirms the latch is set only on the wrapper's cleanup path. The outermost frame records the latch it observed. For (b), records state only read-back results and never claim cleanup. A normal return raises the integrity refusal instead of returning. An exceptional exit preserves the original exception's identity and descriptors. The state is `integrity`, with the integrity exit status. There are no adapter entries after the fault, no corpus or manifest file and no `atexit` marker. Where a cleanup failure follows a failed check, its cleanup and latch events are among those (10.3) permits as P2. (c): the integrity exit status still results. If the artifact write succeeded, the outcome is Refused; if it also failed, the outcome is an unclassified termination. No harness event is required to exist. |
| MV-7 Observation binding | A fixture-owned delivery medium that logs, for every call and by calling process, its own count and the exact tuple returned. Adversarial (U3) defective children that insert a published but unreturned request, omit a returned one, substitute one ID for another, mutate the list after its digest, or read the medium twice. Medium results that raise, are non-tuples, contain non-integers, contain IDs below 1 or contain duplicates. After a failed check (U3): a further check event, a request attempt, or a later-stage boundary event. | A-1: the check events whose kind involved a medium call match the fixture's entries one for one and in order; each `observed_request_ids` equals the sorted tuple returned; there is no **child** access to the medium outside the single read function, and orchestrator operations M-op1 to M-op3 are recorded separately and excluded. A-2: a medium that returns different data to any second read does not change A-1's result. Every adversarial variant fails A-1, and so fails P-T6. A mutation after the digest is also contradictory under (10.3). Each failure kind produces the event of (10.7): it is the last check event, followed only by the events (10.3) permits, with exactly one fixture entry where its kind requires one and no acting decision; the state is `refused` and the outcome is Refused. Each adversarial sequence after a failed check is an unclassified termination. |
| MV-8 Timing and visibility | Causally ordered exchanges between the two processes; a request delivered before `check_started_qpc`; a mock-only delay injected between `check_started_qpc` and the read, with a delivery inside it; a stale request planted before launch; a medium written by a non-orchestrator process; deliveries completing before `read_started_qpc`, starting after `read_returned_qpc` (defective child) and overlapping the read; V3 alone; V3 together with a digest mismatch, an O1 omission or an ordinal gap; V3 with P-T6 marked not established | P-T1: no causally later reading is ever the smaller. P-T2\*: the request completed before the boundary is returned, and both outcomes are accepted for the in-delay delivery. P-T4 and (9.3)(a) to (d): the stale request is detected by M-op2 and the child is not launched; the foreign write is refused; neither M-op2 nor the child's read removes a request; the orchestrator never publishes before launch. Classifications are V1 compatible, V2 contradictory and V3 E. V3 alone is E. V3 together with a contradiction is an unclassified termination. V3 with P-T6 unestablished is E. |
| MV-9 Terminal artifact and classification | One valid artifact; envelope failures (missing, partial only, wrong magic, truncated, appended bytes, wrong length, wrong digest, a non-ASCII byte); payload failures (non-canonical form, a duplicate key, a float, NaN, a missing or extra field); a valid envelope with contradictory content (run ID, launch digest, status against state, status against exit code, PID, latch against state, trigger against exit snapshot, a contradicting child event); launch-file faults (missing at T1, T2 or TV2, changed after T1, changed between T2 and TV2, non-canonical, wrong run ID, authorization reference, harness digest or run kind, a wrong or duplicate status mapping); a synthetic secret in an exception's message, `args`, cause and context; an invalid class name; a deep chain; one case per mapping rule and branch of (12.5); one violation of each invariant of (12.2); the evidence files modified after TV7 and before re-check 1; the same files modified after the classification record is written; post-exit events appended after a valid artifact; the child killed abruptly | Only the valid artifact, including the case with later appended events, is classified under its state. Every other artifact case is an unclassified termination, except a launch-file fault detected at T1, which prevents launch. The secret appears in no artifact, log or output. The invalid class name and the deep chain map under rule 6. **Before initial classification:** re-check 1 finds E-pre; the first mapping records an unclassified termination, and the classification record names the mismatch. **After classification:** a later re-check finds E-post; the classification record is unchanged; the event records both check times and claims no time of modification; the file is preserved; the evidence is marked compromised; acceptance, prerequisite use and continuation are refused. The lifecycle and continuation cases MV-9a to MV-9z30 are in the table below. |
| MV-10 Child identity | The launch method of (4.2) | The orchestrator's handle and the child's self-reported PID are the same process, and `sys.prefix` is the venv's. |
| MV-11 Sample attribution | Orchestrator samples whose brackets fall inside, across and between child boundary brackets | Attribution follows (5.4) exactly, and samples that are ambiguous at a boundary are excluded from every stage. |

**Lifecycle, identity and continuation cases (Level 1).** These use mock authorization
fixtures, which authorize nothing.

| # | Case | Passes when |
|---|---|---|
| MV-4a | A first execution | It runs under its own mock authorization and unique run identifier; its claim is created and owned; its directory is created exclusively at L1; TV2 and TV5 bind its identity. |
| MV-4b, MV-4c | A continuation after a reviewed interruption, and after an ordinary one | It runs under a fresh mock authorization and a new run identifier, with a `continues` binding; L0 to L5 pass; D46 to D49 continuity holds; its final acquisition (wall-clock fields masked), corpus records and outcomes equal an uninterrupted hooked run's. |
| MV-4d | The prior evidence is preserved | The prior `terminal/` directory is byte-identical before and after; its run record changed only by appended audit and evidence-status entries. |
| MV-4e | After L0 passes, a change before L4 to the acquisition directory, the authorization file, the commit, the harness or the package freeze | Refused at L4; a launch-refusal record is written; the run identifier is consumed. |
| MV-4f | A later continuation after MV-4e's launch-refused directory | E3 does not count the launch-refused directory. |
| MV-9a to MV-9l | Refusals before launch: (a) the original authorization or run identifier; (b) a missing or mismatched authorization; (c) a missing, mismatched or unverifiable prior classification; (d) a prior refusal, unclassified termination, non-compliant completion or budget refusal; (e) compromised prior evidence; (f) a missing or mismatched owner's review record; (g) a reused identifier, refused at the claim create before any directory check, or a pre-existing directory; (h) a differing command, run kind, window, corpus root, commit, harness digest or package freeze; (i) an acquisition directory that differs from the prior manifest; (j) a second continuation of one prior run, or a concurrent execution; (k) a prior execution with no classification record; (l) a child writing into another execution's directory (refusal-class) | Each is refused, and each failed check is named in the record. |
| MV-9m | An L0 failure | A preflight-refusal record in `_preflight/`, named by its own SHA-256; no run directory. |
| MV-9n | A foreign preparing or indeterminate directory naming the same prior run | E3 refuses. |
| MV-9o, MV-9p, MV-9q, MV-9w | Manifest policy: missing, added, renamed and changed entries; an added empty directory; a missing, unavailable, non-canonical or digest-mismatched prior manifest; a wrong count or sum; duplicate or unsorted paths; a missing parent; a wrong `run_id` or `acquisition_dir`; every rejected path form of (11.9); the prior manifest changed after classification | Each fails as (11.9) and (12.7.5) define; the last is E-post. Ordinary names are accepted. These establish the validator's decisions only, and nothing about Windows. |
| MV-9r | The prior `terminal/` directory across L0 to L5 | Byte-identical. |
| MV-9s | No E6 acceptance: every continuation, whatever else holds | Refused at L0, with E6 among the recorded failures and a preflight-refusal record. No continuation reaches L1 while O-2 is unresolved. Once E6's evidence is defined, a second variant — acceptance present at L0 but absent at L4 — must be refused at L4. |
| MV-9t, MV-9u, MV-9v | F1 to F4; a launch-refusal record that cannot be written; a refusal whose `_preflight/` record cannot be written safely | F1 to F4 are recorded as found, launch-refused, with the identifier consumed and no outcome. The second leaves the directory indeterminate. The third launches nothing and reports that no durable refusal record exists. |
| MV-9x, MV-9y | An E-post event; a launch-start argument vector that differs from the child's first event | The event records both check times and claims no modification time. The mismatch is contradictory at TV2. |
| MV-9z1, MV-9z2 | Another instance presents a valid-looking preparing directory; the orchestrator restarts between L1 and L4 | E1b fails in both: the in-memory nonce does not hash to `nonce_sha256`, or no nonce is held. Nothing is adopted or launched; the directory is indeterminate to every observer. |
| MV-9z3 | Reserved. Its former content is covered by MV-9z30. | — |
| MV-9z4 | F0-a to F0-f | Each is recorded as its own case, with the identifier's status as (11.1a) states. |
| MV-9z5 | The L0 and L4 fresh manifests | Their bytes are persisted where (11.9) states, and match their recorded digests. |
| MV-9z6, MV-9z7, MV-9z8 | A search of every artifact for the raw nonce; `preparing.json` replaced with a different `nonce_sha256`; a mismatch in `preparing_record.A_D`, `launch_file.authorization_ref`, `launch-start record.A_D`, or between the claim's and the preparing record's `A_D` or `nonce_sha256` | The raw nonce appears nowhere. The creator fails E1b and writes a launch-refusal record. Each mismatch detectable at L4 is refused there; a launch-start mismatch is contradictory at TV2. |
| MV-9z9 | Two lifecycles reserve the same identifier at once | Exactly one owns the claim. The other is refused as F0-a and creates no directory. |
| MV-9z10 | Another lifecycle attempts the same identifier after the first has reserved it, and writes a preflight-refusal record | The second lifecycle's claim create fails (F0-a). Its record never reserves or consumes anything. The first lifecycle is unaffected. |
| MV-9z11 | A crash after the claim file is created and before its content is complete | The identifier is consumed. A later lifecycle naming it is refused as F0-a. No directory is ever created for it. |
| MV-9z12 | A claim owned by another lifecycle, with a valid-looking preparing directory | E1b fails; nothing is adopted. |
| MV-9z13 | The five provisioning cases of (11.1b); an unreadable run directory | Each provisioning case behaves as its table row states. The unreadable directory refuses the lifecycle (F0-d). |
| MV-9z14 | An uppercase or mixed-case new identifier; each reserved name in any case; an identifier that collides by case with an existing claim, run directory, `run_id`, `continues.prior_run_id` or `prior_run_id` | The first two are refused before any filesystem action. A collision with an existing claim is detected at E1a step 3, before any create. Each other collision is refused as already used. |
| MV-9z15 | Under `run-records`: a stray file; a directory with a non-conforming name; a reparse point | The lifecycle refuses in each case. |
| MV-9z16 | The seven steps of E1a, with a fault injected at each | Each failure is classified as (12.7.5) states. A failure at step 5 leaves the claim in place, creates no run directory, and is recorded as F0-c. No step runs out of order. |
| MV-9z17 | `_claims` holding, in turn: a valid complete claim; a valid empty claim; a valid claim with invalid content; a claim in another case; a subdirectory; a file with a wrong suffix; a file whose identifier part is non-conforming; a reserved name | The first four each mark their identifier as used. Each of the others makes the enumeration untrustworthy, and the lifecycle refuses before creating a claim. |
| MV-9z18 | A launched directory with a valid `launch.json` whose `continues.prior_run_id` names P, and a matching valid `continuation-eligibility.json` | P is marked as already continued. A continuation naming P has E3 among its recorded failures; a lifecycle naming an unrelated prior run does not, on account of this directory. |
| MV-9z19 | A launched directory whose `launch.json` is malformed, non-canonical or truncated | The lifecycle refuses (F0-d). The record is never read as "no binding". |
| MV-9z20 | A launched directory with a valid `launch.json` whose `continues` is an object, and a `continuation-eligibility.json` that is malformed, missing, or names a different `prior_run_id` | The lifecycle refuses (F0-d) in each variant. |
| MV-9z21 | A launched directory with a valid `launch.json` whose `continues` is null, and no `continuation-eligibility.json` | Accepted as a first execution. With a `continuation-eligibility.json` present, the lifecycle refuses. |
| MV-9z22 | A preparing directory with `preparing.json` valid and `launch.json` absent (Case A) | A first-execution lifecycle is not refused at E1a step 6 on this ground. A continuation lifecycle is refused at step 6 (F0-d). A present but invalid record refuses both. |
| MV-9z23 | A launch-refused directory holding a partial `launch.json` left by F2 | Not a refusal. The directory's name marks its identifier as used, and it does not count under E3. |
| MV-9z24 | A launch-refused directory with a valid `launch.json` whose `continues.prior_run_id` names P | P is recorded as an observed historical reference. This directory does not put E3 among the failures of a continuation naming P. Its own identifier stays consumed. |
| MV-9z25 | A preparing directory with `continues` null and a `continuation-eligibility.json` present | The lifecycle refuses (F0-d). |
| MV-9z26 | A preparing directory whose `continues.prior_run_id` and `prior_run_id` differ | The lifecycle refuses (F0-d). |
| MV-9z27 | A preparing directory with `continuation-eligibility.json` present and `launch.json` absent | The lifecycle refuses (F0-d). |
| MV-9z28 | An indeterminate directory with contradictory bindings | The lifecycle refuses (F0-d) in each variant. |
| MV-9z29 | E3's decision logic (Case B): a preparing directory, and separately an indeterminate one, with a valid `launch.json` naming P and no `continuation-eligibility.json`. Two scanning continuations, one naming P and one naming an unrelated Q. | Both pass E1a step 6. The continuation naming P has E3 among its recorded L0 failures. The continuation naming Q does **not** have E3 among them on account of this directory. Both are still refused at L0, because E6 is unresolved; neither reaches L1 or L4. |
| MV-9z30 | O-3, through a first execution: an otherwise eligible first-execution lifecycle, with a foreign preparing directory, and separately a foreign indeterminate one, present | The lifecycle is not refused at E1a on this ground. It is refused at L4 under (8), with a launch-refusal record. |

**E6-dependent cases.** Any case that requires a continuation to pass L0 — MV-4b, MV-4c,
MV-4e and MV-4f, the second variant of MV-9s, and the passing branch of MV-9z5 — cannot be
built until E6's evidence is defined (O-2). Every other case, including every continuation
refusal case, can be: L0 records every check that fails, so a refusal case is judged by
which checks appear among the recorded failures.

*(4.6a) Validation activities and measured executions.*

**Three validation levels.**
- **Level 1, unit and mock tests.** These establish the policy decisions, the fail-closed
  logic and the lifecycle rules. They establish nothing about operating-system behaviour.
- **Level 2, Windows filesystem integration tests** (class MV-W). These use actual Windows
  filesystem calls on real fixtures. They are the only evidence that can resolve the
  filesystem parts of D-5.
- **Level 3, authorization of a measured execution.** It is possible only after the
  applicable Level 1 and Level 2 evidence exists (16).

**Validation activities** are the unit tests, the mock validation of (4.6) and class MV-W.
- **Their own authorization.** They may run only under their own explicit **validation
  authorization**, which names the commit, the harness digests and the test classes to be
  run. D51 does not grant it.
- **No production data and no network.** They never read or write `data/`, including
  `data/recon/`, any acquisition directory, any corpus or any run record. They never
  contact a source or the network.
- **Isolated fixtures only.** They run only in isolated temporary directories outside the
  repository, on fixtures they create themselves. They change no tracked file, and they
  remove only what they created.
- **No system change.** They change no Windows privilege, system policy, registry setting,
  power setting or security configuration to make a test pass.
- **An unavailable fixture has not passed.** A test whose fixture cannot be built is
  recorded as **unestablished**. It is never counted as passing, skipped as passing, or
  replaced by a mock.
- **They are not runs.** A validation activity is not a measured execution, produces no
  outcome under (12), and consumes no run identifier.

**Measured executions** remain prohibited until every validation prerequisite that applies
to them has passed, been reviewed, and been recorded at the authorized commit and harness
digests — and each then still needs its own explicit authorization under (13) and (16).
**Completing any validation activity authorizes nothing.**

**Class MV-W.** Each test uses real fixtures in an isolated temporary directory.

| # | Fixture | Acceptance criterion |
|---|---|---|
| MV-W1 | Exclusive creation of a directory, and of a file: create a new one; create it again; two processes create the same name at once | The first create succeeds. Every later or losing create fails without altering what exists, and exactly one of two concurrent creators succeeds. This establishes concurrency only, not durability. |
| MV-W2 | Reserved names, with and without extensions and in mixed case; names ending in a dot or a space. In the fixture's stand-ins for `run-records` and `_claims`: a directory named each reserved name, and a file named `<name>.claim`, in lowercase, uppercase and mixed case | The test records what the filesystem does with each. The rejection lists of (11.1) and (11.9) must cover every name the filesystem alters, refuses or maps to a device. If the filesystem treats any further name as reserved, the lists are corrected by a later document change. |
| MV-W3 | Two names differing only in ASCII case, as files, as claim files and as directories | The test records whether both can exist in the fixture's directory. Either way, the comparison rule of (11.1) must report them as one identifier, and the manifest's case-collision rule must reject any pair the filesystem treats as one name. |
| MV-W4 | A symbolic link to a file, a symbolic link to a directory, a junction, and any other reparse-point type the test can create, each as an entry and as an ancestor | Each is detected from the opened handle and makes the manifest unavailable. Capture never follows one. |
| MV-W5 | An entry whose final on-disk path lies outside the acquisition directory, reached through each fixture of MV-W4 | The handle-based final path shows the escape, and the manifest is unavailable. No path-string check is relied on. |
| MV-W6 | A captured directory, then separately: a file added, removed, renamed and changed in content; an empty directory added and removed | A second manifest differs from the first in exactly the affected entries. |
| MV-W7 | During capture: a file is modified, replaced, renamed or deleted; an entry is added; an ancestor is swapped for a reparse point | Capture reports the manifest unavailable in every case. It never returns a manifest that mixes two states. |
| MV-W8 | Rename without overwrite onto an existing file, and `fsync` of the files D51 writes, including a newly created file | Rename fails and leaves the destination unchanged. The test records what the calls return. It states explicitly that it does **not** establish the durability of a file's directory entry across an OS crash or power loss, and performs no such test. |
| MV-W9 | Claims-directory path integrity. (a) Each of the four components — the fixture's stand-ins for `C:\sentinel`, `data`, `run-records` and `_claims` — in turn as a junction and as a symbolic link. (b) While all four handles are held: an attempt to rename, delete and replace each component in turn. (c) A claim created relative to the verified `_claims` handle, while a path-based create of the same name would have been redirected. (d) A handle that cannot be opened in the required mode. (e) A claim whose final path differs from the expected one. (f) The same traversal for `_preflight`; `_preflight` absent with `run-records` populated. (g) A claim read back through a handle verified by the traversal. | (a) Each reparse point is detected from its own handle, and the lifecycle refuses before any claim is created. (b) **Every attempt fails**, and the directory is unchanged; a run in which any attempt succeeds does not pass, whatever is detected afterwards. (c) The claim is created in the directory the handle refers to, and nowhere else. (d) The lifecycle refuses, and no claim file exists afterwards. (e) Refused after creation, as F0-c; this part is evidence for the additional safeguard only and cannot stand in for (b). (f) The record is written under verified handles, and `_preflight` is created without triggering the claims refusal; with an unsafe ancestor, no record is written and the absence of a durable record is reported. (g) The read-back reads the file the lifecycle created. |
| MV-W10 | Process-crash persistence: a process creates a claim and is killed (a) before writing content, (b) mid-write, (c) before closing | The claim file exists afterwards in every case, and a second creator fails. The test records the file's state as found. It makes no claim about OS crashes. |
| MV-W11 | A stand-in `_claims` directory containing a regular file, an empty file, a subdirectory, a junction, a symbolic link to a file and to a directory, and files with mixed-case names and suffixes | Handle-based enumeration reports every entry, with its type and its exact name. No entry is missed, and no reparse point is followed. The harness's classification of each entry matches the contract of (11.1b). |

- A symbolic-link fixture may need a privilege the account lacks. The harness obtains
  none. If a fixture cannot be built, its case stays unestablished (O-4).
- If MV-W9(b) cannot be established on this platform, the protection is unestablished, D-5
  stays open, and a later decision is needed. D51 offers no weaker substitute.
- The results are evidence for this machine, this Windows build and this volume only.
- Where a recorded behaviour differs from what D51 assumes, the rule is corrected by a
  later document change. The test is never weakened to fit.

*(4.7) Isolation of mock-only behaviour (U3).*

Any substitute for production code (such as `_Run.values`), injected fault, injected
delay, defective-child variant or in-process comparator:
- exists only in the mock test suite, outside the harness scripts whose digests an
  execution's authorization names;
- runs only with synthetic exports, a fake adapter and a closed network;
- never appears in MV-1, or in any measured execution.

At the start of a measured execution, the harness refuses unless `_Run.values` is, by
identity, the function `ingest.fetch.cfpb` defines at the authorized commit, and unless no
mock-test module is loaded. Every artifact of an adversarial test is marked as
adversarial. If this separation cannot be maintained, the affected cases are tested only
by calling H6's decision function directly, and the gap in integration coverage is
recorded.

*(4.8) Status and recorded limits.*

**None of these tests exists. Every criterion above is specified, not established, and
each test is described as passing only once the harness has been built and the test has
been run under a validation authorization.** Until the applicable validation passes, no
measured execution may be conducted under the harness, and none may be conducted without
it. Simulated readings exist only in mock mode, and the harness refuses a measured
execution that would use them.

Mock mode records the hooks' timing overhead. It builds no index of any kind, and no mock
reading is a measurement of any production cost.

The harness's recorded limits stand: response sizes are decoded sizes; NYC 311's fetch
sub-stage times are inferred from the request timeline; and writes outside the repository
and the run's temporary directory are not observed.

*Now — (5) Memory readings, stages and sample attribution.*

*(5.1) Counters.*

Process readings are `GetProcessMemoryInfo` with `PROCESS_MEMORY_COUNTERS_EX` on the
measured child (4.2). The definitions below are as Microsoft documents them and D40.2
adopted; they are not re-verified here (D-5).
- `WorkingSetSize` is the **current resident working set**.
- `PrivateUsage` is the **current private commit charge**, whether resident or paged out.
- `PeakWorkingSetSize` and `PeakPagefileUsage` are the **lifetime peaks** of those two;
  neither resets at a stage boundary.

A read-only file mapping is expected to add pages to the working set and nothing to the
commit; D51 relies on the counters, not on that expectation.

System readings are `GetPerformanceInfo`'s `CommitTotal`, `CommitLimit` and
`PhysicalAvailable`, multiplied by its page size. System headroom is `CommitLimit` minus
`CommitTotal`.

*(5.2) Who reads.*

The orchestrator reads the four process counters, the three system values and free disk on
C: once per second, from the child's launch to its exit, from outside the child. The child
reads its own four counters at every stage boundary below, and at the start and end of
every request attempt.

A current reading, however frequent, is never evidence of a stage's exact peak. A
difference of current readings is memory held across an interval. A difference of lifetime
peaks is not a stage's usage (D41).

*(5.3) Stages.*

The stages are contiguous: each begins at the boundary reading where the previous one
ends.

*For a CFPB real run with a new acquisition,* the stages are:
- imports, from interpreter start to after imports (`B0`);
- setup, to `fetch_into_cache`'s entry;
- discovery, to the first `download`'s entry;
- one download stage per export, from its `download`'s entry to the next one's, or to
  `read_archive`'s entry;
- one stage per export but the last, from its `read_export`'s entry to the next one's,
  including its sort;
- the last export and the merge, from the last `read_export`'s entry to `read_archive`'s
  exit;
- the start snapshot, to the first `fetch_day`'s entry;
- one stage per API day, from its `fetch_day`'s entry to the next one's or, for the last,
  to the end `snapshot`'s entry. It includes the caching and journaling of its page and
  the by-ID guard's lookups of (6.8), which read the archive's existing keys and must
  retain nothing after the stage's `fetch_day` returns;
- finishing, to `fetch_into_cache`'s exit, covering the end snapshot, reconciliation,
  completion of the record and release of the fetcher's state;
- verification, to `iter_cached_pages`' entry;
- one stage per normalization page, from the request for that page to the request for the
  next or, for the last, to the pass's end;
- the post-normalization checks, to `clear_corpus`' entry;
- clearing the corpus, to the first partition writer's entry;
- one stage per partition write (record or outcome), from its writer's entry to the next
  writer's, or to `build_manifest`'s entry;
- the manifest, to `write_manifest`'s exit;
- exit, to the final reading.

The archive keys the guard searches are built within the archive stages — each export's
sorted part in that export's stage, and the merged array in the last export and the merge
— and are held from `read_archive`'s exit until finishing releases the fetcher's state.
The guard adds no copy.

*An offline rerun* has no fetch stage and sends no request: setup runs from `B0` to
verification's start. *An archive replay's* stages are: imports; setup, to the first
`read_export`'s entry; one stage per export but the last; the last export and the merge,
to `read_archive`'s exit; and exit. *NYC 311's* stage list is not yet defined, and is a
prerequisite of Stage C's own authorization (18).

Every boundary reading is bracketed by counter readings, `boundary_pre_qpc` immediately
before the counters are read and `boundary_post_qpc` immediately after; `*_utc` is kept
for audit.

*(5.4) Attribution of orchestrator samples.*

Each orchestrator sample is bracketed by `sample_pre_qpc` and `sample_post_qpc` around its
reads. A sample is attributed to stage `s` only if:
- `sample_pre_qpc` is later than the `boundary_post_qpc` of `s`'s opening boundary; and
- `sample_post_qpc` is earlier than the `boundary_pre_qpc` of `s`'s closing boundary.

A sample whose bracket overlaps a boundary bracket is **boundary-ambiguous**. It is
retained and used for triggers and run-level maxima, but is attributed to no stage.
Excluding it can only lower a stage's lower bound, never invalidate it. Attribution relies
on P-T1 (18). UTC is never used to attribute a sample.

*(5.5) Stage quantities.*

For each stage `s`:
- `S_s` is the current commit at its start.
- `Δ_s`, its **retained change**, is the current commit at its end less `S_s`.
- Its **peak** `P_s` is exact when the lifetime peak at its end exceeds the lifetime peak
  at its start; `P_s` is then that end-boundary lifetime peak. Otherwise it is an
  interval, bounded below by the largest current reading attributed to the stage and above
  by the lifetime peak at its start.
- Its **transient excess** is `X_s = P_s − S_s`, exact only when `P_s` is.

Because the stages are contiguous, `S_(s+1) = S_s + Δ_s`.

For a repeated stage type: `m` is its number of instances; `r_i` is instance `i`'s driver
value; `R_k` is the cumulative driver through instance `k`; `R` is the total; `C_k` is the
cumulative retained change through instance `k`; and `Q = C_m`. For normalization pages,
`p_k` is page `k`'s row count and `n` is the manifest's `record_count`.

Every quantity is computed the same way for the working set, from `WorkingSetSize` and
`PeakWorkingSetSize`, and reported beside the commit.

*Now — (6) The stage-specific guards.*

*(6.1) H6.*

- **What it permits.** H6 permits a request only if it matches the stage's allowlist
  exactly and, for a CFPB stage, the request-class and by-ID rules of (6.7) and (6.8). It
  refuses anything else before the transport adapter receives it.
- **The terminal marker.** While the guarded transport's terminal marker is set, H6
  refuses every request before evaluating any other rule. Only H3's cleanup sets the
  marker, on a cleanup failure (4.4), and nothing clears it within the process. It is
  defence in depth: fail-closed termination does not depend on it.
- **How a refusal is recorded.** A refusal is recorded before it is raised:
  1. H1 derives the request's sanitized representation (6.3);
  2. H5 appends and flushes a refusal event holding it, the rule that failed, and the
     context's sequence number and active day if a context exists;
  3. only then is the refusal exception raised.

  By-ID refusals identify the rule by its condition number in (6.8). Every other refusal
  records its rule as unavailable until D-6 is resolved. The check returns a decision and
  raises nothing itself; the exception is raised from a frame that holds no reference to
  the archive or its keys.
- **Why a refusal ends the run.** Because the refusal precedes the real transport's call,
  the adapter receives no attempt. Because the exception is not a `TransportError`, it
  leaves `HttpClient.get` at once, with no further attempt and no backoff wait; the pacing
  wait that precedes every attempt has already been taken, as in production.
- **If the refusal event cannot be written,** the refusal is still raised, and the missing
  event makes the run a measurement that could not be completed as specified.

*(6.2) Headers.*

- **What the request carries.** The application supplies exactly one header, Sentinel's
  `User-Agent`. `requests` adds exactly `Accept-Encoding: gzip, deflate`, `Accept: */*`
  and `Connection: keep-alive` in this environment. `Host` is not part of the prepared
  request: the connection layer adds it from the URL's host when the request is sent,
  after H6 has checked that URL.
- **What H6 checks.** Before H1 hands a request to the real transport, H6 prepares it from
  the transport's own session, exactly as the transport's call would, and checks that
  actual prepared request, unredacted and in memory.
- **What it refuses.** H6 refuses any other header or any different value: among them
  `Authorization`, `Proxy-Authorization`, an app token, an explicit `Host`, and an
  outgoing `Cookie`.
- **Cookies.** The session's cookie jar gains cookies only from responses, so no cookie
  can be added between the check and the call. An incoming `Set-Cookie` is not itself
  refused. The session keeps such a cookie, as `requests` always does. Whether a later
  request carries it depends on its domain, path, `Secure` flag and expiry. If a prepared
  request carries it as an outgoing `Cookie` header, H6 refuses that request. Provided H6
  checks every outgoing request this way, no stored cookie is ever sent.

*(6.3) Sanitized records.*

- **Order of work.** H6 decides on the actual prepared request before anything about it is
  recorded; only then does H1 derive the sanitized representations that H1 and H5 persist.
- **The request.** The representation keeps the URL and every header's name. It persists a
  header's value only when that exact name-and-value pair is one H6 permits: `User-Agent`
  with Sentinel's value, `Accept-Encoding: gzip, deflate`, `Accept: */*` or
  `Connection: keep-alive`. Every other request-header value is redacted before
  persistence, keeping only the name and the fact that a value was present. The host
  recorded is the one taken from the URL H6 validated, never one from a header.
- **The response.** H6's request allowlist does not apply. The harness persists only its
  **harness response metadata**: the status; the values of `retry-after`, `content-length`
  and `content-encoding`; and, for each `Set-Cookie`, the cookie's name, domain and path.
  It never persists a `Set-Cookie` value or any other response header's value.
- **Redaction applies everywhere.** It happens in memory before any record, log line or
  event is written, and applies equally to run records, the event log, refusal messages,
  captured output, test artifacts and the terminal artifact. H6 never evaluates a
  sanitized copy. The session's jar and the raw prepared request are never persisted.
- **Production acquisition evidence is unchanged by D51.** It holds no request headers.
  Among response headers it persists only `etag` and `last-modified` in each CFPB export's
  pin (`download` in `ingest/fetch/cfpb.py`), and NYC 311's `x-soda2-truth-last-modified`
  and `x-soda2-data-out-of-date` in its snapshots (`ingest/fetch/nyc311.py`). It reads
  `content-length` and `retry-after` only to check a body's size and to time a retry.
- **No existing repository test covers cookie handling or this redaction.** MV-2 is
  intended to cover them once built and passed.

*(6.4) Redirects are never followed.*

`RequestsTransport` sends every request with `allow_redirects=False`, so no redirected
request is ever prepared or sent, and `HttpClient` refuses every 3xx as permanent (D47),
which is a `FetchFailed`. The repository's
`test_the_real_transport_never_follows_a_redirect` drives `requests`' own session
machinery to one request and no second. H4 would in any case refuse to resolve a redirect
target outside the allowlist.

*(6.5) H7.*

H7 compares each pinned artifact's byte count and SHA256 with its pin when its download
attempt returns, before the body reaches the fetcher, and refuses on any difference, so no
later request is made.

*(6.6) Allowlist files.*

Each execution's allowlist is a file generated from that execution's authorization alone
and named in it by digest. For each permitted request, it states whether the request is
permitted outside or inside an active `fetch_day` context. The harness refuses to start
without the file, or if its digest does not match. No stage inherits, merges or defaults
to another stage's allowlist, and H4 resolves only the hosts the allowlist names.

*(6.7) CFPB request classes.*

H6 classifies each prepared request by its exact shape, and refuses a request of no class.

*Outside an active `fetch_day` context*, the permitted requests are:
- `GET` the reading room,
  `https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/`,
  with no parameters;
- `GET` each export of the stage's roster at its exact
  `https://files.consumerfinance.gov/f/documents/` URL, with no parameters;
- the **window snapshot**: `GET`
  `https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/` with
  exactly `date_received_min` and `date_received_max` equal to the window's start and end,
  `size=1`, `no_aggs=true` and `no_highlight=true`.

*Inside an active `fetch_day` context*, the permitted requests are:
- the **day count**: the same count, with both dates equal to the active API day;
- the **day CSV**: exactly `date_received_min` and `date_received_max` both equal to the
  active API day, `format=csv` and `no_aggs=true`;
- the **by-ID request**: the search API's URL followed by a canonical decimal Complaint
  ID, with no parameters, under (6.8).

The active API day must be a permitted API day: from the day before the window to the day
after it.

These phases are exactly those in which the fetcher issues each request
(`ingest/fetch/cfpb.py`):
- discovery, the downloads and the start snapshot happen before the first `fetch_day`, and
  only on a new acquisition;
- the day count, the day CSV and every by-ID request happen inside `fetch_day`;
- the end snapshot happens after the last `fetch_day`.

When the window is a single day, a window snapshot and a day count carry the same
parameters, and are distinguished by context alone.

*(6.8) The by-ID guard.*

- **When the fetcher sends one.** The fetcher sends a by-ID request only from inside
  `fetch_day`; only for an API row of that day whose Complaint ID is among the day's
  included archive rows, which it reads from the day's spill file; and only when that row
  carries a literal `None` in a field the join needs (`fetch_day`, `values` and `by_id` in
  `ingest/fetch/cfpb.py`).
- **H6's separate check.** H6 adds a check against the production archive's keys,
  `_Archive.keys`: the signed 64-bit array that `read_archive` returns, sorted ascending
  by its merge, before the first `fetch_day`, and which nothing modifies afterwards.
- **The key layout.** Each key is `id << ID_SHIFT | ordinal << 1 | included`, with
  `ID_SHIFT` = 22 and `DAY_MASK` = 2^21 − 1. Bits 22 to 62 hold the Complaint ID. Bits 1
  to 21 hold the received date's ordinal. Bit 0 holds the production inclusion flag, set
  when the row is dated inside the run's window and its narrative is not blank after
  `strip()`. The flag depends on the window, so each run's guard uses that run's own
  archive.
- **The active API-day context** belongs to one run and one guarded transport (4.4). Every
  attempt of a request, retries included, runs synchronously inside the `HttpClient.get`
  call that issued it, and `fetch_day` returns before the fetcher yields its page. So
  every by-ID attempt lies inside the `fetch_day` call that issued it.

H6 evaluates these conditions in order, and refuses at the first that fails:
1. The URL is exactly the search API's URL followed by one path segment and nothing else.
   The segment satisfies the production `_canonical_id` rule, and `n = int(segment)` is
   below `ID_LIMIT` (2^41).
2. Exactly one context is active on this transport. Its day is a `date`. Its archive's
   `keys` is an `array` of typecode `q`. The archive is, by identity, this run's
   `read_archive` result, confirmed through the weak reference of (4.4).
3. The active day lies inside the stage's window.
4. With `lo = bisect_left(keys, n << ID_SHIFT)` and
   `hi = bisect_left(keys, (n + 1) << ID_SHIFT)`, on the existing array, `hi − lo == 1`.
   Let `k = keys[lo]`.
5. `k & 1 == 1`.
6. `(k >> 1) & DAY_MASK` equals the active day's ordinal.
7. The request satisfies every other allowlist rule, headers included.

Because the keys are ascending, `hi − lo` is exactly the number of keys carrying `n`.
Anything missing, malformed, nested, inconsistent or ambiguous is refused before the
adapter. The check copies no key and retains nothing.

**Limitations.**
- The count is exact only if the array is sorted, which follows from `read_archive`'s
  merge and is not re-verified by the guard.
- H6 cannot observe which run issued a request. The association rests on the context being
  held by the transport that run's own client calls, and on D41's rule of one measured run
  per child.
- H3's use of the private attribute `HttpClient._transport` fails closed.
- **The check is not independent of the parser.** The spill that triggers a by-ID request
  and the keys H6 consults come from the same production parse of the same pinned bytes.
  H6 establishes agreement with the fetcher's own archive facts, not their correctness,
  which rests on H7's pins and on D43's, D44's and D49's validation and reconciliation.

**No separate index of the archive is built, copied, written or mapped.**

*(6.9) The guard's cost, and the exact-day audit.*

*Cost.* The archive keys are part of `read_archive`'s cost, measured in the archive stages
of (5). The guard's lookups lie within the API-day stages. No part of the guard runs
before the child starts or outside it.

*The audit.* For every by-ID attempt, H1 records the active API day and the context's
sequence number. For every successful logical request, H2 records `response_sha256`, which
names the retained response; a day CSV is retained at `api/<response_sha256>.csv.gz`. For
each by-ID request that reached the transport, the postflight:
1. takes the last day CSV completed in the same context before that request, whose dates
   both equal the active day;
2. confirms that the retained response's decompressed bytes match its digest;
3. requires the requested Complaint ID to appear in its Complaint ID column;
4. where the day was journaled, requires the day's slice to record that same CSV.

Each of these is a refusal-class integrity event:
- no recorded active day;
- no day CSV before the request in its context;
- dates differing from the active day;
- a missing or mismatched retained response;
- an absent ID;
- a slice naming a different CSV.

The audit checks the API side. It complements H6's pre-send check, and neither of them is
the spill, which is what triggers the request. No request in any stage has a permitted
date that is unknown before it is sent, so no date exception exists.

*(6.10) Stage allowlists and pins.*

| Stage | Hosts | Window and permitted days | Artifacts and pins (H7) |
|---|---|---|---|
| A | `www.consumerfinance.gov`, `files.consumerfinance.gov` | 2024-01-08 to 2024-01-14; API days 2024-01-07 to 2024-01-15 | export #5 alone: 100,880,669 bytes, `993747363e52239e4762b804662d7f604f976a139de4f729f613329d3d8c7206`; by-ID guard over A's own archive keys |
| A′ and every offline rerun | none: every resolution and connection refused | none | none; any attempt is a refusal |
| Archive replay | none: every resolution and connection refused | the replayed window, for selecting exports and spill days only | the replayed exports, copied from `data/recon/` and each verified against its pin by the orchestrator before the child starts; any attempt is a refusal |
| A2 | `www.consumerfinance.gov`, `files.consumerfinance.gov` | 2024-01-01 to 2024-01-31; API days 2023-12-31 to 2024-02-01 | export #5 alone, as for A; by-ID guard over A2's own archive keys, whose inclusion bits follow A2's window |
| B | `www.consumerfinance.gov`, `files.consumerfinance.gov` | 2024-01-01 to 2025-12-31 (D37.2); API days 2023-12-31 to 2026-01-01 | the eleven exports below, #5 to #14 as population and #15 as boundary only (D49); by-ID guard over B's own archive keys |
| C | `data.cityofnewyork.us` alone; no CFPB host | New York civil days 2024-11-02 to 2024-11-04; no margin days | none: NYC 311 publishes no pinned artifact and C sends no by-ID request, so the by-ID guard does not apply; integrity rests on the fetcher's count, key and reconciliation checks (D44, D48) and on D46's verification |

| Export (B) | Bytes | SHA256 |
|---|---|---|
| #5 September 2023 – March 2024 | 100,880,669 | `993747363e52239e4762b804662d7f604f976a139de4f729f613329d3d8c7206` |
| #6 April – July 2024 | 77,142,558 | `c8a6fb0d98287958e2c19b47e1a4211c4e43225d27123a3ca08d730a2b4ae0a5` |
| #7 August – October 2024 | 71,365,591 | `f053000c342bfd08bc6f8f382290f623e34b166cb56760c0726ed15e296a6d74` |
| #8 November – December 2024 | 52,563,546 | `24b3326a954c0edeff33318b157d12eef5a59547f6f274f9236f1ceaba988a74` |
| #9 January – February 2025 | 75,475,199 | `11c18bb0973eed8f5e4f66e2979346af0671a3aefcb5a6b9eec73009d53cff64` |
| #10 March – April 2025 | 68,866,998 | `356822a5e9f295398f2e75bfff25b4760365845432c31e87e689be2c2c0c88ef` |
| #11 May – June 2025 | 74,546,009 | `b7c8475a183e5a01f72aec794d6bc9d8a12a98cce272b3b406d9a0b067479adb` |
| #12 July – August 2025 | 84,153,614 | `dcc75521581b124f13d6d4b62c2947ccd06a017a09b1dfa7ddeb2d73548554fa` |
| #13 September – October 2025 | 67,340,083 | `4a48c4e32ab78d8b1c3c910c4e14a62cd9fb50355e60708c1f85e5601858bc9f` |
| #14 November – December 2025 | 63,867,562 | `44fd05c01c9e90ae1abd6e0659898ec7362e7a6a53740e6b18a8d0a0ff3749e7` |
| #15 January – February 2026 (boundary) | 36,832,036 | `437eee7adcc3e4342d1eafd8d716c4ef149e29c6a68c5ce3c379d1dc82c14b22` |

These pins are the bytes retained at reconnaissance. If the source has republished an
export, H7 refuses, and re-pinning is its own decision, never a guard's.

For C, the permitted requests are:
- `GET https://data.cityofnewyork.us/api/views/erm2-nwe9.json`, with no parameters;
- `GET https://data.cityofnewyork.us/resource/erm2-nwe9.json` as a count — exactly
  `$select=count(*) AS n` and a `$where` of `created_date` bounds from one permitted day's
  `T00:00:00` to the next day's, or from 2024-11-02 to 2024-11-05;
- or the same endpoint as data — exactly
  `$select=unique_key,created_date,closed_date,complaint_type,descriptor`, one permitted
  day's `$where`, `$order=unique_key` and `$limit=50000`.

No other dataset, endpoint or parameter is permitted.

*Now — (7) Memory contention and the quiet-period gate.*

- **Contention.** The operator, never the harness, stops `llama-server`, LM Studio and
  Ollama before resource measurement, and keeps them stopped through an authorized
  execution. The harness refuses to launch while any of them runs, records any that start
  during a run, and never terminates a process.
- **The quiet period.** After contention is resolved, the orchestrator reads system
  headroom once per second for ten minutes. **The ten-minute duration is an operational
  policy, not an empirically established fact.** The margin is the largest headroom read
  minus the smallest.
- **The gate.** **Launch is permitted only if every reading, and the final pre-launch
  reading, are at least 6,000,000,000 bytes — the upper end of D44's estimate — plus that
  margin.** A failed or missing reading fails the gate.
- **The pagefile** is system-managed and grew during measurement. Headroom is always read
  against the commit limit in force, and growth that has not happened is never counted as
  headroom.

*Now — (8) The other launch conditions.*

These are evaluated at **L4** of (11.1a), immediately before every launch, including the
launch of a continuation.

- **Repository state.** The repository is at the authorized commit, on `phase-2` and equal
  to `origin/phase-2`, with no tracked change and only the untracked blueprint.
- **Test gate.** The test gate is green at that commit.
- **Validation.** The applicable validation (16) has passed at that commit and harness
  digest.
- **The allowlist file** is present and matches the digest the authorization names.
- **The run directory.** `data/run-records/<run-id>/` was created exclusively at L1. It
  holds only records the orchestrator has written. It holds no launch-start or
  launch-refusal record. Its state is **preparing**, this orchestrator instance holds its
  nonce, and this lifecycle owns the identifier claim (E1b).
- **The terminal directory.** Its `terminal/` directory holds exactly `launch.json`, which
  passes T1. There is no `outcome.rec` and no `outcome.rec.partial`.
- **The delivery medium.** M-op2 of (9.3) has confirmed, immediately before launch, that
  the run-unique medium holds no request.
- **The launch method** is the one of (4.2), as validated under MV-10.
- **For a continuation:** `continuation-eligibility.json` is present and matches its
  recorded digest, and E2 to E6 of (12.7.5) pass when repeated now.
- **Disk.** Free disk on C: is recorded, and is at least the declared disk requirement:
  the larger of the owner's estimate of about 17 GB (3) and any higher measured or
  projected requirement, including a replay's copied exports.
- **Network**, for an execution with permitted hosts: WARP is disconnected, no proxy is
  configured, and the direct route is recorded.
- **Power.** The machine is on AC power, with sleep prevented for the harness's lifetime
  by a process-scoped request and no persistent power-setting or registry change. The
  operator prevents manual sleep and lid-close interruption.
- **Nothing else running.** No other acquisition, ingestion, rerun, replay or continuation
  is running. No run directory other than this execution's own is in the preparing or
  indeterminate state (interim rule, O-3).
- **Directories.** The working directory is `C:\sentinel`, and the run has a fresh, empty,
  process-local temporary directory outside the repository.

The orchestrator checks the WARP state and the default route throughout the run, at an
interval its run record states; neither may change during a run. HTTP 403 stops the run at
once and is never answered by switching route and retrying.

*Now — (9) In-run triggers, checkpoints and graceful interruption.*

*(9.1) Triggers.*

| Trigger | Counter | Read | Fires on |
|---|---|---|---|
| System commit floor | system headroom | by the orchestrator, once per second | the first reading below the quiet-period margin of (7) |
| Acquisition commit budget | the child's `PeakPagefileUsage` | by the orchestrator once per second, and by the child at every safe checkpoint | the first reading at or above 6,000,000,000 bytes |
| Disk | free bytes on C: | by the orchestrator, once per second | the first reading below the requirement declared under (8), less what the run has written |

Each orchestrator trigger fires at most once per run, and is evaluated only from the
child's launch onward.

- **The budget and the floor.** The 6,000,000,000-byte reservation of (7) is the
  acquisition's commit budget; the margin is the floor left for the rest of the machine.
  The budget trigger reads a lifetime peak, so once reached it stays reached.
- **The budget is a trigger, not a cap.** Between the trigger and the next safe
  checkpoint, commit may keep rising, so the run's actual peak can exceed 6,000,000,000
  bytes, and the run record reports it.
- **A hard cap was considered and rejected for the measured child.** A job-object commit
  limit makes any allocation beyond it fail wherever it happens, including inside a
  spill-file append or between `clear_corpus` and `write_manifest`, a sequence D27 makes
  valid only as a whole.

*(9.2) Safe checkpoints.*

The safe checkpoints are:
- `before_request`: before a request attempt is sent;
- `between_pages`: between cached pages in the normalization pass;
- `before_corpus_write`: before the corpus write begins;
- `after_manifest`: after the manifest is written. This is the end checkpoint of a real
  run or an offline rerun.
- `end_of_replay`: after `read_archive`'s exit, before the exit stage. This is an archive
  replay's only checkpoint and its end checkpoint.

None lies inside a request, inside archive streaming, inside a non-atomic file write, or
between `clear_corpus` and `write_manifest`. A trigger whose next safe checkpoint is the
end checkpoint lets the run complete, and its outcome is classified under (12).

*(9.3) Trigger delivery and the medium's lifecycle (D-3).*

The delivery mechanism is not designed. Whatever is chosen must implement and validate the
complete lifecycle below. Until it does, P-T4 cannot be established, and no measured
execution may be authorized.

**Two kinds of access, never confused.**
- **The orchestrator's operations on the medium.** There are exactly three, and none is a
  child visibility read:
  - **M-op1, create.** At L3, the orchestrator creates a fresh medium unique to the
    execution.
  - **M-op2, verify empty.** At L4, immediately before launch, the orchestrator verifies
    through its own read that the medium holds no request. A non-empty medium, or a failed
    check, is a launch refusal.
  - **M-op3, publish.** After launch, the orchestrator publishes trigger requests. Each is
    one delivery action, bracketed as (10.2) specifies, and carrying a `request_id` (from
    1, in firing order) and its `kind`.

  No publication can occur between M-op2 and the child's launch (9.1). No operation
  removes or alters a published request during a run.
- **The child's read.** At each safe checkpoint, the child performs exactly one visibility
  read, under B-1 and the steps of (9.4). It never performs any other access.

**Lifecycle properties D-3 must provide and validate (MV-8):**

| # | Property |
|---|---|
| (a) | A medium unique to the execution, created by M-op1. |
| (b) | An orchestrator-side emptiness check (M-op2) that is distinct from the child's read and does not count as one. |
| (c) | Writer exclusivity: no process other than the orchestrator, through M-op1 and M-op3, can write to the medium. |
| (d) | Non-destructive reads: neither M-op2 nor the child's read removes, consumes or alters a request. |
| (e) | A defined instant at which a delivery action completes (P-T3). |
| (f) | Visibility at the governing boundary (P-T2\*). |
| (g) | No premature visibility: a request becomes visible no earlier than its delivery action begins, and a read returns only requests visible during the call. Request IDs are unique within a run (P-T4). |

If the chosen mechanism cannot provide any one of (a) to (g), the property it lacks
remains an unresolved dependency under D-3, and P-T2\*, P-T3 or P-T4, as applicable, is
unestablished.

*(9.4) The checkpoint check.*

At every safe checkpoint, the child performs these steps:

| Step | Action |
|---|---|
| 0 | Allocate `read_ordinal`, the previous check event's ordinal plus 1, starting at 1, before any reading. Every check event carries exactly one ordinal. |
| 1 | Take `check_started_qpc`. This is the **governing boundary** of (10.5). |
| 2 | The read function: (2a) takes `read_started_qpc`; (2b) makes one call to the delivery medium — the child's single visibility read, B-1; (2c) takes `read_returned_qpc`; (2d) builds and validates `captured`. |
| 3 | Perform the budget read (`budget_reading`, the child's `PeakPagefileUsage`). |
| 4 | Take `check_completed_qpc`. |
| 5 | Make the forced-action decision. |

**The forced-action rule.** The child acts at a check if and only if `budget_reading` is
at least 6,000,000,000 bytes, or `observed_request_ids` is non-empty. When it acts:
- it acts on its own budget (source `child_checkpoint`) if the reading has reached the
  threshold, and otherwise on the lowest observed ID;
- at the run kind's end checkpoint, acting records the trigger and the run completes;
- at any other checkpoint, the child raises the trigger-stop exception (4.5).

**The binding invariant** (specified; established only through P-T6):
- **B-1.** Within the child, exactly one function performs a visibility read. It is called
  from exactly one place in the checkpoint routine, exactly once per check, at step 2b. No
  other child code accesses the medium. The orchestrator's operations M-op1 to M-op3 lie
  outside the child and outside B-1, and never supply, alter or substitute for the child's
  captured result.
- **B-2.** The read builds an immutable `tuple` of `int` directly from the medium's
  response. The routine binds it once, to `captured`, and never rebinds, mutates, copies
  or merges it.
- **B-3.** `captured` is accepted only if it is a `tuple` of integers, each at least 1,
  with no duplicates.
- **B-4.** `observed_request_ids = tuple(sorted(captured))` is derived once. That same
  object feeds the forced-action decision, the choice of the lowest ID, the event's list,
  and `observed_digest` — the SHA-256 of the list's canonical serialization (11.3).
- **B-5.** No check reads the medium twice.
- **B-6.** On any failure at steps 1 to 2d, the routine performs neither step 3 nor step
  4, makes no decision, records the failed-read event of (10.3) where logging works, and
  raises the refusal exception. The run's terminal state is `refused`, and the read is
  never retried.

*(9.5) Latency and crashes.*

For each trigger, the run record holds:
- its firing reading;
- the acting check's `check_started_qpc`;
- the resulting latency, in counter ticks and in the recorded frequency.

This latency is a measured result, because the longest stretch without a checkpoint has no
fixed bound. If the child crashes, its peak is the last lifetime peak read before exit,
which is a lower bound. All evidence is preserved, and nothing restarts automatically.

*Now — (10) Ordering and observation evidence.*

*(10.1) The counter.*

- **The clock.** Every cross-process ordering uses `QueryPerformanceCounter`, read
  directly through `ctypes` in both processes.
- **Frequency.** Each process records its `QueryPerformanceFrequency`. Unequal frequencies
  are E.
- **Brackets.** Each action is bracketed by readings taken immediately before and after
  it. No ordering finer than the brackets is claimed, and "immediately" is never taken to
  mean simultaneous: another process may run inside any bracket.
- **Status.** The counter's comparability across processes is P-T1, and is unverified.
- **UTC** is never used to order events across processes.

*(10.2) Orchestrator trigger entries (M-1, M-4).*

The exit snapshot holds one entry per firing. Each entry has `request_id`, `kind`,
`reading`, `threshold`, `fired_qpc`, `fired_at_utc`, `delivery_started_qpc` (immediately
before the delivery action), `delivered_qpc` (immediately after its completion), and
`delivered_at_utc`. Its shape requires `fired_qpc ≤ delivery_started_qpc ≤ delivered_qpc`.

*(10.3) Child check events (M-2, M-3).*

Every child event carries a contiguous `seq` from 1. On the ordinary path, the outermost
frame's `terminal_begin` event, naming its state, is the child's last event, written
immediately before the terminal artifact. Child evidence is **complete** only if every
`seq` from 1 to `terminal_begin` is present.

Each check event has these fields:
- `seq`, `read_ordinal`, `checkpoint` and `checked_at_utc`;
- `check_started_qpc`, `read_started_qpc`, `read_returned_qpc`, `read_raised_qpc` and
  `check_completed_qpc`;
- `observed_request_ids`, `observed_digest`, `budget_reading` and `acted`, where `acted`
  is null or holds `kind`, `source` and `request_id`;
- `capture_failed` and `capture_failure_kind`.

**`observed_request_ids` records exactly what the check's single read returned.** It never
claims what was visible at the checkpoint's start. A **captured check** has
`capture_failed` false; a **failed check** has it true.

| Event | Required present | Required null |
|---|---|---|
| Success (`capture_failed` false) | `check_started_qpc`, `read_started_qpc`, `read_returned_qpc`, `check_completed_qpc`, `observed_request_ids`, `observed_digest`, `budget_reading` | `read_raised_qpc`, `capture_failure_kind` |
| Failure `check_timing_failed` (step 1) | — | every reading, plus the result fields |
| Failure `read_timing_failed_before` (step 2a) | `check_started_qpc` | `read_started_qpc` onward, plus the result fields |
| Failure `read_raised` (step 2b) | `check_started_qpc`, `read_started_qpc`, and `read_raised_qpc` unless that reading itself failed | `read_returned_qpc`, `check_completed_qpc`, the result fields |
| Failure `read_timing_failed_after` (step 2c) | `check_started_qpc`, `read_started_qpc` | `read_returned_qpc`, `read_raised_qpc`, `check_completed_qpc`, the result fields |
| Failure `result_invalid` (step 2d) | `check_started_qpc`, `read_started_qpc`, `read_returned_qpc` | `read_raised_qpc`, `check_completed_qpc`, the result fields |

Here *the result fields* are `observed_request_ids`, `observed_digest`, `acted` and
`budget_reading`.

- **Shape.** Present readings must satisfy
  `check_started_qpc ≤ read_started_qpc ≤ read_returned_qpc ≤ check_completed_qpc`, and
  `read_started_qpc ≤ read_raised_qpc`. Each check must start no earlier than the previous
  check completed.
- **Malformed evidence is E (R-T6).** This covers a reversed order, a missing required
  field, an unknown kind, and an unordered or duplicated `observed_request_ids`.
- **Contradictory evidence:**
  - a forbidden field that is present;
  - an `observed_digest` that differs from the SHA-256 of its list;
  - an ordinal gap;
  - a failed check followed by a terminal state other than `refused` (exception kind
    `harness_refusal`, clear latch, null trigger) or `integrity` (null trigger);
  - a violation of failed-check sequencing, below.
- **Failed-check sequencing.** After a failed check, no further checkpoint-check event may
  appear. Between a failed check and the child's last event, the only child events
  permitted are:
  - **(P1)** the exit stage's final reading (4.4, 5.3), on the ordinary path;
  - **(P2)** the cleanup, read-back, latch and terminal-marker events of (4.4), when the
    failure occurred inside an active `fetch_day` context;
  - **(P3)** `terminal_begin`, on the ordinary path.

  Any other child event after a failed check is contradictory: a further check event, an
  `acted` record, a request attempt, a stage-boundary reading of any stage other than
  exit, or any H6 refusal event. On the integrity path, neither P1 nor P3 is written; the
  missing `terminal_begin` makes child evidence incomplete, which is not a contradiction.
- **A failed check is excluded from** the success digest, O1 to O3, V1 to V3, R-T3 points
  4 to 6, and the "last captured check" of R-T4.

*(10.4) What readings prove.*

This holds under P-T1 and P-T3. Write `ds` and `dc` for an entry's `delivery_started_qpc`
and `delivered_qpc`; `cs`, `rs` and `rr` for a check's `check_started_qpc`,
`read_started_qpc` and `read_returned_qpc`.

*The readings prove:*
- `dc < rs`: delivery completed before the read call began;
- `ds > rr`: the delivery action began after the read call returned;
- `dc < cs`: delivery completed before the governing boundary;
- `ds > cs`: delivery completed after the governing boundary.

*They cannot prove:*
- the instant within the read at which the medium was sampled;
- for overlapping brackets, whether publication preceded that sampling;
- that a recorded observation is genuine.

*(10.5) Omissions: O1 to O3, against the governing boundary.*

The guarantee is P-T2\* ((9.3)(f)): a request whose delivery completed before the child
took `check_started_qpc` must be returned by that check's read. Nothing is required of a
request that completes later, even before the read begins.

| Relation | Condition | An unobserved T is |
|---|---|---|
| O1 | `dc < cs` | contradictory: a silent omission |
| O2 | `ds > cs` | permitted |
| O3 | `ds ≤ cs ≤ dc` | E |

O1, O2 and O3 partition every pair of entry and check. A required return (O1) always
satisfies V1, because `dc < cs ≤ rs`.

*(10.6) Returns: V1 to V3, temporal compatibility only.*

For each T returned at a check:

| Relation | Condition | Temporal result |
|---|---|---|
| V1 | `dc < rs` | compatible, which includes a legitimate observation of a request that arrived mid-check |
| V2 | `ds > rr` | contradictory: fabricated ((9.3)(g)) |
| V3 | otherwise | E |

A returned T is never treated as visible at the checkpoint's start.

**Authenticity is separate from timing.** A returned request is a **valid observation**
only under V1 together with authenticity. Authenticity requires both P-T6 established at
the run's harness digest, and the event passing every check of (10.3). Timestamps never
establish authenticity.
- V1 with authenticity unavailable is E.
- V1 with a demonstrated mismatch between the record and the read is contradictory.
- P-T5 (a publication sequence number, or orchestrator-served reads) would resolve V3
  causally. It is not adopted.

*(10.7) Failed reads.*

A failed read is represented and validated as in (10.3). Ordinals come from step 0. Every
timestamp recorded is a real reading; none is invented. A consistent failure follows the
refusal path of B-6.

*(10.8) Trigger reconciliation.*

This applies when child evidence is complete; otherwise R-T6 applies. K_act, the **acting
check**, is the check whose `acted` is non-null.

- **R-T1 (orchestrator evidence).** Each of these is E:
  - more than one entry of a kind;
  - a duplicate `request_id`;
  - a missing field or `delivered_qpc`;
  - a shape violation;
  - unequal frequencies.
- **R-T2 (artifact against snapshot).** If the artifact's `trigger.source` is
  `orchestrator`, exactly one entry must have that kind. Otherwise the evidence is
  contradictory.
- **R-T3 (checks).** Any violation of the following is contradictory:
  1. There is one acting check if the artifact has a trigger, and none otherwise. A failed
     check never acts.
  2. If an acting check exists, it is the last check event. An acting check and a failed
     check cannot coexist.
  3. K_act's checkpoint, `acted.kind` and `acted.source` equal the artifact's.
  4. Every **captured** check before K_act — or every captured check, if there is no
     acting check — returned nothing and had a reading below the threshold.
  5. Across captured checks, in `seq` order, `budget_reading` never decreases.
  6. At K_act, the child's own budget takes priority. Otherwise `acted.request_id` is the
     lowest observed ID.
- **R-T3a (`source` is `orchestrator`).** `acted.request_id` is non-null, matches exactly
  one entry of the same kind, and is a valid observation at K_act. A null, absent or
  wrong-kind request is contradictory.
- **R-T3b (`source` is `child_checkpoint`).** `acted.request_id` is null, the kind is
  `budget`, and K_act's `budget_reading` is at least the threshold. No snapshot entry is
  required. A simultaneous orchestrator budget entry is reconciled by R-T4.
- **R-T4 (other entries).** Every entry T other than the acted one is exactly one of:
  - **Covered:** an acting check exists, and T is a valid observation at K_act. Without an
    acting check, a trigger cannot be covered.
  - **Late:** T was never returned, and O2 holds against the last captured check; or no
    check was captured.
  - **Contradictory:** V2, an O1 omission, or a forced-action violation.
  - **E:** V3, an O3 omission, unavailable authenticity, or a shape or prerequisite
    failure.
- **R-T5 (budget claim on incomplete evidence).** When child evidence is incomplete, an
  artifact's `child_checkpoint` budget claim is not contradictory for lack of
  corroboration. The run is E, which gives the same outcomes as a budget trigger.
- **R-T6 (insufficient evidence).** If the snapshot has any entry, or the artifact has a
  trigger, the evidence is E when any one of these holds:
  - child evidence is incomplete or malformed;
  - any of M-1 to M-4 is absent;
  - P-T1, P-T2\*, P-T3, P-T4 or P-T6 is not established at the run's harness digest.
- **R-T7 (checkpoint against state).**
  - `completed` permits a trigger only at the run kind's end checkpoint.
  - `stopped_by_trigger` requires a trigger at any other checkpoint the run kind reaches.
  - All other states require `trigger` to be null.
  - The exit status depends on the state alone.

*Now — (11) Run identifiers, the launch lifecycle and the terminal artifact.*

*(11.1) The run identifier, the run directory, the launch file, and T0 to T2.*

**The run identifier.**
- **A newly authorized run identifier** must match `^[a-z0-9-]{1,64}$`: lowercase ASCII
  letters, digits and hyphens only.
- **It must not be a reserved Windows device name.** An identifier is rejected if it
  equals `con`, `prn`, `aux`, `nul`, `com1` to `com9`, or `lpt1` to `lpt9`. Because an
  identifier contains no dot, the whole identifier is the base name of both its run
  directory and its claim file `<run-id>.claim`.
- **Comparison is always ASCII case-insensitive.** Two identifiers are the same identifier
  if they are equal after every ASCII letter `A` to `Z` is mapped to `a` to `z`. No other
  folding or normalization is applied. This one rule is used for claim-file names (with
  the `.claim` suffix removed); run-directory names; `run_id` in `launch.json`,
  `preparing.json`, the claim and the terminal artifact; `continues.prior_run_id` in
  `launch.json`; and `prior_run_id` in `continuation-eligibility.json`.
- **A case-only collision is an already-used identifier,** whether the directory is
  case-sensitive or case-insensitive, because uniqueness is decided by this comparison and
  never by the filesystem's own name matching.
- **Retained identifiers** that predate this rule are read under the wider pattern
  `^[A-Za-z0-9-]{1,64}$` and compared by the same rule. No retained record is renamed or
  rewritten.

**The run directory.** Each execution has its own `data/run-records/<run-id>/`, which
holds its run record and its `terminal/` directory. The terminal directory holds
`launch.json`, `outcome.rec` and `outcome.rec.partial`, and nothing else uses it. No
execution writes into another execution's run directory, except for the orchestrator's
appended audit and evidence-status entries of (11.7).

**The launch file.** `launch.json` is canonical JSON (11.3), with no BOM and no trailing
newline, of at most 4,096 bytes. It holds exactly these keys:

| Key | Value |
|---|---|
| `format` | `"sentinel-launch/1"` |
| `run_id` | the run identifier, under the rules above |
| `run_kind` | `real_run`, `offline_rerun` or `archive_replay` |
| `authorization_ref` | **A_D**, the SHA-256 of the authorization document |
| `harness_sha256` | the harness digest the authorization names |
| `exit_statuses` | the six terminal states mapped to integers 0–255, pairwise distinct, with `completed` equal to 0, equal to the harness's pinned table, and avoiding 1, 2 and 3 |
| `continues` | null for a first execution. For a continuation, an object with exactly `prior_run_id`, `prior_classification_sha256`, `prior_outcome` (`reviewed_interruption` or `ordinary_interruption`) and `owner_review_sha256` (non-null exactly for a reviewed interruption) |

**T0 (phase L1).**
1. The orchestrator confirms that it still owns the identifier claim (E1a step 5) and that
   the identifier still passes E1a step 1. It then creates `data/run-records/<run-id>/`
   with exclusive-create semantics. A failure here is F0-d or F0-e (11.1a).
2. It writes the preparing record (11.1a). A failure here is F1.
3. It creates `terminal/` inside the new directory.
4. It records, from this execution's own authorization: `run_id`, `run_kind`, A_D, the
   harness digest and the `continues` binding.
5. It writes `terminal/launch.json` by exclusive create, flush, `fsync` and close.
6. It re-reads all of the file's bytes, confirms they equal what it serialized, and
   records their SHA-256, **D_L**.

A failure at steps 2 to 6 is a launch refusal, recorded and classified by state as (11.1a)
specifies.

**T1 (at L4).** The orchestrator re-reads and re-hashes the launch file. A digest other
than D_L is a launch refusal.

**T2 (child startup).** The child reads all of the launch file's bytes and hashes them
itself, giving **D_C**. It requires canonical form, the schema, and a table equal to its
own pinned table. On any failure it exits with no artifact. Its first event records its
PID, `sys.prefix` and the argument vector it received.

*(11.1a) The launch lifecycle, the identifier claim and run-directory states.*

**Phases.** Every launch goes through these phases in order. A phase that fails stops the
lifecycle at that point.

| Phase | Applies to | Action | If it fails |
|---|---|---|---|
| **L0** Continuation eligibility | continuations only; before T0 | Generates the lifecycle nonce. Evaluates E1a, steps 1 to 7 of (12.7.5), which reserves the identifier at step 4 and confirms ownership at step 5. If E1a passes, evaluates every one of E2 to E6, recording each that fails. It evaluates neither (7) nor (8). | A preflight-refusal record. No run directory is created. The claim, if created, remains, and the identifier is consumed. |
| **L1** T0 | every execution | A first execution generates the lifecycle nonce and evaluates E1a, steps 1 to 7. Every execution then performs T0 (11.1). | F0 if no directory was created; otherwise F1 to F3. |
| **L2** Persist eligibility | continuations only | Writes `continuation-eligibility.json` and `acquisition-manifest.l0.json` (11.9) into the new run directory, by exclusive create, flush, `fsync` and rename. The eligibility record is canonical JSON holding L0's inputs and results, including the fields `A_D`, `prior_run_id` and `prior_classification_sha256`, and every digest L0 read. Its SHA-256 is appended to the run record's audit section. | A launch-refusal record (F4). |
| **L3** Preparation | every execution | The allowlist file (6.6); M-op1 (9.3); the contention check; then the ten-minute quiet period of (7), with every reading stored in the run record. | A launch-refusal record (F4). |
| **L4** Fresh launch gate | every execution; immediately before launch | T1; M-op2; the final pre-launch headroom reading of (7); every condition of (8); E1b and the launch binding. For a continuation, E2 to E6 are repeated, and `acquisition-manifest.l4.json` is persisted. L4 never requires L4. | A launch-refusal record (F4). |
| **L5** Launch | every execution | The orchestrator writes and `fsync`s the **launch-start record**, `launch-start.json`, in the run directory. It then creates the child (4.2) with exactly the recorded argument vector. | If the child cannot be created after the launch-start record exists, the run is an execution with no valid artifact: an unclassified termination (11.8). |

**The lifecycle nonce** is 128 bits from a cryptographically secure random source,
generated once per lifecycle. The raw nonce is held **only in the creating orchestrator's
memory**. It is never written to any file, record, log, event or console output.

**The identifier claim** is `data/run-records/_claims/<run-id>.claim`. The directory name
starts with `_`, so it cannot collide with a run identifier.
- **Reservation.** An identifier is reserved at the instant its claim file is created with
  exclusive-create semantics (E1a step 4). That single create is the reservation, and
  exclusive file creation is the only reservation mechanism.
- **Existence is the authority.** An identifier has been used if its claim file exists,
  whatever the file contains. A claim is never deleted, renamed, rewritten or reused by
  the harness. If a second lifecycle attempts the same identifier, its exclusive create
  fails, and it is refused. A preflight-refusal record never reserves or consumes an
  identifier.
- **Contents.** The claim is canonical JSON (11.3) with exactly these keys:

| Key | Value |
|---|---|
| `format` | `"sentinel-claim/1"` |
| `run_id` | the run identifier |
| `A_D` | the authorization digest |
| `nonce_sha256` | the SHA-256 of the lifecycle nonce's 16 raw bytes |
| `claimed_at_utc` | the UTC time, for audit only |

- **Ownership.** A lifecycle owns a claim only if the claim is canonical, matches this
  schema, names its run identifier and A_D, and its `nonce_sha256` equals the SHA-256 of
  the nonce the lifecycle holds in memory (E1a step 5).
- **An incomplete claim.** If the claim file exists but its content is missing, partial or
  invalid, the identifier is consumed, no lifecycle can own it, and no run directory may
  be created for it. It is preserved as found.

**What "consumed" is established against.**
- **A process crash.** If the orchestrator or child crashes after the exclusive create
  returns, the claim file is expected to remain, complete or not, and the identifier stays
  consumed. This expectation is unverified and is established only by MV-W1 and MV-W10.
- **An OS crash or sudden power loss.** Not established. `flush` and `fsync` are applied
  to the claim's contents; D51 does not claim that they make the claim's directory entry
  durable. After such an event a claim may be missing, or present and incomplete. No
  specified test establishes this, and D51 defines no power-loss validation.

**Fail-closed rules that do not depend on durability.**
- An identifier is also treated as used if a directory `data/run-records/<run-id>/`
  exists, or if a retained record names it (E1a step 6), whether or not a claim file
  exists.
- A run directory whose claim file is missing or incomplete is **indeterminate**.
- **After any OS crash, forced restart or power loss during a lifecycle, the authorization
  that lifecycle was using is spent.** Any further attempt needs a new explicit
  authorization naming a new identifier, whatever survives on disk. This rule is applied
  by the operator and the owner; the harness cannot verify it from records. Claims and run
  records that survive remain subject to the normal fail-closed rules.

**The preparing record** is `data/run-records/<run-id>/preparing.json`, written at L1 by
exclusive create, flush, `fsync` and close. It is canonical JSON with exactly these keys:

| Key | Value |
|---|---|
| `format` | `"sentinel-preparing/1"` |
| `run_id` | the run identifier |
| `A_D` | the authorization digest, 64 lowercase hexadecimal digits |
| `nonce_sha256` | the SHA-256 of the nonce's 16 raw bytes |
| `created_at_utc` | the UTC time of creation, for audit only |

An orchestrator instance owns a preparing directory only if the SHA-256 of the nonce it
holds in memory equals `nonce_sha256`. A restarted or different instance cannot recover
ownership from the record. If the creator loses the nonce, no instance can satisfy E1b for
that directory.

This is not evidence of operating-system process liveness. It does not protect against a
process that can read the orchestrator's memory, or one that can replace `preparing.json`.
A replaced record makes the creator fail E1b, which is fail-closed for the creator.

**The launch binding.** No authorization file or configuration is assumed to be immutable.
1. At E1a step 1 — in L0 for a continuation, in L1 for a first execution — the
   orchestrator reads all of the authorization file's bytes once, computes **A_D**, and
   derives from that one read every launch field: `run_id` and run kind; the exact
   argument vector of the command, including window and corpus root; the commit; the
   harness digests; the allowlist digest; the `continues` binding. It records A_D in the
   claim (field `A_D`), in `continuation-eligibility.json` (field `A_D`), in the preparing
   record (field `A_D`) and in the launch file (field `authorization_ref`).
2. At L4, the orchestrator:
   - re-reads the authorization file, and requires its SHA-256 to equal A_D;
   - re-derives every launch field, and requires each to equal the launch file's;
   - requires that the repository is at the derived commit, the running harness has the
     derived digests, and the allowlist file has the derived digest;
   - captures the current package-freeze digest; for a continuation, it must equal the
     prior execution's (E4).

   Any mismatch is a launch refusal.
3. At L5, the launch-start record names: the exact argument vector; the interpreter and
   launch method of (4.2); `A_D`; the commit; the harness digests; the package-freeze
   digest; and, for a continuation, the SHA-256 of the L4 fresh manifest.

**Run-directory states.** The state of a run directory is decided by the records it holds:
`preparing.json`, `launch-start.json`, `launch-refusal.json`, and the contents of
`terminal/`.

| State | Defined by the records present | Treatment |
|---|---|---|
| **Preparing** | a valid preparing record; no launch-start record; no launch-refusal record; no `outcome.rec` or `.partial` | The active pre-launch state, below. |
| **Launched** | a launch-start record | An **execution**: TV0 to TV7 apply, and (11.8) governs a missing or invalid artifact. |
| **Launch-refused** | a verified launch-refusal record; no launch-start record; no `outcome.rec` or `.partial` | **Not an execution.** No outcome under (12). Never continued. Not counted under E3. Its run identifier is consumed. Every partial state is preserved as found. |
| **Indeterminate** | anything else, including: neither a preparing nor a launch-refusal record; a present but invalid preparing record; both a launch-start and a launch-refusal record; an `outcome.rec` or `.partial` without a launch-start record; a missing or incomplete claim; or a state that cannot be established | Blocks fail-closed, below. |

**A preparing directory is active only for the orchestrator instance that holds its
nonce.** That instance may take it through L2 to L5, and it is the only preparing
directory permitted at L4. **To every other observer, a preparing directory is treated
exactly as indeterminate.** The records cannot show whether its orchestrator is still
running, was restarted, or crashed. D51 defines no process-liveness guarantee and infers
none from elapsed time.

**A foreign preparing or indeterminate directory:**
- under E3, blocks every continuation that names the same prior run;
- is never continued, adopted or completed by another instance;
- under (8), blocks every other launch as well (interim rule, O-3);
- is released only by the owner's review under a later decision, which D51 does not define
  (O-1).

**Failure cases before a run directory exists (F0).**

| Case | What happened | The identifier |
|---|---|---|
| **F0-a** | A claim file for the identifier already exists: found by enumeration at E1a step 3, including a case-only collision, or shown by the exclusive create failing at step 4 | **Already consumed** by that claim. This lifecycle never owned it. The existing claim and any directory are untouched. |
| **F0-b** | The claim create failed for another reason, and no claim file exists afterwards | **Not reserved.** The preflight-refusal record states that the reservation did not occur. |
| **F0-c** | The claim file exists afterwards, but this lifecycle cannot confirm it owns it: the create returned ambiguously, the content could not be written, the read-back or ownership check failed, or the claim's final path differs from the expected one | **Consumed.** No run directory is created. |
| **F0-d** | The lifecycle owns the claim, but a later check fails before any directory exists — the retained-record scan (E1a step 6), or E2 to E6 at L0 — or the directory create fails and no directory exists afterwards | **Consumed** by this lifecycle's claim, which remains. |
| **F0-e** | The lifecycle owns the claim, the directory create failed or returned ambiguously, and a directory exists afterwards | **Consumed.** The directory is indeterminate and is never adopted. |
| **F0-f** | The claims directory is absent while other records exist, cannot be opened in the required mode, fails a path-integrity check, or its enumeration is untrustworthy (11.1b) | **Not reserved.** No claim is created. |

A refusal at E1a step 1 — an invalid authorization or identifier — attempts no
reservation, so no F0 case applies; a preflight-refusal record is written where it can be.

**Failure cases after the run directory exists (F1 to F4).**

| Case | What happened | Recorded as |
|---|---|---|
| **F1** | Failure after the directory exists and before `launch.json` is attempted (T0 steps 2 and 3) | A launch-refusal record stating that the preparing record or `terminal/` is absent. |
| **F2** | Failure while creating or writing `launch.json` (T0 step 5) | A launch-refusal record: `terminal/` empty, or `launch.json` partial or invalid, with its SHA-256 as found. |
| **F3** | Failure of the read-back or digest check (T0 step 6) | A launch-refusal record: `launch.json`'s SHA-256 as found, and whether it equals the serialized bytes. |
| **F4** | Failure at L2, L3 or L4 | A launch-refusal record naming the phase and check, and the state of `terminal/` as found. |

In every F0 to F4 case nothing is launched, no outcome under (12) arises, and any further
attempt needs a new explicit authorization naming a new identifier. No file is
manufactured, repaired or deleted. A `launch.json` left by F2 or F3 is never used. If a
launch-refusal record cannot be written, the directory is indeterminate.

**Preflight-refusal records** live in `data/run-records/_preflight/`, whose name cannot
collide with a run identifier. Each is canonical JSON holding the UTC time, `A_D`, the
attempted run identifier, the prior run identifier if any, the phase and every check that
failed, and every digest read. Its file name is the SHA-256 of its own bytes plus `.json`.
It is written by exclusive create, flush, `fsync` and rename, is not an execution, and is
never modified.

**`_preflight/` uses the common traversal, not the claims policy.** Before a
preflight-refusal record is written, `C:\sentinel`, `data` and `run-records` are traversed
and verified exactly as in (11.1b), and `_preflight` is then opened the same way. If
`_preflight` is absent, it is created by exclusive create relative to the verified
`run-records` handle; its absence is never treated as lost evidence. The record is then
created relative to the verified `_preflight` handle. The "absent while `run-records`
holds records" refusal applies to `_claims` only.

**If the record cannot be written safely** — because a component failed traversal,
`run-records` is absent, or the write failed — the lifecycle stops. It writes nothing
else, leaves any `.partial` as found, reports the refusal on its console, and states
explicitly that **no durable refusal record exists**.

*(11.1b) The claims directory.*

**When.** Opening the claims directory is the **first filesystem action of every
lifecycle, after nonce generation and authorization reading** (E1a step 2).

**Trusted traversal, component by component.** The orchestrator walks four components in
order: `C:\sentinel`, then `data`, then `run-records`, then `_claims`.
- `C:\sentinel` is opened by its full path. Each later component is opened **by its single
  name, relative to its parent's verified handle** — never by a full path string.
- Every open is made without following a reparse point.
- For each component, on its own handle, the orchestrator:
  1. confirms it is a directory and not a symbolic link, junction or other reparse point;
  2. resolves its final on-disk path and requires it to equal the expected path exactly;
  3. keeps the handle open, in a mode that does not permit the directory to be renamed,
     deleted or replaced.
- All four handles stay open until the claim has been created and verified.
- `C:\sentinel` and `data` must already exist. They are never created by the harness.

The drive letter `C:` and the volume root are trusted and are not verified.

**Provisioning.**

| Situation | Action | Result |
|---|---|---|
| `_claims` exists and passes traversal | Proceed to enumeration | — |
| `_claims` is absent, and `run-records` is absent or empty | Create the missing directories by exclusive create, each relative to its parent's verified handle; then verify them as above | Proceed. A lifecycle that loses a creation race opens what the winner created. |
| `_claims` is absent, and `run-records` holds anything | **Refuse** (F0-f). Claims may have been lost. | Nothing is reserved. |
| `_claims` exists but cannot be opened in the required mode | **Refuse** (F0-f) | Nothing is reserved. |
| Any component fails a path-integrity check | **Refuse** (F0-f) | Nothing is reserved. |

**The enumeration contract.** Enumeration is made through the verified `_claims` handle
and covers every entry in the directory. Each entry is examined on its own handle, opened
by name relative to `_claims` without following a reparse point.
- **A valid claim entry** is a regular file — not a directory and not a reparse point —
  named `<identifier>.claim`, where the suffix `.claim` is matched under the ASCII
  case-insensitive rule and the identifier part matches the retained pattern
  `^[A-Za-z0-9-]{1,64}$`.
- **Every valid claim entry consumes its identifier,** whether its content is complete,
  partial, empty or invalid. Content is never a reason to disregard a claim. The
  identifier is compared under the rule of (11.1).
- **No entry is ever silently ignored.** The enumeration is **untrustworthy**, and the
  lifecycle refuses before creating a new claim (F0-f), if `_claims` contains any of: a
  directory; a symbolic link, junction or other reparse point; a non-regular file; a name
  that does not have the form above — a wrong or missing suffix, an identifier part
  outside the retained pattern, or a reserved device name; or any entry that cannot be
  opened or examined.
- Two valid claims whose identifiers differ only in case both stand; each marks that
  identifier as used. That is not by itself a refusal.
- **Between enumeration and creation.** A newly created claim is always lowercase, so a
  concurrent lifecycle attempting the same identifier is stopped by the exclusive create.
  The enumeration exists to catch retained claims in another case.

**Anchored creation.** The claim is created by exclusive create **relative to the verified
`_claims` handle**, by its file name alone. It is therefore created in the directory that
handle refers to, and a change to any path string cannot redirect it.

**Protection must be established before any claim is created.** Two things are required,
and neither substitutes for the other:
- the anchored creation above;
- the held handles actually preventing rename, deletion and replacement of all four
  components.

The second is unverified platform behaviour. It is established for a harness digest only
by MV-W9(b). If MV-W9 has not established it, no measured execution may be authorized
(D-5). Within a lifecycle, if any handle cannot be opened in the required mode, the
lifecycle refuses **before creating the claim**. Detecting a swap afterwards is not an
acceptable substitute.

**Verification after creation — an additional safeguard, not the guarantee.** On the
claim's own handle, the orchestrator confirms it is a regular file, not a reparse point,
and that its final on-disk path is the expected `_claims` path joined with
`<run-id>.claim`. A failure here refuses the lifecycle; the claim exists, so the
identifier is consumed (F0-c).

**Every later claim check** — at L1, L4 and TV2 — repeats the traversal and reads the
claim relative to the verified `_claims` handle. No check relies on a path string alone.

**Status.** The Windows calls, sharing modes and enumeration behaviour these steps need
are unverified (D-5) and are established only by MV-W9 and MV-W11.

*(11.2) Writer, timing and write protocol of the terminal artifact.*

The outermost frame writes the artifact once, as its last act before exit (4.4). It:
1. creates `outcome.rec.partial` exclusively (mode `xb`) in its **own** run's `terminal/`
   directory;
2. writes the envelope;
3. flushes and calls `os.fsync`;
4. closes it;
5. renames it to `outcome.rec` with `os.rename`, which on Windows refuses an existing
   destination (unverified, D-5).

If any step fails, no `outcome.rec` exists, nothing is retried, and the child exits with
the status it had determined. A leftover `.partial` is evidence, never an outcome. No
other process writes the artifact; the orchestrator only reads it.

*(11.3) The envelope.*

The file is exactly 107 + L bytes, with 2 ≤ L ≤ 65,536, in three parts:
- the magic: the 9 bytes of ASCII `SNTL-TO1` followed by LF;
- the payload: L bytes;
- the trailer: 98 bytes, consisting of LF, `LEN`, a space, L as exactly 20 zero-padded
  ASCII decimal digits, a space, `SHA256`, a space, 64 lowercase hexadecimal digits, and
  LF.

The digest is the SHA-256 of the payload bytes only.

**Canonical serialization** — used for the payload and every canonical JSON record in D51
— means sorted keys, `,` and `:` as separators, ASCII-only escaping, no NaN or infinity,
and integers as the only numbers.

*(11.4) The payload.*

It holds exactly these keys:
- `format`: `"sentinel-terminal-outcome/1"`;
- `run_id`: this execution's own run identifier;
- `launch_sha256`: D_C;
- `child_pid`;
- `state`: `completed`, `stopped_by_trigger`, `refused`, `integrity`, `operator_interrupt`
  or `crash`;
- `exit_status`;
- `latch`: exactly `cleanup_failed`, `cleanup_verified`, `reason` (null, `primary_raised`
  or `primary_readback_failed`), `seq` (null or ≥ 1) and `day` (null or ISO date);
- `integrity_reason`: null, `cleanup_recovered`, `cleanup_unverified`, `slot_occupied`,
  `integrity_exception` or `checks_failed`;
- `trigger`: null, or exactly `kind` (`system_floor`, `disk` or `budget`), `source`
  (`orchestrator` or `child_checkpoint`) and `checkpoint` (one of the five values of
  (9.2));
- `exception`: null, or a descriptor (11.5).

*(11.5) Exception descriptors.*

- **What is never persisted.** No persisted record holds exception text. `str()`, `repr()`
  and `format()` of an exception, its `args`, notes, cause or context are never persisted;
  nor are `FetchFailed.url`, `params` or `reason`, `OSError.filename`, bodies, headers,
  cookies, credentials or other request details.
- **Why.** `TransportError`'s message embeds text from `requests` that can quote a header
  value.

**The descriptor** is built by one pure function from `type(exc)` and these type-checked
attributes:
- **`kind`** is the first that applies, in this order: `harness_refusal`, `trigger_stop`,
  `fetch_failed`, `transport_error`, `typed_refusal`, `memory`, `os_error`,
  `keyboard_interrupt`, `other`. `typed_refusal` means an `AcquisitionError`, or any class
  defined in an `ingest` module (D-2). `kind` is `unknown` if any step of determining it
  fails, and is `other` only when every earlier test has positively failed.
- **`class`** is `module.qualname`: printable ASCII, at most 256 characters, matching
  `^[A-Za-z_][A-Za-z0-9_]*(\.([A-Za-z_][A-Za-z0-9_]*|<locals>))*$`. Otherwise it is null.
- **`harness_reason`** is always null (D-6).
- **Kind-specific fields**, null otherwise: `fetch_status` and `fetch_attempts` for
  `fetch_failed`; `transient` for `transport_error`; `errno` and `winerror` for
  `os_error`.
- **`identity_matches_exit_record`**: a boolean, or null.
- **`complete`**: false if any field could not be produced safely.
- **`cause` and `context`**: each null, or a link of the same fields minus
  `harness_reason`, `identity_matches_exit_record`, `cause` and `context`, plus
  `chain_continues`. The chain is kept to depth 1.

**Failure handling.** Nothing is truncated. A field that cannot be produced is null, and
`complete` becomes false; a `chain_continues` that cannot be produced is null and is
treated as true. A descriptor that cannot be produced at all is
`{"kind": "unknown", "complete": false}`, with every other field null.

**The in-process exception is untouched** (4.4).

*(11.6) Terminal verification (TV0 to TV7).*

After the child exits:
- **TV0.** The orchestrator observes the exit and its exit code.
- **TV1.** It writes the **exit snapshot**: the trigger entries of (10.2); route and WARP
  state; contention; the launched PID; whether its readings were complete; and the exit
  code. It records the snapshot's SHA-256.
- **TV2.** It reads all of `launch.json` and hashes it; the hash must equal D_L. The file
  must be canonical and match the schema. `run_id`, `run_kind`, `harness_sha256` and
  `continues` must equal the T0 values, and `exit_statuses` must equal the pinned table.
  `claim.A_D`, `preparing_record.A_D`, `launch_file.authorization_ref` and
  `launch-start record.A_D` must all equal the SHA-256 of the authorization file as
  re-read, and `claim.nonce_sha256` must equal `preparing_record.nonce_sha256`. The
  launch-start record's argument vector must equal the one re-derived from the
  authorization and the one the child recorded as its first event. For a continuation, the
  run record must contain `continuation-eligibility.json`, whose `prior_run_id` and
  `prior_classification_sha256` equal the launch file's.
- **TV3.** It checks the artifact's envelope: its size is 109 to 65,643 bytes; it starts
  with the magic; its last 98 bytes match the trailer pattern exactly; 9 + L + 98 equals
  its size; the payload's SHA-256 equals the trailer's.
- **TV4.** It checks the payload: every byte is between `0x20` and `0x7F`; it is one JSON
  object, with duplicate keys, floats, NaN and infinity rejected; canonical
  re-serialization equals it byte for byte; keys and types are exactly as in (11.4) and
  (11.5).
- **TV5.** It checks the bindings: `run_id` equals the launch file's and names this
  execution's own run directory; `launch_sha256` equals D_L; `exit_status` equals
  `exit_statuses[state]` and the observed exit code; `child_pid` equals the snapshot's
  PID.
- **TV6.** It checks the invariants of (12.2), and that no child event contradicts the
  artifact. Absent events are not a contradiction.
- **TV7.** It records the result, with SHA-256 digests of: the artifact bytes as verified,
  computed from the buffer actually checked; D_L; the exit snapshot; and the child-written
  event portion as read at TV6. For a run kind that uses an acquisition directory, it
  captures the acquisition manifest (11.9) and records its SHA-256; if the manifest cannot
  be captured, TV7 records it as **unavailable**, which changes neither validity nor the
  outcome mapping but makes the execution ineligible for continuation.

A failure of TV2 to TV6 is governed by (11.8) and (12.3). Postflight begins only after
TV7, and records the digest of its results.

**Validity** depends only on the artifact's bytes, the launch file, the exit code, the
exit snapshot and the child's events. Post-exit readings and postflight results never
affect validity; they enter only the outcome mapping. Nothing appends to the artifact, and
it need not be the last of anything in the event log.

*(11.7) Validity, classification and acceptance.*

**Three layers, kept distinct.**
- **Validity** (TV0 to TV7) establishes whether the terminal artifact is valid and binds
  it to the launch file, exit code, exit snapshot and child events.
- **Classification** (12.3 to 12.5) maps the verified evidence to one outcome.
- **Acceptance** is the owner's later review, which decides whether the evidence may be
  relied on.

**The classification record.** The first outcome mapping is computed only from the
evidence verified at TV0 to TV7, together with the recorded postflight results. It is
written **once**, as the classification record, which names the outcome, the mapping rule
applied, every digest the mapping relied on, and — as bound but not mapped — the
acquisition manifest's digest. It is never re-derived from files read later. Its own
digest is recorded in the run record, in the event log and in the stage-review report.

**Re-checks.** At each point below, each recorded digest is recomputed from a fresh read
of its file, and compared with the recorded digest and size. The result is appended to the
run record's audit section.

| # | When | Covers |
|---|---|---|
| 1 | Immediately before the first outcome mapping | Artifact, launch file, exit snapshot, child-event portion, postflight results |
| 2 | When the run record is closed | All of the above, plus the classification record, the acquisition manifest, the identifier claim, `continuation-eligibility.json`, the L0 and L4 fresh manifests, and any launch-refusal record |
| 3 | At the owner's stage review | As 2 |
| 4 | Before any decision that relies on the outcome: prerequisite use, a continuation's authorization, or a continuation's launch (L0 and L4) | As 2 |

- **E-pre, "terminal-evidence mismatch found before classification."** Found at
  re-check 1. It establishes that a file's bytes differed from its recorded digest when
  re-check 1 ran. Under (12.3) it is a contradiction: the first mapping records an
  unclassified termination, and the classification record names the mismatch.
- **E-post, "terminal-evidence mismatch found after classification."** Found at re-checks
  2 to 4. It establishes only that the bytes differed when that check ran, and that the
  last matching check — or the original recording — preceded it. Each event records both
  check times. It does not establish when the change happened.
  - The classification record stands as written.
  - The modified file is preserved as found.
  - A separate **evidence-status record** marks the evidence **compromised**.
  - While compromised, the execution cannot be accepted, cannot satisfy any prerequisite,
    and cannot be continued, without the owner's review and new explicit authorization.

**Limits.** A digest detects a change only when it is checked. A change reverted between
checks, or a consistent change to both a file and its recorded digest, is not detected. No
tamper-proofing is claimed. The TV7 record and the classification record stand only as
records of what was verified and classified then.

**A continuation never alters the prior execution's evidence.** The prior `terminal/`
directory is never written. Its run record receives only the appended audit entries and
evidence-status updates of this clause.

*(11.8) A missing or invalid artifact.*

A missing or invalid artifact is an **unclassified termination**, whatever the exit
status. That covers no `outcome.rec`, a `.partial` only, a TV2 failure, an envelope or
payload failure, and a failed binding or invariant. Nothing assumes that an artifact could
have been written.

*(11.9) The acquisition manifest.*

**Where it is stored.** `data/run-records/<run-id>/acquisition-manifest.json`, written by
the orchestrator at TV7 by exclusive create, flush, `fsync` and rename. It is detached
from the terminal artifact, whose schema and 65,536-byte payload bound are unchanged. The
child never writes or reads it, and D51 sets it no size bound.

**Contents.** Canonical JSON with exactly these keys:

| Key | Value |
|---|---|
| `format` | `"sentinel-acquisition-manifest/1"` |
| `run_id` | the identifier of the execution that captured it |
| `acquisition_dir` | the acquisition directory's path relative to `C:\sentinel` |
| `entries` | a list of entries, each with exactly `path`, `type` (`file` or `dir`), `size` and `sha256` |
| `entry_count` | the number of entries |
| `file_bytes` | the total size of the files |

**Canonical paths.** A path is one or more segments joined by `/`, with no leading or
trailing `/`, no empty segment, and no `\`. A segment is valid only if:

| Rule | Rejects |
|---|---|
| Every character is ASCII `0x20` to `0x7E`, excluding `< > : " / \ \| ? *` | control characters; non-ASCII names; `:` and with it drive-qualified paths, drive-relative paths and alternate data streams; UNC, `\\?\` and `\\.\` forms |
| It is not `.` or `..` | relative traversal |
| It does not end in `.` or a space | names Windows silently alters |
| Its base name — the part before the first `.`, compared without regard to ASCII case — is not `CON`, `PRN`, `AUX`, `NUL`, `COM1` to `COM9`, `LPT1` to `LPT9`, `CONIN$` or `CONOUT$` | reserved device names |

Production writes only ASCII names into an acquisition directory: digests, ISO dates and
export names matching `[A-Za-z0-9_]+` (`ingest/fetch/cfpb.py`). Admitting other names
would need normalization and case-folding rules D51 does not define, so a non-ASCII name
makes the manifest unavailable. No ordinary ASCII filename is rejected.

**The validator.** A manifest is valid only if:
1. it is canonical JSON with exactly the keys above, the stated `format`, and a `run_id`
   matching the retained identifier pattern of (11.1);
2. `acquisition_dir` obeys the path rules and its first segment is `data`;
3. each entry's `path` obeys the path rules; a file has an integer `size` of at least 0
   and a `sha256` of 64 lowercase hexadecimal digits; a directory has null for both;
4. paths are unique and strictly ascending, compared byte by byte as recorded; no two
   paths are equal after ASCII lower-casing; every proper parent of an entry is itself a
   `dir` entry, and empty directories are entries;
5. `entry_count` equals the number of entries, and `file_bytes` equals the sum of the
   files' sizes.

**Capture by handle.** Every check is made on an opened object, never on a path string
followed by a separate open. For the acquisition directory and every entry, the capturing
code:
1. opens the object once, without following a reparse point, and keeps that handle open;
2. reads the object's attributes from the handle, and rejects a symbolic link, junction or
   any other reparse point;
3. resolves the final on-disk path from the handle, and requires it to lie strictly below
   the resolved acquisition directory and to equal that directory joined with the recorded
   path;
4. for a file, reads its size and hashes its content through the same handle;
5. for a directory, enumerates its children through the same handle, recording each name
   exactly as enumeration returns it.

The acquisition directory itself must resolve strictly below `C:\sentinel\data\`, with no
reparse point at any component.

**Change during capture.** Each file's identity, size and last-write time are read from
the handle before and after hashing. Each directory is enumerated before and after its
children are processed. Any difference, or any entry that appears, disappears or cannot be
opened, makes the manifest **unavailable**. Capture never retries until it "settles".

**Comparison.** Two manifests are compared on `acquisition_dir`, `entries`, `entry_count`
and `file_bytes`, byte for byte. A missing entry, an added entry, a rename (one missing
and one added) and a changed `size` or `sha256` are each a mismatch.

**Binding.** The manifest's SHA-256 is recorded at TV7 and bound by the classification
record, whose digest a continuation's authorization binds as
`prior_classification_sha256`.

**Fresh manifests.**

| Captured at | Where its bytes are kept | Where its SHA-256 is recorded |
|---|---|---|
| L0, and L0 passes | Held in memory until L2, then `acquisition-manifest.l0.json` in the new run directory | `continuation-eligibility.json` |
| L0, and L0 fails | `_preflight/<sha256>.manifest.json`, if it was captured and can be written | The preflight-refusal record, with whether E5's comparison matched |
| L4, pass or fail | `acquisition-manifest.l4.json` in the run directory, before the launch-start or launch-refusal record | That record |

If a fresh manifest cannot be persisted, the lifecycle stops at that phase, and the
refusal record says the bytes were not kept.

**What a match establishes, and what it does not.** A match establishes only that the
directory's names, types, sizes and contents are the same as when TV7 captured them. It
does **not** establish: that the directory was unchanged between the child's exit and TV7,
an interval that rests only on (8) and the operator (O-2); that the acquisition's contents
are correct; its `acquisition_id`; its immutable start state or pins; or that D46 to D49
will accept the resume. Those remain production's own checks inside the child.

**Status.** The Windows behaviours these rules rely on fall under D-5 and are established
only by class MV-W.

*Now — (12) Classification and outcomes.*

*(12.1) Refusal-class events.*

Each of these is a refusal-class event:
- any typed source, join, roster, identity or acquisition refusal;
- HTTP 403;
- `FetchFailed`, including after retries are exhausted and on any 3xx;
- a guard refusal under (6), including:
  - a pin mismatch;
  - an export outside the stage's roster;
  - a request outside its permitted context or day;
  - a by-ID request the guard refuses: no context; a malformed, nested, foreign or
    stale-archive context; an active day outside the window; a non-canonical or
    out-of-range ID; an absent or repeated ID; an unset inclusion bit; or a different
    archive date;
  - a disallowed header or header value, such as an outgoing `Cookie`, a credential, an
    explicit `Host`, or an unexpected value of a permitted header;
  - a refusal while the terminal marker is set;
- a harness-integrity failure of H3's context handling — a refused ownership, overwrite or
  archive-identity check, or a cleanup failure, recovered or unverified;
- a child exit with the integrity status;
- a capture failure (B-6);
- any failure of the exact-day audit (6.9);
- a failed count or reconciliation invariant;
- an integrity failure;
- the acquisition commit budget reached;
- a change of route or WARP state;
- a write outside the execution's authorized `data/` roots; any write by a child to
  another execution's run directory; any change by the orchestrator to another execution's
  run directory beyond (11.7)'s appended entries; any change to `data/recon/`; or any
  change to a tracked file;
- a continuation whose own TV7 manifest has an `acquisition_dir` other than the prior
  execution's;
- E-pre, which (12.3) treats as a contradiction;
- E-post, which marks the evidence compromised;
- a measurement that cannot be completed as specified, which includes every E condition of
  (12.4);
- an unclassified termination.

A refusal's record names a refused header and keeps its presence and the refusal, never
its value or any cookie value.

*(12.2) Invariants.*

These are checked at TV6, and any violation is contradictory.
- **I1.** If `cleanup_failed` is false, then `cleanup_verified` is false and `reason`,
  `seq` and `day` are null. If it is true, they are all non-null.
- **I2.** If `cleanup_failed` is true, the state is `integrity`, with `integrity_reason`
  `cleanup_recovered` (if verified) or `cleanup_unverified` (if not).
- **I3.** The state is `integrity` exactly when `integrity_reason` is non-null.
- **I4.** `exit_status` equals `exit_statuses[state]`, which equals the exit code.
- **I5.** `completed` requires `exception` to be null, and `trigger` to be null or at the
  run kind's end checkpoint.
- **I6.** `stopped_by_trigger` requires a `trigger` at a non-end checkpoint the run kind
  reaches, and an exception of kind `trigger_stop`.
- **I7.** `refused` requires `trigger` to be null, a clear latch, a null
  `integrity_reason`, and an exception of kind `harness_refusal`, `fetch_failed`,
  `transport_error` or `typed_refusal`.
- **I8.** `crash` requires `trigger` to be null and an exception of kind `memory`,
  `os_error`, `other` or `unknown`.
- **I9.** `operator_interrupt` requires `trigger` to be null and an exception of kind
  `keyboard_interrupt`.
- **I10.** `integrity` requires `trigger` to be null, and permits any exception, or none.
- **I11.** Trigger kind and source agree with R-T2, R-T3a, R-T3b and R-T7.
- **I12.** An `identity_matches_exit_record` of false requires `integrity`.
- **I13.** Kind-specific descriptor fields are non-null only for their own kind, and
  `harness_reason` is null.

*(12.3) Precedence.*

Every applicable check is evaluated in full before classification, and an E finding never
stops evaluation of any other check. The order is then:
1. **An invalid artifact** (11.8) is an **unclassified termination**.
2. **Any demonstrated contradiction**, anywhere, is an **unclassified termination**,
   regardless of any E. Contradictions include V2, an O1 omission, a forced-action or R-T3
   violation, an invariant violation, a digest or ordinal inconsistency, a field-presence
   conflict, a failed-check sequencing violation, and E-pre.
3. **Otherwise, E** leads to mapping rule 1 or 2.
4. **Otherwise,** mapping rules 3 to 13 apply.

V3 alone is E. V3 with any contradiction is an unclassified termination. Unavailable
authenticity is E unless a contradiction exists. A continuation is classified from its own
evidence alone, under the same precedence.

*(12.4) Terms used by the mapping.*

- **E** is any of: an evidence defect from (10) or (11); a postflight refusal-class event;
  a live refusal-class event in the snapshot; incomplete orchestrator readings; R-T6.
- **B** means a budget trigger in the snapshot or in the artifact.
- **FD** means a system-floor or disk trigger in the snapshot.
- **IU** holds when `complete` is false and `identity_matches_exit_record` is null.
- **R\*** holds when any of these holds: the exception's kind is `unknown`; any link's
  kind is refusal-class (`harness_refusal`, `fetch_failed`, `transport_error` or
  `typed_refusal`) or `unknown`; any link's `chain_continues` is true or null.

**Descriptor completeness.** Classification reads only `kind`,
`identity_matches_exit_record` and each link's `kind` and `chain_continues`. All other
descriptor fields are classification-neutral. An incomplete descriptor therefore never
yields a more permissive outcome than a complete one.

*(12.5) The outcome mapping.*

It is applied only to a valid artifact with no contradiction. The first matching rule
decides.

| # | Condition | Outcome |
|---|---|---|
| 1 | E, with state `completed` | Completion, not compliant |
| 2 | E, with any other state | Refused |
| 3 | `integrity` or `refused` | Refused |
| 4 | B, with `completed` | Completion, not compliant |
| 5 | B, with any other state | Refused |
| 6 | `stopped_by_trigger`, `crash` or `operator_interrupt`, with R\* or IU | Refused |
| 7 | `stopped_by_trigger` | Reviewed interruption |
| 8 | `completed`, with no FD | Compliant completion |
| 9 | `completed`, with FD | Completion after a reviewed trigger |
| 10 | `crash`, with FD | Reviewed interruption |
| 11 | `crash`, kind `memory` or `os_error` | Reviewed interruption (D-4) |
| 12 | `crash`, kind `other` | Ordinary interruption |
| 13 | `operator_interrupt` | Reviewed interruption (D-1) |

**Totality.** `integrity` and `refused` end at rule 2 or 3. `stopped_by_trigger` ends at
rule 2, 5, 6 or 7. `completed` ends at rule 1, 4, 8 or 9; under rule 8, R-T2 forces
`trigger` to be null, and under rule 9 every FD entry was acted on, covered or late.
`crash` ends at rule 2, 5, 6 or 10 to 12. `operator_interrupt` ends at rule 2, 5, 6 or 13.
Taking the first match makes the outcome unique. A `completed` run with FD is never
compliant.

*(12.6) Outcomes.*

| Outcome | Status |
|---|---|
| Compliant completion | Eligible for the owner's stage review. Satisfies no prerequisite until the review accepts it. |
| Completion after a reviewed trigger | Eligible for stage review only after the owner reviews the trigger. The review decides whether its measurements stand. |
| Completion, not compliant | A refusal: not accepted as the stage's result. Never eligible for continuation. |
| Refused | A refusal. Never eligible for continuation. |
| Reviewed interruption | Eligible for continuation after the owner's review, **only** by a new execution under a fresh explicit authorization and a new run identifier (12.7). Eligibility is not permission. |
| Ordinary interruption | Eligible for continuation **only** by a new execution under a fresh explicit authorization and a new run identifier (12.7). Eligibility is not permission. |
| Unclassified termination | An integrity refusal. Never eligible for continuation. Any further execution is a new, non-continuing execution, under new explicit authorization after the owner's review. |

An execution whose evidence-status record shows it compromised (11.7) cannot be accepted,
used as a prerequisite or continued under any outcome.

*(12.7) Continuation, refusal and disposition.*

**(12.7.1) No resumption under an existing authorization.** No execution is ever resumed
under its own authorization or run identifier. A continuation is a new execution in every
respect: its own fresh explicit authorization, run identifier, identifier claim, run
record, terminal directory, launch file, delivery medium, terminal artifact,
classification record and evidence-status record. **Without a fresh authorization there is
no continuation.**

**(12.7.2) Eligibility.** An execution's outcome makes it eligible for continuation only
if it is a reviewed interruption whose owner's review is recorded, or an ordinary
interruption. These block continuation:
- a refusal of any kind, including Completion, not compliant;
- an unclassified termination;
- compromised evidence;
- a classification or evidence status that cannot be verified;
- an unavailable acquisition manifest;
- a prior run already continued, or blocked, under E3;
- a prior run with no classification record.

**(12.7.3) What a continuation's authorization must contain.** Everything (13) requires,
plus the continuation binding of (11.1). A continuation has the prior execution's run
kind. Its command is identical, argument for argument, to the prior execution's command.

**(12.7.4) Acquisition continuity is unchanged.** For a real run, a continuation continues
the prior execution's acquisition **under D46 to D49 unchanged**: the same command,
acquisition directory, `acquisition_id`, immutable start state and pins, and journal
continuity, with D27's corpus-replacement semantics. D51 neither redefines nor weakens
those rules, and production's own refusals still apply inside the child. A continuation of
an offline rerun reuses the same completed acquisition. A continuation of an archive
replay uses newly copied, pin-verified exports.

**(12.7.5) The checks E1 to E6.**

| Phase | First execution | Continuation |
|---|---|---|
| L0 | not run | E1a; then, if E1a passes, every one of E2 to E6, each failure recorded |
| L1 (start) | E1a | ownership of the claim confirmed again (T0 step 1) |
| L4 | E1b and the launch binding | E1b, E2, E3, E4, E5, E6 repeated, and the launch binding |

E2 to E6 apply to continuations only, so first executions are unaffected by E6. Any
failure is a refusal before launch.

**E1a. Reservation and authorization, before the run directory exists.** The steps run in
this order, after the lifecycle nonce has been generated, and the lifecycle proceeds only
if all seven pass.
1. **Validate before any filesystem operation.** The authorization is present and
   complete, and A_D is computed from one read. Its run identifier matches the lowercase
   pattern and is not a reserved device name (11.1).
2. **Open and verify `_claims`** by the trusted traversal and provisioning rules of
   (11.1b).
3. **Enumerate the existing claims** through the verified handle, under the contract of
   (11.1b). If the enumeration is untrustworthy, refuse. If any claim's identifier equals
   this identifier under the ASCII case-insensitive rule, the identifier is already used.
4. **Create the claim** by exclusive create, relative to the verified `_claims` handle.
5. **Write and confirm ownership.** The lifecycle writes canonical claim JSON with exactly
   the schema of (11.1a); flushes, `fsync`s and closes it; reads it back through a
   verified handle; confirms it is canonical and matches the schema; and confirms that its
   `run_id`, `A_D` and `nonce_sha256` equal the identifier, A_D and the SHA-256 of the
   nonce the lifecycle holds in memory. Only then does the lifecycle own the claim.
6. **Scan retained records** under `data/run-records/`, as specified below.
7. **Proceed** only if steps 1 to 6 all passed. No run directory is created otherwise.

| Step | Failure | Case | The identifier |
|---|---|---|---|
| 1 | Authorization or identifier invalid | No F0 case; no reservation was attempted | Not reserved |
| 2 | `_claims` absent with other records present, cannot be opened, or fails path integrity | F0-f | Not reserved |
| 3 | Enumeration untrustworthy | F0-f | Not reserved |
| 3 | A case-insensitive collision with an existing claim | F0-a | Already consumed |
| 4 | The create fails because the claim exists | F0-a | Already consumed |
| 4 | The create fails otherwise, and no claim exists afterwards | F0-b | Not reserved |
| 4 | The create returns ambiguously, and a claim exists afterwards | F0-c | Consumed |
| 5 | The claim exists, but writing, read-back or ownership confirmation fails | F0-c | **Consumed.** No run directory is created. |
| 6 | A retained record marks the identifier as used, or the scan refuses | F0-d | Consumed by this lifecycle's claim |

**The retained-record scan (E1a step 6).** Working through the verified `run-records`
handle, the orchestrator enumerates every entry directly under `data/run-records/` and
classifies each:

| Entry | Treatment |
|---|---|
| The directories `_claims` and `_preflight` | Skipped by this scan; each is verified by its own rules. |
| A directory whose name matches the retained pattern | A run directory. Its **name** marks that identifier as used, and it is examined by state, below. |
| Anything else: a non-directory entry, a directory with any other name, or a reparse point | **Its significance cannot be established. The lifecycle refuses** (F0-d). |

If any entry cannot be enumerated, opened or read safely, the lifecycle refuses (F0-d).

*Fields examined*, all under the comparison rule of (11.1), and no others:

| Location | Field |
|---|---|
| `<run-dir>/preparing.json` | `run_id` |
| `<run-dir>/terminal/launch.json` | `run_id` and `continues.prior_run_id` |
| `<run-dir>/continuation-eligibility.json` | `prior_run_id` |

*Binding-bearing records.* Two records carry a continuation binding, and no other does:
`terminal/launch.json`, in `continues.prior_run_id`; and `continuation-eligibility.json`,
in `prior_run_id`. The launch-start record carries no continuation binding under its
schema, so it is not examined for one. **A record is valid** only if it can be read
safely, is canonical, and matches its schema. Anything else that is present — malformed,
non-canonical, incomplete or unreadable — is **invalid**, and is never read as "no
binding".

*Launched directories.* `launch.json` must be present and valid. If its `continues` is
null, the directory is a first execution, and `continuation-eligibility.json` must be
absent. If its `continues` is an object, `continuation-eligibility.json` must be present
and valid, and its `prior_run_id` must equal `continues.prior_run_id`. Otherwise the
lifecycle **refuses** (F0-d).

*Launch-refused directories.* Nothing is required. The records may be absent, partial or
invalid, exactly as F1 to F4 left them. This is not a refusal.

*Preparing and indeterminate directories.* The two records are judged together:

| `terminal/launch.json` | `continuation-eligibility.json` | Treatment |
|---|---|---|
| Absent | Absent | A legitimately incomplete state. No binding is established. (Case A) |
| Valid, `continues` null | Absent | A valid first-execution state. No binding. |
| Valid, `continues` null | Present | **Refuse** (F0-d): inconsistent. |
| Valid, `continues` an object | Absent | An incomplete continuation state. The binding in `launch.json` is an observed reference and blocks under E3. (Case B) |
| Valid, `continues` an object | Valid, with `prior_run_id` equal to `continues.prior_run_id` | A valid continuation state. It blocks under E3. |
| Valid, `continues` an object | Valid, with a different `prior_run_id` | **Refuse** (F0-d): contradictory binding. |
| Absent | Present | **Refuse** (F0-d): the write order makes this impossible. |
| Present but invalid | Either | **Refuse** (F0-d). |
| Valid | Present but invalid | **Refuse** (F0-d). |

*When the scanning lifecycle is a continuation:*
- **Case A — both records absent.** The directory's prior-run binding is unknown.
  Exclusivity cannot be established, so the lifecycle **refuses** (F0-d).
- **Case B — `launch.json` valid with `continues.prior_run_id` naming P, and
  `continuation-eligibility.json` absent.** The prior run is known from the valid launch
  file. P is recorded as an observed reference, and E3 is applied normally: a continuation
  naming P is refused under E3; a continuation naming a different prior run Q is **not**
  refused by this directory. The missing eligibility record changes nothing about the
  `prior_run_id` the valid launch file already records, and no other binding is assumed.

*When the scanning lifecycle is a first execution,* it does not refuse merely because a
preparing or indeterminate directory is incomplete. The interim rule of (8) and O-3 still
blocks its launch at L4.

*Discovery and eligibility are separate.*
- **Discovery.** Any valid `continues.prior_run_id` or `prior_run_id`, in a directory of
  any state, is recorded as an **observed reference** to that prior identifier, and marks
  that identifier as used. An observed reference decides nothing about E3 on its own.
- **Eligibility.** A binding counts as a continuation of its prior run **only if its
  directory is Launched**. A valid binding in a preparing or indeterminate directory
  counts as a **blocking** reference. A valid binding in a launch-refused directory is an
  observed historical reference only and does **not** count under E3; that directory's own
  run identifier remains permanently consumed.
- `launch.json`.`run_id`, `preparing.json`.`run_id` and each run directory's name mark
  their identifier as used. A valid `continues: null` contributes no binding.

**The launch block is separate.** Passing E1a does not permit a launch. Under (8) and the
interim rule O-3, a foreign preparing or indeterminate directory blocks every launch at
L4.

**E1b. Identity of this execution's own directory, at L4.** All of these hold:
- the directory exists and its name equals the run identifier;
- the claim file exists, is canonical, and names this run identifier and A_D;
- `preparing.json` is canonical, matches its schema, and its `run_id` equals the run
  identifier;
- `claim.nonce_sha256 == preparing_record.nonce_sha256 == SHA-256(in-memory nonce)`;
- the authorization file, re-read now, hashes to A_D;
- `claim.A_D == preparing_record.A_D == A_D`;
- `launch_file.authorization_ref == A_D`;
- `launch.json` hashes to D_L;
- every launch field re-derived from the authorization equals the launch file's;
- for a continuation, `continuation-eligibility.json` matches its recorded digest and its
  `A_D` field equals A_D.

No pre-existing directory is ever adopted or reused.

**E2. The prior evidence.** The prior run directory is in the launched state. Its
classification record's SHA-256 equals `prior_classification_sha256`. Re-check 4 of (11.7)
matches over all of the prior evidence. The prior evidence is not compromised. The prior
outcome equals the binding and is eligible. For a reviewed interruption, the owner's
review record exists and its SHA-256 equals `owner_review_sha256` (G-2).

**E3. Exclusivity.** No other run directory in the launched, preparing or indeterminate
state has a valid `launch.json` whose `continues.prior_run_id` names the same prior run.
Launch-refused directories do not count. E3 relies on the bindings established by E1a step
6; where step 6 refused, E3 is not reached.

**E4. Same command and identity.** The run kind, command, window, corpus root, commit,
harness digests and package freeze equal the prior execution's (G-1).

**E5. The acquisition directory.** For run kinds that use one:
- the prior manifest is valid and not unavailable; its `run_id` is the prior run's; its
  `acquisition_dir` equals the path recorded in the prior run record at TV7; its SHA-256
  equals both the digest recorded at the prior TV7 and the digest bound by the prior
  classification record;
- a fresh manifest, captured now by handle with `run_id` set to the continuation's, is
  valid and not unavailable, and its `acquisition_dir`, `entries`, `entry_count` and
  `file_bytes` are byte-identical to the prior manifest's (G-3).

**E6. Acceptance of the exit-to-TV7 interval.** A later decision recorded in the addendum
explicitly accepts the limitation stated in (11.9), and the continuation's authorization
cites that decision. **No such decision exists.** D51 does not define how a decision
expresses that acceptance, nor how an authorization cites it and the orchestrator verifies
the citation (O-2). Until a later decision defines them, E6 cannot be satisfied, and
**every continuation is refused at L0**. No continuation reaches L1 or L4.

**(12.7.6) Classification and memory.** A continuation is classified from its own evidence
alone. The prior execution's classification record is never altered. A continuation's
memory figures describe its own process only, and are never presented as an uninterrupted
run's peak.

**(12.7.7) Stage acceptance.** A stage review considers the completed execution together
with every execution it continues. Each is identified by its own run identifier, and they
are linked only by the prior-run reference in each authorization. Compromised evidence in
any of them blocks acceptance.

**(12.7.8) Disposition.**
- **After any refusal,** including an unclassified termination, any further execution
  requires the owner's review and new explicit authorization.
- **No bypass.** No other outcome may be reclassified as an interruption. An operator
  interrupt counts only when delivered so that the child records it; a forced kill or a
  power loss is an unclassified termination.
- **Preservation.** For an unclassified termination, every artifact is preserved exactly
  as found. No partial record, missing event or unread latch is taken as evidence that
  cleanup succeeded, that the child stopped at a safe checkpoint, or that any write
  completed.
- **Completion, not compliant.** The corpus and its manifest are technically complete
  under D27 and are left exactly as written. The harness modifies, moves and deletes
  nothing, and compliance is recorded only in the run record. The corpus is not the
  stage's result, and no stage may treat it as a prerequisite. Its disposition —
  including, for Stage B, an authoritative manifest at `data/corpus/` — is decided by the
  owner under a new authorization.
- **Acquisition directories.** A complete acquisition directory may be reused by a later
  stage's offline rerun only as that stage's own authorization specifies, subject to D46's
  verification; such reuse is not a continuation. An incomplete acquisition left by an
  ineligible outcome is never continued; it is preserved as evidence.

*Now — (13) The stage sequence, frozen and not authorized.*

The order is frozen: A, A′, A2, the further measurement, B, then C. No two executions run
concurrently. **A stage may be authorized only after every earlier stage in the frozen
order has been accepted.** A refused or unaccepted stage blocks every later stage until it
is accepted under a new authorization. **A prerequisite stage counts only when the owner's
review has accepted a compliant completion, or a completion after a reviewed trigger whose
measurements the review accepts.** Offline reruns and archive replays contact no source
and are never acquisitions.

**Every execution, including every continuation, has its own explicit authorization.** It
names its run kind, command, window, corpus root, commit, harness digests and allowlist
digest; its own new run identifier; and, for a continuation, the continuation binding. A
continuation belongs to the stage of the execution it continues, and never advances the
frozen order.

| Stage | Run | Prerequisite |
|---|---|---|
| A | CFPB bounded validation: `python -m ingest.cli --source cfpb --start 2024-01-08 --end 2024-01-14 --fetch --corpus-root data/corpus-validation`, through the harness, as a real run; no `--limit`; never `data/corpus/` | Before execution, every authorization prerequisite in (16) must be met; Stage A must then have its own explicit authorization, sought only after those prerequisites are met; and the applicable requirements of (4) to (11), with A's allowlist, must be satisfied. All launch conditions must be rechecked immediately before execution. |
| A′ | Offline rerun of A's completed acquisition into `data/corpus-validation-rerun`, through the harness under A′'s allowlist, with no source contact, measuring verification and every later stage without fetch-stage allocations and verifying records, outcomes and `acquisition_id` | A accepted; (4) to (11) with A′'s allowlist; its own authorization |
| A2 | CFPB bounded validation over January 2024 into `data/corpus-validation-january`, through the harness, as a new acquisition | A and A′ accepted; (4) to (11) with A2's allowlist; its own authorization |
| Further measurement | Designed after A2's review, using the offline rerun of (13.8) and the archive replay of (13.9): offline reruns of A2's and of any later completed acquisition whose stages are projected, each into a corpus root its authorization names (never `data/corpus/` and never an existing root), and archive replays of two windows comparable under C1-X, on pin-verified reconnaissance copies, with no source contact. It supplies exact values for every component those procedures can measure, and cannot supply the API-day or finishing excesses. | A, A′ and A2 accepted; its own design decision, allowlists and authorizations under every rule of D51 |
| B | CFPB full D37.2 window into `data/corpus/`, whose unbounded manifest becomes the authoritative roster lock (D24, D26) | A, A′, A2 and the further measurement accepted; every Rule C model supported; a later decision establishing exact measurement of the API-day and finishing transient excesses, which no method D51 proposes can provide; B's peak-memory and disk projections then found supported and fitting at B's review; (4) to (11) with B's allowlist; a new authorization naming every permitted host, the eleven exports and their pins, the window dates and the allowlist digest. **Under D51 alone, Stage B cannot be authorized.** |
| C | NYC 311 bounded validation over New York civil days 2024-11-02 to 2024-11-04 into `data/corpus-validation-nyc311/`, exercising D50's ambiguous-local-time exclusion on the 2024-11-03 fold | Every preceding stage accepted, including B; NYC 311's stage list under (5.3); its own resource, disk and network readiness reviewed; (4) to (11) with C's allowlist; its own authorization. **Under the frozen order, Stage C cannot be authorized under D51, because Stage B cannot.** |

*(13.1) Retained and peak memory are distinct.*

Retained memory is commit still held when a stage ends; peak memory is the highest commit
a run reaches. A linear retained model does not imply a linear peak model, and per-page
current-commit increments are no evidence of transient peaks.

*(13.2) The recurrence.*

B's peak commit is projected by carrying the recurrence of (5.5) through B's own stage
sequence, each repeated stage instance counting as a stage:
- `Ŝ_1` is the larger interpreter-start commit measured in A and A2.
- `Ŝ_(s+1) = Ŝ_s + Δ̂_s`, where `Δ̂_s` is always the incremental retained change of stage
  `s` alone.
- `P̂_s = Ŝ_s + X̂_s`.
- The projected run peak is the largest `P̂_s` over all of B's stages.

The recurrence covers the measured child only. Each `Δ̂_s` and `X̂_s` falls into exactly
one class:
- **carried:** the same stage at the same driver value as measured; its exact value is
  used, or an interval's upper bound;
- **projected:** the driver differs; it is computed only from exact measured values, by
  Rule P or Rule C, under assumptions a later decision must accept;
- **unknown:** an extrapolated component measured only as an interval, or a structure no
  run has measured.

A retained decrease is credited only for a carried stage. A projected decrease is credited
as zero.

*(13.3) Rule P, the proportional bound.*

For a quantity measured exactly as values `v_j` at driver values `r_j`:
`v̂(r) = max(max_j v_j⁺, r · max_j (v_j⁺ / r_j))`, with `v⁺ = max(v, 0)`.
- If the quantity is non-negative, non-decreasing in its driver, and grows no faster than
  in proportion to it beyond the measured range — the assumption a later decision must
  accept — then `v̂` is never below the true value.
- Observations at a single driver value cannot test that assumption.
- Rule P gives every transient excess, the retained change of every single stage, and the
  retained change of each instance of a repeated stage type that has no Rule C model.

*(13.4) Rule C, the cumulative envelope.*

- **RC-1, the form.** At every instance boundary `k ≥ 1`, the true cumulative retained
  change is `C_k = α + γ·k + β·R_k`, with `γ ≥ 0` and `β ≥ 0`. RC-1 does not apply at
  `k = 0`, where `C_0 = 0` by definition; the once-per-observation cost `α` enters at the
  first instance. So `Q = C_m = α + γ·m + β·R`, and the retained increment of every
  non-first instance `i` is `γ + β·r_i`, which is non-negative and non-decreasing in
  `r_i`. RC-1 is the only assumption that requires a non-decreasing relationship with the
  driver.
- **The envelope.** Two observations cannot separate `γ` from `β`. With `ΔQ`, `Δm` and
  `ΔR` taken between them:
  - `β_R = ΔQ / ΔR`;
  - `γ_m = ΔQ / Δm`;
  - `α̂ = max(Q_2 − β_R·R_2, Q_2 − γ_m·m_2, 0)`.

  `γ_m` and `β_R` are the extreme slopes the observations allow, **not estimates** of `γ`
  and `β`.
- **Conversion to increments, before the recurrence uses it:**
  - `Δ̂_1 = α̂ + max(β_R·r_1, γ_m)` for the first instance;
  - `Δ̂_i = max(β_R·r_i, γ_m)` for every later instance;
  - so `Ĉ_k = Ŝ_T + α̂ + Σ_(i≤k) max(β_R·r_i, γ_m)`, where `Ŝ_T` is the projected start of
    the type's first instance.
- **Why it is an upper envelope.** Every `(γ, β)` the two observations allow is a convex
  combination of `(γ_m, 0)` and `(0, β_R)`. So `γ + β·r ≤ max(β_R·r, γ_m)`, and `α ≤ α̂`.
- **Requirements.** Two observations suffice, provided they differ in both `m` and `R`.

**Model kinds.**
- **A shared-archive model** — normalization pages and API days — compares A and A2. Both
  read the same pinned export #5, at the same commit and freeze. **Their page and API-day
  inputs are not identical:** A's window is a subset of A2's, and the live API records for
  shared days may differ between the runs.
- **The export model** compares two archive replays: `Q` is the sum of `Δ` over their
  "each export but the last" stages, `m` is the number of those stages, and `R` is their
  summed archive rows.

| | Criterion | Applies to | Evidence |
|---|---|---|---|
| C1-S | Shared archive and freeze: both runs used the same pinned exports, read the same archive rows, and ran at the same commit, package freeze and harness digests, each a fresh process. This establishes identity of the archive-phase inputs only. | shared-archive models | equality of recorded values |
| C1-X | Export comparability: the same commit, package freeze, harness digests and interpreter, each run a fresh process; every replayed export verified against its pin; shared exports with the same pin, byte count and row count; at least one shared export in the same "each export but the last" position | the export model | equality of recorded values; archive facts |
| C2 | Count audit: driver values sum to the run's recorded totals (the manifest's `record_count`, the acquisition's API rows, or the archive facts' rows, equal to reconnaissance) | all | readings, manifest, record, archive facts |
| C3 | Positivity: `m_2 > m_1`, `R_2 > R_1`, `Q_2 > Q_1` | all | retained readings |
| C4 | Envelope audit: at every instance boundary `k ≥ 1`, `α̂ + Σ_(i≤k) max(β_R·r_i, γ_m) ≥ C_k` | all | per-instance readings |
| C5 | No absorption: for pages, `H_A ≥ H_A′`, where `H = C_m`; for API days, the assumption AD-2; for the export model, satisfied by construction in a replay | all | A's and A′'s readings; a later decision; the replay's stage list |

Each criterion is a comparison of recorded values, or a named assumption, and needs no
tolerance. **The export model's criteria can be met.** A window from 2024-10-31 to
2025-01-01 selects #7, #8 and #9, whose two exports before the last hold 1,415,572 rows. A
window from 2024-10-31 to 2025-03-01 selects #7 to #10, whose three exports before the
last hold 2,226,591 rows. These windows are illustrations, not authorizations.

**Model assumptions.** Each must be accepted explicitly by a later decision. None is
evidence.
- **PG-1 (pages, driver sufficiency):** a page's retained change depends on the page only
  through `p_k`, plus a per-instance cost.
- **AD-1 (API days, driver sufficiency):** a day's retained change depends on the day only
  through `r_d`, plus a per-instance cost.
- Driver sufficiency alone implies nothing about monotonicity; that follows only from
  RC-1.
- **AD-2:** no absorption by memory a preceding stage freed. Not testable: no run executes
  API days without a preceding archive phase.
- **AD-3:** drift neutrality between A's and A2's live API responses. No procedure D51
  proposes can test it.
- **AD-4:** representativeness up to the full window's largest day (34,091 rows, against
  A2's 6,958). No procedure D51 proposes can test it.

**Falsifiers.**
- **F-1:** C4 fails in either observation. This falsifies the envelope as an upper bound
  for that observation, and so RC-1 as fitted to it.
- **F-2:** within one observation, two non-first instances `i` and `j` with `r_i ≤ r_j`
  have measured retained increments `Δ_i > Δ_j`. This falsifies RC-1 for that observation.
  It cannot show whether driver sufficiency or the linear form failed.

Any falsifier that holds makes the model unsupported.

*(13.5) Repeated stage types.*

| Type | Instances in B | Driver | Observations | Retained increments | Transient excess | Status |
|---|---|---|---|---|---|---|
| Normalization page `k` | 731 pages, 2,036,435 records | `p_k` | A and A2: 7 and 31 pages; 10,050 and 43,920 records | shared-archive envelope from `Ŝ_N0`, the projected commit at verification's end; `Δ̂_1 = α̂ + max(β_R·p_1, γ_m)`, `Δ̂_k = max(β_R·p_k, γ_m)` | Rule P on page rows from exact page excesses in offline reruns; page peak `Ĉ_(k−1) + X̂(p_k)` | retained: supported, in Rule C's sense, only if C1-S and C2 to C5 hold, F-1 and F-2 do not, and a later decision accepts PG-1 and RC-1; transient: supported if at least one page excess is exact |
| API day `d` | 733 days, 8,188,022 rows | `r_d` | A and A2: 9 and 33 days; 37,422 and 150,837 rows | shared-archive envelope from `Ŝ_API`, the projected commit at the start snapshot's end; `Δ̂_1 = α̂ + max(β_R·r_1, γ_m)`, `Δ̂_d = max(β_R·r_d, γ_m)` | unknown: no procedure proposed in D51 measures it exactly | retained: **not supportable under D51**; a later decision could support it only after C1-S and C2 to C4 hold, F-1 and F-2 do not, and it accepts AD-1 to AD-4 and RC-1; transient: unknown |
| Each export but the last | 10 exports, 8,643,971 rows | `e_j` | two archive replays comparable under C1-X | export envelope from the projected commit at `read_archive`'s entry | Rule P on export rows from the replays' exact excesses | unknown on A, A′ and A2; supported only if the replays satisfy C1-X and C2 to C5, F-1 and F-2 do not hold, and a later decision accepts RC-1 |
| Export download | 11 | the export's bytes | A and A2, for export #5 | Rule P; every B export is no larger than #5 | Rule P, equal to #5's exact excess | supported if #5's values are exact |
| Partition write (record or outcome) | 2 of each | the partition's records | offline reruns | Rule P per instance; 814,385 and 1,222,050 records | Rule P per instance | supported if exact |

*(13.6) Single stages.*

| Stage | Driver | Rule | Measured exactly in | B |
|---|---|---|---|---|
| Imports, setup, discovery, start snapshot, exit | none | carried | A, A′, A2 | carried |
| Last export and merge, where the merged key array is built | all archive rows of the selected exports | P | archive replays: 945,532; 2,226,591; 3,029,664 | 9,682,354 |
| Finishing | API days, and the largest page | — | no procedure proposed in D51 measures this exactly | unknown |
| Verification | pages, and the largest page | P on each, the larger | offline reruns | 731 pages; 20,524 |
| Post-normalization checks | held records `n` | P | offline reruns: 10,050 and 43,920 | 2,036,435 |
| Clearing the corpus; the manifest | `n` | P | offline reruns | 2,036,435 |

*(13.7) Masked transient peaks.*

`PeakPagefileUsage` and `PeakWorkingSetSize` never reset, so a stage whose peak stays
below an earlier one yields only an interval.

| Component | Masked in a fetch run by | Procedure | Result |
|---|---|---|---|
| Verification; normalization pages; post-normalization checks; clearing the corpus; partition writes; the manifest | the fetch stages | **Offline rerun** (13.8) | exact where a stage sets a new lifetime high; otherwise an interval, and unknown when extrapolated |
| Each export but the last; the last export and the merge | the downloads | **Archive replay** (13.9) | exact where a stage sets a new lifetime high |
| API days | the archive phase, in every process that can reach one | **None.** `fetch_day` needs the archive the production `read_archive` builds in the same process; a resumed fetch re-reads every pinned export first; a replay of `fetch_day` alone would need a harness-built archive and a fake transport, and would omit the transport's handling of the body and the page's caching and journaling | unknown |
| Finishing | the archive phase | **None.** Completion needs the fetcher's archive and every journaled slice in the same process | unknown |

*(13.8) Offline rerun.*

Production `ingest.cli.main` with `--fetch` on a completed acquisition reuses it with zero
requests, with the network refused. It reproduces the production code of every stage from
verification to exit, on the run's own retained pages and record. No fetch stage precedes
it, so its excesses are applied to B's projected starts under the assumption that memory
freed by earlier stages can only absorb a transient, never enlarge it.

*(13.9) Archive replay.*

- **Inputs.** The orchestrator copies the window's pinned exports from `data/recon/` into
  the replay's directory under its run record, and verifies each against its pin.
- **The child.** A fresh child calls the production `select_exports` on the retained
  reading-room page, and the production `read_archive` of a `_Run` built with the
  production constructor. That `_Run`'s context has an HTTP client that refuses every
  request, and a directory that is the replay's.
- **What it reproduces.** `read_export`, each sort, the spill and the merge, on
  byte-identical exports, for the same window and spill days.
- **What it does not run.** No acquisition is opened, and no API day, finishing or later
  stage runs.
- **The audit.** Its spill files and archive facts must equal a fetch run's on the same
  exports where one exists, and the reconnaissance row counts otherwise.
- **Its checkpoint** is `end_of_replay` only.

*(13.10) Supportability and composition.*

- **Stage B's projection is unsupported.** The API-day and finishing transient excesses
  cannot be measured exactly by any procedure D51 proposes, so they remain unknown, and a
  projection with any unknown component is unsupported. The API-day retained model is also
  unsupportable under D51. **Stage B cannot be authorized under D51**, and nothing (13.4)
  allows a later decision to accept weakens that. Only a later decision adopting a method
  able to observe those peaks could change it — for example a trace of the process's
  commit charge over time, or instrumentation inside the fetcher, which would be a code
  change. D51 proposes neither, and assumes neither would succeed.
- **The further measurement** still serves the retained models and every component its
  procedures can measure. It establishes nothing beyond its observed range.
- **Composition rules:** the recurrence receives only increments; every Rule C envelope is
  converted before use, and Rule P is applied to each stage's or instance's own change;
  every retained change enters later stages' starts exactly once; a transient enters only
  its own stage's peak; a projected release enters as zero.
- **If every component were known,** `P̂` would be an upper projection that fits only
  below the 6,000,000,000-byte budget, and the same treatment would apply to the working
  set against B's preflight available memory.
- **The protections are not projections.** The budget trigger and the system floor protect
  a run; they never substitute for a supported projection.

*(13.11) B's disk projection.*

It scales each component measured in A and A2 by its own driver: exports by the pinned
bytes of B's roster (773,033,865); API responses by API rows (8,188,022 over 733 days);
and spill files, pages and corpus by records. It must fit the free disk at launch, and is
decided at B's review. If the memory or disk projection, once supportable, does not fit,
D44's redesign decision comes first. C does not exercise a spring gap, and whether one
must be validated before NYC 311's full window is left open. NYC 311's full window needs
its own go/no-go on its measured memory and disk. The sequence is CFPB first, then
NYC 311.

*Now — (14) Stop conditions.*

A run stops, and its evidence is preserved, on every trigger of (9) and every
refusal-class event of (12.1), with no fix-forward and no automatic rerun. Its outcome
under (12) alone decides whether it is eligible for continuation. A continuation, if any,
needs its own fresh authorization (12.7).

*Now — (15) Retention.*

All of the following is kept under the gitignored `data/` tree.

- **The acquisition directory, complete as produced:** its journal, start state, record,
  rewinds, pages, archive exports, API responses, spill files and quarantine.
- **Each validation corpus,** with its outcome sidecar and version-4 manifest, left as
  written whatever the outcome.
- **`data/run-records/<run-id>/`, one per execution, holding:**
  - the command, commit, environment and package freeze, and the network path;
  - `preparing.json`; the launch-start or launch-refusal record; and every partial or
    invalid `launch.json` or `.partial`, exactly as found;
  - the allowlist file and its digest;
  - a replay's copied exports, spill output and audit;
  - the preflight and quiet-period readings;
  - every orchestrator reading and every child reading of (5), with their counter
    brackets;
  - every prepared request's sanitized representation — never the raw request, and no
    request-header value except one of H6's exact permitted pairs;
  - each response's harness response metadata, including each `Set-Cookie`'s name, domain
    and path, never its value;
  - attempt, status, retry and wait evidence;
  - every refusal event;
  - every by-ID guard decision, with its context sequence number, active API day, the
    number of keys carrying the ID, and the matched key's inclusion bit and archive date;
  - every successful logical request's `response_sha256`;
  - every child check event, including failed reads;
  - every stage's boundaries, start, retained change, peak and excess, for commit and
    working set;
  - every instance's driver value;
  - each trigger's entry, acting check and latency;
  - the exact-day audit's result for each by-ID request;
  - the disk breakdown by subdirectory and the longest created path;
  - the exit status;
  - the `terminal/` directory as found;
  - the exit snapshot, the verification result and its digests;
  - the acquisition manifest; for a continuation, `continuation-eligibility.json` and the
    L0 and L4 fresh manifests;
  - the classification record and its digest, the evidence-status record, and the audit
    section of every re-check;
  - the harness copies and digests, and the authorization's reference.
- **`data/run-records/_claims/`,** holding every identifier claim, permanently and
  unmodified, including incomplete ones.
- **`data/run-records/_preflight/`,** holding every preflight-refusal record and any
  `<sha256>.manifest.json`, unmodified.

**No run identifier, claim or run directory is ever reused**, including a launch-refused
one. A prior execution's run directory is preserved as found, apart from (11.7)'s appended
entries, and its `terminal/` directory is never written.

**No retained artifact, harness or production, holds a cookie value, the value of any
credential or other secret-bearing header, any exception text, or the raw lifecycle
nonce.** Production acquisition evidence keeps only the response-header fields of (6.3),
and no request header. The projection worksheets of (13) are retained alongside, every
value traced to a run record. Nothing under `data/` is committed, and `data/recon/` is
only ever read, never modified.

*Now — (16) The order of work, authorization prerequisites and launch conditions.*

1. D51 was conditionally approved on 2026-11-16. Its entry is committed as a
   documentation-only change, under its own separate authorization.
2. The trigger-delivery mechanism (D-3) and the child launch method (U6) are then
   designed.
3. Building the harness, and running its validation activities (4.6a), are each a separate
   authorized step. Neither contacts a source or touches `data/`. A validation
   authorization permits validation activities only; it permits no measured execution.

**A first execution's authorization may be sought only when all of these hold:**
- the test gate is green at the commit to be authorized;
- every mock-validation class of (4.6) has passed at that commit and at the harness
  digests the authorization will name, except the E6-dependent cases, which cannot yet be
  built. Every continuation refusal case is included;
- the Windows filesystem integration tests (MV-W1 to MV-W11) have passed, with no case
  unestablished, at that commit and those digests, and their evidence has been reviewed;
- the child-identity requirement of (4.2) is met and validated (U6 resolved);
- M-1 to M-4 are implemented, and D-3's mechanism has been validated;
- for the stage concerned, every earlier stage has been accepted (13).

**A continuation's authorization may be sought only when, in addition:**
- its prior execution has been classified as an eligible interruption, with the owner's
  review recorded where required;
- the later decision required by E6 exists;
- the E6-dependent cases of (4.6) have passed.

None of these exists.

**Separately, the launch lifecycle** of (11.1a), L0 to L5 — including the quiet-period
gate of (7) and the conditions of (8) — is mandatory for every launch, whatever an
authorization says. No phase waives any prerequisite, integrity check or launch gate.

**Neither gate is met now:**
- the harness does not exist, so its validation has not passed;
- U6 and D-3 are unresolved;
- the launch gate has not been evaluated by any quiet period, and the only headroom
  evidence is two 2026-10-10 snapshots below 6,000,000,000 bytes (1);
- D51 authorizes no execution.

**A fresh ten-minute quiet-period evaluation immediately before launch remains
mandatory.**

*Now — (17) What the conditional approval of D51 covers, and what it does not authorize.*

**D51 was conditionally approved by the owner on 2026-11-16, as a specification only.
Nothing in it is implemented by appearing in it or by being approved, and the approval
authorizes no activity.**

**What is approved, conditionally.** The **resource-readiness framework** is approved
conditionally, as a specification for a future external harness and for the conduct of
separately authorized executions. It covers:
- the environment record (1);
- the harness requirements of (4), including the child-identity requirement and the
  validation rules of (4.6a);
- the memory readings, stages and counter-based attribution (5);
- the guard specifications, the active-day by-ID guard and the exact-day audit (6);
- the launch gate and conditions (7, 8);
- the triggers, checkpoints, forced-action rule, delivery-medium lifecycle and binding
  invariant (9);
- the ordering and observation-evidence rules (10);
- the run-identifier rules, the identifier claim, the launch lifecycle, the terminal
  artifact and the acquisition manifest (11);
- the classification, precedence, outcome and continuation rules (12);
- the stage sequence and the authorization gates;
- the recurrence, Rules P and C with their stated assumptions, the offline-rerun and
  archive-replay procedures, and the further-measurement requirement (13);
- the stop conditions and retention rules (14, 15).

Conditional approval means: the harness, once built, becomes usable only after its
validation passes; usability authorizes no execution; every execution, including a
continuation, needs its own explicit authorization; and the prerequisites and dependencies
of (18) remain unestablished until shown.

**Not authorized by D51 or by its conditional approval:**
- any measured execution;
- the building of the harness, which needs its own authorization;
- any validation activity, which needs its own validation authorization;
- any repository change other than appending D51's own entry, which itself requires
  separate authorization;
- any request to, or download from, either source;
- ingestion, model training, threshold fitting or real-corpus evaluation;
- README metrics;
- Phase 3 wiring.

**Stage B cannot be authorized under D51**, and under the frozen order, neither can Stage
C.

*Now — (18) Register of prerequisites, dependencies, owner decisions and blockers.*

None is established. "Blocked" below means that no measured execution may be authorized; a
validation activity under its own validation authorization is never blocked by this
register.

| Item | Statement | Interim consequence |
|---|---|---|
| **U6** | The measured child must be the interpreter process, not the venv launcher (4.2) | Blocked |
| P-T1 | `QueryPerformanceCounter` is one counter, comparable across both processes, non-decreasing, at equal frequency | Blocked; attribution (5.4) unavailable |
| P-T2\* | A delivery completed before `check_started_qpc` is returned by that check's read ((9.3)(f)) | Blocked; R-T6 makes any trigger evidence E |
| P-T3 | The counter readings bracket their actions and the visibility-read call; delivery completion is defined by the mechanism | As P-T2\* |
| P-T4 | No premature or stale return. Mechanism part: (9.3)(c), (d) and (g). Per-run part: M-op2 at each launch | Blocked until the mechanism part is established; a failed per-run check is a launch refusal |
| P-T6 | The binding invariant B-1 to B-6 holds at the harness digest | Authenticity unavailable, so any trigger evidence is E |
| P-T5 (option, not adopted) | A publication sequence number, or orchestrator-served reads | V3 remains E; authenticity remains procedural |
| M-1 to M-4 | Delivery brackets; check events at every safe checkpoint; contiguous `seq` with `terminal_begin`; trigger entries with `request_id`, `reading` and `threshold` | Any run with a trigger is E |
| D-1 | No orchestrator record of an operator's interrupt request | `operator_interrupt` maps only to a reviewed interruption |
| D-2 | Typed-refusal classes not enumerated from source | Every `ingest`-defined exception is refusal-class |
| D-3 | The delivery mechanism, which must implement and validate (9.3)(a) to (g) | Blocked |
| D-4 | Windows disk-full error codes unverified | Every `os_error` crash is a reviewed interruption |
| D-5 | Untested platform semantics. The filesystem parts are established only by MV-W1 to MV-W11: exclusive creation of files and directories; reserved names; trailing dots and spaces; case-insensitive matching; reparse points; handle-relative opens and creates; handle-based final paths; the hold protection of MV-W9(b); handle-based enumeration; behaviour under concurrent change; rename without overwrite; `fsync`; persistence of a created file across a process crash. The remainder — the counter definitions, CPython weak references, exception identity, `os._exit` — is established only by the other validation classes. Resolved only when every required case has passed, with none unestablished, and the evidence has been reviewed | Blocked |
| D-6 | No authoritative `harness_reason` enumeration | `harness_reason` stays null |
| O-1 | How a foreign preparing or indeterminate directory is released | It stays blocking |
| O-2 / E6 | The owner's acceptance of the exit-to-TV7 limitation, and the form of its decision and citation | Every continuation is refused at L0; first executions are unaffected |
| O-3 | Whether a foreign preparing or indeterminate directory blocks non-conflicting launches | It blocks every launch |
| O-4 | Whether a symbolic-link fixture may be built if it needs a privilege | That case is unestablished; D-5 stays open |
| G-1 | Whether a continuation may change commit, harness digests or package freeze | Equality is required |
| G-2 | The format of the owner's review record | A record referenced by digest is required |
| G-3 | Production's D46 acquisition-identity fields were not re-read | Byte equality of the whole acquisition directory is required |
| RC-1, PG-1, AD-1 to AD-4 | Rule C's form and model assumptions | No Rule C model is supported until a later decision accepts them; the API-day retained model is unsupportable under D51 |
| Disk basis | The ~17 GB figure is an owner estimate | It remains the declared lower bound of (8) |
| C's stage list | NYC 311 stages are undefined (5.3) | A prerequisite of C's authorization |
| Launch gate | Not evaluated; two historical snapshots below the minimum | A fresh quiet period is mandatory |

**Platform limitations that no specified test can remove.**
- **Durability of a directory entry across an OS crash or power loss is unestablished.**
  If a claim, its run directory and every record naming an identifier were all lost in
  such an event, the harness could not detect reuse. Only the spent-authorization rule of
  (11.1a) covers that case, and it is not machine-verifiable.
- The drive letter `C:` and the volume root are trusted, not verified.
- The lifecycle nonce does not establish process liveness, and does not defend against a
  process that can read the orchestrator's memory or replace its records.

*D42 is preserved.*

Plan Tasks 23 to 25 implemented the fetchers of D42's first deferred item. Its formal
closure is left to a separate decision entry, and D51 closes none of D42's items. Items 2
to 7 remain open. **Phase 2 is not complete, and no document may describe it as
complete.**

*Scope:* D51 is a documentation-only decision, conditionally approved by the owner on
2026-11-16 as a specification. Approval of the framework is not authorization to implement
or execute it. Its only repository change is its own entry in the addendum, which is
appended only under a separate authorization. **D1–D50 are unaltered**, and every existing
repository requirement — code, tests, `requirements/`, CI and other documents — is
unchanged. The requirements D51 states apply to a future external harness, kept outside
the repository, and to the conduct of future authorized executions. They are approved as a
specification only. None of them is implemented by appearing in D51 or by its approval.
Building the harness, running any validation activity and conducting any execution each
require their own explicit authorization, and none is authorized.

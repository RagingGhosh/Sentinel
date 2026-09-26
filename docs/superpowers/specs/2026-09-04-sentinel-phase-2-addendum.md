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

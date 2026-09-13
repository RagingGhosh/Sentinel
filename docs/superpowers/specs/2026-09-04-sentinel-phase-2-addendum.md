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
read by training and evaluation code only. They are never migrated into, joined
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

This is one of two conditions under which ingest refuses before writing
anything. The other is §2.6: a `--limit`ed run whose target corpus root already
holds an authoritative corpus (D26).

### 2.6 Corpus replacement

Ingest writes a source's partitions and manifest into its target corpus root,
replacing the files it writes rather than merging into them. An unbounded rerun over
an unbounded corpus is therefore idempotent — the same window and raw cache
produce the same part files and the same `corpus_id` — and that rerun is the
resume path. What was never stated is what a `--limit`ed run may do to a root
that already holds a full corpus. Left unstated, it replaced it: a truncated run
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
| an authoritative corpus (`limit: null`) | **Allowed.** The resume path, and idempotent | **Refused** with `AuthoritativeCorpusExists` |

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

**Deliberately not decided here.** An unbounded run over a narrower window than
an existing authoritative corpus also replaces it, discarding records outside
the new window. That is visible in the manifest's `window_start` and
`window_end`, and is a different question from truncation. This section does not
address it.

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

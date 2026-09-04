# Sentinel — Phase 2 Specification Addendum

**Date:** 2026-09-04
**Status:** Approved with amendments (C1/C2 resolved 2026-09-04; see D15–D18)
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
    timely_response: bool        # see §4.1

@dataclass(frozen=True)
class NYC311Outcome:
    external_id: str
    closed_at: datetime | None
    resolution_hours: float | None   # see §4.2
```

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
| **`TransferFeaturesV1`** (a separate, smaller subset) | The three features the reduced-feature cross-domain experiment accepts, per §5.4. Versioned independently of `RiskFeaturesV1`. |
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

Permitted use: the secondary transfer experiment of §5.4, with imbalance-aware
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
through the explicitly-labelled transfer experiment in §5.4.

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

### 5.4 Cross-domain transfer — demoted to a secondary robustness experiment

Phase 1 §6.4 made cross-domain transfer a headline claim. **The measurement
does not support that weight.** The transfer target — CFPB `timely` — is
0.61% minority overall and 1.12% within narratives, a ratio of roughly 163:1
and 88:1 respectively.

Transfer is therefore **retained only as a secondary robustness experiment**,
reported after the in-domain results and never as the project's ML headline.

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
evaluated on CFPB records against `cfpb_timely_response`.

**Naming and reporting are binding.** The experiment is called the
**reduced-feature cross-domain robustness experiment** wherever it appears — in
artifact metadata, the README, and any published table. It must never be
described, labelled, or tabulated as the performance of the primary risk model.
Its artifact carries its own `model_name`, its own `model_version`, and a
`feature_spec` of exactly the three names above, so the two models cannot be
confused at load time.

**What it can and cannot tell you.** A reduced-feature model that transfers
poorly may be failing because of domain shift, because three weak features are
insufficient, or both — the design cannot separate those. To make the comparison
interpretable, the same three-feature model is **also evaluated in-domain on
held-out 311**, so the transfer gap is measured against that reduced-feature
ceiling rather than against the five-feature primary model. Comparing a
three-feature cross-domain score to a five-feature in-domain score would
attribute to domain shift what may simply be missing features.

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

**The transfer result is never reported as a standalone number.** Every figure
appears beside the trivial baseline computed on the same split, in the same
table, so a reader sees the model and the do-nothing comparison together. A
PR-AUC quoted alone invites the reader to supply their own intuition about what
is good, and at a 1.12% base rate that intuition will be wrong. The baseline is
part of the result, not a footnote to it.

A poor transfer result is published as a legitimate finding about domain shift.

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
construct ten fields in order to call a six-field model would be asked for
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
(§5.4).**
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

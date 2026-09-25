# Task 19 — Reduced-feature cross-domain cross-target robustness probe: contract

**Status: FROZEN.** The seven decisions O1–O7 that this document previously
carried as open are recorded verbatim in §16 and folded into the clauses they
govern. Implementation may begin; one residual wording question is noted in §17
and does not block it.

The contract is reconstructed from what the repository fixes — addendum §4, §4.3,
§5.4, D15, D16, D19, D20, D34, D35, D37, D38, and plan Task 19 — plus the O1–O7
rulings. Nothing here is invented. The rulings belong in the addendum's decision
log as **D39** when Task 19 is implemented, following the pattern D36, D37 and
D38 set; this document is the working contract D39 ratifies.

---

## 1. What the probe is, and what it is not

A **distinct** three-feature `HistGradientBoostingClassifier` trained on NYC 311
against `nyc311_sla_breach`, evaluated twice: once in-domain on the NYC 311 test
period, once cross-domain on the CFPB test period against
`cfpb_timely_response`.

It is **not** the primary risk model scored on other data. §5.4 establishes that
the primary `RiskFeaturesV1` model *cannot* be scored on CFPB at all — two of its
five features need resolution times CFPB does not publish, and §4.3 prohibits
substituting `timely` for a breach rate. The impossibility is structural, not a
preference.

**Binding prohibition (§5.4, D19).** The probe is never described, labelled,
summarised or tabulated as evidence that the 311 SLA-risk model transfers to the
CFPB task. "Transfers to", "generalises to", "works on CFPB" are defects in the
report, not stylistic choices.

---

## 2. Feature schema — FROZEN

`TRANSFER_FEATURES_V1`, already implemented at `ml/training/features.py:101`:

| # | Name | Source |
|---|---|---|
| 1 | `submitted_hour` | the record's submitted instant in its source's local representation (§2.4, D21, D32) |
| 2 | `submitted_weekday` | the same instant's weekday |
| 3 | `text_length` | character length of `CorpusRecord.text` |

Order is the spec's order and is load-bearing. `feature_spec_version` is
`transfer_features_v1`, versioned independently of `risk_features_v1` so the
probe can never be read as the primary model.

No fourth feature is added. No aggregate feature appears: `build_features` is
called with `aggregates=None`, and a spec naming an aggregate raises
`FeatureUnavailable` — which is how §5.4's impossibility is an error rather than
a silent fallback.

**Target-derived aggregates are excluded by construction**, not by convention:
`category_mean_resolution_hours` and `category_breach_rate` are simply not in the
spec, and the `None` aggregates argument makes requesting them raise.

---

## 3. Populations and boundaries — FROZEN

**Source (training).** NYC 311, loaded through `load_corpus("nyc311")`. The
window is Task 17's: `2024-01-01T00:00:00Z` through `2025-12-31T23:59:59Z`
inclusive (D37.2), and the split is Task 17's: `temporal_split` over the windowed
records at `DEFAULT_FRACTIONS` (70/15/15). The probe **reuses those exact
boundaries and does not compute its own** (plan Task 19, finding I6) — otherwise
the reduced-feature in-domain reference and the primary model's in-domain figure
would rest on different data and the reader's obvious question, "what did
dropping two features cost?", would be answered with an invalid comparison.

Because `temporal_split` is deterministic given the same timestamps and
fractions, recomputing it over the same windowed population reproduces Task 17's
cut dates exactly. The probe does **not** import from
`ml/training/experiments/risk.py`; it applies the same frozen inputs.

**Training uses the NYC 311 TRAIN period only** (O5). The validation period is
produced by the split and then left unused: this probe tunes nothing and selects
no threshold, so there is nothing for validation to decide.

**In-domain evaluation.** The NYC 311 **TEST** period from that same split,
evaluated exactly once (O5).

**Cross-domain evaluation.** The **CFPB Task 16 test population with persisted
`timely_response` outcomes** — that is, the test period of `temporal_split` over
the CFPB corpus at `DEFAULT_FRACTIONS`, restricted to records whose outcome is
present in the CFPB outcome sidecar (§5). Evaluated exactly once. No record that
tuned Task 16's abstention threshold appears here, because that threshold was
selected on CFPB validation.

Both period identifiers — the 311 cut dates and the CFPB cut dates — are recorded
in the report.

---

## 4. Targets — FROZEN, and deliberately non-equivalent

| | Source | Evaluation |
|---|---|---|
| Domain | NYC 311 | CFPB |
| Target | `nyc311_sla_breach` | `cfpb_timely_response` |
| Means | resolution exceeded the request type's training p75 (Task 13 thresholds, frozen) | the company replied inside CFPB's 15-day window |

§4 defines these as non-equivalent constructs and §4.3 prohibits combining them
under a single name. The probe is the one explicitly-labelled exception, and is
exploratory analysis rather than a claim that either target substitutes for the
other.

**Polarity mapping (§5.4 fact 5).** The model outputs P(adverse). "Adverse" is
`nyc311_sla_breach == True` in the source and `cfpb_timely_response == False` in
the evaluation domain. This correspondence is an interpretive choice, not a fact
about the data, and is stated in the report.

**Mechanically**, CFPB records enter every metric as the derived adverse boolean
`not timely_response`, so both evaluations share one roster `(False, True)` and
one positive label `True`. That is what allows a single frozen baseline (§7) to
score both. The derivation happens at the metric boundary and creates no field,
column or variable combining the two targets — §4.3 is satisfied.

**Source labels.** Produced by Task 13's frozen thresholds fitted on the 311
**training period only**, exactly as Task 17 does. No threshold mathematics is
recomputed here.

**Open records (O6).** The existing frozen semantics are reused unchanged. An
open NYC 311 request — one with no resolution time — remains part of the corpus
and of split construction, receives **no fabricated label**, contributes **no
supervised training or evaluation label**, and may still carry the three
aggregate-free structural features. CFPB has no analogous case and none is
invented: `timely_response` is an explicitly persisted target, and
`ingest/sources/cfpb.py` already refuses any row whose `timely` is not exactly
`Yes` or `No` (`InvalidTimelyValue`), so every persisted CFPB record carries a
definite outcome.

---

## 5. CFPB outcome persistence (O1) — FROZEN

Task 19 is the task that first consumes CFPB outcomes, and D37's scope clause
assigns the sidecar schema to exactly that task: "a future source's sidecar
schema belongs to the task that first consumes it, not to this one."

**A source-specific CFPB outcome sidecar is persisted**, carrying exactly three
fields per record:

| Field | Meaning |
|---|---|
| `external_id` | the record this outcome belongs to |
| `timely_response` | the boolean target, `CFPBOutcome.timely_response` |
| `date_sent_to_company` | `CFPBOutcome.sent_to_company_at` — provenance only |

`date_sent_to_company` is **provenance evidence and nothing else**. `schema.py`
already states it is "never a feature and never a target"; persisting it does not
change that, and the probe must not read it into any feature or label.

**Constraints, all binding:**

- `timely_response` is **never** written into `CorpusRecord.label`.
  `CorpusRecord.label` remains the CFPB product/category taxonomy that Task 16's
  roster consumes.
- The sidecar is **manifest-backed and integrity-checked**, on the same terms as
  Task 17's: the manifest decides which files exist, and every listed file's
  bytes are verified before a single outcome is yielded.
- **NYC 311's `load_outcomes()` API and semantics are preserved unchanged**,
  including its `Iterator[NYC311Outcome]` return type and its
  `OutcomeSidecarNotFound` absence error.
- A **CFPB-specific loader/API** is added rather than making the established
  NYC 311 outcome type contract generic.
- **Task 17's NYC 311 outcome schema is not redesigned.** Its three columns,
  its resolved/open pairing rule and its error vocabulary stand.
- `ingest/storage.py`, `ingest/manifest.py` and `ingest/cli.py` may be modified
  **only as far as this source-specific sidecar requires**.

**The sidecar's symbols are frozen**, each mirroring its NYC 311 counterpart so
the two sidecars read as one design with two schemas rather than two designs:

| CFPB symbol | NYC 311 counterpart | Meaning |
|---|---|---|
| `ingest.storage.CFPB_OUTCOME_ARROW_SCHEMA` | `OUTCOME_ARROW_SCHEMA` | the three columns of §5, in that order |
| `ingest.storage.write_cfpb_outcome_partition(outcomes, source, year, part_index, root=)` | `write_outcome_partition` | writes one part file, sorted by `external_id` |
| `ingest.storage.read_cfpb_outcome_parts(paths)` | `read_outcome_parts` | yields `CFPBOutcome` in stored order |
| `ingest.manifest.load_cfpb_outcomes(years=None, root=CORPUS_ROOT)` | `load_outcomes` | returns `(manifest, Iterator[CFPBOutcome])`, manifest-backed |

These names are authoritative: the RED suite at
`tests/ingest/test_cfpb_outcomes.py` is written against them, so renaming one
means amending this contract and those tests together, never the code alone.
The NYC 311 symbols keep their names, signatures and return types untouched.

**Two superseded Task 17 assertions are updated, and only those two.**
`tests/ingest/test_cli.py` carries two tests written to stop Task 17 from
pre-empting the schema this decision now defines — one asserting that CFPB
ingest creates no outcome sidecar, the other that `load_outcomes("cfpb")`
raises the typed absence error. Task 19 has arrived, so both describe a scope
that no longer holds. They are **inverted rather than deleted**, so the same
two properties stay pinned in the same place:

1. CFPB ingestion **declares and persists** its CFPB outcome sidecar, listed in
   the manifest.
2. The NYC 311-specific `load_outcomes()` **never successfully deserialises a
   CFPB sidecar as `NYC311Outcome`**. The property is failure and
   non-deserialisation, **not** a particular exception class: no existing
   contract fixes one for a cross-source read, and the typed schema rejection
   already in `ingest/storage.py` is sufficient fail-closed behaviour.

**NYC 311 production behaviour is unchanged by both**: no schema, loader,
writer or error type moves to make either test pass, and no writer-level
source-slug guard is added — `ingest/sources/__init__.py` keeps source names out
of storage. **No other Task 16–18 test may be modified**, and the CI exception
of §14 is unchanged.

---

## 6. Model and preprocessing (O2) — FROZEN

`HistGradientBoostingClassifier` with **exactly Task 17's configuration**:

```
learning_rate=0.1        max_iter=100           max_leaf_nodes=31
max_depth=None           min_samples_leaf=20    l2_regularization=0
early_stopping=False     class_weight=None      random_state=17
```

Every parameter not named takes the pinned scikit-learn 1.9.0 default, and the
report records the versions that supplied them. Reusing Task 17's configuration
is a decision taken here, not an inheritance: the probe is a distinct model and
D37.10's scope is Task 17.

- **No hyperparameter tuning**, of any kind.
- **No preprocessing.** No scaling, no imputation, no encoding. The three
  features are numeric by construction and the estimator handles `NaN` natively.
- **No resampling and no class weighting.** §5.4 requires the imbalance to be
  reported, not engineered away — the rule D36 and D37 applied.
- **No tuning against the CFPB evaluation population**, and no tuning at all:
  the probe fits once on NYC 311 TRAIN and scores twice.
- **No threshold tuning and no banding.** The probe publishes ranking metrics.

---

## 7. Metrics and baseline (O4) — FROZEN

Required for **each** of the two evaluations (§5.4):

- **PR-AUC**, the headline.
- **ROC-AUC**, secondary only. It is prohibited as a headline at these ratios.
- **Minority-class precision, recall and F1**, reported explicitly.
- **The majority/prior baseline** on the same split.
- **The absolute count of minority-class instances** in the evaluation split.
- **`base_rate`** for each evaluation.

All come from Task 14's `ml/training/metrics.py`, unchanged.

**The baseline is a source-training prior.** It is derived from **NYC 311 TRAIN
labels only** — the positive-class prior is the fraction of `True` breach labels
in the training population — and is **frozen before any evaluation runs**. The
same frozen source prior scores both the 311 test evaluation and the CFPB test
evaluation. **CFPB labels never fit a baseline.** This resolves the tension with
D34 explicitly: D34 fixes baseline priors to training labels, and the probe's
training labels are the 311 ones, in both evaluations.

Mechanically this is `majority_baseline(nyc311_train_labels, roster)` computed
once and passed to both `pr_auc` calls, which is exactly the existing Task 14
API — no new metric function is required.

**Each evaluation is reported alongside that source-trained baseline.**

**The two headline figures are never compared to each other, and never presented
as a before/after improvement claim.** Measured base rates differ by roughly 27×
(311 breach 26.9–31.1%; CFPB not-timely 1.12%), and PR-AUC is base-rate
dependent, so a lower cross-domain figure is expected arithmetic before any
question of shift. Each figure is interpretable **only as lift over its own
baseline**; both baselines and both base rates sit beside them.

**No figure is ever published alone.**

**The prohibition is on comparative presentation inside the metrics, not on a
vocabulary anywhere in the document.** What is forbidden is publishing the two
figures as one before/after pair: an `improvement`, `delta`, `degradation`,
`gain` or `loss` field, or any equivalent framing, in the metrics the report
publishes. The check therefore applies to the serialized `metrics` section.

It is **not** a ban on those words appearing anywhere in the report. Task 8's
provenance evidence, which §9 requires the report to copy, is itself built from
field names such as `median_delta_seconds`, `frac_delta_le_1min`,
`count_delta_negative` and `delta_percentiles_seconds`. Those names are that
diagnostic's own vocabulary, they carry no comparison between the two
evaluations, and they are explicitly permitted. A guard scanning the whole
serialized report for the substring would be satisfiable only by dropping the
evidence §9 and §11 require, so the scope is fixed here rather than left to a
test to imply.

---

## 8. Distribution-shift diagnostics — FROZEN

For **each** feature in `TRANSFER_FEATURES_V1`, the report records:

- source-training quantiles at min, p01, p05, p25, p50, p75, p95, p99, max;
- evaluation-set quantiles at the same nine points;
- `pct_outside_source_range` — percentage of evaluation records outside the
  source training [min, max];
- `pct_outside_source_iqr` — percentage outside the source training [p25, p75];
- an `out_of_range` flag where the evaluation median falls outside the source
  training IQR.

Recorded as `feature_distribution_shift`.

`text_length` is expected to flag: measured medians are 15 characters (311
descriptor) against 1,202 (CFPB narrative), so the classifier's training-derived
bins place essentially every CFPB record in the topmost bin, making the feature
effectively constant at inference. **That diagnostic is preserved and published,
never normalised away.** No ad-hoc rescaling, standardisation or per-domain
transformation is applied to rescue the comparison.

---

## 9. Result classification — FROZEN

The probe reads Task 8's `timestamp_diagnostic` verdict for **CFPB** from the
manifest and sets its own `result_classification`:

| `timestamp_diagnostic` verdict | `result_classification` |
|---|---|
| `strongly_suspicious_load_timestamp` | `non-informative / diagnostic` |
| `suspicious_insufficient_evidence` | `substantive_with_stated_caveat` |
| `supported_plausible_event_time` | `substantive` |

The three verdict strings exist today at `ingest/cli.py:83-85`.

**The downgrade fires only on the pre-specified field-delta rule, never on
histogram shape.** A non-diurnal hour histogram is secondary evidence that can
raise doubt but cannot establish artifact status. The probe copies the verdict,
the delta metrics that produced it, and the rule thresholds into its own report,
so which branch fired and why is visible without opening the manifest.

Rationale recorded for the non-informative branch: with `submitted_hour`
unusable and `text_length` degenerate across domains, the model would rest almost
entirely on `submitted_weekday`, and any resulting figure would describe nothing.

---

## 10. The six framing facts — FROZEN

Emitted in the probe's own report structure, not only in prose:

1. `source_domain` and `source_target`, with the threshold rule defining it.
2. `evaluation_domain` and `evaluation_target`, with what that field measures.
3. `feature_set` — the three names, and why the five-feature set was unusable.
4. `target_semantics_differ: true`, naming both constructs and §4's
   non-equivalence.
5. `polarity_mapping` — see §4.
6. `analysis_type: "exploratory robustness, not same-task transfer"`.

A report missing any of the six is incomplete.

---

## 11. Artifact and report (O3, O7) — FROZEN

**Artifact identity.** The probe writes its own artifact through Task 15's
`write_artifact`:

| Field | Value |
|---|---|
| `model_name` | `xdomain_xtarget_probe` |
| `model_version` | `xdomain_xtarget_probe_v1` |
| `feature_spec` | the three names, under `transfer_features_v1` |
| `experiment_label` | `reduced-feature cross-domain cross-target robustness probe` |
| `thresholds` | `null` — the probe bands nothing |

The naming is binding (§5.4) so the probe cannot be confused with the primary
model at load time. This path is already proven:
`tests/ml/training/test_artifacts.py:306` writes and loads an artifact carrying
`TRANSFER_FEATURES_V1`, and `_require_producible` accepts it because all three
names are record features.

**D35's closed metadata schema is NOT extended.** `artifacts.py:355-357` refuses
unknown top-level keys, and no Task 19 diagnostic is added to it.

**A standalone Task 19 experiment report is persisted instead**, following the
convention D38 established for Task 18: the report is **returned as a frozen
object and serialized to a caller-supplied path**, and nothing is written inside
the repository by default.

The report contains, at minimum:

- the source training population;
- the 311 in-domain evaluation population;
- the CFPB cross-domain evaluation population;
- the exact feature names, in order;
- the estimator and its version;
- the frozen baseline prior;
- the metrics for each evaluation, each beside that baseline;
- `feature_distribution_shift`, with all nine quantiles per feature and both
  percentage-outside measures;
- the six framing facts;
- `result_classification`;
- the CFPB `timestamp_diagnostic` verdict used to set it, with the delta metrics
  and rule thresholds that produced it, under the diagnostic's own field names
  and exempt from §7's comparative-vocabulary check;
- the interpretation limitations of §12.

---

## 12. Interpretation limits — FROZEN

A reduced-feature model that scores poorly here may be failing because of domain
shift, because the target means something different, because three weak features
are insufficient, or any combination — **the design cannot separate them**. The
result is reported as an observation with its causes enumerated and unresolved,
never as a measurement of domain shift alone.

The in-domain figure is the **reduced-feature in-domain reference performance**,
not a "ceiling": a ceiling would imply the cross-domain number measures the same
quantity less well, and it does not.

A poor result is published rather than buried.

---

## 13. Failure behaviour — FROZEN

- A feature the inputs cannot supply raises `FeatureUnavailable` — never a
  silent fallback, never a substituted column (§5.4).
- A naive `submitted_at` raises `ValueError` from Task 12's assembly.
- A CFPB row whose `timely` is not exactly `Yes`/`No` was already refused at
  ingest by `InvalidTimelyValue`; no such record reaches the probe.
- A missing CFPB outcome sidecar raises the CFPB loader's own absence error —
  an absence, never an empty iterator.
- A missing CFPB `timestamp_diagnostic` verdict is a refusal, not a default:
  `result_classification` has no fallback value.
- Non-finite feature values: the three features are finite by construction. A
  non-finite value indicates an upstream defect and raises rather than being
  imputed.
- An empty labelled population in either evaluation raises, following D34's rule
  that a metric over no data describes nothing. `pr_auc` already raises when the
  evaluation population holds no positive example.

---

## 14. Dependencies — FROZEN

**None added.** `scikit-learn==1.9.0` supplies `HistGradientBoostingClassifier`
and `numpy==2.5.2` the arrays, both already pinned in `requirements/ml.txt`;
`pyarrow==25.0.1` already backs the sidecar storage layer. No new package and no
tier change.

**CI is otherwise unchanged, with exactly one permitted addition.**
`tests/ingest/test_cfpb_outcomes.py` exercises Parquet I/O and is therefore not
marked `ml` — the same reasoning the workflow already records for
`test_storage.py`, `test_manifest.py` and `test_cli.py`, which it selects by path
in the ML job because the marker means "exercises an ML model artifact" and
Parquet I/O does not. Left out, the new module would run only in the application
job, where `pyarrow` is absent, and would skip silently: a job green having
tested nothing, which is the exact false pass that path selection exists to
prevent.

**A single minimal path-selection addition is permitted, solely to include
`tests/ingest/test_cfpb_outcomes.py` in the existing PyArrow-capable ML job's
path-selected command.** Nothing else about CI may change: not the workflow's
structure, not its jobs, not its dependency installation, not its commands, not
its markers, and not any unrelated path in that list.

---

## 15. Files

**Expected to change**

| File | Change |
|---|---|
| `ingest/schema.py` | possibly — only if the CFPB sidecar needs a schema constant beside `CFPBOutcome` |
| `ingest/storage.py` | CFPB outcome Arrow schema, writer, part reader |
| `ingest/manifest.py` | CFPB sidecar checksums in the manifest, CFPB-specific loader |
| `ingest/cli.py` | persist the CFPB outcome stream |
| `tests/ingest/test_storage.py`, `test_manifest.py` | cover the above |
| `tests/ingest/test_cli.py` | **exactly two** superseded Task 17 assertions inverted, per §5 |
| `ml/training/experiments/robustness_probe.py` | create |
| `tests/ml/training/test_robustness_probe.py` | create |
| `docs/superpowers/specs/2026-09-04-sentinel-phase-2-addendum.md` | D39 recording O1–O7 |

**Forbidden to change**

`ml/training/experiments/triage.py`, `risk.py`, `dedup.py` and their tests;
`ml/embedders/*` and their tests; `ml/training/{metrics,artifacts,features,labels,thresholds,aggregates,splits}.py`;
`ml/base.py`, `ml/registry.py`, `ml/null.py` and every serving path;
`requirements/`; `pyproject.toml`; `Sentinel_Complete_Blueprint.md`. NYC 311's
outcome schema, `load_outcomes()` and their semantics are unchanged.

`.github/workflows/ci.yml` is forbidden **except** for the single path-selection
addition §14 permits: appending `tests/ingest/test_cfpb_outcomes.py` to the ML
job's existing path-selected test command, and nothing else.

---

## 16. The frozen decisions

**O1 — CFPB target persistence.** A source-specific CFPB outcome sidecar,
carrying exactly `external_id`, `timely_response` and `date_sent_to_company`.
`timely_response` never enters `CorpusRecord.label`, which remains the CFPB
product/category taxonomy. The sidecar is integrity-checked and manifest-backed.
NYC 311's `load_outcomes()` API and semantics are preserved; a CFPB-specific
loader is added rather than changing the established NYC 311 outcome type
contract. `ingest` storage/manifest/CLI may be modified only as far as this
source-specific sidecar requires. Task 17's NYC 311 outcome schema is not
redesigned.

**O1 addendum — frozen symbols and the CI carve-out.** The sidecar's four public
symbols are `CFPB_OUTCOME_ARROW_SCHEMA`, `write_cfpb_outcome_partition`,
`read_cfpb_outcome_parts` and `load_cfpb_outcomes(years=None, root=)`, each
mirroring its NYC 311 counterpart (§5). CI is otherwise unchanged, with one
permitted addition: `tests/ingest/test_cfpb_outcomes.py` joins the ML job's
existing path-selected command, because it needs the PyArrow-capable environment
and carries no `ml` marker. No workflow structure, dependency, command, marker or
unrelated path changes (§14).

**O1 addendum — the two superseded assertions.** The two `tests/ingest/test_cli.py`
tests that assert CFPB has no outcome sidecar are inverted to assert that it now
has one and that `load_outcomes("cfpb")` fails closed without deserialising it.
They are superseded Task 17 scope assertions, updated because Task 19 introduced
the sidecar they were written to defer. NYC 311 production behaviour is
unchanged, no exception class is fixed for the cross-source read, and no other
Task 16–18 test is modified (§5).

**O2 — estimator.** Task 17's exact `HistGradientBoostingClassifier`
configuration: `learning_rate=0.1`, `max_iter=100`, `max_leaf_nodes=31`,
`max_depth=None`, `min_samples_leaf=20`, `l2_regularization=0`,
`early_stopping=False`, `class_weight=None`, `random_state=17`. No hyperparameter
tuning.

**O3 — model version.** `model_version = "xdomain_xtarget_probe_v1"`;
`model_name` remains `xdomain_xtarget_probe`; `experiment_label` remains
`reduced-feature cross-domain cross-target robustness probe`.

**O4 — baseline.** A source-training prior baseline, derived from NYC 311 TRAIN
labels only, the positive-class prior being the fraction of `True` breach labels
in the training population. Frozen before evaluation and used unchanged for both
the 311 test and CFPB test evaluations. CFPB labels never fit a baseline. Each
evaluation is reported alongside that source-trained baseline. The two PR-AUC
values are never presented as a before/after improvement claim.

**O5 — in-domain evaluation population.** The exact NYC 311 TEST period from
Task 17's temporal split. Training uses NYC 311 TRAIN only. Validation tunes
nothing, because the probe has no tuning and no threshold selection. One
evaluation on NYC 311 TEST, one on CFPB TEST.

**O6 — open records.** The existing frozen semantics, reused: open NYC 311
records stay in the corpus and split population, receive no fabricated label,
contribute no supervised train or evaluation label, and may still receive
aggregate-free structural features. No analogous "open" interpretation is
invented for CFPB, whose `timely_response` is an explicit persisted target.

**O7 — reporting.** D35's closed artifact metadata schema is **not** extended for
Task 19 diagnostics. A standalone Task 19 experiment report is persisted using
the convention D38 established for Task 18 experiment reports — a frozen object
serialized to a caller-supplied path — carrying the source training population,
both evaluation populations, the exact feature names and order, the estimator and
version, the baseline prior, the metrics for each evaluation, the
`feature_distribution_shift` block with all nine quantiles per feature and both
percentage-outside measures, the six framing facts, `result_classification`, the
CFPB timestamp-diagnostic verdict used for that classification, and the
interpretation limitations.

---

## 17. Residual wording note — not blocking

O7 specifies "the same repository convention/location established by Task 18".
D38 established a **convention** — "the report is returned as a frozen object and
serialized to a caller-supplied path; nothing is written inside the repository by
default" — and deliberately established **no default in-repository location**.
This contract therefore records the convention and leaves the path to the
caller, exactly as Task 18 does. If a fixed on-disk location is wanted for
Task 19, it is a new decision rather than an inheritance, and Task 21 is where a
published report would be cited from.

One naming detail for the implementer: the sidecar field is named
`date_sent_to_company` after CFPB's own column, while the dataclass field it
carries is `CFPBOutcome.sent_to_company_at`. Both names refer to the same value;
the contract uses CFPB's column name for the persisted field.

---

## 18. Acceptance criteria

1. The probe completes on a fixture corpus and publishes its figures whatever
   they show.
2. Both evaluations report PR-AUC, ROC-AUC, minority precision/recall/F1, the
   frozen source-trained baseline, minority count and base rate — each figure
   beside that baseline, never alone.
3. All six framing facts appear in the report structure.
4. `feature_distribution_shift` carries all nine quantiles per feature plus both
   percentage-outside measures and the flag.
5. `result_classification` matches the CFPB `timestamp_diagnostic` verdict per
   §9, with the verdict and its delta metrics copied into the report.
6. The 311 split boundaries in the report equal Task 17's.
7. The CFPB evaluation population is the Task 16 test population with persisted
   `timely_response` outcomes.
8. The baseline prior is fitted on NYC 311 TRAIN labels and is identical across
   both evaluations.
9. A test asserts the report's `metrics` section never presents the two PR-AUCs
   as a single before/after pair — no `improvement`, `delta`, `degradation`,
   `gain` or `loss` field — and that no prohibited transfer wording appears
   anywhere in the report. A companion test asserts the copied provenance
   evidence of §9 is still present, so the guard can never be satisfied by
   removing it (§7).
10. The artifact loads through `load_artifact` with `feature_spec` of exactly the
    three names and `model_version` `xdomain_xtarget_probe_v1`.
11. D35's metadata schema is unchanged and carries no Task 19 diagnostic.
12. NYC 311's `load_outcomes()` behaviour is unchanged, proven by its existing
    tests still passing untouched.
13. Tasks 16, 17 and 18 remain byte-identical.
14. The sidecar exposes exactly the four frozen symbols of §5, and NYC 311's
    symbols are unchanged.
15. The only CI change is `tests/ingest/test_cfpb_outcomes.py` appended to the
    ML job's path-selected command; the workflow is otherwise byte-identical.
16. The only historical test change is the two inverted assertions in
    `tests/ingest/test_cli.py`; every other Task 16–18 test is byte-identical.
17. A test observes the arguments `fit` actually receives and asserts they are
    exactly the labelled NYC 311 TRAIN matrix and labels, in the frozen feature
    order. This is how O5's "training uses NYC 311 TRAIN only" is proven:
    the reported training population is assembled separately from the matrix the
    estimator is given, so a report claiming train-only establishes nothing about
    the fit. A fixture guard keeps TRAIN, TEST, TRAIN + validation, TRAIN + TEST
    and all three periods at five distinct row counts, so no swap or union can
    satisfy the assertion by accident.

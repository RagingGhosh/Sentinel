# Sentinel Phase 2 — Data and Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, leakage-controlled offline ML pipeline — corpus ingestion, temporal splitting, feature engineering, three model experiments and a representation benchmark — producing versioned artifacts with published metrics, without touching the running application.

**Architecture:** Two new Django-independent packages. `ingest/` turns public source APIs into a versioned on-disk Parquet corpus keyed by `(source, external_id)`. `ml/training/` consumes that corpus through explicitly time-ordered splits, fits every statistic on training data only, and emits artifacts whose `metadata.json` is authoritative about their own feature contract. Nothing in Phase 2 imports from `complaints/`, writes to the database, or is wired into serving.

**Tech Stack:** Python 3.13, pandas + pyarrow (corpus I/O), scikit-learn (TF-IDF, calibrated linear, histogram gradient boosting), onnxruntime (MiniLM inference), numpy. Existing: pytest, ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-09-04-sentinel-phase-2-addendum.md` (authoritative), amending `docs/superpowers/specs/2026-09-03-sentinel-design.md` §6.

> **STATUS: revised 2026-09-04. Both Critical findings resolved in the spec
> (addendum D15–D19); no task remains blocked.** `RiskFeaturesV1` is **five**
> features — `sla_hours` was removed as target-derived. What Phase 1 called
> transfer is now the **reduced-feature cross-domain cross-target robustness
> probe** (`TransferFeaturesV1`): a distinct three-feature model, and explicitly
> not evidence that the 311 SLA-risk model transfers to the CFPB task, because
> the two targets are non-equivalent constructs. `precision@k` is removed from
> the risk metrics. Embedding dimension is recorded and validated rather than
> assumed. Awaiting final approval before execution.

## Global Constraints

Copied from the addendum and the Phase 1 design. Every task's requirements implicitly include this section.

- **Phase 2 adds no schema migration, writes no `Prediction` rows, populates no `Complaint.embedding`, wires nothing into the serving registry, and changes no UI, authorization rule or lifecycle code.**
- **No corpus record ever enters the operational database.** Corpus identity is `(source, external_id)`. No synthetic `Complaint.pk` is created for a corpus record.
- **Random and stratified-random splits are prohibited.** All evaluation is time-ordered.
- **Every statistic is fitted on the training period only** — preprocessors, scalers, imputers, encoders, TF-IDF vocabulary and IDF, feature selection, target-derived aggregates, SLA thresholds, decision thresholds, resampling, evaluation indexes.
- **Target-derived aggregates use out-of-fold construction with time-ordered forward-chaining folds.** Not random K-fold. Not leave-one-out.
- **Decision thresholds are tuned on validation and reported on test.** Test is never used for tuning.
- **CFPB `cfpb_timely_response` and NYC 311 `nyc311_sla_breach` are never unified** under one field or a generic `sla_met`.
- **Accuracy is never a headline metric.** Every metric is reported beside a majority-class baseline on the same split.
- **The CFPB roster is derived from data and asserted**, never transcribed from a document. Out-of-roster or missing labels fail ingest loudly; no dropping, remapping or auto-expansion.
- **The artifact's `feature_spec` is authoritative at inference**, in exact order; loading fails loudly on mismatch.
- Python 3.13. Line length 100. `ruff check`, `ruff format --check`, and `mypy complaints domains ml accounts ingest` must be clean.
- TDD: failing test first, watched fail, then implement. One coherent commit per task.

---

## A. Executive summary

Phase 1 shipped a working complaint platform with an ML-shaped hole: protocols, null implementations and a registry that always resolves to nulls. Phase 2 fills the offline half of that hole — data and models — and deliberately stops short of serving.

The work divides into three layers. **Ingestion** (Tasks 3–8) turns two public APIs into a versioned Parquet corpus with derived-and-asserted taxonomy. **Leakage machinery** (Tasks 9–13) provides the temporal splitting, forward-chaining folds, threshold fitting and out-of-fold aggregation that every experiment must route through — built and tested before any model exists, because retrofitting leakage controls onto a working model is how leakage survives. **Experiments** (Tasks 14–19) train the triage classifier, the risk model and the two benchmark arms, each producing a versioned artifact whose metadata is authoritative about its own contract.

The critical path runs through the leakage machinery, not the models. A model trained on leaked features is worse than no model, because it looks like success.

## B. Scope

In scope: corpus ingestion and validation for CFPB and NYC 311; temporal split and fold machinery; SLA threshold fitting; risk feature assembly; the CFPB triage classifier; the 311 risk model; the `TextEmbedder` abstraction and both benchmark arms; the dedup retrieval benchmark; artifact format, metadata and load-time guards; resource measurement; reproducibility controls; a dependency split preceding all of it.

## C. Non-goals

Wiring artifacts into `ml/registry.py`. Populating `Complaint.embedding`. Writing `Prediction` rows. Any schema migration. Any change to `complaints/services.py`, the API, the UI or authorization. Retraining on Sentinel's own operational history. Making production inference depend on anything introduced here. Downloading complete corpora during CI.

## D. Repository changes by package/file

| Path | Responsibility | Status |
|---|---|---|
| `requirements/base.txt` | Runtime dependencies for the deployed app | Create |
| `requirements/ml.txt` | ML inference dependencies — installed in Phase 3, not by `build.sh` | Create |
| `requirements/train.txt` | Training/benchmark-only dependencies | Create |
| `requirements/dev.txt` | Test and static-analysis tooling | Create |
| `requirements.txt` | Compatibility shim: `-r requirements/base.txt` | Modify |
| `build.sh` | Installs `requirements/base.txt` only | Modify |
| `.github/workflows/ci.yml` | Installs base+dev for app tests; base+dev+train for ML tests | Modify |
| `.gitignore` | Exclude the whole corpus tree and artifact binaries | Modify |
| `ingest/schema.py` | `CorpusRecord`, `CFPBOutcome`, `NYC311Outcome`, `SCHEMA_VERSION` | Create |
| `ingest/identity.py` | `RecordRef`, `make_ref`, `parse_ref` | Create |
| `ingest/storage.py` | Parquet partition read/write, corpus discovery | Create |
| `ingest/manifest.py` | `CorpusManifest`, checksums, resumability state | Create |
| `ingest/roster.py` | Roster derivation and the both-directions failure policy | Create |
| `ingest/sources/base.py` | `SourceAdapter` protocol | Create |
| `ingest/sources/cfpb.py` | CFPB API → `CorpusRecord` + `CFPBOutcome` | Create |
| `ingest/sources/nyc311.py` | Socrata → `CorpusRecord` + `NYC311Outcome` | Create |
| `ingest/cli.py` | `python -m ingest.cli` entry point | Create |
| `ml/base.py` | Add `TextEmbedder` protocol only | Modify |
| `ml/training/splits.py` | Date-cut temporal split, forward-chaining folds | Create |
| `ml/training/aggregates.py` | Out-of-fold target-derived aggregates | Create |
| `ml/training/thresholds.py` | 311 per-type p75 threshold fitting and freezing | Create |
| `ml/training/features.py` | `FeatureSpec`, risk feature assembly | Create |
| `ml/training/artifacts.py` | Artifact write/load, metadata schema, feature guard | Create |
| `ml/training/metrics.py` | macro-F1, PR-AUC, recall@k, baselines | Create |
| `ml/embedders/tfidf.py` | TF-IDF `TextEmbedder` | Create |
| `ml/embedders/minilm.py` | ONNX MiniLM `TextEmbedder` | Create |
| `ml/training/experiments/triage.py` | CFPB triage experiment | Create |
| `ml/training/experiments/risk.py` | 311 risk experiment | Create |
| `ml/training/experiments/dedup.py` | Perturbation benchmark harness | Create |
| `ml/training/experiments/robustness_probe.py` | Reduced-feature cross-domain cross-target robustness probe | Create |
| `tests/ingest/`, `tests/ml/training/`, `tests/ml/embedders/` | Mirrored test packages, each with `__init__.py` | Create |
| `docs/phase-2-reproducibility.md` | Environment, seeds, corpus versions, rerun instructions | Create |

**Two structural decisions, stated once here.**

*Training code is Django-independent.* `ingest/` and `ml/training/` import no Django. They are invoked as `python -m ingest.cli` / `python -m ml.training.cli`, never as management commands. This keeps `INSTALLED_APPS` untouched (so no migration risk), keeps training dependencies out of the app's import graph, and makes the dependency split enforceable rather than aspirational. `ml/` remains unregistered as an app exactly as Phase 1 left it.

*Serving must never import training.* `ml/base.py`, `ml/null.py` and `ml/registry.py` may not import from `ml/training/`, `ml/embedders/` or `ingest/`. Task 2 adds a test enforcing that import direction, because this is the boundary that keeps a 90MB ONNX dependency out of the web process.

## E. Dependency changes

Four files replace one. The split is by *when the dependency is needed*, not by whether it feels heavy.

| File | Contents | Installed by |
|---|---|---|
| `base.txt` | Django, DRF, allauth, gunicorn, whitenoise, psycopg2-binary, django-environ, requests, and their transitive pins | `build.sh` (production), CI |
| `ml.txt` | `-r base.txt` plus numpy, scikit-learn, onnxruntime | **Phase 3** production; CI's ML job |
| `train.txt` | `-r ml.txt` plus pandas, pyarrow | CI's ML job; local training |
| `dev.txt` | `-r base.txt` plus pytest, pytest-django, pytest-cov, factory_boy, Faker, ruff, mypy, django-stubs, coverage | CI, local |

**`ml.txt` is deliberately separate from `train.txt` even though Phase 2 always installs both.** scikit-learn and onnxruntime become *runtime* dependencies in Phase 3 when artifacts are loaded for inference; pandas and pyarrow never do. Merging them now would force a re-split later, and would make the Phase 3 memory measurement include a dataframe library the web process never uses.

**`requirements.txt` remains as a one-line shim** (`-r requirements/base.txt`) so that any external tooling or documentation referring to it keeps working.

## F. Data ingestion architecture

Each source implements `SourceAdapter`:

```python
class SourceAdapter(Protocol):
    slug: ClassVar[str]
    def fetch(self, window: DateWindow, page: PageCursor | None) -> FetchPage: ...
    def normalize(self, raw: Mapping[str, Any]) -> tuple[CorpusRecord, Outcome | None]: ...
```

`fetch` is paginated and resumable; `normalize` is pure and independently testable against fixture rows. Adapters attach to their domain pack in `domains/packs.py` via a `dataset_adapter` ClassVar, so no source slug literal is ever written in `complaints/` — the Phase 1 domain-pack invariant carries forward unchanged.

Ingestion is a three-stage pipeline with a durable boundary between each stage:

```
fetch  → data/raw/<source>/<window>/page-NNNN.json.gz   (gitignored, resumable)
normalize → CorpusRecord / Outcome streams              (pure, in memory)
load   → data/corpus/<source>/v<N>/year=<YYYY>/part-NNNN.parquet + manifest.json
```

Interrupting `fetch` loses at most one page. Re-running `fetch` skips pages already on disk (verified by checksum, not existence). `normalize` and `load` are deterministic given the same raw pages, so the corpus is reproducible from the raw cache without re-hitting an API.

## G. Corpus representation

```
data/
  raw/                                 # gitignored: API responses, resumable cache
  corpus/
    cfpb/v1/
      manifest.json
      year=2024/part-0000.parquet
      year=2025/part-0000.parquet
    nyc311/v1/
      manifest.json
      year=2024/part-0000.parquet
```

Partitioned by source, then schema version, then year. Year partitioning matters because every consumer filters by time; schema version in the path means an incompatible `CorpusRecord` change produces a new tree rather than silently mixing shapes.

`manifest.json` per source records: `schema_version`, `source_slug`, `window_start`/`window_end`, `ingested_at`, `record_count`, `per_year_counts`, the observed label roster with per-label counts, `part_files` with a SHA256 per file, the adapter's `source_api_version` string, and `corpus_id` — a stable hash over the part-file checksums that artifacts cite to name exactly which corpus they trained on.

Records carry `(source, external_id)` and never an integer id. `RecordRef` is a frozen dataclass with `source: str` and `external_id: str`; `make_ref`/`parse_ref` round-trip it as `"<source>:<external_id>"` for use as a dictionary key and in evaluation output.

## H. Temporal split architecture

Default 70/15/15 **by record count, realised as date cuts**. Row-index slicing is prohibited: both sources carry duplicate timestamps (CFPB `date_received` and 311 `created_date` both include time-of-day, but ties are common at scale), and a boundary falling inside a tie would place identical timestamps in two periods.

```python
def temporal_split(timestamps: Sequence[datetime],
                   fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)
                   ) -> TemporalSplit:
    """Cut a time-ordered dataset into train/val/test at *date* boundaries.

    Records sharing a timestamp always land in the same period. Achieved
    fractions therefore differ from requested ones; both are recorded.
    """
```

The algorithm: sort unique timestamps ascending; build the cumulative record count per unique timestamp; for each requested boundary, choose the largest timestamp whose cumulative count does not exceed the target, and assign every record at that timestamp to the earlier period. `TemporalSplit` reports `train_end`, `val_end`, per-period counts, and the achieved fractions.

## I. Leakage-control architecture

Seven paths, each with an owning module and an explicit test (Task 11 and the per-experiment tasks):

| Path | Control | Test asserts |
|---|---|---|
| Preprocessing | fitted inside the train period, applied to val/test | fitted statistics change when train changes, not when test changes |
| Vocabulary / IDF | `TfidfVectorizer.fit` on train text only | a token appearing only in test is absent from the vocabulary |
| Feature selection | any target-consulting selection on train folds only | selection output identical when test labels are permuted |
| Threshold tuning | fitted on validation, applied to test | test metrics unchanged when test labels are permuted *before* threshold fitting |
| Resampling | train only | val/test class balance equals the natural balance |
| Evaluation index | index contains only records at or before the query's period | a query cannot retrieve a record with a later timestamp |
| Target-derived aggregates | out-of-fold, forward-chaining (§J) | a training row's own outcome does not change its own feature |

## J. Feature engineering

**`RiskFeaturesV1` — five features**, in this exact order. The order is part of the contract and is recorded in `feature_spec`:

1. `submitted_hour` — from the source timestamp
2. `submitted_weekday` — from the source timestamp
3. `text_length` — character count of `CorpusRecord.text`
4. `category_mean_resolution_hours` — out-of-fold
5. `category_breach_rate` — out-of-fold

`sla_hours` is **not** a feature (addendum D15). The per-type threshold is the p75 of training resolution hours — the same outcomes that define the target — so a training row's own resolution time would contribute to the p75 becoming that row's own feature. The frozen-label design admits no leakage-safe construction for it, and maintaining a separate out-of-fold feature threshold alongside the frozen label threshold was rejected. The threshold's only role is to produce the label.

**`TransferFeaturesV1` — three features**, versioned independently of `RiskFeaturesV1`, used only by the reduced-feature cross-domain cross-target robustness probe (§O):

1. `submitted_hour`
2. `submitted_weekday`
3. `text_length`

These are the only features computable in both corpora.

**Forward-chaining out-of-fold construction, specified concretely.** `sklearn.model_selection.TimeSeriesSplit` is **rejected**: it splits by index position, so with duplicate timestamps a fold boundary can straddle a timestamp — the same defect the temporal split avoids. A date-cut equivalent is implemented instead:

```python
def forward_chaining_folds(timestamps: Sequence[datetime],
                           n_folds: int = 5,
                           warmup_fraction: float = 0.20) -> list[Fold]:
    """Expanding-window folds over the TRAIN period, cut at date boundaries.

    Fold i's `fit` block is every training record strictly before the fold's
    start date; its `apply` block is the records inside the fold. Records in
    the warm-up prefix appear in no fold's apply block.
    """
```

- **Number of folds: 5**, fixed. Enough that each apply block holds ~16% of the training period; few enough that the earliest fold still has a 20% warm-up behind it.
- **Warm-up: the first 20% of the training period by record count**, date-cut as in §H. Nothing before the first fold boundary receives an out-of-fold value.
- **Fold boundaries** are the four date cuts dividing the remaining 80% into four equal-count blocks, plus the warm-up boundary — five apply blocks total.
- **Warm-up rows are retained with `NaN`** for both out-of-fold aggregates. They are *not* dropped, and are *not* given the training global mean — that mean is computed from their own outcomes and would reintroduce the leak this construction exists to prevent. `HistGradientBoostingClassifier` handles `NaN` natively, so the rows still contribute their other four features. The warm-up row count is recorded in metadata.
- **Assembly:** each fold's aggregate is computed only from its `fit` block and written to its `apply` block. No row is ever assigned a value derived from itself or from any later record.

For validation and test rows, both aggregates are computed once from the **entire train period** and applied as constants. Categories unseen in training receive the train-period global mean. That global mean is leakage-free for val/test rows because it uses no val/test outcome.

## K. SLA threshold fitting

`nyc311_sla_breach` is derived from `nyc311_resolution_hours` against a per-complaint-type threshold:

```
TRAIN     fit per-type p75 of resolution_hours  →  freeze
VALIDATION apply frozen thresholds
TEST       apply frozen thresholds
```

Types with fewer than **100 training records** fall back to the global train-period p75. Thresholds are frozen into an immutable mapping before any validation or test record is touched; the fitting function returns a frozen object and the applying function accepts one, so recomputation on val/test is a type error rather than a discipline question. Per-type thresholds, the global fallback value, the number of types that hit the fallback, and the resulting overall breach rate on each period are written to metadata.

## L. Risk model training

`HistGradientBoostingClassifier` on the five `RiskFeaturesV1` features, target `nyc311_sla_breach`, 311 corpus only. Baselines: majority-class and stratified-random on the same split. Metrics: **PR-AUC (headline)**, ROC-AUC (secondary only), a calibration curve, and the absolute minority count per period. Decision banding thresholds are tuned on validation and applied unchanged to test.

`precision@k` is **not** required (addendum D17). It was named in the Phase 1 design because the model ranks a queue, but no `k` was defined, and `k` is an operational quantity Sentinel has no history to ground. The calibration curve carries the "is the top of the ranking trustworthy" question without inventing a constant.

The plan does not assume this model will be good. Task 15's acceptance criterion is that metrics are produced and published honestly, not that they clear a bar.

## M. CFPB triage training

`TfidfVectorizer` (word 1–2 grams plus char 3–5 grams, fitted on train text only) into `LogisticRegression` wrapped in `CalibratedClassifierCV`, because the UI shows a confidence and an uncalibrated one is a lie. Target: the derived 11-label roster. Baselines: majority-class and stratified-random. Metrics: macro-F1 (headline), per-class precision/recall/F1, confusion matrix, top-3 accuracy. Abstention threshold tuned on validation.

## N. MiniLM vs TF-IDF benchmark

Both arms implement `TextEmbedder`. The benchmark builds a retrieval index over held-out CFPB records, perturbs a sample of them (paraphrase by synonym substitution, truncation to 60% length, and typo injection at 2% of characters — each perturbation type reported separately), and measures recall@k for retrieving the original by `RecordRef`.

Everything except the representation is held identical and asserted by a test: the same temporal boundaries, the same evaluation population, the same perturbation seed and set, the same similarity function (cosine over L2-normalised vectors), the same `k`, the same index population, the same metric, the same baselines, and an equal tuning budget. Task 18 encodes these as a single frozen `BenchmarkConfig` that both arms consume, so divergence requires editing shared config rather than one arm's code.

## O. Reduced-feature cross-domain cross-target robustness probe

**Not transfer.** A transfer experiment holds the task fixed and varies the data. This varies both (addendum D19):

| | Source | Evaluation |
|---|---|---|
| Domain | NYC 311 civic service requests | CFPB consumer financial complaints |
| Target | `nyc311_sla_breach` | `cfpb_timely_response` |
| Target means | resolution slower than the type's training p75 | company replied inside CFPB's 15-day window |

Addendum §4 defines those targets as deliberately non-equivalent and §4.3 prohibits unifying them, so a model trained on one and scored against the other is being asked a **different question** in a new domain.

A distinct `HistGradientBoostingClassifier` is trained on 311 using only `TransferFeaturesV1`, targeting `nyc311_sla_breach`, then evaluated twice:

1. **In-domain on held-out 311** against its own target — the **reduced-feature in-domain reference performance**. Not a "ceiling": a ceiling implies the cross-domain figure measures the same quantity less well, and it does not measure the same quantity at all.
2. **Cross-domain, cross-target on CFPB** against `cfpb_timely_response`.

Metrics: PR-AUC headline, minority precision/recall/F1, absolute minority count, majority-class baseline, on the same split. ROC-AUC secondary only. Every figure beside its baseline in the same table.

**Binding prohibition.** This probe must never be described, labelled, summarised or tabulated as evidence that the 311 SLA-risk model transfers to the CFPB task. Wording such as "transfers to", "generalises to" or "works on CFPB" is a defect in the report. Task 21's doc test enforces it.

**Six facts the output must carry itself**, so they cannot be lost in transcription: source domain and source target (with the threshold rule defining it); evaluation domain and evaluation target (with what that field measures); the reduced feature set and why the five-feature set was unusable; that the target semantics differ and §4 defines them as non-equivalent; the polarity mapping (which class counts as adverse in each domain); and that this is exploratory robustness analysis rather than same-task transfer.

## P. Artifact / metadata format

```
ml/artifacts/<domain>/<model>/<version>/
    model.joblib | model.onnx
    metadata.json
```

`metadata.json` fields: `model_name`, `model_version`, `trained_at`, `git_sha`, `corpus_id`, `corpus_schema_version`, `source_window`, `split` (cut dates, per-period counts, requested vs achieved fractions), `feature_spec` (ordered list of feature names), `feature_spec_version`, `label_roster`, `thresholds`, `metrics` (per period, each beside its baseline), `warmup_row_count`, `seeds`, `dependency_versions`, and — for embedder artifacts — `embedding_dimension`, `embedding_model_id` and `embedding_model_sha256`. Experiment artifacts additionally carry `experiment_label`, which for the transfer artifact is `"reduced-feature cross-domain robustness"`.

`load_artifact()` reads `feature_spec` and returns a loader that builds exactly those features in exactly that order. Requesting a feature the environment cannot produce raises at load time. This is the compatibility guard from addendum §3.3, and Task 20 tests it against a deliberately mismatched artifact.

## Q. Testing strategy

Unit tests for normalization, identity round-trips, roster validation, date parsing, text normalization, split boundaries, fold construction, threshold fitting, aggregate construction, feature assembly, `feature_spec` enforcement, artifact metadata, and each metric against hand-computed values. Integration tests for source→corpus, corpus→dataset, training→artifact, artifact→evaluation, and benchmark reproducibility. Every leakage path in §I gets a test that fails if the implementation learns from the future.

**Network is never touched in tests.** Adapters are tested against committed fixture pages (small, redacted, under `tests/fixtures/`), and a test asserts no test performs an outbound request.

**Long-running ML tests carry the existing `ml` marker** (already declared in `pyproject.toml`) and are excluded from the default run, so the app suite stays fast.

## R. Reproducibility strategy

Seeds are set explicitly for numpy, scikit-learn estimators and the perturbation sampler, and recorded in metadata. Corpus ordering is deterministic (sorted by `(submitted_at, external_id)`) before any split. `corpus_id` ties an artifact to exact input bytes. `dependency_versions` records the installed versions of numpy, scikit-learn and onnxruntime.

**What is not claimed:** bitwise reproducibility across machines. `HistGradientBoosting` and BLAS-backed operations vary with thread count and CPU. The claim made in the docs is narrower and true: *given the same corpus, seeds and dependency versions on the same platform, results reproduce; across platforms, metrics reproduce to within a stated tolerance.* Task 21 measures that tolerance rather than asserting it.

## S. Resource measurement

Measured in an environment installed from `requirements/ml.txt` only — never one polluted by `train.txt` — because the Phase 3 memory budget covers inference, not training. Measured: peak RSS during MiniLM load, embedding throughput (records/sec at batch sizes 1/8/32), artifact sizes on disk, index build time and memory for 10k/50k vectors, and single-query latency. Environment (CPU model, core count, Python and library versions) is recorded with every figure. No figure is written down that was not produced by a run.

## T. Documentation

`docs/phase-2-reproducibility.md` covers environment setup per dependency tier, corpus ingestion commands with expected runtimes, how to rerun each experiment, and how to interpret each artifact's metadata. The project README's "Model performance" placeholder is filled with the measured table — every figure beside its baseline, the dedup benchmark labelled as synthetic retrieval, and any experiment that did not clear its baseline reported as such.

---

## U. Task-by-task implementation sequence

Each task: objective, files, prerequisites, behavior, tests, acceptance criteria, commit boundary, and what must not change.

### Task 1: Dependency split

**Objective:** Replace the flat `requirements.txt` with four tiers before any ML dependency is introduced.
**Files:** Create `requirements/{base,ml,train,dev}.txt`. Modify `requirements.txt`, `build.sh`, `.github/workflows/ci.yml`.
**Prerequisites:** none.
**Behavior:** Partition the current 36 pinned packages by tier per §E, preserving exact versions. `requirements.txt` becomes `-r requirements/base.txt`. `build.sh` installs `requirements/base.txt`. CI gains two jobs: `app` (base+dev, runs the existing suite and static checks) and `ml` (base+dev+train, runs `pytest -m ml`).

**The `ml` job tolerates pytest exit code 5 (`NO_TESTS_COLLECTED`) while the `ml` marker has no users.** No test carries the marker until Task 16, so `pytest -m ml` collects nothing and exits 5, which GitHub Actions reads as failure. The step checks the return code explicitly and accepts 5 alone; 1, 2, 3 and 4 all still fail the job. The job is kept rather than deferred because installing `requirements/train.txt` on a clean Linux runner is itself useful coverage — those pins were resolved locally and are otherwise never exercised. Task 16 removes the tolerance.
**Tests:** `tests/test_requirements.py` — asserts every package in the original freeze appears in exactly one tier; asserts `base.txt` contains no test or training package (explicit deny-list: pytest, ruff, mypy, factory_boy, Faker, coverage, pandas, pyarrow, scikit-learn, onnxruntime, numpy); asserts `ml.txt` and `train.txt` both start with a `-r` line.
**Acceptance:** existing 125 tests pass under base+dev alone; `pip install -r requirements/base.txt` in a clean venv can run `manage.py check`.
**Commit:** `chore: split requirements into base/ml/train/dev tiers`
**Must not change:** any application code; the pinned versions themselves.

### Task 2: Package skeletons and the import-direction guard

**Objective:** Create `ingest/` and `ml/training/` as importable Django-independent packages, and lock the boundary that keeps training out of the web process.
**Files:** Create `ingest/__init__.py`, `ml/training/__init__.py`, `ml/embedders/__init__.py`, `tests/ingest/__init__.py`, `tests/ml/training/__init__.py`, `tests/ml/embedders/__init__.py`. Modify `.gitignore`.
**Prerequisites:** Task 1.
**Behavior:** `.gitignore` gains `data/` (the whole tree — the existing entries cover only `data/raw/` and `data/interim/`, which would let a Parquet corpus be committed) and `ml/artifacts/**/*.joblib` alongside the existing ONNX rule. Small metadata files remain committable.
**Also in this task (moved from Task 1, see M2):** add `ingest` to the CI mypy target in `.github/workflows/ci.yml`, now that the package exists. Verify with `mypy complaints domains ml accounts ingest` before committing.
**Tests:** `tests/test_import_boundaries.py` — walks the AST of `ml/base.py`, `ml/null.py`, `ml/registry.py` and asserts none imports `ml.training`, `ml.embedders` or `ingest`; asserts `ingest` and `ml.training` import no `django` module. Uses AST inspection, not import side effects, so it fails deterministically without needing the heavy packages installed.
**Acceptance:** `mypy ingest ml` clean; boundary test passes; `git status` clean after creating a dummy `data/corpus/x.parquet`.
**Commit:** `feat: add ingest and ml.training packages with import-direction guard`
**Must not change:** `ml/base.py` contents; `INSTALLED_APPS`.

### Task 3: Corpus schema and identity

**Objective:** Define the record types and the `(source, external_id)` identity.
**Files:** Create `ingest/schema.py`, `ingest/identity.py`, `tests/ingest/test_schema.py`, `tests/ingest/test_identity.py`.
**Prerequisites:** Task 2.
**Behavior:** Frozen dataclasses `CorpusRecord(source, external_id, text, label, submitted_at)`, `CFPBOutcome(external_id, timely_response, sent_to_company_at)`, `NYC311Outcome(external_id, closed_at, resolution_hours)`, and `SCHEMA_VERSION = 1`. `RecordRef(source, external_id)` frozen and hashable; `make_ref(record) -> RecordRef`; `parse_ref("cfpb:12345") -> RecordRef`; `RecordRef.__str__` produces that form.
**Tests:** `CFPBOutcome` carries `sent_to_company_at` and accepts `None` for it; round-trip `parse_ref(str(ref)) == ref`; a ref containing a colon in the external id round-trips correctly (split on the first colon only); `CorpusRecord` rejects mutation; refs from different sources with the same external id are unequal and hash differently.
**Acceptance:** no integer identifier exists anywhere in `ingest/schema.py`.
**Commit:** `feat: corpus record schema and (source, external_id) identity`
**Must not change:** `ml/base.py`.

### Task 4: Corpus storage and manifest

**Objective:** Write and read the partitioned Parquet corpus with a checksummed manifest.
**Files:** Create `ingest/storage.py`, `ingest/manifest.py`, `tests/ingest/test_storage.py`, `tests/ingest/test_manifest.py`.
**Prerequisites:** Task 3.
**Behavior:** `write_partition(records, source, year, part_index) -> Path`; `read_corpus(source, years=None) -> Iterator[CorpusRecord]` yielding in deterministic `(submitted_at, external_id)` order; `CorpusManifest` with the §G fields; `compute_corpus_id(part_checksums) -> str` as a SHA256 over sorted `(path, sha256)` pairs.
**Tests:** a written-then-read corpus round-trips records exactly; read order is deterministic across two calls and independent of file order on disk; `corpus_id` changes when a part file's bytes change and is stable when only mtime changes; a manifest whose checksum does not match its part file raises.
**Acceptance:** `read_corpus` never loads more than one part file into memory at a time (asserted by reading a two-part corpus with a memory-bounded assertion on chunk size).
**Commit:** `feat: partitioned Parquet corpus storage with checksummed manifest`
**Must not change:** schema from Task 3.

### Task 5: Source adapter protocol and CFPB normalization

**Objective:** Define `SourceAdapter` and implement CFPB's pure normalization against fixtures.
**Files:** Create `ingest/sources/__init__.py`, `ingest/sources/base.py`, `ingest/sources/cfpb.py`, `tests/fixtures/cfpb_page.json`, `tests/ingest/test_cfpb_normalize.py`.
**Prerequisites:** Task 3.
**Behavior:** `normalize` maps `complaint_id → external_id`, `complaint_what_happened → text`, `product → label`, `date_received → submitted_at` (ISO-8601 with timezone, parsed to aware UTC), `timely` (`"Yes"`/`"No"`) → `CFPBOutcome.timely_response`, and **`date_sent_to_company → CFPBOutcome.sent_to_company_at`** (aware UTC, `None` when absent). That last field exists only so Task 8's field-delta provenance measurement is computable from the corpus rather than only from the raw cache; it is never a feature and never a target — it is downstream of intake and unavailable for a live complaint. Records with a null or empty narrative are rejected with a typed error rather than silently skipped, so the caller decides.
**Tests:** normalization of a fixture row produces exact expected values; a `"Yes"`/`"No"` string maps to `True`/`False` and any other value raises; a missing narrative raises `MissingNarrative`; a naive timestamp is rejected; **`sent_to_company_at` is parsed when present and `None` when the field is absent, and a fixture row exercises each case**; the fixture is small (≤5 rows) and contains no personal data.
**Acceptance:** `normalize` is pure — no network, no filesystem, no clock.
**Commit:** `feat: CFPB source adapter normalization`
**Must not change:** anything under `complaints/`.

### Task 6: NYC 311 normalization

**Objective:** Implement 311's pure normalization, including resolution hours.
**Files:** Create `ingest/sources/nyc311.py`, `tests/fixtures/nyc311_page.json`, `tests/ingest/test_nyc311_normalize.py`.
**Prerequisites:** Task 5.
**Behavior:** `unique_key → external_id`, `descriptor → text`, `complaint_type → label`, `created_date → submitted_at`, `closed_date → NYC311Outcome.closed_at`, and `resolution_hours = (closed_at - submitted_at).total_seconds() / 3600`. A missing `closed_date` yields `closed_at=None, resolution_hours=None`. **A negative duration raises** rather than being clamped — the reconnaissance measured zero negatives, so one appearing means an assumption broke.
**Tests:** resolution hours computed correctly to within a second; missing close date yields `None` for both fields, not zero; a negative duration raises `NegativeResolutionTime`; a fixture row with a null descriptor raises.
**Acceptance:** pure, as Task 5.
**Commit:** `feat: NYC 311 source adapter normalization`
**Must not change:** CFPB adapter.

### Task 7: Roster derivation and failure policy

**Objective:** Derive the CFPB label roster from data and fail loudly on any membership difference.
**Files:** Create `ingest/roster.py`, `tests/ingest/test_roster.py`.
**Prerequisites:** Task 5.
**Behavior:** `derive_roster(labels_by_year: Mapping[int, set[str]]) -> frozenset[str]` returns the intersection across years. `assert_roster(observed, locked)` raises `RosterMismatch` naming the unexpected and the missing labels separately, and does so *before* any record is processed. The locked roster lives in the corpus manifest of the first successful ingest, not in source code.
**Tests:** an unexpected label raises and the message names it; a missing label raises and the message names it; an exact match passes and returns the roster; a same-size roster with one label swapped raises (this is the case a count assertion would miss); the error carries per-label counts.
**Acceptance:** no label string from the spec document appears as a literal in `ingest/roster.py`.
**Commit:** `feat: derive-and-assert CFPB label roster with both-direction failure`
**Must not change:** adapters.

### Task 8: Ingestion CLI with resumable fetch

**Objective:** Wire fetch → normalize → load into a resumable command.
**Files:** Create `ingest/cli.py`, `tests/ingest/test_cli.py`.
**Prerequisites:** Tasks 4, 6, 7.
**Behavior:** `python -m ingest.cli --source {cfpb,nyc311} --start YYYY-MM-DD --end YYYY-MM-DD [--limit N]`. Fetch writes gzipped pages under `data/raw/`; a page whose checksum matches an existing file is skipped. Normalize and load are re-run from the raw cache without network access. `--limit` bounds records for development and is recorded in the manifest so a truncated corpus can never be mistaken for a full one.
**Required output — timestamp provenance diagnostic (finding I2, addendum §2.3).** Ingest writes `timestamp_diagnostic` into every manifest. It has two parts with **different evidential weight**, and the distinction is load-bearing rather than presentational.

*Part A — distributional anomaly diagnostic (secondary evidence).* Per source: the 24 hour-of-day counts, the 7 weekday counts, a chi-square against uniform with its p-value, and `hour_concentration` (share in the single busiest hour). **This part is explicitly labelled `evidence_class: "distributional_anomaly"` and may never on its own establish that a timestamp is an artifact.** Complaints aggregated across time zones, batch forwarding, and automated retries all produce flat or spiky hour distributions from genuine event times; a diurnal-looking histogram would prove nothing either.

*Part B — field-delta measurement (primary evidence), CFPB only.* Over records where both `date_received` and `date_sent_to_company` are present: `pair_coverage`; the delta distribution in seconds at p5, p25, p50, p75, p95, p99 with the median stated explicitly; `frac_delta_le_1min`, `frac_delta_le_10min`, `frac_delta_le_1h`; `count_delta_negative`, `count_delta_zero`; and `frac_identical_timestamps`. Labelled `evidence_class: "field_delta"`.

*The verdict, from a rule fixed before any data is seen:*

| Verdict | Rule (delta metrics only) |
|---|---|
| `strongly_suspicious_load_timestamp` | `median_delta_seconds <= 60` **or** `frac_delta_le_1min >= 0.50` **or** `frac_identical_timestamps >= 0.20` |
| `supported_plausible_event_time` | `median_delta_seconds >= 3600` **and** `frac_delta_le_1min < 0.05` **and** `count_delta_negative == 0` |
| `suspicious_insufficient_evidence` | anything else, **and always** when `pair_coverage < 0.50` |

Part A does not enter this rule. It may downgrade `supported_plausible_event_time` to `suspicious_insufficient_evidence` when extreme — movement toward doubt only. It can never produce `strongly_suspicious_load_timestamp` and can never upgrade a verdict. The rule thresholds are recorded in the manifest beside the verdict, so a reader sees which branch fired.

*NYC 311 has no testable pair.* Its `created_date`→`closed_date` interval **is** the target variable, so using it for provenance would test the label with the label. 311 receives Part A only and is recorded `suspicious_insufficient_evidence` by construction, with `not_directly_testable: true` — never defaulted to supported. Absence of evidence is not recorded as evidence of soundness.

**Tests:** with a stub adapter, an interrupted run resumes without duplicating records; re-running with the raw cache present performs zero fetches; `--limit` is recorded in the manifest; the roster assertion from Task 7 runs before any Parquet file is written (asserted by checking no part file exists after a `RosterMismatch`); **a synthetic CFPB corpus with a 3-second median delta yields `strongly_suspicious_load_timestamp`; one with a multi-hour median and no sub-minute mass yields `supported_plausible_event_time`; one with 40% coverage yields `suspicious_insufficient_evidence` regardless of its deltas**; **a corpus with a uniform hour histogram but healthy multi-hour deltas does NOT yield `strongly_suspicious_load_timestamp`** (distribution alone cannot establish artifact status); a corpus with extreme hour concentration downgrades `supported` to `suspicious_insufficient_evidence` but never further; 311 is recorded `suspicious_insufficient_evidence` with `not_directly_testable: true`; both parts carry their `evidence_class`; the rule thresholds appear in the manifest.
**Acceptance:** no test performs a network call; CLI is importable without Django; `timestamp_diagnostic` present in every manifest with both evidence classes labelled and the verdict rule recorded.
**Commit:** `feat: resumable corpus ingestion CLI with timestamp provenance diagnostic`
**Must not change:** normalization logic.

### Task 9: Temporal split machinery

**Objective:** Implement the date-cut split that never straddles a timestamp.
**Files:** Create `ml/training/splits.py`, `tests/ml/training/test_splits.py`.
**Prerequisites:** Task 2.
**Behavior:** `temporal_split(timestamps, fractions=(0.70, 0.15, 0.15)) -> TemporalSplit` per §H. `TemporalSplit` exposes `train_end`, `val_end`, `period_of(ts) -> Period`, per-period counts, and `achieved_fractions`.
**Tests:** with all-unique timestamps the achieved fractions are within one record of requested; with heavy ties (10,000 records across 3 timestamps) no timestamp appears in two periods and the achieved fractions are reported honestly rather than forced; periods are contiguous and disjoint; an empty input raises; a single-timestamp input puts everything in train and reports it.
**Acceptance:** the function never returns a split where `max(train timestamps) == min(val timestamps)`.
**Commit:** `feat: date-cut temporal split that never straddles a timestamp`
**Must not change:** anything outside `ml/training/`.

### Task 10: Forward-chaining folds

**Objective:** Implement the expanding-window folds used for out-of-fold aggregates.
**Files:** Modify `ml/training/splits.py`. Test: `tests/ml/training/test_folds.py`.
**Prerequisites:** Task 9.
**Behavior:** `forward_chaining_folds(timestamps, n_folds=5, warmup_fraction=0.20) -> list[Fold]` per §J. `Fold` exposes `fit_indices`, `apply_indices`, `fit_end`, `apply_start`, `apply_end`.
**Tests:** every fold's `fit` block is strictly earlier than its `apply` block; fold `apply` blocks are disjoint and together cover everything after the warm-up; no timestamp straddles a fold boundary; the warm-up prefix appears in no `apply` block and its size is reported; `n_folds=1` and degenerate tiny inputs behave or raise explicitly. **One test documents why `TimeSeriesSplit` was rejected** by constructing a tied-timestamp case where index-based splitting straddles and the date-cut implementation does not.
**Acceptance:** no `sklearn.model_selection` import in `splits.py`.
**Commit:** `feat: date-cut forward-chaining folds for out-of-fold aggregation`
**Must not change:** `temporal_split` semantics.

### Task 11: Out-of-fold target-derived aggregates

**Objective:** Build the two target-derived features without letting any row see its own outcome.
**Files:** Create `ml/training/aggregates.py`, `tests/ml/training/test_aggregates.py`.
**Prerequisites:** Task 10.
**Behavior:** `oof_category_aggregates(records, outcomes, folds) -> AggregateColumns` producing `category_mean_resolution_hours` and `category_breach_rate` for training rows, `NaN` for warm-up rows. `fit_category_aggregates(train_records, train_outcomes) -> FrozenAggregates` and `apply_category_aggregates(frozen, records)` for validation and test, with the train-period global mean for unseen categories.
**Tests — this is the highest-risk module in the plan:**
- changing one training row's outcome does **not** change that row's own aggregate value (the defining property);
- changing that row's outcome **does** change a later row's aggregate in the same category (proving the aggregate is live, not a constant);
- a warm-up row's aggregates are `NaN`, not the global mean;
- validation aggregates are identical when validation outcomes are permuted (proving they derive only from train);
- an unseen category in test receives the train global mean;
- `FrozenAggregates` rejects mutation, so val/test cannot be refit.
**Acceptance:** every one of the six tests above exists and passes.
**Commit:** `feat: out-of-fold target-derived category aggregates`
**Must not change:** fold construction.

### Task 12: Risk feature assembly and FeatureSpec

**Objective:** Assemble the five-feature `RiskFeaturesV1` matrix and the three-feature `TransferFeaturesV1` matrix, each in a fixed, declared order.
**Files:** Create `ml/training/features.py`, `tests/ml/training/test_features.py`.
**Prerequisites:** Task 11. (**No longer depends on Task 13** — with `sla_hours` removed, no threshold enters the feature matrix.)
**Behavior:** `FeatureSpec(names: tuple[str, ...], version: str)`; module constants `RISK_FEATURES_V1` (five names, ordered per §J) and `TRANSFER_FEATURES_V1` (three names). `build_features(records, aggregates, spec) -> np.ndarray` produces columns in `spec.names` order exactly. Requesting a feature the inputs cannot supply raises `FeatureUnavailable` naming it.
**Tests:** column order matches `spec.names`; permuting `spec.names` permutes the output columns correspondingly; a spec naming `queue_depth` raises `FeatureUnavailable`; **a spec naming `sla_hours` raises `FeatureUnavailable`** (the removal is enforced, not merely documented); `NaN` aggregates survive assembly rather than being imputed; `TRANSFER_FEATURES_V1` is a strict subset of `RISK_FEATURES_V1` and carries a distinct `version` string.
**Acceptance:** both specs' names and orders asserted in tests; `RISK_FEATURES_V1` has exactly five entries and `TRANSFER_FEATURES_V1` exactly three.
**Commit:** `feat: risk feature assembly with declared feature order`
**Must not change:** aggregate semantics.

### Task 13: 311 SLA threshold fitting

**Objective:** Fit, freeze and apply per-type thresholds.
**Files:** Create `ml/training/thresholds.py`, `tests/ml/training/test_thresholds.py`.
**Prerequisites:** Task 9.
**Behavior:** `fit_thresholds(train_records, train_outcomes, min_records=100) -> FrozenThresholds` computing each type's p75 with a global p75 fallback; `apply_thresholds(frozen, records, outcomes) -> np.ndarray[bool]`. `FrozenThresholds` is immutable and records `per_type`, `global_fallback`, `fallback_type_count`.
**Tests:** a type with ≥100 training records gets its own p75; a type with 99 gets the fallback and the fallback count increments; thresholds computed on train are unchanged when validation outcomes are permuted; `apply_thresholds` on a test set does not mutate the frozen object; breach rate is reported per period.
**Acceptance:** no code path recomputes a threshold outside `fit_thresholds`.
**Commit:** `feat: frozen per-type 311 SLA thresholds`
**Must not change:** split machinery.

### Task 14: Metrics and baselines

**Objective:** Implement every metric against hand-computed expectations, with baselines as first-class outputs.
**Files:** Create `ml/training/metrics.py`, `tests/ml/training/test_metrics.py`.
**Prerequisites:** Task 2.
**Behavior:** `macro_f1`, `per_class_report`, `top_k_accuracy`, `pr_auc`, `minority_report` (precision/recall/F1 plus absolute count), `recall_at_k`, `majority_baseline`, `stratified_baseline`. Every function returns a structure that carries its baseline beside the score, so a metric cannot be reported alone.
**Tests:** each metric matches a hand-computed value on a small fixture; `pr_auc` on a 99:1 fixture differs sharply from ROC-AUC on the same data (documenting why the headline changed); `majority_baseline` on the 99:1 fixture scores 0.99 accuracy and near-zero minority recall.
**Acceptance:** no metric function can return a bare float without its baseline.
**Commit:** `feat: imbalance-aware metrics with mandatory baselines`
**Must not change:** anything outside `ml/training/`.

### Task 15: Artifact format, metadata and load guard

**Objective:** Write and load artifacts whose metadata is authoritative about their features.
**Files:** Create `ml/training/artifacts.py`, `tests/ml/training/test_artifacts.py`.
**Prerequisites:** Tasks 12, 14.
**Behavior:** `write_artifact(model, metadata, path)`; `load_artifact(path) -> LoadedArtifact` exposing `feature_spec` and a `build_features(records)` bound to that spec. Loading raises `FeatureSpecMismatch` when the environment cannot produce a named feature, and `ArtifactSchemaError` when metadata lacks a required field.
**Tests:** an artifact declaring `queue_depth` fails to load with a message naming it; metadata missing `corpus_id` fails; a round-tripped artifact reproduces identical predictions on fixed input; feature order from metadata is preserved through loading.
**Acceptance:** the guard is proven by a deliberately mismatched artifact fixture.
**Commit:** `feat: artifact metadata format with feature_spec load guard`
**Must not change:** `ml/registry.py` — Phase 2 wires nothing into serving.

### Task 16: CFPB triage experiment

**Objective:** Train and evaluate the triage classifier end to end.
**Files:** Create `ml/training/experiments/__init__.py`, `ml/training/experiments/triage.py`, `tests/ml/training/test_triage_experiment.py`.
**Prerequisites:** Tasks 9, 14, 15, 8.
**Behavior:** Loads the CFPB corpus, applies `temporal_split`, fits TF-IDF on train text only, trains a calibrated linear classifier, tunes the abstention threshold on validation, evaluates on test, and writes an artifact per §P.
**Tests (marked `ml`, on a small fixture corpus):** a token appearing only in test text is absent from the fitted vocabulary (leakage path 2); the abstention threshold is chosen from validation and unchanged by test labels (leakage path 4); metrics appear beside baselines; the artifact's `feature_spec` and `label_roster` match what was trained.
**Also required in this task:** **remove the exit-code-5 tolerance from the `ml` CI job** (`.github/workflows/ci.yml`), restoring the step to a plain `pytest -m ml`. This task introduces the marker's first users, so from here an empty selection means a mis-typed marker rather than an expected state, and the job must fail on it. Leaving the tolerance in place would let a typo silently pass as success — see the Task 1 note that created it.
**Acceptance:** experiment runs to completion on the fixture corpus and produces a loadable artifact; `pytest -m ml` selects a non-zero number of tests; the CI step no longer special-cases any exit code.
**Commit:** `feat: CFPB triage experiment with leakage-tested pipeline`
**Must not change:** metrics or artifact modules.

### Task 17: 311 risk experiment

**Objective:** Train and evaluate the risk model end to end.
**Files:** Create `ml/training/experiments/risk.py`, `tests/ml/training/test_risk_experiment.py`.
**Prerequisites:** Tasks 11, 12, 13, 15.
**Behavior:** Loads the 311 corpus, splits, fits thresholds on train, builds out-of-fold aggregates for train and frozen aggregates for val/test, trains `HistGradientBoostingClassifier`, tunes banding on validation, evaluates on test, writes the artifact.
**Tests (marked `ml`):** train aggregates differ from val/test construction paths; permuting test outcomes leaves training features unchanged; warm-up `NaN` rows survive into training; thresholds in metadata match those fitted; metrics appear beside baselines.
**Acceptance:** completes on a fixture corpus; publishes metrics whatever they show.
**Commit:** `feat: 311 SLA risk experiment with out-of-fold features`
**Must not change:** aggregate or threshold semantics.

### Task 18: TextEmbedder and both benchmark arms

**Objective:** Add the embedder abstraction and both implementations under one frozen config.
**Files:** Modify `ml/base.py` (add `TextEmbedder` protocol only). Create `ml/embedders/tfidf.py`, `ml/embedders/minilm.py`, `ml/training/experiments/dedup.py`, `tests/ml/embedders/test_tfidf.py`, `tests/ml/embedders/test_minilm.py`, `tests/ml/training/test_dedup_benchmark.py`.
**Prerequisites:** Tasks 9, 14.
**Behavior:** `TextEmbedder` protocol with `embed(texts: Sequence[str]) -> np.ndarray` and `model_version: str`. `BenchmarkConfig` frozen dataclass holding split boundaries, evaluation population refs, perturbation seed and types, `k`, similarity function and tuning budget — consumed identically by both arms.
**Embedding dimension is observed, recorded and validated — never assumed** (addendum D18). Each embedder exposes `embedding_dimension` read from its actual output, plus `embedding_model_id` and, for MiniLM, the ONNX file's SHA256. The index records the dimension it was built with and validates every subsequent embedding against it. `384` appears nowhere as a literal: it is a property of `all-MiniLM-L6-v2`, not of the MiniLM family or of an arbitrary ONNX export whose pooling layer may differ.
**Tests:** both arms receive the identical `BenchmarkConfig` instance (asserted by identity, not equality); TF-IDF vocabulary is fitted on train text only; the retrieval index contains no record later than the query period (leakage path 6); recall@k is computed over `RecordRef`, never an integer; MiniLM output vectors are L2-normalised; **`embedding_dimension` matches the model's observed output width and is recorded in metadata**; **an embedding whose width differs from the index's recorded dimension raises `EmbeddingDimensionMismatch` rather than broadcasting or silently comparing**; a grep-style test asserts no `384` literal in `ml/embedders/`; a test asserts the two arms' configs are the same object so divergence is impossible without editing shared config.
**Acceptance:** benchmark runs both arms on the fixture corpus and reports recall@k per perturbation type with the result labelled synthetic.
**Commit:** `feat: TextEmbedder abstraction with TF-IDF and MiniLM benchmark arms`
**Must not change:** `Match`, `DedupIndex`, or any serving code.

### Task 19: Reduced-feature cross-domain cross-target robustness probe

**Objective:** Train a distinct three-feature model on 311 and evaluate it in-domain and cross-domain/cross-target on CFPB, with diagnostics that make the result's interpretability visible.
**Files:** Create `ml/training/experiments/robustness_probe.py`, `tests/ml/training/test_robustness_probe.py`.
**Prerequisites:** Tasks 8, 12, 13, 14, 15, 17. (**Task 8 is a hard prerequisite** — the probe reads its `timestamp_diagnostic` to set `result_classification`.)
**Behavior:** Trains a separate `HistGradientBoostingClassifier` on the 311 corpus using `TRANSFER_FEATURES_V1` only, targeting `nyc311_sla_breach` (thresholds from Task 13, frozen). Evaluates in-domain on held-out 311 and cross-domain/cross-target on CFPB against `cfpb_timely_response`. Writes its own artifact with `model_name="xdomain_xtarget_probe"`, its own `model_version`, `feature_spec` of exactly the three names, and `experiment_label="reduced-feature cross-domain cross-target robustness probe"`.

**Required output — the six framing facts.** The experiment emits, in its own report structure (not only in prose docs): `source_domain`, `source_target` with its threshold rule, `evaluation_domain`, `evaluation_target` with what the field measures, `feature_set` with the reason the five-feature set was unusable, `target_semantics_differ: true` with both construct definitions, `polarity_mapping` (see I7), and `analysis_type: "exploratory robustness, not same-task transfer"`.

**Required output — feature distribution diagnostic (finding C3), strengthened.** For each feature in `TRANSFER_FEATURES_V1`, the report records:
- source-training quantiles (min, p01, p05, p25, p50, p75, p95, p99, max);
- evaluation-set quantiles at the same points;
- **`pct_outside_source_range`** — the percentage of evaluation records whose value falls outside the source training [min, max] interval;
- **`pct_outside_source_iqr`** — the percentage outside the source training [p25, p75];
- an `out_of_range` flag where the evaluation median falls outside the source training IQR.

Written to metadata as `feature_distribution_shift` and reproduced in the published table. `text_length` is expected to flag: measured medians are 15 chars (311 descriptor) against 1,202 (CFPB narrative), so `HistGradientBoostingClassifier`'s training-derived bins place essentially every CFPB record in the topmost bin, making the feature effectively constant at inference.

**Split reuse (finding I6).** The probe **reuses Task 17's exact 311 split boundaries** — it does not compute its own. Otherwise the reduced-feature in-domain reference performance and the primary model's in-domain figure would rest on different data, and the obvious reader question ("what did dropping two features cost?") would be answered with an invalid comparison. CFPB evaluation uses the **CFPB test period only**, so no record that tuned the triage model's threshold appears here. Both period identifiers are recorded in metadata.

**Polarity mapping is explicit (finding I7).** The model outputs P(adverse outcome). "Adverse" means `nyc311_sla_breach == True` in the source and `cfpb_timely_response == False` in the evaluation domain. That mapping is an interpretive choice, not a fact about the data, so it is stated as a sixth framing fact — `polarity_mapping` — rather than left implicit in the code.

**The two PR-AUCs are not comparable to each other (finding I7).** Measured base rates differ by roughly 27×: 311 breach runs 26.9–31.1% at the p75 threshold, while CFPB not-timely is 1.12% within narratives. PR-AUC is base-rate dependent, so a lower cross-domain PR-AUC is expected arithmetic, not evidence of anything. The report therefore states, in its own output, that each figure is interpretable **only as lift over its own baseline**, records both baselines beside both figures, and includes `base_rate` per evaluation. A test asserts the report never presents the two PR-AUCs as a single before/after pair.

**Required output — result classification, gated on the §2.3 verdict (escalated I2, addendum D20).** The probe reads Task 8's `timestamp_diagnostic` verdict for CFPB and sets:

| Task 8 verdict | `result_classification` |
|---|---|
| `strongly_suspicious_load_timestamp` | `non-informative / diagnostic` — figures must not be interpreted as substantive model evidence |
| `suspicious_insufficient_evidence` | `substantive_with_stated_caveat` — figures reported with the unresolved provenance question beside them |
| `supported_plausible_event_time` | `substantive` |

**The downgrade to non-informative fires only on the pre-specified field-delta rule, never on histogram shape.** A non-diurnal hour distribution is secondary evidence and is insufficient on its own — distribution does not identify provenance. The probe copies the verdict, the delta metrics that produced it, and the rule thresholds into its own output, so which branch fired and why is visible without opening the manifest.

Rationale recorded in the output for the non-informative branch: with `submitted_hour` unusable and `text_length` degenerate across domains per C3, the model would rest almost entirely on `submitted_weekday`, and any resulting figure would describe nothing.

**Tests (marked `ml`):** train aggregates differ from val/test construction paths; permuting test outcomes leaves training features unchanged; warm-up `NaN` rows survive into training; thresholds in metadata match those fitted; metrics appear beside baselines.
**Acceptance:** completes on a fixture corpus; publishes metrics whatever they show.
**Commit:** `feat: 311 SLA risk experiment with out-of-fold features`
**Must not change:** aggregate or threshold semantics.

### Task 18: TextEmbedder and both benchmark arms

**Objective:** Add the embedder abstraction and both implementations under one frozen config.
**Files:** Modify `ml/base.py` (add `TextEmbedder` protocol only). Create `ml/embedders/tfidf.py`, `ml/embedders/minilm.py`, `ml/training/experiments/dedup.py`, `tests/ml/embedders/test_tfidf.py`, `tests/ml/embedders/test_minilm.py`, `tests/ml/training/test_dedup_benchmark.py`.
**Prerequisites:** Tasks 9, 14.
**Behavior:** `TextEmbedder` protocol with `embed(texts: Sequence[str]) -> np.ndarray` and `model_version: str`. `BenchmarkConfig` frozen dataclass holding split boundaries, evaluation population refs, perturbation seed and types, `k`, similarity function and tuning budget — consumed identically by both arms.
**Embedding dimension is observed, recorded and validated — never assumed** (addendum D18). Each embedder exposes `embedding_dimension` read from its actual output, plus `embedding_model_id` and, for MiniLM, the ONNX file's SHA256. The index records the dimension it was built with and validates every subsequent embedding against it. `384` appears nowhere as a literal: it is a property of `all-MiniLM-L6-v2`, not of the MiniLM family or of an arbitrary ONNX export whose pooling layer may differ.
**Tests:** both arms receive the identical `BenchmarkConfig` instance (asserted by identity, not equality); TF-IDF vocabulary is fitted on train text only; the retrieval index contains no record later than the query period (leakage path 6); recall@k is computed over `RecordRef`, never an integer; MiniLM output vectors are L2-normalised; **`embedding_dimension` matches the model's observed output width and is recorded in metadata**; **an embedding whose width differs from the index's recorded dimension raises `EmbeddingDimensionMismatch` rather than broadcasting or silently comparing**; a grep-style test asserts no `384` literal in `ml/embedders/`; a test asserts the two arms' configs are the same object so divergence is impossible without editing shared config.
**Acceptance:** benchmark runs both arms on the fixture corpus and reports recall@k per perturbation type with the result labelled synthetic.
**Commit:** `feat: TextEmbedder abstraction with TF-IDF and MiniLM benchmark arms`
**Must not change:** `Match`, `DedupIndex`, or any serving code.

### Task 19: Reduced-feature cross-domain cross-target robustness probe

**Objective:** Train a distinct three-feature model on 311 and evaluate it in-domain and cross-domain/cross-target on CFPB, with diagnostics that make the result's interpretability visible.
**Files:** Create `ml/training/experiments/robustness_probe.py`, `tests/ml/training/test_robustness_probe.py`.
**Prerequisites:** Tasks 8, 12, 13, 14, 15, 17. (**Task 8 is a hard prerequisite** — the probe reads its `timestamp_diagnostic` to set `result_classification`.)
**Behavior:** Trains a separate `HistGradientBoostingClassifier` on the 311 corpus using `TRANSFER_FEATURES_V1` only, targeting `nyc311_sla_breach` (thresholds from Task 13, frozen). Evaluates in-domain on held-out 311 and cross-domain/cross-target on CFPB against `cfpb_timely_response`. Writes its own artifact with `model_name="xdomain_xtarget_probe"`, its own `model_version`, `feature_spec` of exactly the three names, and `experiment_label="reduced-feature cross-domain cross-target robustness probe"`.

**Required output — the six framing facts.** The experiment emits, in its own report structure (not only in prose docs): `source_domain`, `source_target` with its threshold rule, `evaluation_domain`, `evaluation_target` with what the field measures, `feature_set` with the reason the five-feature set was unusable, `target_semantics_differ: true` with both construct definitions, `polarity_mapping` (see I7), and `analysis_type: "exploratory robustness, not same-task transfer"`.

**Required output — feature distribution diagnostic (finding C3), strengthened.** For each feature in `TRANSFER_FEATURES_V1`, the report records:
- source-training quantiles (min, p01, p05, p25, p50, p75, p95, p99, max);
- evaluation-set quantiles at the same points;
- **`pct_outside_source_range`** — the percentage of evaluation records whose value falls outside the source training [min, max] interval;
- **`pct_outside_source_iqr`** — the percentage outside the source training [p25, p75];
- an `out_of_range` flag where the evaluation median falls outside the source training IQR.

Written to metadata as `feature_distribution_shift` and reproduced in the published table. `text_length` is expected to flag: measured medians are 15 chars (311 descriptor) against 1,202 (CFPB narrative), so `HistGradientBoostingClassifier`'s training-derived bins place essentially every CFPB record in the topmost bin, making the feature effectively constant at inference.

**Split reuse (finding I6).** The probe **reuses Task 17's exact 311 split boundaries** — it does not compute its own. Otherwise the reduced-feature in-domain reference performance and the primary model's in-domain figure would rest on different data, and the obvious reader question ("what did dropping two features cost?") would be answered with an invalid comparison. CFPB evaluation uses the **CFPB test period only**, so no record that tuned the triage model's threshold appears here. Both period identifiers are recorded in metadata.

**Polarity mapping is explicit (finding I7).** The model outputs P(adverse outcome). "Adverse" means `nyc311_sla_breach == True` in the source and `cfpb_timely_response == False` in the evaluation domain. That mapping is an interpretive choice, not a fact about the data, so it is stated as a sixth framing fact — `polarity_mapping` — rather than left implicit in the code.

**The two PR-AUCs are not comparable to each other (finding I7).** Measured base rates differ by roughly 27×: 311 breach runs 26.9–31.1% at the p75 threshold, while CFPB not-timely is 1.12% within narratives. PR-AUC is base-rate dependent, so a lower cross-domain PR-AUC is expected arithmetic, not evidence of anything. The report therefore states, in its own output, that each figure is interpretable **only as lift over its own baseline**, records both baselines beside both figures, and includes `base_rate` per evaluation. A test asserts the report never presents the two PR-AUCs as a single before/after pair.

**Required output — non-informative classification (escalated I2).** The probe reads the ingest-time timestamp diagnostic from Task 8. **If that diagnostic indicates CFPB `date_received` time-of-day is likely an ingestion artifact rather than a genuine event time, the probe sets `result_classification: "non-informative / diagnostic"` and the report states that the figures must not be interpreted as substantive model evidence.** Rationale carried in the output: with `submitted_hour` an artifact and `text_length` degenerate per C3, the model would be trained and scored almost entirely on `submitted_weekday`, and any resulting figure — good or bad — would describe nothing. Otherwise the classification is `"substantive"`, and the criterion that produced it is recorded either way.

**Tests (marked `ml`):**
- the artifact's `feature_spec` has exactly three names and its `model_name` differs from the primary risk artifact's;
- loading the primary five-feature artifact and attempting to score CFPB records raises `FeatureUnavailable` — the impossibility motivating D16 is asserted, not merely described;
- all six framing facts are present in the emitted report structure, each non-empty;
- the probe's 311 split boundaries are identical to Task 17's, and CFPB evaluation draws only from the CFPB test period;
- `base_rate` is recorded for each evaluation, and a test asserts the two PR-AUCs are never presented as a single before/after pair;
- `feature_distribution_shift` includes `pct_outside_source_range` and `pct_outside_source_iqr` for every transfer feature, and flags `text_length`;
- given a stubbed diagnostic whose verdict is `strongly_suspicious_load_timestamp`, `result_classification` is `"non-informative / diagnostic"`; `suspicious_insufficient_evidence` yields `"substantive_with_stated_caveat"`; `supported_plausible_event_time` yields `"substantive"`;
- **a stubbed diagnostic with a non-diurnal hour histogram but a `supported_plausible_event_time` verdict does NOT yield `"non-informative / diagnostic"`** — histogram shape alone cannot downgrade the result;
- the verdict, its delta metrics and the rule thresholds are copied into the probe's own output;
- both evaluations are reported, and the cross-domain figure is accompanied by the reduced-feature in-domain reference performance;
- every metric appears beside its majority-class baseline, with absolute minority counts;
- the report contains none of the strings "transfers to", "generalises to", "generalizes to".

**Acceptance:** two evaluations published with baselines; the strengthened distribution diagnostic emitted with `text_length` flagged; the classification set from a recorded criterion; the primary artifact never scored on CFPB anywhere in the codebase.
**Commit:** `feat: reduced-feature cross-domain cross-target robustness probe`
**Must not change:** the primary risk experiment, `RISK_FEATURES_V1`, or thresholds.

### Task 20: Resource measurement

**Objective:** Measure inference-time resource cost in a clean environment.
**Files:** Create `ml/training/measure.py`, `docs/phase-2-resource-measurements.md`.
**Prerequisites:** Task 18.
**Behavior:** Measures the §S quantities in a venv built from `requirements/ml.txt` only, recording CPU model, core count and library versions with every figure.
**Tests:** the measurement harness reports the environment it ran in and refuses to write a report if `pandas` or `pyarrow` is importable (proving the environment is inference-only).
**Acceptance:** every published number came from a recorded run.
**Commit:** `feat: inference-environment resource measurement`
**Must not change:** experiment code.

### Task 21: Reproducibility documentation and README metrics

**Objective:** Document how to rerun everything, and publish measured metrics.
**Files:** Create `docs/phase-2-reproducibility.md`. Modify `README.md`.
**Prerequisites:** Tasks 16, 17, 18, 20.
**Behavior:** Per §T. The cross-platform metric tolerance is *measured* on at least two environments and stated, not asserted.
**Tests:** a doc test asserts every metric table in `README.md` has a baseline column; **a doc test asserts neither `README.md` nor the reproducibility guide contains the strings "transfers to", "generalises to" or "generalizes to" in any sentence naming the robustness probe** (addendum D19's binding prohibition, enforced in docs as Task 19 enforces it in the report); a doc test asserts the probe's published table carries its `result_classification`.
**Acceptance:** no unmeasured figure appears in either document.
**Commit:** `docs: Phase 2 reproducibility guide and measured metrics`
**Must not change:** experiment results to make them look better.

### Task 22: Whole-branch integration verification

**Objective:** Verify the properties no single task can.
**Files:** Create `tests/test_phase2_integration.py`.
**Prerequisites:** all.
**Behavior:** Asserts the Phase 2 boundary held: no migration exists beyond Phase 1's; `ml/registry.py` still returns null implementations; no `Prediction` row is written by any Phase 2 code path; `complaints/` contains no source slug literal; the app suite passes without `train.txt` installed.
**Tests:** the above, each as a named test.
**Acceptance:** full suite green under both dependency environments; `makemigrations --check` clean.
**Commit:** `test: Phase 2 whole-branch integration guarantees`
**Must not change:** anything — this task only adds tests.

## V. Dependencies between tasks

```
1 → 2 → 3 → 4 ──────────────┐
        3 → 5 → 6 → 8       │
            5 → 7 → 8       │
    2 → 9 → 10 → 11 → 12 ───┼→ 15 → 16
    2 → 9 → 13 ─────────────┤        17 → 19
    2 → 14 ─────────────────┘
    9, 14 → 18 → 20 → 21
    8 ─────────────────────────────→ 19   (timestamp_diagnostic)
    all → 22
```

**Changed by D15:** Task 12 no longer depends on Task 13. With `sla_hours` removed, no threshold enters the feature matrix, so feature assembly and threshold fitting are now independent and may proceed in either order. Task 13 remains a prerequisite of Tasks 17 and 19, which need thresholds to produce the *label*.

Tasks 16 and 17 are independent of each other. Task 19 depends on 17 only for sequencing discipline — it must not be built before the primary risk experiment exists, so that the "primary artifact cannot score CFPB" assertion has a real artifact to test against.

## W. Acceptance criteria

Stated per task above. Phase 2 as a whole is complete when: both corpora ingest reproducibly with asserted rosters; every leakage path in §I has a passing test that fails when leakage is introduced; the triage and risk experiments produce loadable artifacts with metrics beside baselines; the dedup benchmark reports both arms under one config with a ship-or-cut decision recorded; the robustness probe publishes its cross-domain cross-target figures beside both its baselines and its reduced-feature in-domain reference performance, carries all six framing facts, and records a result classification; resource figures come from a measured inference-only environment; and Task 22's integration guarantees pass.

## X. Failure and rollback strategy

Each task is one commit, so rollback is `git revert` of that commit. The corpus is regenerable from `data/raw/` without network access, and from the source APIs given the recorded window. Artifacts are versioned directories, so a bad model is abandoned by not pinning it — nothing in Phase 2 is wired into serving, so no rollback touches the running application. If a leakage test fails after a later change, the responsible task is identified from the failing property and corrected with a regression test rather than patched around.

## Y. Final whole-branch verification checklist

- [ ] Full suite green under base+dev (no training packages installed)
- [ ] `pytest -m ml` green under base+dev+train
- [ ] `ruff check`, `ruff format --check`, `mypy complaints domains ml accounts ingest` clean
- [ ] `makemigrations --check --dry-run` reports no changes
- [ ] `ml/registry.py` unchanged; `registry_status()` still reports `null` for every model
- [ ] No `Prediction` row written by any Phase 2 path
- [ ] No source slug literal in `complaints/`
- [ ] `git ls-files` shows no Parquet, no `.joblib`, no `.onnx`, no `data/` content
- [ ] Every published metric appears beside its baseline
- [ ] Every leakage path has a test that fails when leakage is deliberately introduced
- [ ] `RISK_FEATURES_V1` has five names; a spec naming `sla_hours` raises
- [ ] The primary risk artifact is never scored on CFPB anywhere in the codebase
- [ ] No `384` literal in `ml/embedders/`; `embedding_dimension` recorded and validated
- [ ] No `precision@k` in any risk-model metric output or published table

---

## PLAN PRE-FLIGHT FINDINGS

Adversarial review of this plan against the approved addendum and the repository.

### Critical — both RESOLVED in the spec (addendum D15, D16)

**C1 — RESOLVED. The transfer evaluation is now a distinct reduced-feature model.**

*Original finding:* addendum §5.4 required evaluating the 311-trained risk model against CFPB `cfpb_timely_response`, while §3.2 fixed the feature set at six features, three of which cannot exist for a CFPB record. A model trained on six features cannot be scored on records that supply three.

*Resolution (addendum D16):* a separate `HistGradientBoostingClassifier` is trained on 311 using `TransferFeaturesV1` — `submitted_hour`, `submitted_weekday`, `text_length` — versioned independently, and evaluated both in-domain on held-out 311 and cross-domain on CFPB. The naming is binding so it can never be read as the primary model's performance. Task 19 is now implementable and asserts the original impossibility as a test: loading the primary artifact and attempting to score CFPB raises `FeatureUnavailable`.

**C2 — RESOLVED. `sla_hours` removed; `RiskFeaturesV1` is five features.**

*Original finding:* `sla_hours` is the per-type p75 of training resolution hours — the same outcomes that define the target — so a training row's own resolution time contributes to the p75 becoming its own feature. Target-derived, in a feature §6.3 did not name.

*Resolution (addendum D15):* removed. The threshold's sole role is to produce the label. The justification recorded in the spec is leakage, not redundancy; redundancy with `category_mean_resolution_hours` is noted only as a secondary observation, and the feature would be removed even if it carried unique signal. Task 12 now asserts the removal — a `FeatureSpec` naming `sla_hours` raises `FeatureUnavailable`, so the decision is enforced by code rather than documented in prose.

**C3 — NEW. `text_length` is not comparable across the two corpora, which may make the robustness probe uninterpretable.**

The reduced-feature model trains on 311 and is scored on CFPB. Its only non-calendar feature is `text_length`, and the measured distributions are not merely different — they barely overlap:

| Corpus | `text` source | Median length |
|---|---|---|
| NYC 311 | `descriptor` | **15 chars** |
| CFPB | consumer narrative | **1,202 chars** |

An ~80× shift. `HistGradientBoostingClassifier` bins continuous features at fit time from the training distribution, so every CFPB record would land in the topmost 311-derived bin. The feature becomes effectively constant at inference, leaving a model that is nominally three features and operationally two calendar features.

*Why it matters:* this is a consequence of the C1 resolution, not a pre-existing defect — it only appears once the probe's model is a real trained object rather than a hypothetical. If the probe scores poorly, the cause is unattributable between domain shift, differing target semantics, a degenerate feature, and two weak calendar features. That is precisely the confound the reduced-feature in-domain reference performance was added to remove, and it defeats it by a different route.

*Smallest correction, and it does not require a spec change:* Task 19 must report, for every feature, the train-period distribution beside the evaluation-period distribution, and flag any feature whose evaluation median falls outside the training interquartile range. The limitation then appears in the published output instead of being discovered by a reader. I have added this to Task 19 rather than leaving it implicit.

*What I did not do:* normalise `text_length` per domain. That is a domain-adaptation step, it would change what the experiment measures, and choosing it is a spec decision rather than a planning one. Flagging instead.

### Important

**I1 — The risk model's category aggregates may be unproducible at Phase 3 serving time, making the model unservable.**

`category_mean_resolution_hours` and `category_breach_rate` are keyed to 311 complaint types and computed from 311 history. At Phase 3 inference on a Sentinel complaint in the CFPB domain, neither value has a source: Sentinel has no accumulated resolution history, and the 311 values are keyed to types that do not exist in that domain. The Phase 2 model may therefore be trainable but not deployable.

*Why it matters:* it is invisible until Phase 3 and would waste the risk work.
*Smallest correction:* Task 15's metadata records that these aggregates are 311-derived and names the lookup key, and Task 21's documentation states the Phase 3 prerequisite explicitly. No Phase 2 code change.

**I2 — CFPB timestamps may be ingestion artifacts rather than event times, which would make `submitted_hour` noise.**

I verified `date_received` carries time-of-day (`2024-09-03T22:42:53.000Z`), correcting an assumption I nearly wrote into this plan. But on the same record `date_sent_to_company` is **three seconds later**, which is not plausible as a real business process and suggests both timestamps were stamped at load time.

*Why it matters — escalated by the C1 resolution.* These were two of six risk features when this finding was first written. They are now two of the **three** features in `TransferFeaturesV1`, and finding C3 shows the third is effectively constant at CFPB inference. If the CFPB hour is a load artifact, the probe's model is trained and scored almost entirely on noise, and any result it produces — good or bad — means nothing. The validation step below is no longer a nice-to-have; it determines whether Task 19 is worth running at all.
*Correction, now applied (revised — see I8):* Task 8 emits a required `timestamp_diagnostic` with two evidence classes. The **primary** evidence is the `date_received` → `date_sent_to_company` delta distribution, against a rule fixed before any data is seen. The hour and weekday distributions are retained as **secondary** evidence that may move a verdict toward doubt but can never establish artifact status. Task 19 consumes the resulting three-level verdict to set `result_classification`.

**I3 — The existing `.gitignore` would permit committing the corpus.**

It excludes `data/raw/` and `data/interim/` but not `data/corpus/`, which is where §G places multi-hundred-megabyte Parquet files. It also excludes `ml/artifacts/**/*.onnx` but not `*.joblib`, and the addendum commits sklearn artifacts to git — which is intended for *small* ones but has no size guard.

*Why it matters:* a single `git add -A` commits the corpus.
*Smallest correction:* Task 2 broadens the ignore rules, and Task 22 asserts `git ls-files` shows no Parquet or model binaries. Both are already in the plan; this finding records why.

**I4 — `ml/training/` lives inside a package the serving path imports.**

The Phase 1 design's module layout places training under `ml/`, and this plan follows it. But `ml/base.py`, `ml/null.py` and `ml/registry.py` are imported by Django at startup. A stray `from ml.training import ...` in serving code — or an `__init__.py` that eagerly imports submodules — would pull scikit-learn and onnxruntime into the web process.

*Why it matters:* it silently breaks the dependency split and the Phase 3 memory budget.
*Smallest correction:* already planned — Task 2's AST-based import-direction test. Recorded here so the reason survives.

**I5 — NEW (found in this pass). Task 19 read an output Task 8 did not produce.**

The revised Task 19 sets `result_classification` from "the ingest-time timestamp diagnostic from Task 8". Task 8 emitted no such thing — the diagnostic existed only as a suggested correction inside finding I2, which is prose about the plan rather than a specification of behaviour. Task 19 would have been implemented against a field that does not exist, and the failure would have surfaced at the very end of the plan, in its last experiment.

*Why it matters:* this is the "assumption that only becomes visible after several tasks have already been implemented" class. It arose because a finding's proposed remedy was never promoted into the owning task's specification — a gap that recurs whenever review prose and task specs drift.

*Correction, applied:* `timestamp_diagnostic` is now a required output of Task 8 with a defined structure, stated verdict thresholds, and three tests covering the uniform, diurnal and concentrated cases. Task 8 is added to Task 19's prerequisites and to the dependency graph.

**I6 — NEW. The probe would have computed its own splits, invalidating the comparison a reader will inevitably make.**

Task 19 evaluates its three-feature model in-domain on 311 and Task 17 evaluates the five-feature primary model on the same corpus. Nothing said the two must share split boundaries. A reader seeing both in-domain PR-AUCs will read the difference as the cost of dropping two features — valid only if both rest on identical data. Nothing also said which CFPB period the cross-domain evaluation draws from, leaving it free to reuse records that tuned the triage model's threshold.

*Correction, applied:* Task 19 reuses Task 17's exact 311 boundaries and evaluates on the CFPB test period only; both identifiers go into metadata, and a test asserts the boundaries match.

**I7 — NEW. The probe's two PR-AUCs are not comparable to each other, and the polarity mapping was implicit.**

Measured base rates differ by roughly 27× — 311 breach at 26.9–31.1%, CFPB not-timely at 1.12% within narratives. PR-AUC is base-rate dependent, so the cross-domain figure will be lower than the in-domain one as arithmetic, before any question of domain or target shift. Presenting them as a before/after pair would manufacture an apparent degradation out of the base rate alone — the same category of error as reporting accuracy on a 99:1 split.

Separately, the model outputs P(adverse), and "adverse" means breach in 311 but *not-timely* in CFPB. That mapping is an interpretive choice and was implicit in the plan.

*Correction, applied:* each figure is stated as interpretable only as lift over its own baseline; `base_rate` is recorded per evaluation; a test asserts the two PR-AUCs are never presented as a single pair; and `polarity_mapping` becomes a sixth required framing fact.

**I8 — NEW (found in this pass, and it was my error). Distribution shape was treated as proof of timestamp provenance.**

The previous Task 8 diagnostic emitted `plausible_diurnal` / `suspect_uniform` / `suspect_concentrated` from an hour histogram, and Task 19 downgraded the probe's result to non-informative on that basis. That inference does not hold. A flat or spiky hour distribution is entirely consistent with genuine event times — complaints aggregated across time zones, batch-forwarded submissions, and automated retries all produce one — and a diurnal-looking histogram would not have established genuine provenance either. The design would have let an anomaly signal masquerade as an identification of how the data was produced, and would then have used that to discard or accept a result.

*Why it matters:* it is the same class of error as reporting accuracy on an imbalanced split — a number that looks like it answers the question while answering a different one. It also had teeth: the verdict gates whether Task 19's figures count as evidence at all.

*Correction, applied:* the verdict now derives from a **pre-specified rule over the `date_received` → `date_sent_to_company` delta**, which brackets a real administrative process and therefore carries actual information about provenance — a median measured in seconds is not a forwarding process. Thresholds are fixed in the spec before any data is seen, so the verdict cannot become a judgment made after looking at the answer. Hour and weekday diagnostics are relabelled `evidence_class: "distributional_anomaly"`, may move a verdict only toward doubt, and can never produce the strongly-suspicious branch. NYC 311, whose only timestamp pair *is* the target variable, receives the distributional part only and is recorded `suspicious_insufficient_evidence` by construction rather than defaulted to supported.

**I9 — NEW (found in this pass). Task 8's primary evidence had no data source.**

The revised Task 8 computes its verdict from the `date_received` → `date_sent_to_company` delta. Nothing captured `date_sent_to_company`: Task 5's normalization mapped four CFPB fields and `CFPBOutcome` held only `external_id` and `timely_response`. Task 8 would have had nothing to measure, and the failure would have surfaced only once the verdict rule was implemented — two tasks after the schema was frozen.

*Why it matters:* it is the I5 pattern repeating. A requirement introduced in one task's specification silently assumed an input another task was never told to produce. That both instances came from *late amendments* rather than the original plan is the actual lesson: every amendment needs its own upstream trace, not just a downstream one.

*Correction, applied:* `CFPBOutcome` gains `sent_to_company_at: datetime | None`, Task 5 normalizes it with tests for the present and absent cases, and Task 3's schema test asserts the field. The spec records that it is provenance evidence only — never a feature, never a target — since it is downstream of intake and unavailable for a live complaint, so using it as a feature would leak.

### Minor

**M1 — `ml/apps.py` remains unreachable.** Carried forward from Phase 1; `ml` is not in `INSTALLED_APPS`, correctly. Phase 2 adds no reason to register it. No action.

**M2 — `mypy` targets do not include `ingest`.** CI runs `mypy complaints domains ml accounts`; since `ml/training` is under `ml`, it is covered, but `ingest` is not.

*Corrected during Task 1 execution:* this disposition said Task 1 adds `ingest` to the CI mypy target. It cannot — `ingest/` does not exist until Task 2, and mypy fails hard on a missing path (`mypy: error: Cannot read file 'ingest': No such file or directory`, verified). Adding it in Task 1 would have broken CI on the first push. **The mypy target change moves to Task 2**, which creates the package.

**M3 — Fixture corpora must be small enough to commit but realistic enough to exercise splits.** A five-row fixture cannot exercise a 70/15/15 split with five folds. Tasks 16 and 17 need a synthetic fixture corpus of a few hundred generated records with controlled timestamps — generated by a committed script, not committed as data.

**M4 — The `ml` pytest marker is declared but no test currently uses it.** Phase 2 is the first user. No action; recorded to explain why the marker existed before any marked test.

---

## SPEC REQUIREMENT → TASK → TEST COVERAGE MATRIX

| Spec requirement | Plan task(s) | Tests proving it | Status |
|---|---|---|---|
| CFPB window 2024–2025 only | 8 | `test_cli` window recorded in manifest | Covered |
| Roster derived, not hardcoded | 7 | `test_roster` — no literal in module | Covered |
| Out-of-roster label fails ingest | 7, 8 | `test_roster` unexpected/missing/swapped | Covered |
| No drop / remap / auto-expand | 7 | swapped-label test | Covered |
| Corpus never enters `Complaint` | 2, 22 | import-boundary + integration tests | Covered |
| Identity is `(source, external_id)` | 3 | `test_identity` round-trip; no int id | Covered |
| No synthetic `Complaint.pk` | 3, 22 | schema test; integration test | Covered |
| `RawComplaint` withdrawn | 3 | schema defines `CorpusRecord` only | Covered |
| Per-source outcome types | 3, 5, 6 | normalization tests | Covered |
| Corpus on disk, versioned, checksummed | 4 | `test_manifest` corpus_id stability | Covered |
| Resumable / partial ingestion | 8 | interrupted-run test | Covered |
| `TextEmbedder` is the benchmark abstraction | 18 | embedder tests | Covered |
| `DedupIndex` / `Match` unchanged | 18, 22 | must-not-change + integration | Covered |
| `RiskFeaturesV1` = 5, ordered | 12 | order asserted; permutation test; five-name count | Covered |
| `sla_hours` excluded and enforced | 12 | spec naming `sla_hours` raises `FeatureUnavailable` | Covered |
| `TransferFeaturesV1` = 3, versioned separately | 12 | subset + distinct-version test | Covered |
| `feature_spec` authoritative at load | 15 | mismatched-artifact fixture | Covered |
| Fail loudly on unproducible feature | 12, 15 | `FeatureUnavailable`, `FeatureSpecMismatch` | Covered |
| Phase 3 interface not mutated | 22 | integration assertion | Covered |
| Aggregates OOF for training rows | 11 | own-outcome-invariance test | Covered |
| Forward-chaining, not random K-fold | 10 | fit-before-apply; no sklearn import | Covered |
| Not leave-one-out | 10, 11 | fold structure test | Covered |
| Val/test aggregates from train only | 11 | permuted-val-outcomes test | Covered |
| Unseen category → train global mean | 11 | unseen-category test | Covered |
| Warm-up row handling defined | 10, 11 | `NaN` assertion | Covered |
| CFPB / 311 SLA never unified | 3, 6 | separate outcome types; no `sla_met` | Covered |
| 311 threshold = per-type train p75 | 13 | p75 and fallback tests | Covered |
| <100 records → global fallback | 13 | 99-record boundary test | Covered |
| Threshold frozen before val/test | 13 | immutability + permutation tests | Covered |
| Thresholds and breach rate in metadata | 13, 15 | metadata schema test | Covered |
| Temporal split mandatory | 9, 16, 17 | split tests; experiment tests | Covered |
| No timestamp straddles a boundary | 9, 10 | tied-timestamp tests | Covered |
| 70/15/15 default, actuals recorded | 9 | achieved-fractions test | Covered |
| Preprocessing fitted on train | 16 | vocabulary leakage test | Covered |
| TF-IDF vocab/IDF train-only | 16, 18 | test-only-token absence | Covered |
| Threshold tuning on validation | 16, 17 | permuted-test-labels test | Covered |
| Resampling train only | 17 | natural-balance assertion | Covered |
| Eval index excludes future records | 18 | index leakage test | Covered |
| Triage: macro-F1 headline + per-class | 14, 16 | metric tests | Covered |
| Triage: top-3, confusion matrix | 14, 16 | metric tests | Covered |
| `precision@k` absent from risk metrics | 14, 17 | no `precision_at_k` in risk output | Covered |
| Baselines: majority + stratified | 14 | baseline tests | Covered |
| Accuracy never headline | 14, 21 | README baseline-column doc test | Covered |
| Dedup: recall@k over `RecordRef` | 18 | ref-typed assertion | Covered |
| Dedup labelled synthetic | 18, 21 | report wording test | Covered |
| Embedding dimension recorded, not assumed | 18 | dimension read from output; recorded in metadata | Covered |
| Embedding dimension validated on mismatch | 18 | `EmbeddingDimensionMismatch` test; no `384` literal | Covered |
| Benchmark arms identical but representation | 18 | shared-config identity test | Covered |
| Equal tuning budget | 18 | config identity test | Covered |
| Probe: PR-AUC headline | 14, 19 | `pr_auc` test | Covered |
| Probe: minority P/R/F1 + count | 14, 19 | `minority_report` test | Covered |
| Probe reported beside trivial baseline | 14, 19, 21 | baseline-column doc test | Covered |
| Probe is a distinct model, not the primary artifact | 19 | distinct `model_name`; primary artifact raises on CFPB | Covered |
| Probe evaluated in-domain as reference performance | 19 | both evaluations asserted present | Covered |
| Named cross-domain **and** cross-target (D19) | 19, 21 | `experiment_label` test; naming doc test | Covered |
| Never presented as same-task transfer | 19, 21 | forbidden-wording tests in report and docs | Covered |
| Six framing facts carried in the output | 19 | all six present and non-empty | Covered |
| Polarity mapping stated explicitly (I7) | 19 | `polarity_mapping` fact asserted | Covered |
| Probe reuses Task 17's 311 split (I6) | 19 | boundary-equality test | Covered |
| CFPB evaluation on test period only (I6) | 19 | period identifier asserted | Covered |
| Base rate recorded per evaluation (I7) | 19 | `base_rate` present both sides | Covered |
| Two PR-AUCs never paired as before/after (I7) | 19 | report-structure test | Covered |
| Source/evaluation domain + target stated | 19 | framing-facts test | Covered |
| Target semantics differ, stated explicitly | 19 | framing-facts test | Covered |
| Labelled exploratory robustness analysis | 19 | `analysis_type` test | Covered |
| Probe naming binding in metadata and README | 19, 21 | `experiment_label` test; doc test | Covered |
| Source-vs-target quantiles per feature (C3) | 19 | quantile table asserted for all three features | Covered |
| % of target records outside source range (C3) | 19 | `pct_outside_source_range` + `pct_outside_source_iqr` | Covered |
| `text_length` flagged out-of-range | 19 | flag asserted | Covered |
| Timestamp provenance diagnostic emitted (I2, D20) | 8 | three-verdict rule tests | Covered |
| `sent_to_company_at` captured for the delta (I9) | 3, 5 | present/absent normalization tests | Covered |
| `sent_to_company_at` never a feature or target (I9) | 12 | absent from both feature specs | Covered |
| Field-delta metrics measured (D20) | 8 | quantiles, sub-minute fractions, identical-timestamp share | Covered |
| Distribution labelled secondary, cannot prove artifact (D20) | 8 | uniform histogram + healthy deltas ⇒ not strongly-suspicious | Covered |
| Verdict rule pre-specified and recorded (D20) | 8 | thresholds present in manifest | Covered |
| 311 recorded not-directly-testable (D20) | 8 | `not_directly_testable` asserted | Covered |
| Three-level classification gated on verdict (D20) | 8, 19 | all three branches tested | Covered |
| Histogram shape alone cannot downgrade result (D20) | 19 | non-diurnal + supported verdict ⇒ substantive | Covered |
| ROC-AUC secondary only | 14, 21 | doc test | Covered |
| Artifact metadata fields | 15 | schema test | Covered |
| Reproducibility controls recorded | 15, 21 | metadata + doc tests | Covered |
| Resource measured in clean env | 20 | pandas-absent assertion | Covered |
| No migration, no serving wiring | 22 | integration tests | Covered |
| Dependency split before ML deps | 1 | tier deny-list test | Covered |

**Orphan check.** Every task traces to at least one requirement. Tasks 2 (package skeletons) and 22 (integration) exist to enforce the boundary requirements rather than implement a feature; both are justified by the "no serving wiring" and "corpus never enters `Complaint`" rows.

**Gaps.** None. Every requirement row has an owning task and a proving test. The four rows previously blocked by C1 and C2 are now covered following addendum D15 and D16.

**Open question, not a gap.** Finding C3 (`text_length` incomparability) and the escalated I2 (CFPB timestamps possibly load artifacts) both concern whether Task 19's *result* will be interpretable, not whether the task is implementable. Both are now surfaced in the experiment's own output — C3 through the source-vs-target quantile diagnostic with out-of-range percentages, I2 through a `result_classification` that downgrades the figures to non-informative when the ingest verdict says the timestamps are artifacts. Neither is resolved in the plan, because neither can be resolved without the data in hand.

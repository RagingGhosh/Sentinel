# Phase 2 reproducibility guide

How to set up each environment, what Phase 2 can rerun today, and what it cannot.

This guide is written under **decision D42** in
`docs/superpowers/specs/2026-09-04-sentinel-phase-2-addendum.md`. The plan's
original Task 21 asked for a guide to rerunning everything and for measured model
metrics in the README. Those metrics do not exist yet: no model has been evaluated
on a real corpus, for the reason given in §3. D42 records that as a deviation and
defers the missing work, listed in §8. **Phase 2 is not complete**, and nothing
here should be read as saying otherwise.

The only measured figures Phase 2 has published are the inference-time resource
measurements in [`docs/phase-2-resource-measurements.md`](phase-2-resource-measurements.md).
This guide states none of them; it says how they were produced and how to produce
them again (§6).

---

## 1. What can be rerun today

- **The test suite.** Every experiment, leakage guard and metric is exercised
  against small fixtures written into temporary directories. Those fixtures test
  behaviour; the figures they produce describe the fixtures, not a model, and are
  never published as model performance.
- **The Task 20 resource measurement** (§6). It needs no corpus.

What **cannot** be rerun end to end today is anything that needs a real corpus:
ingestion from source, and therefore every experiment in §4. §3 explains why.

---

## 2. Dependency tiers

`requirements.txt` is a one-line shim that installs `requirements/base.txt`. The
real tiers live in `requirements/`, split by *when* a dependency is needed:

| Tier | File | Adds | Used for |
| --- | --- | --- | --- |
| base | `requirements/base.txt` | Django, DRF, allauth, gunicorn and their pins | the deployed application; the only file `build.sh` installs |
| dev | `requirements/dev.txt` | `-r base.txt` plus pytest, ruff, mypy and test factories | running the application's tests and static checks |
| ml | `requirements/ml.txt` | `-r base.txt` plus numpy, scikit-learn, onnxruntime, tokenizers | model inference; the only environment Task 20 measures in |
| train | `requirements/train.txt` | `-r ml.txt` plus pandas and pyarrow | corpus I/O: ingestion, and every experiment in §4 |

`ml` and `train` are separate on purpose. scikit-learn and onnxruntime become
runtime dependencies once artifacts are served; pandas and pyarrow never do.

Create a virtual environment and install the tier you need. On Windows:

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements/dev.txt                               # application + tests
.venv/Scripts/pip install -r requirements/train.txt -r requirements/dev.txt     # everything, as CI's ML job
```

On macOS or Linux, use `.venv/bin/` in place of `.venv/Scripts/`.

The full suite, including the tests marked `ml`, needs `train.txt` and `dev.txt`
together. `manage.py` needs `SECRET_KEY` in the environment; the tests supply it
through `conftest.py`.

---

## 3. Ingestion, as implemented

Plan §F defines three stages, each with a durable boundary:

```
fetch      -> data/raw/<source>/...                                     (gitignored, resumable)
normalize  -> CorpusRecord / outcome streams                            (pure, in memory)
load       -> data/corpus/<source>/v<N>/year=<YYYY>/part-NNNN.parquet   (manifest.json written last)
```

The command line is:

```bash
python -m ingest.cli --source {cfpb,nyc311} --start YYYY-MM-DD --end YYYY-MM-DD \
    [--limit N] [--corpus-root PATH]
```

`--limit` bounds the record count for development and is recorded in the manifest.
`--corpus-root` chooses where the corpus is written; a `--limit` run needs one that
does not already hold an authoritative corpus.

**No concrete fetcher exists for either source.** Fetching is an injected
boundary: no approved document specifies an endpoint, a pagination scheme, a
retry policy or a rate limit for CFPB or NYC 311, so none was invented. The command
line passes no fetcher and reads only the raw cache under `data/raw/`, relative to
the working directory — and nothing in the repository populates that cache. Run
against an empty cache, the command refuses with `EmptyWindow`, reporting that zero
cached pages were read, and writes nothing.

So **neither corpus can currently be obtained from its source**, and this guide
states **no ingestion runtime**: none has been measured. Specifying and
implementing the fetchers is deferred work (§8).

---

## 4. Running the experiments

None of the four experiments has a command-line entry point. Each is a Python
function, called from a script or an interpreter in a `train`-tier environment.
Each loads its corpus through the manifest-verified loader, so **each needs a real
corpus under `corpus_root`, which cannot currently be produced** (§3). None has
therefore been run on real data, and none of their results is published.

**Task 16 — CFPB triage.**
`ml.training.experiments.triage.run_experiment(*, corpus_root, artifact_root, seed, ...)`
reads the CFPB corpus and writes the `cfpb_triage_tfidf` artifact, version `v1`.
`seed` is required and recorded in the artifact's `seeds` field; decision D36 fixes
no particular value.

**Task 17 — NYC 311 SLA risk.**
`ml.training.experiments.risk.run_experiment(*, corpus_root, artifact_root, seed, ...)`
reads the NYC 311 corpus and its outcome sidecar and writes the `nyc311_sla_risk`
artifact, version `v1`. Decision D37 fixes the estimator's `random_state` at 17,
which is the value to pass as `seed` to reproduce it.

**Task 18 — duplicate retrieval.**
`ml.training.experiments.dedup.run_benchmark(*, corpus_root=None, config=BENCHMARK_CONFIG)`
reads the CFPB corpus and compares the TF-IDF and MiniLM representations under one
frozen configuration, whose seed is 18. It needs the MiniLM assets (§6.2). It
**returns** a `BenchmarkReport` object; it writes no file and no artifact, because
decision D38 writes no embedder artifact in Phase 2. Its results, when they exist,
are a synthetic retrieval exercise, not real-world duplicate-detection accuracy.

**Task 19 — the reduced-feature cross-domain cross-target robustness probe.**
`ml.training.experiments.robustness_probe.run_probe(*, corpus_root, artifact_root, report_path)`
reads both corpora and both outcome sidecars, writes the `xdomain_xtarget_probe`
artifact, version `xdomain_xtarget_probe_v1`, and writes its report to the path the
caller supplies. Its seed is 17. Its report classifies its own result from the CFPB
timestamp diagnostic, and decision D19's binding prohibition on how that result may
be described applies wherever it is reported.

Artifacts are written to `<artifact_root>/<source>/<model_name>/<model_version>/`.
Each experiment records the commit it ran from by calling `git rev-parse HEAD`,
unless a `git_sha` argument is passed.

---

## 5. Reading artifact metadata

Every artifact is one version directory holding `model.joblib` and
`metadata.json`. **`metadata.json` is written last**, through a temporary file, so
a directory without it is not an artifact and a failed write leaves nothing
loadable. The directory's name must equal `model_version`, and its parent's name
`model_name`.

Decision D35 fixes a **closed** metadata schema:

- **Sixteen required fields:** `model_name`, `model_version`, `trained_at`,
  `git_sha`, `corpus_id`, `corpus_schema_version`, `source_window`, `split`,
  `feature_spec`, `feature_spec_version`, `label_roster`, `thresholds`, `metrics`,
  `warmup_row_count`, `seeds`, `dependency_versions`.
- **Three of them may be `null`,** meaning "not applicable to this kind of
  artifact": `label_roster`, `thresholds`, `warmup_row_count`.
- **Four optional fields,** absent or valid: `embedding_dimension`,
  `embedding_model_id`, `embedding_model_sha256`, `experiment_label`.
- **Any other key is refused**, and so is anything strict JSON cannot represent,
  such as a `NaN`.

`corpus_id` ties an artifact to the exact bytes of the corpus it was trained on.
`feature_spec` is authoritative: the artifact builds exactly those features, in
that order, and nothing else.

Load an artifact with `ml.training.artifacts.load_artifact(path)`. It refuses a
directory whose name disagrees with the metadata, invalid metadata, and — before
unpickling anything — a `feature_spec` naming a feature this environment cannot
produce. **Load only artifacts you trust:** `model.joblib` is a pickle, and loading
it runs code it contains.

---

## 6. Rerunning the Task 20 resource measurement

The harness is `ml.training.measure`. Its authoritative published results are in
[`docs/phase-2-resource-measurements.md`](phase-2-resource-measurements.md), which
names the run and the report key behind each figure. What follows is how to
produce such a run; it restates no result.

### 6.1 A clean `ml`-tier environment

Build the environment from `requirements/ml.txt` **alone**:

```bash
python -m venv <somewhere outside the repository>
<that venv>/Scripts/pip install -r requirements/ml.txt
```

The harness proves the environment rather than trusting it: if `pandas` or
`pyarrow` is importable, it **refuses to write a report**. It therefore refuses in
any `train`-tier environment, including a normal development one. That is the rule
working, not a fault.

### 6.2 The MiniLM assets

The embedder's files are an external prerequisite, pinned by decision D38. They are
**not committed and not downloaded by the repository**: obtain them from the
checkpoint `sentence-transformers/all-MiniLM-L6-v2` at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` — its `onnx/model.onnx` and its
`tokenizer.json`, both from that revision. Every load verifies them against the
pinned digests:

```
model.onnx      6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452
tokenizer.json  be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037
```

Place them in `ml/artifacts/embedders/all_minilm_l6_v2/v1/`, or point
**`SENTINEL_MINILM_DIR`** at the directory that holds them. A missing file raises
`ModelAssetUnavailable` and a digest mismatch raises `ModelAssetMismatch`; no
measurement is taken of a model that did not load.

### 6.3 One run

From the repository root, in the clean environment:

```python
from pathlib import Path
from ml.training.measure import run_measurements

run_measurements(
    artifact_root=Path("<an artifact root to measure>"),
    report_path=Path("<where to write the report>"),
)
```

Both arguments are required and keyword-only, and there is no default report path:
nothing is written inside the repository unless you choose to. The report is
written whole or not at all. An artifact root holding no experiment artifacts is
valid — the three experiment classes are then reported absent, never as zero bytes.

### 6.4 The fresh-process rule

**Every memory figure intended for publication must come from a fresh interpreter
that has performed no earlier measurement run** (decision D41, second addendum).
Both backends report the process's *peak* resident set, a high-water mark for its
whole lifetime, so a second run in one interpreter inherits whatever the first
reached. Run each measurement you mean to publish as its own process.

Even from a fresh process, a memory figure is the process peak observed *through*
the measured stage — the interpreter, the libraries and everything loaded so far —
and never the incremental cost of that stage alone.

---

## 7. A Phase 3 prerequisite for the risk model

Plan finding I1 requires this to be stated explicitly.

The Task 17 risk model uses two target-derived features,
`category_mean_resolution_hours` and `category_breach_rate`. Both are computed from
NYC 311 resolution history and looked up by the record's category label, which for
NYC 311 is its complaint type.

At Phase 3 serving time, a Sentinel complaint in another domain has **no meaningful
source for either value**. Sentinel has no accumulated resolution history of its
own yet, and its categories are not NYC 311 complaint types. Mechanically, the
Phase 2 lookup falls back to the NYC 311 training period's global mean and global
breach rate for any category it has not seen — values that describe NYC 311, not
the domain being served.

**So the Phase 2 risk model may be trainable but not servable.** Before it can be
served in Phase 3, those two aggregates need a source for the serving domain. The
artifact's metadata does not currently record that the aggregates are NYC 311
derived, so this guide is where the prerequisite is written down.

---

## 8. Deferred under D42

These remain required. They are not delivered by this guide or by the README, and
Phase 2 is not complete until they are:

- a **concrete CFPB and NYC 311 fetcher** — endpoint, pagination, retry and rate
  limit — specified and implemented;
- **real ingestion** of both corpora, and the ingestion runtimes this guide would
  then state;
- **real-corpus triage evaluation** (Task 16);
- **real-corpus risk evaluation** (Task 17);
- **real-corpus duplicate-retrieval evaluation** (Task 18);
- **real-corpus evaluation of the reduced-feature cross-domain cross-target
  robustness probe** (Task 19);
- the **MiniLM ship-or-cut decision**, which rests on the Task 18 benchmark having
  run on a real corpus;
- publication of the reduced-feature cross-domain cross-target robustness probe's
  real-corpus table with its `result_classification`;
- the risk model's **README verdict against its majority baseline**;
- the **cross-platform metric tolerance**, until a second environment is actually
  measured. The only published figures so far come from one Windows machine.

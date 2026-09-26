# Task 20 — Inference-environment resource measurement: contract

**Status: FROZEN.** The six decisions this document previously carried as open
are recorded in §13 as **D40** and folded into the clauses they govern.
Implementation may begin. Nothing here is invented: every clause is either
reconstructed from plan §S, plan §U's Task 20 entry, plan §E and addendum §10,
or is one of the six rulings §13 records.

The rulings belong in the addendum's decision log as **D40**, following the
pattern D36–D39 set; this document is the working contract D40 ratifies.

---

## 1. What Task 20 is, and what it is not

A measurement harness. It produces **figures about cost**, in an environment
built from `requirements/ml.txt` alone, and writes them to a caller-supplied
path. It trains nothing, fits nothing, scores nothing and publishes no model.

Its reason to exist is plan §E's rationale for keeping `ml.txt` separate from
`train.txt`: scikit-learn and onnxruntime become *runtime* dependencies in
Phase 3 when artifacts are served, while pandas and pyarrow never do. A memory
figure measured with a dataframe library resident would describe an environment
the web process will never run in. So the measurement environment is the point,
not an implementation detail.

**It is not a benchmark of quality.** Task 18 owns retrieval quality; this task
owns cost. No recall figure, no metric, no baseline comparison and no
ship-or-cut decision appears in its output.

**It is not a serving path.** Addendum §10 keeps artifact wiring, `Prediction`
rows, `Complaint.embedding` and migrations in Phase 3. Nothing here is imported
by `ml/registry.py`, `ml/base.py` or `ml/null.py`, and the existing static
boundary test in `tests/test_import_boundaries.py` keeps it that way unchanged.

---

## 2. The five measurements — FROZEN

Exactly plan §S's quantities, no more and no fewer.

| | Measurement | Unit |
|---|---|---|
| **A** | MiniLM peak process RSS during model load | bytes |
| **B** | MiniLM embedding throughput at batch sizes **1, 8 and 32** | records/sec |
| **C** | Artifact sizes on disk | bytes, per artifact |
| **D** | Index **build time** and **peak process RSS** for **10,000** and **50,000** vectors | seconds, bytes |
| **E** | Single-query latency | seconds |

**D is four figures, not two:** build time at 10k, peak memory at 10k, build
time at 50k, peak memory at 50k. The task is **not** narrowed to drop them
(D40.1); the coupling that made them unreachable is resolved by §7's extraction
instead.

**B measures embedding, D and E measure the index.** They are reported
separately and never combined, because conflating them would make an index
figure depend on tokenizer and ONNX cost that has nothing to do with index
construction.

---

## 3. The measurement environment — FROZEN

- The environment is created and run from **`requirements/ml.txt` only**.
- **`pandas` must not be importable.**
- **`pyarrow` must not be importable.**
- **If either is importable, the harness refuses to write the report.** This is
  the proof that the environment is inference-only, and it is a refusal rather
  than a warning: a report written from a polluted environment would understate
  the Phase 3 budget and there would be nothing in the file to say so.
- Every figure carries its environment provenance: **Python version, CPU model,
  core count, and the versions of every library the figure depended on.**

The check is on importability, not on a requirements file: an environment that
happens to have pyarrow installed is refused whatever its `pip freeze` says.

---

## 4. Peak RSS — FROZEN

**No new dependency.** `psutil` is not added, and neither is anything else:
plan §E's four tiers stand, and `ml.txt` becomes a Phase 3 *runtime* tier, so a
measurement-only package must not enter it (D40.2).

The measured quantity is **peak process RSS**, obtained from platform-specific
standard-library mechanisms:

| Platform | Mechanism |
|---|---|
| Windows | `ctypes` calling `GetProcessMemoryInfo`, reading `PeakWorkingSetSize` |
| POSIX | `resource.getrusage(RUSAGE_SELF).ru_maxrss` |

**The report names the backend that produced each memory figure.** The two are
not interchangeable — `ru_maxrss` is kilobytes on Linux and bytes on macOS, and
`PeakWorkingSetSize` is a Windows working set rather than a Unix resident set —
so a figure without its backend cannot be compared with another machine's.

`tracemalloc` is explicitly **not** used for these figures. It measures Python
heap allocation, and the quantity of interest is dominated by the ONNX runtime's
native allocation, which `tracemalloc` cannot see.

**An unsupported platform is a refusal, not a zero.** A figure of zero bytes
would read as "no memory used".

### 4.1 One fresh process per publishable memory figure

**Any memory figure intended for publication must be produced in a fresh
process that has performed no earlier measurement run.**

Both backends report a process **peak** — a high-water mark for the lifetime of
the process — so a second run inside one interpreter inherits whatever the first
run reached. **Repeated runs in a single interpreter are therefore not
independent memory measurements**, and the later ones are not publishable as
such. This was observed, not anticipated: in the clean-environment validation,
a second run reported a MiniLM *load* peak identical to the first run's
*50,000-vector index* peak, because the process had already been there. The
first run's figures were sound; the second run's memory figures were not.

Time and throughput figures do not have this property and may be repeated
freely in one process; the rule is about memory alone.

**Even in a fresh process, a memory figure is the process peak observed through
the measured stage — not an incremental allocation attributable to that stage
alone.** A load figure includes the interpreter, the imported libraries and the
ONNX session; an index figure includes everything the process had already
reached. **The published documentation must not imply otherwise**, and must not
describe such a figure as the cost of the stage by itself.

This clause is a **measurement procedure**, and nothing more. It changes neither
backend, nor the quantity measured, nor the standard-library-only rule, nor the
harness, nor the five categories of §2.

---

## 5. Artifact sizes, and what absence means — FROZEN

- **The harness receives a caller-supplied artifact root.** No prior training
  run is required as hidden setup, and the harness never triggers one (D40.3).
- For each expected artifact class that **exists**: its actual on-disk byte
  size, measured during the recorded run.
- For each expected artifact class that **does not exist**: recorded as
  **absent/unavailable**.
  - **No size is invented.**
  - **Absence is never reported as zero bytes.**
- The report **distinguishes measured sizes from absent artifacts** structurally,
  not only in prose, so a reader cannot mistake one for the other. The two
  serialized shapes are frozen in §15.

The expected classes are those Phase 2 produces: the Task 16 triage artifact,
the Task 17 risk artifact and the Task 19 probe artifact, plus the external
MiniLM assets the embedder loads.

**Phase 2 writes no embedder *artifact*** — D38 decided that, and D18's
artifact-metadata clause is deferred to the first serving embedder artifact,
which Phase 3 creates. That is a statement about artifacts, **not** about the
assets: the pinned `model.onnx` and `tokenizer.json` the embedder loads are real
files with a real on-disk size, and their cost is exactly the kind of figure §S
asks for. So `minilm_assets` is an **external model-asset class** rather than a
run-produced experiment artifact, and §15.8 fixes where it resolves. The three
experiment classes are the ones expected to be absent in a fresh clone.

Artifacts and assets are git-ignored and are produced by runs, so absence is the
normal state of a fresh clone. The harness must be useful in that state.

---

## 6. Vector benchmark input — FROZEN

For the 10k and 50k index measurements (D40.4):

- **Deterministic synthetic `float32` vectors.** Seeded, so a rerun on the same
  machine produces the same input.
- **The dimension comes from the actual MiniLM embedder** —
  `load_minilm().embedding_dimension`, which `ml/embedders/minilm.py` sets from
  an observed forward pass rather than a checkpoint-family stereotype (D18).
  **The dimension is never hard-coded**, and in particular the literal `384`
  appears nowhere, exactly as D18 requires of `ml/embedders/`.
- **All benchmark vectors are generated before timing starts.**
- **Index timing excludes vector generation and excludes embedding generation.**
  What is timed is Sentinel's index construction and query, and nothing else.
- **Deterministic synthetic `RecordRef` identities**, as the index API requires.
  `build_index` refuses a repeated reference, so the generated identities must be
  distinct by construction.

Synthetic vectors are used because the mandated environment cannot load a
corpus — `pyarrow` is absent by design — and because the quantity being measured
is index cost as a function of population size, which is a property of the index
rather than of any particular corpus.

---

## 7. The index extraction — the one authorised change to experiment code — FROZEN

Plan §U's Task 20 entry says **"Must not change: experiment code."** D40.1
records the single narrow exception, and its scope is fixed here.

**The problem, stated exactly.** Task 20 must measure the real Sentinel index,
and the index primitives live in `ml/training/experiments/dedup.py`, which
imports `ingest.manifest` at module level; `ingest.manifest` imports
`ingest.storage`, which imports `pyarrow`. So importing the index primitives
requires `pyarrow`, and the environment §3 mandates is precisely the one where
`pyarrow` is absent — and where the harness is required to refuse to run if it
is present. Without an extraction the four §2.D figures and §2.E are
unreachable, not merely inconvenient.

**The resolution.** A narrow, behaviour-preserving extraction:

- The pyarrow-free retrieval and index primitives Task 20 needs are extracted
  into **`ml/training/index.py`**, importable with neither `pandas` nor
  `pyarrow` installed.
- **Exactly these four symbols move:** `cosine_similarity`, `RetrievalIndex`,
  `build_index`, and the `_require_record_refs` helper `build_index` calls.
  Their bodies, signatures, defaults, error types and error messages are
  unchanged.
- `recall_at_k` and `random_ranking_baseline` **stay in `dedup.py`**: Task 20
  measures no metric, so they are not "required by Task 20" and moving them
  would widen the exception past its justification.
- **`dedup.py` consumes the extracted primitives** by importing them, and
  **re-exports them** so that `module.build_index`, `module.RetrievalIndex` and
  `module.cosine_similarity` keep resolving on the dedup module. Task 18's tests
  reach every symbol as an attribute of the dynamically imported dedup module,
  so they continue to pass **untouched**.
- **No index implementation is duplicated inside `measure.py`**, or anywhere
  else. There is one definition, in one place, and both callers import it.
- `EmbeddingDimensionMismatch` is **not** moved. It stays in
  `ml/embedders/minilm.py`.
- **Exception-identity preservation.** The four extracted symbols retain their
  observable signatures, defaults, validation, exception types, exception
  messages, ordering and numerical behaviour. Within that, `RetrievalIndex`
  **may resolve `EmbeddingDimensionMismatch` through the currently loaded
  `ml.embedders.minilm` module at raise time**, rather than capturing the class
  object when `ml/training/index.py` is imported.

  This is **required**, not preferred. `ml.embedders.minilm` is reloadable and
  is in fact reloaded by its own test suite; `importlib.reload` rebinds
  `EmbeddingDimensionMismatch` to a new class object, so a class captured at
  index-module import time becomes stale and an `except` clause reading the
  name off the module no longer matches what `rank` raises. Before the
  extraction the question could not arise, because `rank` and the benchmark
  resolved the name from one namespace.

  **This changes implementation binding, not observable behaviour.** The
  exception type and its message are exactly Task 18's. The purpose is
  compatibility preservation, and the alternative would break Task 18's tests.
  **No other extracted body may receive an analogous change without a separate
  contract decision**, and the deviation is confined to that one raise site.
- The location is `ml/training/`, not `ml/`, so the existing serving boundary is
  untouched: `tests/test_import_boundaries.py` already forbids `ml/base.py`,
  `ml/null.py` and `ml/registry.py` from importing `ml.training`, and that test
  needs no change. Whether the index eventually belongs beside a serving path is
  **Phase 3's question**, and this contract does not answer it.
- **Regression coverage proves Task 18 behaviour is unchanged**: the existing
  Task 18 suite passes with no edit, and a new test asserts the extracted module
  imports with `pandas` and `pyarrow` blocked — the property the extraction
  exists for, which no existing test can express because the current module
  cannot satisfy it.

**The extraction exists solely to separate inference and index mechanics from
corpus and Parquet I/O, so that the mandated clean measurement environment can
measure the actual Sentinel index rather than a copy of it.** It changes no
benchmark contract, no metric semantics, no report semantics, and no observable
behaviour of Task 18. It is not a refactor undertaken for its own sake, and it
authorises no other movement of experiment code.

---

## 8. The report — FROZEN

- **The report path is caller-supplied.** There is **no default in-repository
  report path**, following the convention D38 established and D40.5 re-affirms.
- `docs/phase-2-resource-measurements.md` is the document plan §U names, and it
  is written from a recorded run's output. It is not the harness's default
  destination.
- **That document may publish only figures produced by fresh-process runs in the
  clean environment** (§3, §4.1). A memory figure from a second run inside an
  interpreter that had already measured is evidence, not a publishable
  measurement, and none may be transcribed.
- **Every numeric figure carries its environment provenance** (§3).
- **Every figure is produced by an actual measurement run.** No hand-entered
  value appears in the harness's output, and no figure is transcribed into the
  document that did not come from a run.
- The report records which measurements were taken and which were recorded as
  absent (§5), so a reader can tell an unavailable figure from an omitted one.

---

## 9. Failure behaviour — FROZEN

- `pandas` or `pyarrow` importable → **refusal**, and no report is written (§3).
- An unsupported platform for peak RSS → **refusal**, never a zero (§4).
- A missing artifact → **recorded as absent**, never a size (§5).
- Missing MiniLM assets → the embedder's own `ModelAssetUnavailable` /
  `ModelAssetMismatch`, uncaught. A measurement of a model that was not loaded
  describes nothing.
- An index queried at the wrong width → `EmbeddingDimensionMismatch`, unchanged
  by §7's extraction.
- Nothing is written partially: the report is serialized whole to the caller's
  path or not at all, following D27's pattern as Tasks 18 and 19 do.

---

## 10. The plan's §P cross-reference — already satisfied, no Task 20 work

Plan §P states: *"This is the compatibility guard from addendum §3.3, and Task
20 tests it against a deliberately mismatched artifact."*

**That work is already done, by Task 15.** `tests/ml/training/test_artifacts.py`
carries the deliberately mismatched artifact fixture, whose own docstring names
it *"the deliberately mismatched artifact fixture (plan Task 15 acceptance)"*,
and asserts `FeatureSpecMismatch` is raised naming the unproducible feature
before the model is unpickled.

**Task 20 therefore adds no compatibility test and writes no code for this
sentence** (D40.6). The reference is a stale forward-pointer from an earlier
draft of the plan, resolved here so that a later reader does not implement it
twice. No plan file is edited to say so; this clause is the record.

---

## 11. Files

**Expected to change**

| File | Change |
|---|---|
| `ml/training/measure.py` | create — the harness |
| `docs/phase-2-resource-measurements.md` | create — the recorded figures |
| `ml/training/index.py` | create — §7's extraction, four symbols moved |
| `ml/training/experiments/dedup.py` | import and re-export the extracted four; **no other change** |
| `tests/ml/training/test_measure.py` | create — the harness's tests |
| `docs/superpowers/specs/2026-09-04-sentinel-phase-2-addendum.md` | D40 recording the six rulings |

**Forbidden to change**

`ml/training/experiments/{triage,risk,robustness_probe}.py` and their tests;
`tests/ml/training/test_dedup_benchmark.py`; `ml/embedders/*` and their tests;
`ml/training/{metrics,artifacts,features,labels,thresholds,aggregates,splits}.py`;
`ml/base.py`, `ml/registry.py`, `ml/null.py` and every serving path;
`ingest/*`; `tests/test_import_boundaries.py`; `requirements/`; `pyproject.toml`;
`.github/workflows/ci.yml`; `Sentinel_Complete_Blueprint.md`.

`dedup.py`'s benchmark contract, metric semantics and report semantics are
unchanged, and the only edit it receives is the import and re-export §7 fixes.

---

## 12. Dependencies

**None added.** `numpy==2.5.2`, `scikit-learn==1.9.0`, `onnxruntime==1.29.0` and
`tokenizers==0.23.2` are already pinned in `requirements/ml.txt`, and the peak-RSS
backends are standard library. No new package, no tier change, and no CI change.

---

## 13. The frozen decisions — D40

**D40.1 — the five measurement categories are retained, and the index coupling
is resolved by extraction rather than by narrowing.** All five §S categories
stand, including index build time and peak memory at 10k and 50k and
single-query latency. Because the index primitives are coupled at module-import
time to the ingest/Parquet stack and therefore cannot be imported from the
required `ml.txt`-only environment, the pyarrow-free retrieval and index
primitives Task 20 needs are extracted into a dedicated importable module;
Task 18's public benchmark behaviour is unchanged; `dedup.py` consumes the
extracted primitives rather than duplicating them; no index implementation is
duplicated inside `measure.py`; the Task 18 benchmark contract, metric semantics
and report semantics do not change; and regression coverage proves Task 18
behaviour remains unchanged. **This is the only permitted exception to plan
§U's "must not change experiment code".** The extraction exists solely to
separate inference and index mechanics from corpus and Parquet I/O so that the
mandated clean measurement environment can measure the actual Sentinel index.

**D40.1 addendum — exception identity under a reloadable module.** The four
extracted symbols keep their observable signatures, defaults, validation,
exception types, messages, ordering and numerical behaviour. `RetrievalIndex`
may resolve `EmbeddingDimensionMismatch` through the currently loaded
`ml.embedders.minilm` module at raise time rather than capturing the class
object at index-module import time, because that module is reloadable and a
captured exception class becomes stale after `importlib.reload()`. This changes
implementation binding, not observable exception type or message behaviour; its
purpose is compatibility preservation rather than refactoring; and no other
extracted body may receive an analogous change without a separate contract
decision.

**D40.2 — peak RSS from the standard library.** No `psutil` and no other new
runtime dependency. Peak process RSS is measured with platform-specific
standard-library mechanisms: `ctypes` and `GetProcessMemoryInfo` on Windows,
`resource.getrusage` on POSIX. The report identifies the backend used. The
reported quantity is peak process RSS.

**D40.3 — artifact size and absence.** No prior training run is required as
hidden setup. The harness receives a caller-supplied artifact root. Each
expected artifact class that exists is measured for its actual on-disk byte size
from the recorded run; each that does not exist is recorded as absent or
unavailable, with no invented size and with absence never treated as zero bytes.
The report distinguishes measured sizes from absent artifacts. No figure appears
unless the measurement run generated it.

**D40.4 — vector benchmark input.** The 10k and 50k index measurements use
deterministic synthetic `float32` vectors whose dimension is obtained from the
actual MiniLM embedder rather than hard-coded. All benchmark vectors are
generated before timing; index timing excludes vector generation and embedding
generation; deterministic synthetic `RecordRef` identities are used as the index
API requires. The purpose is to measure Sentinel index construction and query
cost rather than conflate it with embedding throughput.

**D40.5 — spec authority and the report rule.** Task 20's requirements are
frozen in this contract, which D40 ratifies, rather than resting on the plan
alone. The report path is caller-supplied with no default in-repository path;
every numeric figure carries its environment provenance; figures are produced by
an actual measurement run; no hand-entered measured value is published. The
environment rule is as §3 states, including the refusal to write a report when
`pandas` or `pyarrow` is importable.

**D40.6 — the stale compatibility cross-reference.** Plan §P's sentence assigning
the artifact compatibility-guard test to Task 20 is already satisfied by Task
15's deliberately mismatched artifact test. Task 20 adds no duplicate
compatibility test and performs no implementation or test work for it.

---

## 14. Acceptance criteria

1. The harness runs to completion in an environment installed from
   `requirements/ml.txt` only, and produces all five measurement categories.
2. It refuses to write a report when `pandas` or `pyarrow` is importable, and
   the refusal is asserted by a test.
3. Every numeric figure in the output carries Python version, CPU model, core
   count and the relevant library versions.
4. Memory figures name the backend that produced them.
5. The index figures cover 10,000 **and** 50,000 vectors, each with both build
   time and peak memory, plus single-query latency.
6. The vector dimension in the output equals the dimension the MiniLM embedder
   reports, and no dimension literal appears in the harness.
7. Index timing excludes vector and embedding generation, asserted by a test
   rather than by comment.
8. Artifact sizes are measured for present artifacts and recorded as absent for
   missing ones; a test proves absence is not reported as zero.
9. The report is written only to a caller-supplied path, and the harness's
   signature carries no default path.
10. `ml/training/index.py` imports with `pandas` and `pyarrow` blocked, asserted
    by a test.
11. Task 18's test suite passes **byte-identical**, and `dedup.py`'s only change
    is the import and re-export of the extracted primitives.
12. The width error `RetrievalIndex.rank` raises is the class the currently
    loaded `ml.embedders.minilm` exposes, proven by a test that reloads that
    module first; the type and message are Task 18's unchanged.
13. No dependency, `pyproject.toml` or CI change.
14. Tasks 16, 17 and 19 remain byte-identical, as do the serving path and the
    existing import-boundary test.
15. The module exposes exactly the identifiers §15 freezes, spelled as §15
    spells them, and adds no further public API.
16. A `Figure` cannot be constructed without its environment provenance, and a
    figure whose provenance is incomplete does not serialize (§15).
17. The five `measure_*` stage functions exist as independently patchable seams,
    so a failure injected at any one of them leaves no report (§13's
    whole-or-nothing rule, §15's seam classification).
18. `ARTIFACT_CLASSES` is a declared tuple, and the report's artifact order and
    membership equal it regardless of what the filesystem holds or in what order
    it enumerates (§15).
19. An existing artifact serializes as `{"value": <int>, "unit": "bytes"}` and an
    absent one as `{"status": "absent"}`, with no `value` key — never zero, never
    null, never a sentinel number (§15).

---

## 15. The frozen API surface — FROZEN

This section was added after the RED phase, which is why it sits last: §§1–14
fixed the file, the five measurements and their semantics but **no identifiers**,
and `tests/ml/training/test_measure.py` could not express the frozen behaviour
without naming things. The names below are therefore **authoritative for Task
20**, and the RED suite is written against them: renaming one means amending this
section and those tests together, never the code alone.

### 15.1 Constants and configuration

| Identifier | Meaning | Frozen value |
|---|---|---|
| `BATCH_SIZES` | §2.B's batch sizes, in order | `(1, 8, 32)` |
| `INDEX_POPULATIONS` | §2.D's populations, in order | `(10_000, 50_000)` |
| `FORBIDDEN_PACKAGES` | §3's clean-environment proof | `("pandas", "pyarrow")` |
| `SEED` | the deterministic seed for §6's synthetic input | `20`, the task number, as Task 17 used 17 and Task 18 used 18 |
| `WINDOWS_RSS_BACKEND` | §4's Windows backend name, recorded with each memory figure | a name containing `GetProcessMemoryInfo` |
| `POSIX_RSS_BACKEND` | §4's POSIX backend name | a name containing `getrusage` |
| `ARTIFACT_CLASSES` | §5's expected classes, in report order | a declared tuple (§15.5) |

**These are the measurement protocol, not runtime knobs.** Nothing exposes them
as parameters, environment variables or command-line options. A production run
always uses the frozen values.

The RED suite patches `INDEX_POPULATIONS` down to two small populations, and
patches `BATCH_SIZES` and `FORBIDDEN_PACKAGES` in the few tests that need to, so
that a unit test never waits for a real 50,000-vector build. That is a test
affordance and **not** a supported production configuration: patching a module
constant is available to any test in Python, and choosing it over a production
parameter is precisely how the protocol stays frozen for real runs.

### 15.2 The figure

```
Figure(value, unit, environment)
```

`environment` is **required**: see §15.3.

### 15.3 Functions

| Identifier | Role |
|---|---|
| `run_measurements(*, artifact_root, report_path) -> ResourceReport` | **the entry point** |
| `environment()` | the provenance every figure carries (§3) |
| `peak_rss_bytes()` | §4's peak process RSS, in bytes |
| `rss_backend()` | which of the two §4 backends this platform uses |
| `require_clean_environment()` | §3's refusal |
| `synthetic_vectors(count, dimension)` | §6's deterministic `float32` vectors |
| `synthetic_refs(count)` | §6's deterministic, distinct `RecordRef` identities |
| `load_embedder()` | the pinned MiniLM loader, whose observed dimension §6 uses |
| `measure_minilm_load()` | stage seam — §2.A |
| `measure_embedding_throughput(embedder)` | stage seam — §2.B |
| `measure_artifact_sizes(artifact_root)` | stage seam — §2.C |
| `measure_index_build(dimension)` | stage seam — §2.D |
| `measure_query_latency(index)` | stage seam — §2.E |

**Public versus internal.**

- **`run_measurements` is the harness's entry point**, and the only function a
  caller outside this module is expected to use. Both its arguments are
  keyword-only and neither has a default (§5, §8).
- **`Figure` is the required provenance-bearing figure representation.** Every
  numeric figure the report publishes is one.
- **The five `measure_*` functions are internal stage seams.** They exist so that
  §13's whole-or-nothing failure contract can be exercised: a test injects a
  failure at exactly one stage and asserts no report survives. They are **not**
  intended as stable external APIs, and nothing outside this module and its tests
  should call them.
- The remaining functions are the harness's own mechanics, named because the RED
  suite asserts on them directly — the RSS backend, the synthetic input's
  determinism, the environment guard's refusal.

**No additional public API is invented.** A harness that needed a further public
name would be exceeding this contract, not extending it.

### 15.4 Provenance is not optional

**A `Figure` cannot exist without its environment provenance.** `environment` is
a required constructor argument, so an unprovenanced figure is a construction
error rather than a serialization-time omission — which is what makes §8's "every
numeric figure carries its environment provenance" checkable rather than
aspirational.

The environment attached to every figure records:

- the **CPU model**;
- the **CPU core count**;
- the **Python version**;
- the **versions of the libraries the figure depended on**;
- and, for a memory figure, the **RSS backend** that produced it (§4).

**A figure whose provenance is missing or incomplete does not serialize
successfully.** A non-finite value does not either: a `NaN` or an infinity
describes nothing.

### 15.5 The two artifact shapes

An artifact that **exists** serializes as a measurement:

```json
{"value": 2098, "unit": "bytes"}
```

An artifact that **does not exist** serializes as a status, carrying **no**
`value` key:

```json
{"status": "absent"}
```

Absence is **never** represented as zero bytes, as a null value, or as an
ordinary measurement holding a sentinel number. The distinction **survives
serialization**, so a reader of the persisted report — not only a caller holding
the in-memory object — can tell an unavailable figure from a measured one.

The converse binds equally: a genuinely empty artifact directory measures **zero
bytes** and is reported as a measurement, not as absent. Zero is a fact; absence
is the lack of one.

### 15.6 Why `ARTIFACT_CLASSES` is declared rather than discovered

A filesystem scan cannot report a class that is missing, and §5 requires exactly
that — absence is a first-class result, and in a fresh clone it is the *normal*
result, since artifacts are git-ignored and produced by runs. A scan would also
let directory enumeration order decide the report's order, which §14's
reproducibility rule forbids.

So `ARTIFACT_CLASSES` is **a declared, deterministic tuple and part of the
measurement protocol**, not a list inferred at runtime. The report's artifact
membership and order equal that tuple, whatever the filesystem holds.

### 15.7 The environment guard is not relaxed for convenience

§3's rule stands exactly as written: `pandas` importable refuses the report,
`pyarrow` importable refuses it, either one refuses it, and only an environment
where **both are genuinely unavailable** permits the positive path.

The development environment for this repository installs the training tier, so
the harness **correctly refuses there**. That is the rule working, not a problem
with the rule, and it is not weakened to make local runs convenient: the RED suite
tests the refusals in-process against that real environment, neutralises the guard
for the tests whose subject is something else, and proves the positive path in a
subprocess where both packages are genuinely blocked.

### 15.8 Why `minilm_assets` resolves outside the artifact root

Three of `ARTIFACT_CLASSES` are experiment artifacts and resolve **under the
caller-supplied `artifact_root`**, which is where Tasks 16, 17 and 19 write them.
`minilm_assets` does not, and that is deliberate:

- It is an **external model-asset class, not a run-produced experiment artifact.**
  Nothing in Phase 2 writes it; the embedder *reads* it, from wherever it lives.
  Its location is the embedder's own — `SENTINEL_MINILM_DIR` when configured, the
  packaged default otherwise — and an `artifact_root` a caller chose for
  experiment output has no authority over it.
- **It is therefore resolved through the MiniLM asset directory**, not through
  `artifact_root`. Requiring it under `artifact_root` would mean either copying a
  90MB binary into a directory that exists for experiment output, or reporting the
  class absent on a machine where the assets are demonstrably present — measuring
  nothing while a real cost sits on disk.
- **Its measured size is still a real on-disk byte measurement**, taken during the
  recorded run by the same traversal and reported in the same shape as any other
  measured class.
- **If the asset directory is unavailable, the class follows the ordinary
  absence/failure semantics of §5 and §9.** It is recorded absent, or the run
  fails, exactly as the rest of the contract says. **No value is invented**, and
  absence is still never zero.

`ARTIFACT_CLASSES` does not change, and the assets are not moved into
`artifact_root`. This clause records where the one external class resolves; it
alters no other rule.

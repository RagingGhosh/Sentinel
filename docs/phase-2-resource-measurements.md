# Phase 2 resource measurements

Inference-time resource cost, measured by `ml/training/measure.py` (Task 20).

**Every number below is copied verbatim from a recorded measurement run.** None is
rounded, averaged, converted, estimated or typed by hand. Each row names the run
that produced it and the key path at which that run's JSON report holds it, so any
figure can be checked against its source without trusting this document.

The measurement contract is
`docs/superpowers/specs/2026-09-25-task-20-resource-measurement-contract.md`;
the decisions behind it are D40 and D41 in the Phase 2 addendum.

---

## 1. Measurement environment

The environment was built from **`requirements/ml.txt` only** — the inference
tier. The training tier was not installed, and the harness refuses to write a
report at all unless both `pandas` and `pyarrow` are genuinely unimportable, which
is how the environment proves itself rather than being asserted to be clean.

| Property | Value | Source | Key path |
|---|---|---|---|
| Python | 3.13.3 | A, B, C | `environment.python_version` |
| Platform | Windows-11-10.0.26200-SP0 | A, B, C | `environment.platform` |
| CPU model | Intel64 Family 6 Model 154 Stepping 4, GenuineIntel | A, B, C | `environment.cpu_model` |
| CPU cores | 12 | A, B, C | `environment.cpu_cores` |
| numpy | 2.5.2 | A, B, C | `environment.library_versions.numpy` |
| scikit-learn | 1.9.0 | A, B, C | `environment.library_versions.scikit-learn` |
| onnxruntime | 1.29.0 | A, B, C | `environment.library_versions.onnxruntime` |
| tokenizers | 0.23.2 | A, B, C | `environment.library_versions.tokenizers` |
| scipy | 1.18.1 | A, B, C | `environment.library_versions.scipy` |
| joblib | 1.6.0 | A, B, C | `environment.library_versions.joblib` |
| pandas | not importable | A, B, C | proven by the run completing |
| pyarrow | not importable | A, B, C | proven by the run completing |
| Peak-RSS backend | windows_GetProcessMemoryInfo_PeakWorkingSetSize | A, B, C | `rss_backend` |
| Synthetic-input seed | 20 | A, B, C | `seed` |

Runs A, B and C recorded **byte-identical** environment blocks, and every figure in
every report carries that block inline — figure-level provenance is in the reports
themselves, not only in this section.

The embedder and its assets:

| Property | Value | Source | Key path |
|---|---|---|---|
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 | A, B, C | `embedding_model_id` |
| Model version | all_minilm_l6_v2_onnx_v1 | A, B, C | `model_version` |
| Observed embedding dimension | 384 | A, B, C | `embedding_dimension` |

The dimension is **observed** from the loaded model, never assumed: D18 requires
it, and the literal appears nowhere in the harness. The assets were exposed through
`SENTINEL_MINILM_DIR` and verified against the digests D38 pins:

```
model.onnx      6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452
tokenizer.json  be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037
```

---

## 2. MiniLM load

| Figure | Value | Unit | Source | Key path |
|---|---|---|---|---|
| Peak process RSS after load | 200724480 | bytes | A | `minilm_load.peak_rss_bytes.value` |
| Load duration | 0.4106049999827519 | seconds | A | `minilm_load.seconds.value` |
| Backend | windows_GetProcessMemoryInfo_PeakWorkingSetSize | — | A | `minilm_load.peak_rss_bytes.rss_backend` |

Run A had performed no earlier measurement when this figure was taken, as §4.1 of
the contract requires of a publishable memory figure. Read §7 before interpreting
it: this is the process's peak *through* the load, not the model's own footprint.

---

## 3. Embedding throughput

Each batch size embeds the same number of records, so the three rates are
comparable with one another. Both the record count and the elapsed time are
recorded beside the rate, so the arithmetic is checkable rather than asserted.

| Batch size | Throughput | Unit | Records | Seconds | Source | Key path |
|---|---|---|---|---|---|---|
| 1 | 297.54519402244006 | records_per_second | 256 | 0.8603735000360757 | A | `embedding_throughput.1` |
| 8 | 606.944584045988 | records_per_second | 256 | 0.42178480001166463 | A | `embedding_throughput.8` |
| 32 | 705.1127562508533 | records_per_second | 256 | 0.3630624999059364 | A | `embedding_throughput.32` |

Throughput rises with batch size across the three measured points. That is an
observation about these three measurements on this machine, not a claim about an
optimum: the contract fixes exactly these batch sizes and no search over them.

---

## 4. Artifact sizes

| Artifact class | Size | Unit | Status | Source | Key path |
|---|---|---|---|---|---|
| cfpb_triage_tfidf | — | — | absent | A, B, C | `artifact_sizes.cfpb_triage_tfidf` |
| nyc311_sla_risk | — | — | absent | A, B, C | `artifact_sizes.nyc311_sla_risk` |
| xdomain_xtarget_probe | — | — | absent | A, B, C | `artifact_sizes.xdomain_xtarget_probe` |
| minilm_assets | 90871461 | bytes | measured | A, B, C | `artifact_sizes.minilm_assets.value` |

The three experiment artifacts are **absent**, and that is the correct result
rather than a gap: artifacts are git-ignored and produced by experiment runs, and
no Task 16, 17 or 19 experiment had been run in this environment. Absence is
recorded as absence — never as zero bytes, which would be a measurement.

`minilm_assets` is an external model-asset class rather than a run-produced
artifact, so it resolves through the MiniLM asset directory rather than the
caller-supplied artifact root (contract §15.8). Its figure is the real on-disk size
of the two pinned files. All three runs measured the same byte count.

---

## 5. Index build

Both populations use deterministic synthetic `float32` vectors at the observed
embedding dimension, generated in full **before** timing begins, so the timed
interval holds index construction and nothing else — no vector generation, no
identity generation, no embedding.

| Population | Build time | Unit | Source | Key path |
|---|---|---|---|---|
| 10000 | 0.0022055000299587846 | seconds | A | `index_build.10000.seconds.value` |
| 50000 | 0.013611900038085878 | seconds | A | `index_build.50000.seconds.value` |

| Population | Peak process RSS | Unit | Source | Key path |
|---|---|---|---|---|
| 10000 | 200806400 | bytes | B | `index_build.10000.peak_rss_bytes.value` |
| 50000 | 278134784 | bytes | C | `index_build.50000.peak_rss_bytes.value` |

The two memory figures come from **different runs, each in its own fresh
interpreter**, because a peak is a per-process high-water mark: a 50,000-vector
figure taken after a 10,000-vector build in the same process would be measuring
both. Run B supplies the 10,000 figure and run C the 50,000 figure, and each run
is the first and only measurement its process performed.

The 10,000-vector figure is close to the load figure of §2 because, within run B,
building that index **did not raise the process peak above what loading the model
had already reached** — the two figures are equal inside every one of the three
runs. A 10,000 × 384 `float32` matrix is small next to the ONNX session, so the
high-water mark did not move. That is a property of a cumulative peak, not evidence
that index construction is free; §7 says what these figures are and are not.

---

## 6. Query latency

| Figure | Value | Unit | Source | Key path |
|---|---|---|---|---|
| Single-query latency | 0.017625499982386827 | seconds | A | `query_latency.value` |

One `rank` call against the index actually built at the larger population, with
the query vector generated before the clock starts. Construction time is not
included.

---

## 7. Methodology and limitations

**Peak RSS is a process high-water mark, not a stage's own allocation.** Both
backends the contract permits — `GetProcessMemoryInfo` on Windows,
`getrusage` on POSIX — report the maximum resident set the *process* has reached.
So the load figure in §2 includes the interpreter, the imported libraries and the
ONNX session, and the index figures in §5 include everything the process had
already reached. **No figure here is the incremental cost of the stage it sits
under**, and none should be read that way.

**Every memory figure was produced in a fresh process** that had performed no
earlier measurement run, as contract §4.1 requires. The three runs are separate
interpreters; nothing was measured twice in one process and published as two
independent results.

**Time and throughput figures are not subject to that rule** and are all taken
from run A, so the timings in §3, §5 and §6 describe one coherent run.

**Nothing here is derived.** No value was averaged, converted to other units,
rounded, adjusted or entered by hand. Where a figure could not be measured, it is
reported absent rather than filled in. No number came from this repository's own
development environment, which installs the training tier and in which the harness
correctly refuses to write a report at all.

**These are measurements of one machine at one moment**, with the CPU, core count
and library versions of §1. They bound nothing on other hardware and no tolerance
across machines is claimed here; Task 21 owns that question.

**No figure is compared with another as an improvement or a regression.** The
document reports what was measured. Where two numbers differ — the three batch
sizes, the two index populations — the difference is stated as an observation about
those measurements and nothing more.

---

## 8. Source reports

Three runs, each a separate interpreter executing the production entry point
`run_measurements(artifact_root=..., report_path=...)` against temporary paths.

| Run | Report | Supplies |
|---|---|---|
| A | `run-A.json` | the load figures, all throughput, both build times, the latency |
| B | `run-B.json` | the 10000-vector peak RSS |
| C | `run-C.json` | the 50000-vector peak RSS |

The reports are measurement evidence rather than repository content, so they are
not committed: the harness writes to a caller-supplied path and has no default
location inside the repository. Regenerating them is one command per run in an
environment built from `requirements/ml.txt`, with `SENTINEL_MINILM_DIR` pointing
at the pinned assets.

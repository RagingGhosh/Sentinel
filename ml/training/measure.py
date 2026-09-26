"""Inference-environment resource measurement (Task 20, D40, D41, plan §S, §U).

One run, end to end::

    require_clean_environment()          pandas and pyarrow must not be importable
      ->  measure_minilm_load()          peak process RSS around the real load
      ->  measure_embedding_throughput() records/sec at batch 1, 8 and 32
      ->  measure_artifact_sizes(root)   measured, or recorded absent
      ->  measure_index_build(dimension) build time and peak RSS at 10k and 50k
      ->  measure_query_latency(index)   one rank() against the 50k index
      ->  ResourceReport                 serialized whole to a caller's path

**The environment is the point, not a detail.** Plan §E keeps `ml.txt` separate
from `train.txt` because scikit-learn and onnxruntime become *runtime*
dependencies in Phase 3 while pandas and pyarrow never do. A memory figure taken
with a dataframe library resident would describe an environment the web process
will never run in, so the harness refuses to write a report if either package is
importable. That refusal fires in this repository's own development environment,
which installs the training tier — which is the rule working, not a fault in it.

**Every figure carries its own provenance.** `Figure` takes `environment` as a
required argument, so an unprovenanced figure is a construction error rather than
a serialization-time omission, and a memory figure additionally names the backend
that produced it (D41). A non-finite value is refused: a `NaN` describes nothing.

**Peak RSS comes from the standard library** — `GetProcessMemoryInfo` on Windows,
`getrusage` on POSIX — because no measurement-only package may enter `ml.txt`,
which is a Phase 3 runtime tier (D40.2). Both mechanisms report a **process**
peak rather than an interval peak, so a figure here is the process's peak observed
at the end of that stage, not a delta attributable to it alone. That is what the
frozen mechanism can honestly provide, and the report names which one gave it.

**Absence is a result, not a zero.** Artifacts are git-ignored and produced by
runs, so a fresh clone has none; §5 requires the harness to be useful in that
state. A class that exists is measured, a class that does not is recorded absent,
and a genuinely empty directory measures zero bytes — zero is a fact, absence is
the lack of one.

**Nothing is fitted, trained, ingested or written except the report.** No corpus
is read: the mandated environment cannot read one, and index cost is a property of
the index rather than of any corpus, so the index populations are deterministic
synthetic vectors at the embedder's own observed width (D18, D40.4).

The five `measure_*` functions are **internal stage seams** (D41). They exist so
the whole-or-nothing failure contract can be exercised one stage at a time; only
`run_measurements` is an entry point.

Training-side and Django-independent: invoked as ``python -m``, never as a
management command, and nothing here is imported by serving.
"""

from __future__ import annotations

import ctypes
import importlib.util
import json
import math
import os
import platform
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed_version
from pathlib import Path
from typing import Any

import numpy as np

from ingest.identity import RecordRef
from ml.embedders import minilm
from ml.training.index import build_index

BATCH_SIZES: tuple[int, ...] = (1, 8, 32)
"""§2.B: records per batch, in this order. Not a runtime knob (D41)."""

INDEX_POPULATIONS: tuple[int, ...] = (10_000, 50_000)
"""§2.D: both populations, each measured for build time **and** peak memory."""

FORBIDDEN_PACKAGES: tuple[str, ...] = ("pandas", "pyarrow")
"""§3: the clean-environment proof. Either one present refuses the report."""

SEED = 20
"""The deterministic seed for §6's synthetic input: the task number, as Task 17
used 17 and Task 18 used 18."""

WINDOWS_RSS_BACKEND = "windows_GetProcessMemoryInfo_PeakWorkingSetSize"
POSIX_RSS_BACKEND = "posix_resource_getrusage_ru_maxrss"
"""§4's two backends. Named in every memory figure, because they do not measure
quite the same quantity and differ in units by platform."""

ARTIFACT_CLASSES: tuple[str, ...] = (
    "cfpb_triage_tfidf",
    "nyc311_sla_risk",
    "xdomain_xtarget_probe",
    "minilm_assets",
)
"""§5's expected classes, in report order: Task 16's triage artifact, Task 17's
risk artifact, Task 19's probe artifact, and the external MiniLM assets. A
declared tuple rather than a filesystem scan, because a scan cannot report a class
that is missing and enumeration order must not decide the report's order (D41)."""

PROVENANCE_KEYS: tuple[str, ...] = (
    "cpu_model",
    "cpu_cores",
    "python_version",
    "library_versions",
)
"""§3: recorded with every figure. A figure missing any of these does not exist."""

MEASURED_LIBRARIES: tuple[str, ...] = (
    "numpy",
    "scikit-learn",
    "onnxruntime",
    "tokenizers",
    "scipy",
    "joblib",
)
"""Read through `importlib.metadata`, so recording a version never imports a
library the measured figure did not need."""

_THROUGHPUT_TARGET = 256
"""How many records each batch size embeds, so a throughput figure is a ratio of
two measured quantities rather than one clock tick. Deterministic: the batch count
is ``ceil(target / batch_size)`` and the record count follows from it."""


# --- the figure ----------------------------------------------------------------------


@dataclass(frozen=True)
class Figure:
    """One measured number, with the environment that produced it (D41).

    `environment` has no default, so a figure cannot be constructed without its
    provenance — which is what makes §8's "every numeric figure carries its
    environment provenance" a property of the type rather than a habit of the
    caller.
    """

    value: float
    unit: str
    environment: Mapping[str, Any]
    rss_backend: str | None = None
    """Set on a memory figure, per §4. Absent on a time or rate figure."""
    details: Mapping[str, Any] = field(default_factory=dict)
    """The quantities a derived figure was computed from — a throughput's records
    and seconds — so a reader can check the arithmetic rather than trust it."""

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ValueError(f"a figure's value must be a number, got {self.value!r}")
        if not math.isfinite(float(self.value)):
            raise ValueError(
                f"a figure's value must be finite, got {self.value!r}; a non-finite "
                "measurement describes nothing and must never reach the report"
            )
        if not self.unit:
            raise ValueError("a figure must name its unit")
        missing = [key for key in PROVENANCE_KEYS if key not in self.environment]
        if missing:
            raise ValueError(
                f"this figure's provenance is missing {', '.join(missing)}; every "
                "numeric figure carries the environment that produced it (§8, D41)"
            )


# --- provenance ------------------------------------------------------------------------


def environment() -> Mapping[str, Any]:
    """The environment recorded with every figure (§3).

    The CPU core count is refused rather than defaulted when the platform will not
    report it: a figure whose core count was guessed cannot be compared with
    another machine's.
    """
    cores = os.cpu_count()
    if not cores:
        raise RuntimeError(
            "this platform does not report a CPU core count, and a measurement "
            "whose core count was invented could not be compared with another run"
        )
    return {
        "cpu_model": platform.processor() or platform.machine() or "unknown",
        "cpu_cores": int(cores),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "library_versions": _library_versions(),
    }


def _library_versions() -> Mapping[str, str]:
    """Installed versions of the libraries a figure could have depended on."""
    versions = {}
    for name in MEASURED_LIBRARIES:
        try:
            versions[name] = installed_version(name)
        except PackageNotFoundError:
            continue
    return versions


# --- peak RSS --------------------------------------------------------------------------


def rss_backend() -> str:
    """Which of §4's two backends this platform uses.

    An unsupported platform is a refusal, never a zero: zero bytes would read as
    "no memory used" (§4, D40.2).
    """
    if os.name == "nt":
        return WINDOWS_RSS_BACKEND
    if os.name == "posix":
        return POSIX_RSS_BACKEND
    raise RuntimeError(
        f"no peak-RSS backend for os.name {os.name!r}; the two supported mechanisms "
        f"are {WINDOWS_RSS_BACKEND} and {POSIX_RSS_BACKEND}, and a figure of zero "
        "bytes would read as 'no memory used'"
    )


def peak_rss_bytes() -> int:
    """This process's peak resident set size, in bytes.

    Both mechanisms report a **process** peak rather than the peak of an interval,
    so a caller measuring a stage reads this at the end of it and reports the
    process peak observed there. Neither samples the Python heap: a heap-only
    sampler would miss almost all of an ONNX session's native allocation, which is
    the quantity of interest here.
    """
    backend = rss_backend()
    if backend == WINDOWS_RSS_BACKEND:
        return _windows_peak_rss()
    return _posix_peak_rss()


class _ProcessMemoryCounters(ctypes.Structure):
    """`PROCESS_MEMORY_COUNTERS`, in declaration order (Windows only)."""

    _fields_ = (
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    )


def _windows_peak_rss() -> int:
    """`GetProcessMemoryInfo`'s `PeakWorkingSetSize`, already in bytes."""
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
        ctypes.c_void_p(handle), ctypes.byref(counters), counters.cb
    )
    if not ok:
        raise RuntimeError(
            f"GetProcessMemoryInfo failed with error {ctypes.get_last_error()}; "  # type: ignore[attr-defined]
            "no peak RSS was measured, and none is invented"
        )
    return int(counters.PeakWorkingSetSize)


def _posix_peak_rss() -> int:
    """`getrusage(RUSAGE_SELF).ru_maxrss`, converted to bytes.

    Imported here rather than at module scope because `resource` does not exist on
    Windows. The unit differs by platform — bytes on macOS, kibibytes on Linux and
    the BSDs — and the report's unit is bytes either way.
    """
    import resource

    # `resource` is POSIX-only, so a type checker running on Windows sees no
    # attributes on it -- the same reason `ctypes.windll` is ignored above.
    usage = resource.getrusage(resource.RUSAGE_SELF)  # type: ignore[attr-defined]
    maximum = usage.ru_maxrss
    if sys.platform == "darwin":
        return int(maximum)
    return int(maximum) * 1024


# --- the clean-environment proof -----------------------------------------------------------


def require_clean_environment() -> None:
    """Refuse unless every forbidden package is genuinely unimportable (§3).

    Importability is the test, not a pinned file: an environment that happens to
    have pyarrow installed is refused whatever it claims to pin. A package whose
    spec cannot even be located is not importable, which is the outcome this
    function is looking for.
    """
    present = []
    for name in FORBIDDEN_PACKAGES:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if found:
            present.append(name)
    if present:
        raise RuntimeError(
            f"{', '.join(present)} is importable, so this is not the inference-only "
            "environment Task 20 measures. Install the inference tier alone; a figure "
            "taken here would describe an environment the web process never runs in, "
            "and no report is written (§3, D40.5)"
        )


# --- deterministic synthetic input -----------------------------------------------------------


def synthetic_vectors(count: int, dimension: int) -> np.ndarray:
    """`count` deterministic unit-less `float32` rows of width `dimension` (§6).

    Seeded from `SEED` through its own generator, so the global NumPy random state
    cannot move a published figure. Generated **before** any timing starts: index
    cost is what this measures, and generation is not part of it.
    """
    _positive("count", count)
    _positive("dimension", dimension)
    generator = np.random.default_rng(SEED)
    return generator.standard_normal((count, dimension), dtype=np.float32)


def synthetic_refs(count: int) -> tuple[RecordRef, ...]:
    """`count` deterministic, distinct identities (§6).

    Distinct by construction, because `build_index` refuses a repeated reference
    rather than de-duplicating one.
    """
    _positive("count", count)
    return tuple(
        RecordRef(source="synthetic", external_id=f"m{position:08d}") for position in range(count)
    )


def _positive(what: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{what} must be a positive integer, got {value!r}")


# --- the embedder ------------------------------------------------------------------------------


def load_embedder() -> Any:
    """The pinned MiniLM embedder, whose observed width §6 uses.

    The dimension is never assumed from a checkpoint-family stereotype: the loader
    observes it from a forward pass, and this module reads it from the embedder
    (D18). Missing or altered assets raise the loader's own errors, uncaught — a
    measurement of a model that did not load describes nothing.
    """
    return minilm.load_minilm()


# --- the five stage seams -----------------------------------------------------------------------


def measure_minilm_load() -> tuple[Any, Mapping[str, Figure]]:
    """§2.A: peak process RSS around the real model load, and the load's duration.

    Takes no population or batch argument, so no benchmark generation can be timed
    inside it. Nothing synthetic is generated here.
    """
    recorded = environment()
    backend = rss_backend()
    started = time.perf_counter()
    embedder = load_embedder()
    seconds = time.perf_counter() - started
    figures = {
        "peak_rss_bytes": Figure(
            value=peak_rss_bytes(), unit="bytes", environment=recorded, rss_backend=backend
        ),
        "seconds": Figure(value=seconds, unit="seconds", environment=recorded),
    }
    return embedder, figures


def measure_embedding_throughput(embedder: Any) -> Mapping[int, Figure]:
    """§2.B: records per second at each of `BATCH_SIZES`, measured separately.

    Each batch size embeds about `_THROUGHPUT_TARGET` records, so the figure is a
    ratio of two measured quantities and not one clock tick, and both are recorded
    beside it. Embedding cost is reported here and nowhere else: it is not index
    cost and the two are never combined (§6).
    """
    recorded = environment()
    figures: dict[int, Figure] = {}
    for size in BATCH_SIZES:
        _positive("batch_size", size)
        batches = -(-_THROUGHPUT_TARGET // size)
        texts = [f"synthetic measurement record {position}" for position in range(size)]
        started = time.perf_counter()
        for _ in range(batches):
            embedder.embed(texts)
        seconds = time.perf_counter() - started
        records = batches * size
        if seconds <= 0.0:
            raise RuntimeError(
                f"embedding {records} records at batch size {size} took no measurable "
                "time; a throughput computed from a zero interval is not a measurement"
            )
        figures[size] = Figure(
            value=records / seconds,
            unit="records_per_second",
            environment=recorded,
            details={"records": records, "seconds": seconds, "batch_size": size},
        )
    return figures


def measure_artifact_sizes(artifact_root: Path) -> Mapping[str, Figure | None]:
    """§2.C: each declared class measured, or recorded absent.

    `None` is this stage's absence marker and serializes as ``{"status":
    "absent"}`` — never as zero bytes, which is a measurement. Traversal is sorted,
    so the byte total cannot depend on enumeration order, and the reported classes
    come from `ARTIFACT_CLASSES` rather than from what the tree happens to hold.

    The MiniLM assets are external to the caller's artifact root by nature — they
    are the files the embedder loads — so that one class is located through the
    embedder's own asset directory.
    """
    root = Path(artifact_root)
    sizes: dict[str, Figure | None] = {}
    recorded = environment()
    for name in ARTIFACT_CLASSES:
        directory = minilm.asset_dir() if name == "minilm_assets" else _class_directory(root, name)
        if directory is None or not directory.is_dir():
            sizes[name] = None
            continue
        total = sum(path.stat().st_size for path in sorted(directory.rglob("*")) if path.is_file())
        sizes[name] = Figure(value=total, unit="bytes", environment=recorded)
    return sizes


def _class_directory(root: Path, name: str) -> Path | None:
    """The first directory named `name` under `root`, in sorted order, or `None`."""
    if not root.is_dir():
        return None
    found = sorted(path for path in root.rglob(name) if path.is_dir())
    return found[0] if found else None


def measure_index_build(dimension: int) -> tuple[Mapping[int, Mapping[str, Figure]], Any]:
    """§2.D: build time and peak process RSS at each of `INDEX_POPULATIONS`.

    Every vector and every identity is generated **before** the clock starts, so
    the timed interval holds `build_index` and nothing else — not generation, not
    embedding (§6, §9). The index built at the largest population is returned so
    that the latency stage queries a real one rather than building a second.
    """
    recorded = environment()
    backend = rss_backend()
    results: dict[int, Mapping[str, Figure]] = {}
    largest = max(INDEX_POPULATIONS)
    queried: Any = None
    for population in INDEX_POPULATIONS:
        references = synthetic_refs(population)
        vectors = synthetic_vectors(population, dimension)
        started = time.perf_counter()
        index = build_index(references, vectors)
        seconds = time.perf_counter() - started
        results[population] = {
            "seconds": Figure(value=seconds, unit="seconds", environment=recorded),
            "peak_rss_bytes": Figure(
                value=peak_rss_bytes(), unit="bytes", environment=recorded, rss_backend=backend
            ),
        }
        if population == largest:
            queried = index
    if queried is None:
        raise RuntimeError("no index was built, so no population was measured")
    return results, queried


def measure_query_latency(index: Any) -> Figure:
    """§2.E: one `rank` against the index that was actually built.

    The query vector is generated before the clock starts, at the index's own
    width, so the interval holds the query and nothing else — no construction and
    no setup.
    """
    recorded = environment()
    query = synthetic_vectors(1, index.dimension)[0]
    started = time.perf_counter()
    index.rank(query)
    seconds = time.perf_counter() - started
    if seconds <= 0.0:
        raise RuntimeError(
            "the query took no measurable time; a latency of zero is not a measurement"
        )
    return Figure(value=seconds, unit="seconds", environment=recorded)


# --- the report ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourceReport:
    """One run's figures, and the identity of what produced them.

    Every numeric member is a `Figure`, so every published number carries its own
    provenance. The five categories are separate members: §6 keeps embedding
    throughput, index build cost and query latency apart so that no reader can take
    one for another.
    """

    environment: Mapping[str, Any]
    rss_backend: str
    embedding_dimension: int
    embedding_model_id: str
    model_version: str
    minilm_load: Mapping[str, Figure]
    embedding_throughput: Mapping[int, Figure]
    artifact_sizes: Mapping[str, Figure | None]
    index_build: Mapping[int, Mapping[str, Figure]]
    query_latency: Figure
    artifact_root: Path
    report_path: Path

    def as_json(self) -> str:
        """The report as strict JSON text — exactly what `report_path` receives.

        No filesystem path is included: a path is the caller's context rather than
        a finding, and a report that travels should not carry one machine's layout.
        """
        return json.dumps(_payload(self), indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def run_measurements(*, artifact_root: Path, report_path: Path) -> ResourceReport:
    """Measure the five §2 categories and serialize them to the caller's path.

    Both arguments are required and keyword-only: §5 and §8 give the harness no
    default artifact root and no default report path, so it can neither measure the
    repository by accident nor write into it.

    The environment proof runs first, and any stage failure propagates: a failed
    measurement is a failed run, never a quietly missing figure. Nothing reaches
    `report_path` unless the whole report can be written at once.
    """
    require_clean_environment()

    root = Path(artifact_root)
    if root.exists() and not root.is_dir():
        raise ValueError(
            f"artifact_root {root} is not a directory; a file where a tree belongs is "
            "a caller error rather than an empty measurement"
        )

    embedder, load_figures = measure_minilm_load()
    throughput = measure_embedding_throughput(embedder)
    sizes = measure_artifact_sizes(root)
    build, index = measure_index_build(int(embedder.embedding_dimension))
    latency = measure_query_latency(index)

    report = ResourceReport(
        environment=environment(),
        rss_backend=rss_backend(),
        embedding_dimension=int(embedder.embedding_dimension),
        embedding_model_id=str(embedder.embedding_model_id),
        model_version=str(embedder.model_version),
        minilm_load=load_figures,
        embedding_throughput=throughput,
        artifact_sizes=sizes,
        index_build=build,
        query_latency=latency,
        artifact_root=root,
        report_path=Path(report_path),
    )
    _write_report(report)
    return report


# --- serialization ------------------------------------------------------------------------------


def _payload(report: ResourceReport) -> dict[str, Any]:
    """The report as strict-JSON-ready data, one block per §2 category."""
    return {
        "environment": dict(report.environment),
        "rss_backend": report.rss_backend,
        "embedding_dimension": report.embedding_dimension,
        "embedding_model_id": report.embedding_model_id,
        "model_version": report.model_version,
        "batch_sizes": list(BATCH_SIZES),
        "index_populations": list(INDEX_POPULATIONS),
        "seed": SEED,
        "minilm_load": {name: _figure(figure) for name, figure in report.minilm_load.items()},
        "embedding_throughput": {
            str(size): _figure(figure) for size, figure in report.embedding_throughput.items()
        },
        "artifact_sizes": {
            name: _artifact(figure) for name, figure in report.artifact_sizes.items()
        },
        "index_build": {
            str(population): {name: _figure(figure) for name, figure in block.items()}
            for population, block in report.index_build.items()
        },
        "query_latency": _figure(report.query_latency),
    }


def _figure(figure: Figure) -> dict[str, Any]:
    payload: dict[str, Any] = dict(figure.details)
    payload["value"] = float(figure.value) if isinstance(figure.value, float) else figure.value
    payload["unit"] = figure.unit
    payload["environment"] = dict(figure.environment)
    if figure.rss_backend is not None:
        payload["rss_backend"] = figure.rss_backend
    return payload


def _artifact(figure: Figure | None) -> dict[str, Any]:
    """A measurement, or an absence — structurally distinct, and never confused.

    An absent class carries no `value` key at all, so absence cannot be read as a
    quantity; a class that exists and holds nothing measures zero bytes, because
    zero is a fact and absence is the lack of one (§5, D41).
    """
    if figure is None:
        return {"status": "absent"}
    return _figure(figure)


def _write_report(report: ResourceReport) -> Path:
    """Serialize to the caller's path, whole or not at all (D27's pattern).

    Written through a temporary file in the same directory and moved into place, so
    a reader finds either a complete report or no report, and a failure leaves
    neither a partial file nor a stale temporary one.
    """
    target = report.report_path
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=".resources-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(report.as_json())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target

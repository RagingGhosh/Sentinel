"""Task 20 slice 2: the inference-environment resource measurement harness.

RED phase. `ml.training.measure` does not exist, so every test that reaches
production fails. The production module is imported late, inside `measure()`, so
collection succeeds and the fixtures and source-level guards still run.

The API surface these tests pin — **none of it named by the frozen contract**,
which fixes the file, the five measurements and their semantics but no
identifiers, so the RED phase fixes them the way Tasks 18 and 19 did:

    BATCH_SIZES / INDEX_POPULATIONS / FORBIDDEN_PACKAGES / SEED
    WINDOWS_RSS_BACKEND / POSIX_RSS_BACKEND / ARTIFACT_CLASSES
    Figure(value, unit, environment)          a figure cannot exist unprovenanced
    environment()                             cpu model, cores, python, libraries
    peak_rss_bytes() / rss_backend()
    require_clean_environment()               refuses when pandas/pyarrow import
    synthetic_vectors(count, dimension) / synthetic_refs(count)
    load_embedder()
    measure_minilm_load() / measure_embedding_throughput(embedder)
    measure_artifact_sizes(artifact_root)
    measure_index_build(dimension) / measure_query_latency(index)
    run_measurements(*, artifact_root, report_path) -> ResourceReport

Two environmental facts shape this suite. First, **this development environment
has pandas and pyarrow installed**, so the harness must refuse here — the
positive paths therefore neutralise `require_clean_environment` through the
`harness` fixture, and one test runs the guard in a subprocess with both
packages genuinely blocked. Second, the pinned MiniLM assets are git-ignored, so
the one test that loads the real embedder skips when they are absent, on the
convention `test_dedup_benchmark.py` established.

The `harness` fixture also shrinks the two index populations from 10k and 50k to
64 and 128, by patching the module constant rather than by asking production for
a knob the contract does not authorise. A separate test pins the frozen values.

Nothing here writes into the repository: every path is a `tmp_path`.
"""

import ast
import hashlib
import importlib
import inspect
import json
import os
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy", reason="numpy lives in requirements/ml.txt")
pytest.importorskip("onnxruntime", reason="onnxruntime lives in requirements/ml.txt")

from ingest.identity import RecordRef  # noqa: E402
from ml.training import index as index_module  # noqa: E402

pytestmark = pytest.mark.ml

ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "ml" / "training" / "measure.py"

# --- the contract's frozen values ------------------------------------------------------

BATCH_SIZES = (1, 8, 32)
"""Contract §2.B: exactly these three, no others."""

INDEX_POPULATIONS = (10_000, 50_000)
"""Contract §2.D: both, each with build time **and** peak memory."""

FORBIDDEN_PACKAGES = ("pandas", "pyarrow")
"""Contract §3: the harness refuses to write a report if either is importable."""

PROVENANCE_KEYS = ("cpu_model", "cpu_cores", "python_version", "library_versions")
"""Contract §3: recorded with every figure."""

CATEGORIES = (
    "minilm_load",
    "embedding_throughput",
    "artifact_sizes",
    "index_build",
    "query_latency",
)
"""Contract §2's five, kept apart in the report so none can be read as another."""

BANNED_IN_SOURCE = ("psutil", "tracemalloc", "import pandas", "import pyarrow", "384")
"""§4 forbids psutil and tracemalloc for peak RSS, §3 the train tier, §6 a
hard-coded dimension."""

STAGES = (
    "measure_minilm_load",
    "measure_embedding_throughput",
    "measure_artifact_sizes",
    "measure_index_build",
    "measure_query_latency",
)

SMALL_POPULATIONS = (64, 128)

ASSET_DIR_ENV = "SENTINEL_MINILM_DIR"
REQUIRE_ENV = "SENTINEL_REQUIRE_MINILM"
DEFAULT_ASSET_DIR = Path("ml/artifacts/embedders/all_minilm_l6_v2/v1")
ONNX_SHA256 = "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"
TOKENIZER_SHA256 = "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"


# --- the production module, imported late ---------------------------------------------


def measure():
    return importlib.import_module("ml.training.measure")


def source() -> str:
    return MODULE_PATH.read_text(encoding="utf-8")


def require_real_assets() -> Path:
    """The one real-embedder test skips when the pinned assets are absent."""
    directory = Path(os.environ.get(ASSET_DIR_ENV) or ROOT / DEFAULT_ASSET_DIR)
    problems = []
    for name, expected in (("model.onnx", ONNX_SHA256), ("tokenizer.json", TOKENIZER_SHA256)):
        path = directory / name
        if not path.is_file():
            problems.append(f"{path} is missing")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            problems.append(f"{path} does not match the pinned digest")
    if not problems:
        return directory
    detail = "; ".join(problems)
    if os.environ.get(REQUIRE_ENV) == "1":
        pytest.fail(f"{REQUIRE_ENV}=1 but the pinned MiniLM assets are unusable: {detail}")
    pytest.skip(f"pinned MiniLM assets unavailable: {detail}")


def boom(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("this stage was made to fail")


# --- fakes and the harness, so no test waits for a real 50k build ---------------------


class FakeEmbedder:
    """The `TextEmbedder` surface the harness reads, and nothing more.

    Its dimension is deliberately **not** 384: a harness that hard-coded the real
    width would disagree with what it was handed.
    """

    def __init__(self, dimension: int = 7) -> None:
        self.embedding_dimension = dimension
        self.model_version = "fake_embedder_v1"
        self.embedding_model_id = "fake/embedder"
        self.batches: list[int] = []

    def embed(self, texts):
        rows = list(texts)
        self.batches.append(len(rows))
        return np.zeros((len(rows), self.embedding_dimension), dtype=np.float32)


@dataclass
class Harness:
    """One run's paths, plus the three patches every positive path needs."""

    artifact_root: Path
    report_path: Path
    monkeypatch: Any
    embedder: FakeEmbedder = field(default_factory=FakeEmbedder)
    populations: tuple[int, ...] = SMALL_POPULATIONS
    _prepared: bool = False

    def prepare(self):
        """Import the harness and neutralise the environment guard and the sizes.

        Deliberately not done at fixture setup: importing the module there would
        turn a RED-phase failure into a fixture *error* instead.

        **Applied once.** `run()` calls this too, and re-applying the patches would
        overwrite a patch the test itself installed afterwards on one of these three
        names -- which silently defeated `test_the_guard_runs_before_any_measurement`
        until the implementation exposed it.
        """
        module = measure()
        if not self._prepared:
            self.monkeypatch.setattr(module, "require_clean_environment", lambda: None)
            self.monkeypatch.setattr(module, "INDEX_POPULATIONS", self.populations)
            self.monkeypatch.setattr(module, "load_embedder", lambda: self.embedder)
            self._prepared = True
        return module

    def run(self):
        return self.prepare().run_measurements(
            artifact_root=self.artifact_root, report_path=self.report_path
        )

    def written(self) -> dict:
        return json.loads(self.report_path.read_text(encoding="utf-8"))

    def artifact(self, model: str, payload: bytes = b"x" * 2048) -> Path:
        directory = self.artifact_root / "nyc311" / model / "v1"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model.joblib").write_bytes(payload)
        (directory / "metadata.json").write_text("{}", encoding="utf-8")
        return directory


@pytest.fixture
def harness(tmp_path, monkeypatch) -> Harness:
    return Harness(
        artifact_root=tmp_path / "artifacts",
        report_path=tmp_path / "reports" / "resources.json",
        monkeypatch=monkeypatch,
    )


def figures(node, path="report"):
    """Every mapping that looks like a figure — one carrying a value and a unit."""
    if isinstance(node, dict):
        if "value" in node and "unit" in node:
            yield path, node
        for key, item in node.items():
            yield from figures(item, f"{path}.{key}")
    elif isinstance(node, list):
        for position, item in enumerate(node):
            yield from figures(item, f"{path}[{position}]")


# --- A. the frozen constants ------------------------------------------------------------


def test_the_batch_sizes_are_exactly_one_eight_and_thirty_two():
    assert tuple(measure().BATCH_SIZES) == BATCH_SIZES


def test_the_index_populations_are_ten_thousand_and_fifty_thousand():
    """Contract §2.D. A harness that measured only 10k would publish half a figure."""
    assert tuple(measure().INDEX_POPULATIONS) == INDEX_POPULATIONS


def test_the_forbidden_packages_are_declared_by_the_module():
    assert set(measure().FORBIDDEN_PACKAGES) == set(FORBIDDEN_PACKAGES)


def test_the_artifact_classes_are_declared_without_repetition():
    declared = tuple(measure().ARTIFACT_CLASSES)
    assert declared, "the expected classes of §5"
    assert len(declared) == len(set(declared)), "a class is listed twice"
    assert "minilm_assets" in declared, "the external assets are an expected class"


def test_the_seed_is_declared_so_synthetic_input_is_reproducible():
    assert isinstance(measure().SEED, int)


# --- B. the clean-environment guard -------------------------------------------------------


def test_the_report_is_refused_when_pandas_is_importable(harness, monkeypatch):
    """§3. This environment has pandas, so an unpatched harness must refuse here."""
    module = measure()
    monkeypatch.setattr(module, "FORBIDDEN_PACKAGES", ("pandas",))
    with pytest.raises(Exception):
        module.run_measurements(
            artifact_root=harness.artifact_root, report_path=harness.report_path
        )
    assert not harness.report_path.exists(), "a refused run leaves no report"


def test_the_report_is_refused_when_pyarrow_is_importable(harness, monkeypatch):
    module = measure()
    monkeypatch.setattr(module, "FORBIDDEN_PACKAGES", ("pyarrow",))
    with pytest.raises(Exception):
        module.run_measurements(
            artifact_root=harness.artifact_root, report_path=harness.report_path
        )
    assert not harness.report_path.exists()


def test_the_report_is_refused_when_either_is_importable(harness):
    """Both are installed here, so the default constant is enough to refuse."""
    module = measure()
    with pytest.raises(Exception):
        module.run_measurements(
            artifact_root=harness.artifact_root, report_path=harness.report_path
        )
    assert not harness.report_path.exists()


def test_the_guard_names_the_package_it_found():
    """A refusal that did not say which package would be unactionable."""
    module = measure()
    with pytest.raises(Exception) as caught:
        module.require_clean_environment()
    message = str(caught.value).lower()
    assert any(package in message for package in FORBIDDEN_PACKAGES)


def test_the_guard_checks_importability_not_a_requirements_file():
    """§3: an environment that happens to have pyarrow is refused whatever it pins."""
    body = inspect.getsource(measure().require_clean_environment)
    assert "requirements" not in body
    assert "find_spec" in body or "import_module" in body


def test_the_guard_runs_before_any_measurement(harness, monkeypatch):
    """Measurement may run; writing is what a failed proof forbids (§3)."""
    checked = []
    module = harness.prepare()
    monkeypatch.setattr(module, "require_clean_environment", lambda: checked.append("first"))
    monkeypatch.setattr(module, "measure_minilm_load", boom)
    with pytest.raises(Exception):
        harness.run()
    assert checked == ["first"], "the environment proof is the run's first act"


_CLEAN_RUN = """
import sys

BLOCKED = {"pandas", "pyarrow"}


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(name + " is not installed")
        return None


sys.meta_path.insert(0, Blocker())

import importlib

measure = importlib.import_module("ml.training.measure")
measure.require_clean_environment()      # must NOT raise: neither package is importable
assert not (BLOCKED & set(sys.modules)), "a blocked package was imported anyway"
print("OK")
"""


def test_the_guard_passes_when_neither_package_is_importable():
    """The positive path, in the environment the contract mandates."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_CLEAN_RUN)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "OK" in result.stdout


# --- C. provenance, per figure -------------------------------------------------------------


def test_the_environment_records_cpu_cores_python_and_libraries():
    recorded = measure().environment()
    for key in PROVENANCE_KEYS:
        assert key in recorded, key
    assert isinstance(recorded["cpu_cores"], int) and recorded["cpu_cores"] >= 1
    assert recorded["python_version"].startswith(str(sys.version_info.major))
    assert recorded["library_versions"], "at least the libraries a figure depended on"


def test_a_figure_cannot_exist_without_its_provenance():
    """§8: *every* numeric figure carries provenance, so it is a required field."""
    module = measure()
    with pytest.raises(TypeError):
        module.Figure(value=1.0, unit="bytes")


def test_a_figure_carries_a_value_a_unit_and_an_environment():
    module = measure()
    figure = module.Figure(value=1.0, unit="bytes", environment=module.environment())
    assert figure.value == 1.0
    assert figure.unit == "bytes"
    for key in PROVENANCE_KEYS:
        assert key in figure.environment


def test_every_serialized_figure_carries_provenance(harness):
    """Not merely once at the top level: each figure, as §3 and §8 require."""
    harness.run()
    published = list(figures(harness.written()))
    assert published, "the report published no figure at all"
    for path, figure in published:
        for key in PROVENANCE_KEYS:
            assert key in figure.get("environment", {}), f"{path} is missing {key}"


def test_a_non_finite_measured_value_is_refused():
    """A NaN or an infinity describes nothing and must never reach the report."""
    module = measure()
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(Exception):
            module.Figure(value=bad, unit="seconds", environment=module.environment())


def test_an_incomplete_provenance_is_refused():
    module = measure()
    with pytest.raises(Exception):
        module.Figure(value=1.0, unit="bytes", environment={"cpu_model": "only this"})


# --- D. peak RSS ----------------------------------------------------------------------------


def test_the_rss_backend_is_the_platform_one():
    module = measure()
    expected = module.WINDOWS_RSS_BACKEND if os.name == "nt" else module.POSIX_RSS_BACKEND
    assert module.rss_backend() == expected


def test_the_two_backend_names_are_distinct_and_descriptive():
    module = measure()
    assert module.WINDOWS_RSS_BACKEND != module.POSIX_RSS_BACKEND
    assert "getprocessmemoryinfo" in module.WINDOWS_RSS_BACKEND.lower()
    assert "getrusage" in module.POSIX_RSS_BACKEND.lower()


def test_peak_rss_is_a_positive_byte_count():
    value = measure().peak_rss_bytes()
    assert isinstance(value, int)
    assert value > 0, "a zero peak RSS would read as 'no memory used'"


def test_peak_rss_is_reported_in_bytes_not_kilobytes():
    """`ru_maxrss` is kilobytes on Linux; the reported unit is bytes either way."""
    assert measure().peak_rss_bytes() > 1_000_000, "a live Python process exceeds 1MB"


def test_an_unsupported_platform_refuses_rather_than_returning_zero(monkeypatch):
    module = measure()
    monkeypatch.setattr(os, "name", "somethingelse")
    with pytest.raises(Exception):
        module.rss_backend()


def test_the_harness_measures_rss_without_psutil_or_tracemalloc():
    text = source()
    for banned in ("psutil", "tracemalloc"):
        assert banned not in text, banned
    assert "getrusage" in text
    assert "GetProcessMemoryInfo" in text


def test_every_memory_figure_names_its_backend(harness):
    """§4: two backends do not measure quite the same thing, so a figure says which."""
    report = harness.run()
    memory = [
        (path, figure)
        for path, figure in figures(harness.written())
        if figure["unit"] == "bytes" and "rss" in path.lower()
    ]
    assert memory, "no RSS figure was published"
    for path, figure in memory:
        assert figure.get("rss_backend") == report.rss_backend, path


# --- E. the five categories, kept apart ------------------------------------------------------


def test_the_report_publishes_all_five_categories(harness):
    harness.run()
    written = harness.written()
    for key in CATEGORIES:
        assert key in written, key


def test_throughput_index_build_and_latency_are_never_combined(harness):
    """§6: B, D and E are separate structures, so no reader can conflate them."""
    harness.run()
    written = harness.written()
    assert set(written["embedding_throughput"]) == {str(size) for size in BATCH_SIZES}
    assert set(written["index_build"]) == {str(size) for size in SMALL_POPULATIONS}
    assert "value" in written["query_latency"]


def test_throughput_is_measured_at_every_batch_size(harness):
    report = harness.run()
    assert set(report.embedding_throughput) == set(BATCH_SIZES)
    for size in BATCH_SIZES:
        figure = report.embedding_throughput[size]
        assert figure.unit == "records_per_second"
        assert figure.value > 0


def test_the_throughput_denominator_is_explicit(harness):
    """§11.E: records and seconds are both recorded, not only their quotient."""
    harness.run()
    written = harness.written()
    for size in BATCH_SIZES:
        figure = written["embedding_throughput"][str(size)]
        assert figure["records"] > 0
        assert figure["seconds"] > 0
        assert figure["batch_size"] == size


def test_index_build_reports_both_time_and_peak_memory(harness):
    """§2.D is four figures: time and memory, at each of two populations."""
    report = harness.run()
    assert set(report.index_build) == set(SMALL_POPULATIONS)
    for population, block in report.index_build.items():
        assert block["seconds"].unit == "seconds", population
        assert block["peak_rss_bytes"].unit == "bytes", population


def test_query_latency_is_measured_in_seconds(harness):
    report = harness.run()
    assert report.query_latency.unit == "seconds"
    assert report.query_latency.value > 0


# --- F. artifact sizes and absence -----------------------------------------------------------


def test_an_existing_artifact_is_measured_at_its_actual_byte_size(harness):
    directory = harness.artifact("nyc311_sla_risk")
    expected = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    report = harness.run()
    measured = report.artifact_sizes["nyc311_sla_risk"]
    assert measured.value == expected
    assert measured.unit == "bytes"


def test_a_missing_artifact_is_never_serialized_as_zero_bytes(harness):
    """§5: absence is never zero — zero is a measurement, absence is not."""
    harness.run()
    entry = harness.written()["artifact_sizes"]["nyc311_sla_risk"]
    assert entry.get("value") != 0
    assert entry.get("status") == "absent"


def test_measured_and_absent_artifacts_are_structurally_distinguishable(harness):
    harness.artifact("nyc311_sla_risk")
    harness.run()
    sizes = harness.written()["artifact_sizes"]
    present, absent = sizes["nyc311_sla_risk"], sizes["cfpb_triage_tfidf"]
    assert "value" in present and present.get("status") != "absent"
    assert absent.get("status") == "absent" and "value" not in absent


def test_a_zero_byte_artifact_is_measured_as_zero_not_absent(harness):
    """The distinction cuts both ways: an empty file was measured, not missing."""
    harness.artifact("nyc311_sla_risk", payload=b"")
    (harness.artifact_root / "nyc311" / "nyc311_sla_risk" / "v1" / "metadata.json").write_bytes(b"")
    harness.run()
    entry = harness.written()["artifact_sizes"]["nyc311_sla_risk"]
    assert entry.get("status") != "absent"
    assert entry["value"] == 0


def test_every_expected_class_appears_in_the_report(harness):
    report = harness.run()
    assert set(report.artifact_sizes) == set(measure().ARTIFACT_CLASSES)


def test_the_artifact_root_is_required_and_keyword_only():
    """§5: caller-supplied. A harness that guessed one could measure the repository."""
    parameter = inspect.signature(measure().run_measurements).parameters["artifact_root"]
    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_the_harness_triggers_no_training_run():
    """§5: no hidden setup. Nothing here fits, trains or writes an artifact."""
    text = source()
    for banned in ("run_experiment", "run_benchmark", "run_probe", "write_artifact", ".fit("):
        assert banned not in text, banned


def test_a_malformed_artifact_root_is_refused(harness, tmp_path):
    """A file where a directory belongs is a caller error, not an empty measurement."""
    harness.prepare()
    harness.artifact_root = tmp_path / "root.txt"
    harness.artifact_root.write_text("not a directory", encoding="utf-8")
    with pytest.raises(Exception):
        harness.run()
    assert not harness.report_path.exists()


# --- G. synthetic index input -----------------------------------------------------------------


def test_synthetic_vectors_are_float32_and_the_requested_shape():
    vectors = measure().synthetic_vectors(5, 7)
    assert vectors.dtype == np.float32
    assert vectors.shape == (5, 7)


def test_synthetic_vectors_are_deterministic():
    assert np.array_equal(measure().synthetic_vectors(16, 7), measure().synthetic_vectors(16, 7))


def test_synthetic_vectors_ignore_global_random_state():
    """§14: a seeded generator, never the global one."""
    module = measure()
    np.random.seed(1)
    first = module.synthetic_vectors(16, 7)
    np.random.seed(99_999)
    second = module.synthetic_vectors(16, 7)
    assert np.array_equal(first, second)


def test_synthetic_refs_are_record_refs_and_distinct():
    references = measure().synthetic_refs(32)
    assert len(references) == 32
    assert all(isinstance(reference, RecordRef) for reference in references)
    assert len(set(references)) == 32, "build_index refuses a repeated reference"


def test_synthetic_refs_are_deterministic():
    assert measure().synthetic_refs(8) == measure().synthetic_refs(8)


def test_an_invalid_dimension_is_refused():
    module = measure()
    for bad in (0, -1):
        with pytest.raises(Exception):
            module.synthetic_vectors(4, bad)


def test_the_dimension_comes_from_the_embedder_not_a_literal(harness):
    """§6: observed, never hard-coded. The fake's width is not 384."""
    report = harness.run()
    assert report.embedding_dimension == harness.embedder.embedding_dimension
    assert report.embedding_dimension != 384


def test_no_embedding_dimension_literal_appears_in_the_harness():
    assert "384" not in source(), "D18: the dimension is observed, never assumed"


def test_the_harness_reads_no_corpus():
    """§6: the mandated environment cannot load one, and index cost is not corpus cost."""
    text = source()
    for banned in ("load_corpus", "load_outcomes", "ingest.manifest", "ingest.storage"):
        assert banned not in text, banned


# --- H. the index measurements ------------------------------------------------------------------


def test_the_harness_uses_the_extracted_index_implementation():
    """Identity, so neither a fake nor a copy can stand in for it."""
    assert measure().build_index is index_module.build_index


def test_the_harness_defines_no_index_of_its_own():
    defined = {
        node.name
        for node in ast.parse(source()).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    assert defined.isdisjoint({"build_index", "RetrievalIndex", "cosine_similarity"})
    assert "argsort" not in source(), "ranking belongs to the index, not to the harness"


def test_vector_generation_is_excluded_from_the_build_timing(harness, monkeypatch):
    """§9: vectors are generated *before* timing starts.

    A generator that sleeps a quarter second would show up in the build figure if
    the timing bracketed it, so the reported seconds must bound the sleep out.
    """
    module = harness.prepare()
    original = module.synthetic_vectors

    def slow(count, dimension):
        time.sleep(0.25)
        return original(count, dimension)

    monkeypatch.setattr(module, "synthetic_vectors", slow)
    report = harness.run()
    for population, block in report.index_build.items():
        assert block["seconds"].value < 0.25, population


def test_the_query_latency_is_measured_against_the_built_index(harness, monkeypatch):
    """Not a fabricated number: a real `rank` call on a real index."""
    ranked = []
    original = index_module.RetrievalIndex.rank

    def spy(self, vector):
        ranked.append(len(self.refs))
        return original(self, vector)

    monkeypatch.setattr(index_module.RetrievalIndex, "rank", spy)
    harness.run()
    assert ranked, "no query was ever executed"
    assert max(ranked) == max(SMALL_POPULATIONS)


def test_an_invalid_batch_size_is_refused(harness, monkeypatch):
    module = harness.prepare()
    monkeypatch.setattr(module, "BATCH_SIZES", (0,))
    with pytest.raises(Exception):
        harness.run()
    assert not harness.report_path.exists()


# --- I. the MiniLM measurements ------------------------------------------------------------------


def test_the_load_figure_is_a_peak_rss_in_bytes(harness):
    report = harness.run()
    assert report.minilm_load["peak_rss_bytes"].unit == "bytes"
    assert report.minilm_load["peak_rss_bytes"].value > 0


def test_the_load_measurement_takes_no_population_or_batch_argument():
    """§11.D: load cost cannot accidentally include benchmark generation work."""
    parameters = set(inspect.signature(measure().measure_minilm_load).parameters)
    assert not parameters & {"population", "populations", "batch_size", "batch_sizes", "vectors"}


def test_no_synthetic_vector_is_generated_while_the_load_is_measured(harness, monkeypatch):
    module = harness.prepare()
    generated = []
    monkeypatch.setattr(
        module, "synthetic_vectors", lambda count, dimension: generated.append(count)
    )
    module.measure_minilm_load()
    assert not generated, "vector generation leaked into the load measurement"


def test_the_report_records_the_embedder_identity(harness):
    report = harness.run()
    assert report.embedding_model_id == harness.embedder.embedding_model_id
    assert report.model_version == harness.embedder.model_version


def test_the_real_embedder_is_loaded_through_the_pinned_loader():
    """One test against the real thing, so the fakes cannot hide a wrong loader."""
    require_real_assets()
    from ml.embedders import minilm

    embedder = measure().load_embedder()
    assert embedder.embedding_dimension == minilm.load_minilm().embedding_dimension
    assert embedder.model_version == minilm.MODEL_VERSION


# --- J. report persistence ------------------------------------------------------------


def test_the_report_is_written_to_the_caller_supplied_path(harness):
    harness.run()
    assert harness.report_path.is_file()


def test_no_default_report_path_is_invented():
    """§8: nothing is written inside the repository by default."""
    parameter = inspect.signature(measure().run_measurements).parameters["report_path"]
    assert parameter.default is inspect.Parameter.empty or parameter.default is None


def test_the_report_is_valid_json(harness):
    harness.run()
    assert isinstance(harness.written(), dict)


def test_the_report_object_serializes_to_what_was_written(harness):
    report = harness.run()
    assert json.loads(report.as_json()) == harness.written()


def test_the_write_leaves_no_temporary_file_behind(harness):
    harness.run()
    leftovers = [
        entry.name for entry in harness.report_path.parent.iterdir() if entry != harness.report_path
    ]
    assert not leftovers, leftovers


def test_a_serialization_failure_leaves_no_report(harness, monkeypatch):
    harness.prepare()
    monkeypatch.setattr(json, "dumps", boom)
    with pytest.raises(Exception):
        harness.run()
    assert not harness.report_path.exists()


def test_the_report_contains_only_the_figures_this_run_produced(harness):
    """§8: no hand-entered value, and no figure carried over from another run."""
    report = harness.run()
    written = harness.written()
    assert set(written["index_build"]) == {str(size) for size in SMALL_POPULATIONS}
    assert set(written["embedding_throughput"]) == {
        str(size) for size in report.embedding_throughput
    }


@pytest.mark.parametrize("stage", STAGES)
def test_a_failure_at_any_stage_leaves_no_partial_report(harness, monkeypatch, stage):
    """§13's whole-or-nothing rule, exercised stage by stage."""
    module = harness.prepare()
    monkeypatch.setattr(module, stage, boom)
    with pytest.raises(Exception):
        harness.run()
    assert not harness.report_path.exists(), f"{stage} left a partial report"
    if harness.report_path.parent.exists():
        assert not list(harness.report_path.parent.iterdir()), "a temporary file survived"


# --- K. reproducibility --------------------------------------------------------------


def test_the_artifact_size_set_does_not_depend_on_filesystem_order(harness, tmp_path):
    """§14: the reported classes come from the declared tuple, not from a scandir."""
    harness.artifact("cfpb_triage_tfidf")
    harness.artifact("nyc311_sla_risk")
    first = harness.run()
    harness.report_path = tmp_path / "again.json"
    second = harness.run()
    assert list(first.artifact_sizes) == list(second.artifact_sizes)
    assert list(first.artifact_sizes) == list(measure().ARTIFACT_CLASSES)


def test_a_rerun_does_not_change_what_was_measured(harness, tmp_path):
    """§14: provenance may carry a timestamp; measurements never depend on one."""
    first = harness.run()
    harness.report_path = tmp_path / "second.json"
    second = harness.run()
    assert first.embedding_dimension == second.embedding_dimension
    assert set(first.index_build) == set(second.index_build)
    assert set(first.artifact_sizes) == set(second.artifact_sizes)


def test_the_batch_sizes_are_not_reordered(harness):
    report = harness.run()
    assert list(report.embedding_throughput) == list(measure().BATCH_SIZES)


# --- L. the dependency rule ----------------------------------------------------------


def test_the_harness_imports_no_forbidden_package():
    imported = set()
    for node in ast.walk(ast.parse(source())):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    for banned in ("pandas", "pyarrow", "psutil", "django"):
        assert banned not in imported, banned
        assert not any(name.startswith(f"{banned}.") for name in imported), banned


def test_the_harness_source_names_no_banned_mechanism():
    text = source()
    for banned in BANNED_IN_SOURCE:
        assert banned not in text, banned


# --- M. guarding this suite against itself ---------------------------------------------


def test_no_raises_block_in_this_file_imports_the_harness():
    """A RED-phase trap, closed: `ModuleNotFoundError` is an `Exception`.

    Six of these tests originally called `measure()` *inside* their
    `pytest.raises(Exception)` block, so the absent module satisfied the very
    assertion that was supposed to pin a refusal -- they passed against no
    implementation at all, and would keep passing against one that refused for
    entirely the wrong reason. Every such call is now hoisted above the block, and
    this test walks the file's own AST to keep it that way.
    """
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        raising = any(
            isinstance(item.context_expr, ast.Call)
            and getattr(item.context_expr.func, "attr", None) == "raises"
            for item in node.items
        )
        if not raising:
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and getattr(inner.func, "id", None) == "measure":
                offenders.append(node.lineno)
    assert not offenders, f"measure() called inside a pytest.raises block at lines {offenders}"

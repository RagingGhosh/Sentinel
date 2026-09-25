"""Task 20 slice 1: the retrieval index, separated from corpus and Parquet I/O.

`ml/training/index.py` holds four definitions that were Task 18's, moved verbatim
out of `ml/training/experiments/dedup.py` so that an environment installed from
`requirements/ml.txt` alone can import them (D40.1). The benchmark still owns
them behaviourally: `dedup.py` imports them back and re-exports them, and Task
18's suite passes unedited.

What these tests pin, in order:

    A, B, C  the module imports with pandas blocked, with pyarrow blocked, and
             with both blocked -- and the primitives still work under the block
    D        cosine_similarity against hand-checked vectors
    E, F     RetrievalIndex construction and build_index's binding
    G        every refusal, with its exception type and message unchanged
    H, I     dedup's re-exports are the same objects, including through the
             dynamic import Task 18's own tests use

The blocked-import cases run in a subprocess. Blocking a package inside this
process would mean purging `ml.*` and `ingest.*` from `sys.modules` while other
tests hold references to them, which would make this file's verdict depend on
collection order. A subprocess gives the same answer whatever ran before it.

Deliberately not duplicated here: Task 18's benchmark suite. `recall_at_k`,
`random_ranking_baseline`, the perturbations and the report stay in
`tests/ml/training/test_dedup_benchmark.py`, unchanged.
"""

import ast
import importlib
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="numpy lives in requirements/ml.txt")
pytest.importorskip("onnxruntime", reason="onnxruntime lives in requirements/ml.txt")

from ingest.identity import RecordRef  # noqa: E402
from ml.embedders import minilm  # noqa: E402
from ml.training.index import (  # noqa: E402
    RetrievalIndex,
    _require_record_refs,
    build_index,
    cosine_similarity,
)

pytestmark = pytest.mark.ml

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "ml" / "training" / "index.py"

EXTRACTED = ("cosine_similarity", "RetrievalIndex", "build_index", "_require_record_refs")

FORBIDDEN_IMPORTS = (
    "ingest.manifest",
    "ingest.storage",
    "pandas",
    "pyarrow",
    "django",
    "ml.registry",
    "ml.null",
)
"""Contract §3. `ingest.manifest` is the one that made the extraction necessary:
it imports `ingest.storage`, which imports `pyarrow`."""


def refs(count: int, source: str = "cfpb") -> list[RecordRef]:
    """Deterministic synthetic identities, distinct by construction."""
    return [RecordRef(source=source, external_id=f"c{position:05d}") for position in range(count)]


def unit_rows(count: int, width: int) -> "np.ndarray":
    """One-hot rows: unit length, so cosine is the dot product and ranking is exact."""
    matrix = np.zeros((count, width), dtype=np.float32)
    for position in range(count):
        matrix[position, position % width] = 1.0
    return matrix


# --- A, B, C. the import boundary the extraction exists for ----------------------------

_UNDER_BLOCK = """
import sys

BLOCKED = {blocked!r}


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(name + " is not installed")
        return None


sys.meta_path.insert(0, Blocker())

import importlib

module = importlib.import_module("ml.training.index")

# The point is not merely that it imported: exercise it.
import numpy as np
from ingest.identity import RecordRef

references = [RecordRef(source="cfpb", external_id="c%05d" % i) for i in range(4)]
vectors = np.eye(4, dtype=np.float32)
index = module.build_index(references, vectors)
assert index.dimension == 4
assert index.rank(vectors[2])[0] == references[2]
assert module.cosine_similarity(vectors, vectors).diagonal().tolist() == [1.0, 1.0, 1.0, 1.0]

leaked = sorted(name for name in BLOCKED if name in sys.modules)
assert not leaked, "blocked package reached sys.modules: %s" % leaked
print("OK")
"""


def _import_under_block(*blocked: str) -> None:
    """Import and exercise `ml.training.index` in a process where `blocked` is absent."""
    script = textwrap.dedent(_UNDER_BLOCK).format(blocked=set(blocked))
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"blocking {sorted(blocked)} broke the import or the primitives:\n"
        f"{result.stdout}\n{result.stderr}"
    )
    assert "OK" in result.stdout


def test_the_index_imports_and_works_with_pandas_blocked():
    _import_under_block("pandas")


def test_the_index_imports_and_works_with_pyarrow_blocked():
    """The binding case: `pyarrow` is what `ingest.storage` pulls in."""
    _import_under_block("pyarrow")


def test_the_index_imports_and_works_with_both_blocked():
    """The environment Task 20 mandates: `requirements/ml.txt` and nothing more."""
    _import_under_block("pandas", "pyarrow")


def test_the_module_imports_nothing_from_the_corpus_layer():
    """Checked statically, so the verdict does not depend on what is installed.

    The dynamic tests above prove the module loads without the train tier; this
    one says *why* it does, and fails on the specific import that would undo the
    extraction rather than on a later ImportError.
    """
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    for forbidden in FORBIDDEN_IMPORTS:
        assert forbidden not in imported, forbidden
        assert not any(name.startswith(f"{forbidden}.") for name in imported), forbidden


# --- D. cosine_similarity ----------------------------------------------------------------


def test_cosine_similarity_is_the_dot_product_on_unit_rows():
    queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    candidates = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=np.float32)
    scores = cosine_similarity(queries, candidates)
    assert scores.shape == (2, 3)
    assert scores[0].tolist() == pytest.approx([1.0, 0.0, 0.6])
    assert scores[1].tolist() == pytest.approx([0.0, 1.0, 0.8])


def test_cosine_similarity_does_not_renormalise_its_inputs():
    """A rescaled row scores proportionally: silently renormalising here would
    hide an arm that stopped normalising (the docstring's own reason)."""
    doubled = np.array([[2.0, 0.0]], dtype=np.float32)
    candidates = np.array([[1.0, 0.0]], dtype=np.float32)
    assert cosine_similarity(doubled, candidates)[0][0] == pytest.approx(2.0)


def test_cosine_similarity_returns_float32():
    scores = cosine_similarity(np.eye(2, dtype=np.float64), np.eye(2, dtype=np.float64))
    assert scores.dtype == np.float32


# --- E, F. the index ----------------------------------------------------------------------


def test_a_retrieval_index_records_the_width_it_was_built_with():
    index = build_index(refs(3), unit_rows(3, 5))
    assert isinstance(index, RetrievalIndex)
    assert index.dimension == 5
    assert index.refs == tuple(refs(3))
    assert index.vectors.dtype == np.float32
    assert index.similarity is cosine_similarity


def test_build_index_ranks_every_candidate_most_similar_first():
    references = refs(4)
    vectors = np.eye(4, dtype=np.float32)
    index = build_index(references, vectors)
    ranking = index.rank(vectors[2])
    assert len(ranking) == 4, "every candidate is ranked, not only the top one"
    assert ranking[0] == references[2]
    assert set(ranking) == set(references)


def test_ties_resolve_stably_to_the_earlier_candidate():
    """`argsort(kind="stable")` is load-bearing: equal scores keep input order."""
    references = refs(3)
    identical = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (3, 1))
    index = build_index(references, identical)
    assert index.rank(np.array([1.0, 0.0], dtype=np.float32)) == tuple(references)


def test_a_caller_supplied_similarity_is_the_one_used():
    """The benchmark passes the function its shared config governs, not a default."""
    calls = []

    def reversed_similarity(queries, candidates):
        calls.append((queries.shape, candidates.shape))
        return -cosine_similarity(queries, candidates)

    references = refs(3)
    vectors = np.eye(3, dtype=np.float32)
    index = build_index(references, vectors, similarity=reversed_similarity)
    assert index.similarity is reversed_similarity
    assert index.rank(vectors[0])[-1] == references[0]
    assert calls, "the supplied similarity was never called"


# --- G. the refusals, with their types and messages unchanged ------------------------------


def test_a_non_two_dimensional_matrix_is_refused():
    with pytest.raises(ValueError, match=r"expected a 2-D embedding matrix, got shape"):
        build_index(refs(2), np.zeros(2, dtype=np.float32))


def test_a_length_mismatch_between_refs_and_vectors_is_refused():
    with pytest.raises(ValueError, match="every candidate must carry exactly one vector"):
        build_index(refs(3), unit_rows(2, 4))


def test_a_positional_identity_is_refused():
    """Identity is a `RecordRef` throughout, never a position in some array."""
    with pytest.raises(ValueError, match="must hold RecordRef values, got int"):
        build_index([0, 1], unit_rows(2, 4))


def test_a_repeated_reference_is_refused_rather_than_de_duplicated():
    repeated = [*refs(2), RecordRef(source="cfpb", external_id="c00000")]
    with pytest.raises(ValueError, match="appears more than once in the candidate population"):
        build_index(repeated, unit_rows(3, 4))


def test_a_query_of_the_wrong_width_raises_the_embedders_own_error():
    """The exception object is `ml.embedders.minilm`'s, not a local equivalent.

    Read off the module rather than captured at import time, because
    `tests/ml/embedders/test_minilm.py` reloads it and a captured class would then
    be a stale object. That is the same reason `RetrievalIndex.rank` resolves it
    on the module, and asserting it this way is what proves the two agree.
    """
    index = build_index(refs(2), unit_rows(2, 4))
    with pytest.raises(minilm.EmbeddingDimensionMismatch) as caught:
        index.rank(np.zeros(3, dtype=np.float32))
    assert "this index was built with embeddings of width 4" in str(caught.value)
    assert "never broadcast, truncated or padded" in str(caught.value)


_AFTER_RELOAD = """
import importlib

import numpy as np

from ingest.identity import RecordRef

index_module = importlib.import_module("ml.training.index")
minilm = importlib.import_module("ml.embedders.minilm")

# What tests/ml/embedders/test_minilm.py does, and what broke the extraction:
# a reload rebinds the module's exception class to a brand-new object.
before = minilm.EmbeddingDimensionMismatch
importlib.reload(minilm)
assert minilm.EmbeddingDimensionMismatch is not before, "the reload did not rebind the class"

references = [RecordRef(source="cfpb", external_id="c%05d" % i) for i in range(2)]
index = index_module.build_index(references, np.eye(2, dtype=np.float32))
try:
    index.rank(np.zeros(3, dtype=np.float32))
except minilm.EmbeddingDimensionMismatch as caught:
    assert "never broadcast, truncated or padded" in str(caught)
    print("OK")
else:
    raise AssertionError("a wrong-width query did not raise")
"""


def test_the_width_error_is_the_class_the_module_currently_exposes():
    """The regression this slice actually hit, pinned.

    `ml/embedders/minilm.py` is reloaded by its own test suite, which rebinds
    `EmbeddingDimensionMismatch` to a new class object. A version of `rank` that
    captured the class at import time raised the stale one, so `except
    EmbeddingDimensionMismatch` read off the module missed it -- and three Task 18
    tests failed in the full run while passing in isolation. Run in a subprocess
    because a reload here would leak into every test after it.
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_AFTER_RELOAD)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "OK" in result.stdout


def test_the_record_ref_guard_names_what_it_was_checking():
    with pytest.raises(ValueError, match="the expected originals must hold RecordRef values"):
        _require_record_refs(["not a ref"], "the expected originals")


# --- H, I. dedup's re-exports are the same objects -----------------------------------------


def test_dedup_re_exports_the_extracted_implementations():
    """One implementation, in one place, reached from both callers (D40.1)."""
    from ml.training import index as extracted
    from ml.training.experiments import dedup

    assert dedup.cosine_similarity is extracted.cosine_similarity
    assert dedup.RetrievalIndex is extracted.RetrievalIndex
    assert dedup.build_index is extracted.build_index
    assert dedup._require_record_refs is extracted._require_record_refs


def test_the_dynamically_imported_dedup_module_still_exposes_them():
    """Task 18's own tests reach every symbol as an attribute of this module."""
    module = importlib.import_module("ml.training.experiments.dedup")
    for name in EXTRACTED:
        assert hasattr(module, name), name
    index = module.build_index(refs(2), unit_rows(2, 6))
    assert isinstance(index, module.RetrievalIndex)
    assert index.dimension == 6


def test_the_benchmark_still_defines_its_own_metric_and_baseline():
    """Only four symbols moved. The metric and its baseline stay with Task 18."""
    source = (ROOT / "ml" / "training" / "experiments" / "dedup.py").read_text(encoding="utf-8")
    defined = {
        node.name
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    assert "recall_at_k" in defined
    assert "random_ranking_baseline" in defined
    assert defined.isdisjoint(EXTRACTED), "a second implementation survives in dedup.py"


def test_the_four_definitions_live_only_in_the_extracted_module():
    """No copy anywhere: the extraction replaced the originals, it did not clone them."""
    index_defined = {
        node.name
        for node in ast.parse(MODULE.read_text(encoding="utf-8")).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    assert index_defined == set(EXTRACTED)
